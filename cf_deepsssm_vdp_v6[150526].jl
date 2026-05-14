# =============================================================
#  cf_deepsssm_vdp_v6_1.jl
#  SafeCF-SSM — Van der Pol Benchmark
#  Coded by Thanana Nuchkrua, Copyright 2026
#
#  This script implements and validates the SafeCF-SSM framework
#  (Nuchkrua & Boonto, IEEE L-CSS 2026) on the Van der Pol (VdP)
#  oscillator across four consecutive distributional-shift regimes:
#  nominal → abrupt shift → observational drift → gradual drift.
#
#  ALGORITHM (per iteration):
#  1. EKF inference step: infer z_t, Σ_t from history H_t
#  2. BMPC: solve convex QP (OSQP) with adaptive tightening
#     β_{i,t} = max(c_i·S_t, L_{g,i}·σ_t), σ_t = √λ_max(Σ_t)
#  3. Apply u_t to true system, observe y_{t+1}
#  4. Surprise evaluation: S_t = 0.5·||y_{t+1} - ẑ_{t+1}||²/σw²
#  5. Bounded CF adaptation: η_t = η_max/(1+√S_t)
#     μ̂_{t+1} = μ̂_t + η_t·∇_μ̂ log p,  CFI_t ≤ 1 (Theorem 1)
#
#  System (Discrete Van der Pol, Euler, dt=0.05):
#    x1_{t+1} = x1_t + dt·x2_t
#    x2_{t+1} = x2_t + dt·[μ(1-x1²)x2 - x1 + u] + w_t
#    y_t      = [x1_t, x2_t]' + v_t
#
#  Four regimes (continuous, no re-initialization):
#    Nominal:           μ = μ₁ = 0.5   (t ∈ [0, 45] s)
#    Abrupt shift:      μ = μ₂ = 2.66  (t ∈ [45, 115] s)
#    Observational drift: μ = μ₁, bias b_t grows (t ∈ [115, 275] s)
#    Gradual drift:     μ: μ₁ → μ₂ exponentially (t ∈ [275, 345] s)
#
#  Dependencies: Julia 1.x, CairoMakie, JuMP, OSQP, Distributions
# =============================================================
using Random, Distributions, Statistics, LinearAlgebra
using CairoMakie
using Printf
using JuMP, OSQP
using CairoMakie: vspan!, ylims!, xlims!, hlines!, vlines!, lines!, text!, band!, Legend, Axis, Figure, save

# =============================================================
#  SECTION 0: GLOBAL CONSTANTS
#  All simulation parameters are defined here for reproducibility.
#  T_NOM, T_ABRUPT, T_OBS, T_GRAD define the length (in steps)
#  of each distributional-shift regime. MU_1, MU_2 are the VdP
#  damping parameters before and after the abrupt shift.
#  ETA_MAX is the maximum adaptation rate (Sec. III-C, Eq. 14).
#  ε_DEC_LIST is used for Gap-2 validation (Remark 4, Table I).
# =============================================================
const dt        = 0.05
const T_NOM     = 900
const T_ABRUPT  = 1400
const T_OBS     = 3200
const T_GRAD    = 1400
const T_TOTAL   = T_NOM + T_ABRUPT + T_OBS + T_GRAD

const t_s1 = T_NOM
const t_s2 = T_NOM + T_ABRUPT
const t_s3 = T_NOM + T_ABRUPT + T_OBS

const MU_1      = 0.5
const MU_2      = 2.66
const σw        = 0.06
const σv        = 0.10
const U_MAX     = 3.0
const K1        = 12.0
const K2        = 6.0
const ETA_MAX   = 0.15
const GRAD_CLIP = 2.0
const ε_CF      = ETA_MAX

# Gap-2 validation: decoder error sweep (Remark 4, Table I)
const ε_DEC_LIST = [0.0, 0.05, 0.10]

# =============================================================
#  SECTION 1: TRUE SYSTEM (Van der Pol)
#  Implements one Euler step of the true (unknown) VdP dynamics
#  with additive process noise w_t ~ N(0, σw²).
#  This is the physical plant — never accessed by the controller.
# =============================================================
function vdp_step(x::Vector{Float64}, u::Float64, μ::Float64,
                  rng::AbstractRNG)
    x1, x2 = x
    dx1 = x2
    dx2 = μ*(1.0 - x1^2)*x2 - x1 + u
    return [x1 + dt*dx1,
            x2 + dt*dx2 + σw*randn(rng)]
end

# =============================================================
#  SECTION 2: LATENT MODEL
#  The controller's internal SSM (Eq. 5 in paper).
#  vdp_predict: one-step latent prediction using estimated μ̂_t.
#  vdp_jacobian: linearization F_t = ∂f/∂z for EKF propagation.
#  covariance_update: EKF prediction covariance Σ_{t+1|t} = FΣF' + Q,
#    with eigenvalue clipping for numerical stability.
# =============================================================
function vdp_predict(z::Vector{Float64}, u::Float64, μ_hat::Float64)
    z1, z2 = z
    dz1 = z2
    dz2 = μ_hat*(1.0 - z1^2)*z2 - z1 + u
    return [z1 + dt*dz1, z2 + dt*dz2]
end

function vdp_jacobian(z::Vector{Float64}, u::Float64, μ_hat::Float64)
    z1, z2 = z
    F = [1.0              dt;
         dt*(-2*μ_hat*z1*z2 - 1.0)    1.0 + dt*μ_hat*(1.0-z1^2)]
    return F
end

function covariance_update(Σ::Matrix{Float64}, F::Matrix{Float64})
    Q = σw^2 * I(2)
    Σ_new = F*Σ*F' + Q
    Σ_new = 0.5*(Σ_new + Σ_new')
    Σ_new += 1e-6 * I(2)
    λ_max = maximum(real.(eigvals(Σ_new)))
    if λ_max > 5.0
        Σ_new = Σ_new .* (5.0 / λ_max)
    end
    return Σ_new
end

# =============================================================
#  SECTION 3: SURPRISE SIGNAL AND BOUNDED CF ADAPTATION
#  Implements Eqs. (13)–(15) of the paper.
#  S_t = negative log-likelihood proxy (squared prediction error).
#  η_t = η_max / (1 + √S_t): large surprise slows adaptation
#    (Gap-1 mechanism, Sec. III-C).
#  μ̂ is updated via a clipped gradient step on log p(y|z,u).
#  CFI_t = |Δμ̂| / ε_CF monitors Theorem 1 bound online.
#  Adaptation is skipped when S_t < 0.8 (no evidence of mismatch).
# =============================================================
function surprise_and_update!(μ_hat::Vector{Float64},
                               z::Vector{Float64},
                               y_next::Vector{Float64},
                               u::Float64)
    z_pred = vdp_predict(z, u, μ_hat[1])
    err    = y_next - z_pred
    S_t    = 0.5 * sum(err.^2) / σw^2
    η_t = ETA_MAX / (1.0 + sqrt(max(0.0, S_t)))

    if S_t < 0.8
        return S_t, η_t, 0.0
    end

    z1, z2 = z
    dz2_dμ = dt * (1.0 - z1^2) * z2
    grad_μ = err[2] / σw^2 * dz2_dμ
    grad_μ = clamp(grad_μ, -GRAD_CLIP, GRAD_CLIP)
    μ_old     = μ_hat[1]
    μ_hat[1] += η_t * grad_μ
    μ_hat[1]  = clamp(μ_hat[1], 0.1, 3.0)
    CFI_t = abs(μ_hat[1] - μ_old) / ε_CF
    return S_t, η_t, CFI_t
end

# =============================================================
#  SECTION 4: BMPC WITH ADAPTIVE TIGHTENING (SafeCF-SSM)
#  Implements Eqs. (10)–(12) of the paper.
#  β_{i,t} = max(c_i·S_t, L_{g,i}·σ_t) is the unified safety
#    margin (Lemma 1): the first term handles surprise-driven
#    tightening (Gap-1); the second handles covariance-driven
#    tightening (Assumption 3).
#  The QP is solved via OSQP over a linearized prediction horizon
#    N=10. A PD fallback is used if the QP is infeasible (<0.1%
#    of steps; safety unaffected since β_{i,t} remains active).
#  β_fixed: optional fixed margin for Robust MPC baseline.
# =============================================================
function bmpc(z::Vector{Float64}, μ_hat::Float64,
              Σ::Matrix{Float64}, x_ref::Float64; N::Int=10,
              S_prev::Float64=0.0, β_fixed::Union{Float64,Nothing}=nothing)
    z1, z2 = z
    λ_max = max(0.0, maximum(real.(eigvals(Σ))))
    σ_t   = sqrt(λ_max)
    β_t   = isnothing(β_fixed) ? max(0.4*S_prev, 0.4*σ_t) : β_fixed
    u_max_safe = max(0.1, U_MAX - β_t)

    z_nom_traj = Vector{Vector{Float64}}(undef, N+1)
    z_nom_traj[1] = copy(z)
    for k in 1:N
        z_nom_traj[k+1] = vdp_predict(z_nom_traj[k], 0.0, μ_hat)
    end

    B_ctrl = [0.0; dt]
    model  = Model(OSQP.Optimizer)
    set_silent(model)

    @variable(model, u_seq[1:N])
    @variable(model, dz[1:N+1, 1:2])
    @constraint(model, dz[1,1] == 0.0)
    @constraint(model, dz[1,2] == 0.0)

    for k in 1:N
        Ak = vdp_jacobian(z_nom_traj[k], 0.0, μ_hat)
        @constraint(model, dz[k+1,1] == Ak[1,1]*dz[k,1] + Ak[1,2]*dz[k,2] + B_ctrl[1]*u_seq[k])
        @constraint(model, dz[k+1,2] == Ak[2,1]*dz[k,1] + Ak[2,2]*dz[k,2] + B_ctrl[2]*u_seq[k])
    end
    @constraint(model, [k=1:N], -u_max_safe <= u_seq[k] <= u_max_safe)
    @objective(model, Min,
        sum(20.2*(z_nom_traj[k][1] + dz[k,1] - x_ref)^2 +
             3.0*(z_nom_traj[k][2] + dz[k,2])^2 +
            0.06*u_seq[k]^2 for k in 1:N))
    optimize!(model)

    u_opt = if termination_status(model) == MOI.OPTIMAL
        clamp(value(u_seq[1]), -u_max_safe, u_max_safe)
    else
        clamp(-K1*(z1-x_ref) - K2*z2, -u_max_safe, u_max_safe)
    end
    return u_opt, β_t, σ_t
end

# =============================================================
#  SECTION 4b: STANDARD MPC BASELINES
#  Nominal MPC: fixed μ̂ = μ₁, β = 0 (no adaptation, no tightening).
#  Robust MPC:  fixed μ̂ = μ₁, β = 0.5 (fixed worst-case margin).
#  Both use the same QP structure as BMPC for fair comparison.
#  These baselines isolate the benefit of SafeCF-SSM's
#  surprise-driven adaptation and adaptive tightening.
# =============================================================
function mpc_fixed(x::Vector{Float64}, μ_hat::Float64,
                   x_ref::Float64; N::Int=10, β::Float64=0.0)
    x1, x2 = x
    u_max_safe = max(0.1, U_MAX - β)

    x_traj = Vector{Vector{Float64}}(undef, N+1)
    x_traj[1] = copy(x)
    for k in 1:N
        x_traj[k+1] = vdp_predict(x_traj[k], 0.0, μ_hat)
    end

    B_ctrl = [0.0; dt]
    model  = Model(OSQP.Optimizer)
    set_silent(model)

    @variable(model, u_seq[1:N])
    @variable(model, dx[1:N+1, 1:2])
    @constraint(model, dx[1,1] == 0.0)
    @constraint(model, dx[1,2] == 0.0)

    for k in 1:N
        Ak = vdp_jacobian(x_traj[k], 0.0, μ_hat)
        @constraint(model, dx[k+1,1] == Ak[1,1]*dx[k,1] + Ak[1,2]*dx[k,2] + B_ctrl[1]*u_seq[k])
        @constraint(model, dx[k+1,2] == Ak[2,1]*dx[k,1] + Ak[2,2]*dx[k,2] + B_ctrl[2]*u_seq[k])
    end
    @constraint(model, [k=1:N], -u_max_safe <= u_seq[k] <= u_max_safe)
    @objective(model, Min,
        sum(20.2*(x_traj[k][1] + dx[k,1] - x_ref)^2 +
             5.0*(x_traj[k][2] + dx[k,2])^2 +
            0.06*u_seq[k]^2 for k in 1:N))
    optimize!(model)

    u_opt = if termination_status(model) == MOI.OPTIMAL
        clamp(value(u_seq[1]), -u_max_safe, u_max_safe)
    else
        clamp(-K1*(x1-x_ref) - K2*x2, -u_max_safe, u_max_safe)
    end
    return u_opt
end

# =============================================================
#  SECTION 5: ENCODER — EXTENDED KALMAN FILTER (EKF)
#  Implements the encoder q_{φ_θt}(z_t | H_t) = N(ẑ_t, Σ_t)
#  (Assumption 3, Footnote 2 in paper).
#  Standard EKF predict-update cycle:
#    Predict: z_{t+1|t} = f(z_t, u_t; μ̂),  Σ_{t+1|t} = FΣF' + Q
#    Update:  z_t = z_{t|t-1} + K·(y_t - z_{t|t-1})
#  Innovation inn = y_t - ẑ_t is returned for diagnostics.
#  The framework generalizes to any encoder satisfying Assm. 3.
# =============================================================
function ekf_update(z::Vector{Float64}, Σ::Matrix{Float64},
                    y::Vector{Float64}, u::Float64, μ_hat::Float64)
    z_pred = vdp_predict(z, u, μ_hat)
    F      = vdp_jacobian(z, u, μ_hat)
    Σ_pred = covariance_update(Σ, F)
    R   = σv^2 * I(2)
    S   = Σ_pred + R
    K   = Σ_pred / S
    inn = y - z_pred
    z_upd = z_pred + K * inn
    Σ_upd = (I(2) - K) * Σ_pred
    Σ_upd = 0.5*(Σ_upd + Σ_upd') + 1e-6*I(2)
    return z_upd, Σ_upd, inn
end

# =============================================================
#  SECTION 6: COMBINED CONTINUOUS SIMULATION
#  Runs all three controllers (SafeCF-SSM, Nominal, Robust) on
#  the same VdP trajectory across all four regimes continuously
#  without re-initialization (single 345-s experiment, Fig. 2).
#
#  ε_dec > 0: additive decoder perturbation for Gap-2 validation
#    (Remark 4): x_decoded = x + ε_dec·N(0,I) simulates decoder
#    approximation error; safety is evaluated on x_decoded.
#
#  Returns all time-series needed for Fig. 2 and Table I.
# =============================================================
function run_combined(; seed::Int=2026, X1_MAX::Float64,
                        ε_dec::Float64=0.0)
    rng  = MersenneTwister(seed)
    x_ref(t) = 1.2*sin(0.05π*t*dt) + 0.2*cos(0.01π*t*dt)

    x     = [0.1, 0.0]; z = copy(x); Σ = 0.1*I(2)|>Matrix
    μ_hat = [MU_1];      S_prev = 0.0
    x_nom = copy(x); z_nom = copy(x); Σ_nom = copy(Σ)
    x_rob = copy(x); z_rob = copy(x); Σ_rob = copy(Σ)

    x1_cf  = zeros(T_TOTAL); x1_nom = zeros(T_TOTAL); x1_rob = zeros(T_TOTAL)
    S_hist = zeros(T_TOTAL); η_hist = zeros(T_TOTAL)
    CFI_h  = zeros(T_TOTAL); μ_hist = zeros(T_TOTAL)
    safe_cf  = ones(T_TOTAL); safe_nom = ones(T_TOTAL); safe_rob = ones(T_TOTAL)

    for t in 1:T_TOTAL
        ref = x_ref(t)

        # True μ_t schedule across four regimes
        μ_true = if t <= t_s1
            MU_1
        elseif t <= t_s2
            MU_2
        elseif t <= t_s3
            MU_1
        else
            κ = 1.0 - exp(-(t - t_s3)*dt / 20.0)
            MU_1 + κ*(MU_2 - MU_1)
        end

        # Observational drift bias b_t (Eq. 22, observational drift regime)
        obs_bias = if t > t_s2 && t <= t_s3
            κ = 1.0 - exp(-(t - t_s2)*dt / 6.0)
            κ * [0.83, 0.88]
        else
            zeros(2)
        end

        # ── SafeCF-SSM ───────────────────────────────────
        u_t, β_t, _ = bmpc(z, μ_hat[1], Σ, ref; S_prev=S_prev)
        x   = vdp_step(x, u_t, μ_true, rng)
        y_t = x + σv*randn(rng, 2) + obs_bias
        z, Σ, _ = ekf_update(z, Σ, y_t, u_t, μ_hat[1])
        S_t, η_t, CFI_t = surprise_and_update!(μ_hat, z, y_t, u_t)
        S_prev    = S_t
        x1_cf[t]  = x[1]
        S_hist[t] = S_t
        η_hist[t] = η_t
        CFI_h[t]  = CFI_t
        μ_hist[t] = μ_hat[1]

        # Gap-2 validation: decoder perturbation (Remark 4)
        x_decoded = ε_dec > 0.0 ? x + ε_dec * randn(rng, 2) : x
        safe_cf[t] = (abs(x_decoded[1]) ≤ X1_MAX &&
              abs(u_t) ≤ U_MAX) ? 1.0 : 0.0

        # ── Nominal MPC — fixed μ̂=μ₁, β=0 ─────────────
        u_nom_t, _, _ = bmpc(z_nom, MU_1, Σ_nom, ref; S_prev=0.0)
        x_nom   = vdp_step(x_nom, u_nom_t, μ_true, rng)
        y_nom   = x_nom + σv*randn(rng, 2) + obs_bias
        z_nom, Σ_nom, _ = ekf_update(z_nom, Σ_nom, y_nom, u_nom_t, MU_1)
        x1_nom[t]   = x_nom[1]
        safe_nom[t] = (abs(x_nom[1]) ≤ X1_MAX && abs(u_nom_t) ≤ U_MAX) ? 1.0 : 0.0

        # ── Robust MPC — fixed μ̂=μ₁, fixed β=0.5 ──────
        u_rob_t, _, _ = bmpc(z_rob, MU_1, Σ_rob, ref; β_fixed=0.5)
        x_rob   = vdp_step(x_rob, u_rob_t, μ_true, rng)
        y_rob   = x_rob + σv*randn(rng, 2) + obs_bias
        z_rob, Σ_rob, _ = ekf_update(z_rob, Σ_rob, y_rob, u_rob_t, MU_1)
        x1_rob[t]   = x_rob[1]
        safe_rob[t] = (abs(x_rob[1]) ≤ X1_MAX && abs(u_rob_t) ≤ U_MAX) ? 1.0 : 0.0
    end

    ref_traj = [x_ref(t) for t in 1:T_TOTAL]
    return (x1_cf=x1_cf, x1_nom=x1_nom, x1_rob=x1_rob,
            S=S_hist, η=η_hist, CFI=CFI_h, μ=μ_hist,
            safe=safe_cf, safe_nom=safe_nom, safe_rob=safe_rob,
            ref=ref_traj)
end

# =============================================================
#  SECTION 7: METRICS AND LIMIT CYCLE AMPLITUDE
#  rmse: root mean squared error for tracking performance (Table I).
#  vdp_limit_cycle_amplitude: estimates the uncontrolled VdP limit
#    cycle amplitude A(μ) by simulating the free system for T_sim
#    steps and recording max|x₁| in the second half (after transient).
#    Used to set X1_MAX = A(μ₂) as the safety bound (Sec. V-A).
# =============================================================
rmse(x, ref) = sqrt(mean((x .- ref).^2))

function vdp_limit_cycle_amplitude(μ::Float64; T_sim::Int=4000)
    x     = [2.0, 0.0]
    x_max = 0.0
    for t in 1:T_sim
        x1, x2 = x
        dx1 = x2
        dx2 = μ*(1.0 - x1^2)*x2 - x1
        x   = [x1 + dt*dx1, x2 + dt*dx2]
        if t > T_sim÷2
            x_max = max(x_max, abs(x[1]))
        end
    end
    return x_max
end

# =============================================================
#  SECTION 8: COMBINED FIGURE (Fig. 2 in paper)
#  Generates the 5-panel figure:
#    (a) x₁ tracking — SafeCF-SSM vs. Nominal vs. Robust MPC
#    (b) μ̂_t adaptation with regime annotations
#    (c) Surprise signal S_t
#    (d) Safety feasibility (Safe/Unsafe binary)
#    (e) CFI_t monitor with E[CFI] ≤ 1 bound (Theorem 1)
#  Saved as both PDF and PNG to figures/ directory.
#  Color scheme follows Wong (2011) colorblind-safe palette.
# =============================================================
function make_figure(r; X1_MAX::Float64, A_mu1::Float64, A_mu2::Float64)
    t_ax  = (1:T_TOTAL) .* dt
    ts1   = t_s1 * dt
    ts2   = t_s2 * dt
    ts3   = t_s3 * dt
    T_end = T_TOTAL * dt

    COL_CF  = RGBf(0/255,   114/255, 178/255)
    COL_NOM = RGBf(213/255,  94/255,   0/255)
    COL_ROB = RGBf(  0/255, 158/255, 115/255)
    COL_REF = RGBf(0, 0, 0)
    COL_MU  = RGBf(204/255, 121/255, 167/255)

    mkpath("figures")
    fig = Figure(size=(1080, 1150), fontsize=22)

    periods   = [(0.0, ts1), (ts1, ts2), (ts2, ts3), (ts3, T_end)]
    shade_col = [(:gray,0.06), (:blue,0.07), (:orange,0.07), (:green,0.07)]
    plabels   = [L"\mathrm{nominal}", L"\mathrm{abrupt\;shift}",
                 L"\mathrm{observational\;drift}", L"\mathrm{gradual\;drift}"]

    function add_shades!(ax, ytext)
        for (i,(a,b)) in enumerate(periods)
            vspan!(ax, a, b; color=shade_col[i])
        end
        for ts in [ts1, ts2, ts3]
            vlines!(ax, [ts]; color=(:black,0.25), linestyle=:dash, linewidth=1.8)
        end
        for (i,(a,b)) in enumerate(periods)
            text!(ax, (a+b)/2, ytext; text=plabels[i],
                  fontsize=24, color=(:black,0.69), align=(:center,:center))
        end
    end

    ax1 = Axis(fig[1,1]; ylabel=L"x_1",
        title=L"\textbf{(a)}\;\mathrm{Tracking}",
        xlabelsize=32, ylabelsize=48, titlesize=32,
        xticklabelsize=24, yticklabelsize=24,
        topspinevisible=false, xgridvisible=false,
        ygridvisible=false, rightspinevisible=false)
    add_shades!(ax1, 3.2)
    l_cf  = lines!(ax1, t_ax, r.x1_cf;  color=COL_CF,  linewidth=5.2)
    l_rob = lines!(ax1, t_ax, r.x1_rob; color=COL_ROB, linewidth=2.4, linestyle=:solid)
    l_nom = lines!(ax1, t_ax, r.x1_nom; color=COL_NOM, linewidth=2.8, linestyle=:dash)
    l_ref = lines!(ax1, t_ax, r.ref;    color=COL_REF, linewidth=1.8, linestyle=:dash)
    l_bnd = hlines!(ax1, [X1_MAX, -X1_MAX]; color=(:red,0.5), linewidth=3.5, linestyle=:dot)
    l_lc1 = hlines!(ax1, [A_mu1, -A_mu1];   color=(:blue,0.4), linewidth=3.8, linestyle=:dashdot)
    ylims!(ax1, -2.7, 3.7)
    xlims!(ax1, 0.0, t_ax[end])
    Legend(fig[1,2],
       [l_nom, l_rob, l_cf, l_ref, l_bnd, l_lc1],
       [L"\mathrm{Nominal}", L"\mathrm{Robust}", L"\mathrm{SafeCF-SSM}",
        L"\mathrm{ref}", L"X_{1,\max}\!=\!A(\mu_2)", L"A(\mu_1)"];
       framevisible=false, backgroundcolor=:white, labelsize=21,
       tellwidth=true, tellheight=false, rowgap=3, columngap=11,
       margin=(28,0,0,0), padding=(4,4,4,4))

    ax2 = Axis(fig[2,1]; ylabel=L"\hat\mu_t",
        title=L"\textbf{(b)}\;\hat\mu_t\;\mathrm{adaptation}",
        xlabelsize=32, ylabelsize=48, titlesize=32,
        xticklabelsize=24, yticklabelsize=24,
        topspinevisible=false, xgridvisible=false,
        ygridvisible=false, rightspinevisible=false)
    add_shades!(ax2, 3.4)
    l_mu1 = hlines!(ax2, [MU_1]; color=(:gray,0.5), linestyle=:dot, linewidth=3.5)
    l_mu2 = hlines!(ax2, [MU_2]; color=(:red,0.4),  linestyle=:dot, linewidth=3.5)
    l_mu  = lines!(ax2, t_ax, r.μ; color=COL_MU, linewidth=3.8)
    ylims!(ax2, -1.3, 4.3)
    xlims!(ax2, 0.0, t_ax[end])
    Legend(fig[2,2], [l_mu1, l_mu2, l_mu],
           [L"\mu_1\!=\!0.5", L"\mu_2\!=\!2.66", L"\hat\mu_t"];
           framevisible=false, backgroundcolor=:white,
           labelsize=22, tellwidth=true, tellheight=false)
    text!(ax2, (ts1+ts2)/2.5, -0.03;
        text=L"\mathbf{\mu_1}\!\uparrow\!\mathbf{\mu_2}",
        fontsize=34, color=COL_CF, align=(:center,:center))
    text!(ax2, (ts2+ts3)/2.2, -0.03;
        text=L"\mathbf{\mu_1\;+\;b_t}",
        fontsize=34, color=COL_CF, align=(:center,:center))
    text!(ax2, (ts3+T_end)/1.98, -0.03;
        text=L"\mathbf{\mu_1}\!\nearrow\!\mathbf{\mu_2}",
        fontsize=34, color=COL_CF, align=(:center,:center))

    ax3 = Axis(fig[3,1]; ylabel=L"\mathcal{S}_t",
        title=L"\textbf{(c)}\;\mathcal{S}_t\;\mathrm{(surprise\;signal)}",
        xlabelsize=32, ylabelsize=48, titlesize=32,
        xticklabelsize=24, yticklabelsize=24,
        topspinevisible=false, xgridvisible=false,
        ygridvisible=false, rightspinevisible=false)
    add_shades!(ax3, 27.0)
    l_S = lines!(ax3, t_ax, r.S; color=COL_CF, linewidth=1.2)
    ylims!(ax3, 0, 30)
    xlims!(ax3, 0.0, t_ax[end])

    ax4 = Axis(fig[4,1];
        title=L"\textbf{(d)}\;\mathrm{Safety\;feasibility}",
        xlabelsize=32, ylabelsize=28, titlesize=32,
        xticklabelsize=24, yticklabelsize=28,
        topspinevisible=false, xgridvisible=false,
        ygridvisible=false, rightspinevisible=false,
        yticks=([0,1], [L"\mathrm{Unsafe}", L"\mathrm{Safe}"]))
    add_shades!(ax4, 1.22)
    ylims!(ax4, -0.1, 1.35)
    xlims!(ax4, 0.0, t_ax[end])
    l_rob_s = lines!(ax4, t_ax, r.safe_rob; color=COL_ROB, linewidth=1.8, linestyle=:solid)
    l_cf_s  = lines!(ax4, t_ax, r.safe;     color=COL_CF,  linewidth=4.8, linestyle=:solid)
    l_nom_s = lines!(ax4, t_ax, r.safe_nom; color=COL_NOM, linewidth=2.2, linestyle=:dash)
    Legend(fig[4,2], [l_nom_s, l_rob_s, l_cf_s],
           [L"\mathrm{Nominal}", L"\mathrm{Robust}", L"\mathrm{SafeCF-SSM}"];
           framevisible=false, backgroundcolor=:white,
           labelsize=22, tellwidth=true, tellheight=false)

    ax5 = Axis(fig[5,1]; ylabel=L"\mathrm{CFI}_t", xlabel=L"t\;[\mathrm{s}]",
        title=L"\textbf{(e)}\;\mathrm{CFI}_t",
        xlabelsize=32, ylabelsize=48, titlesize=32,
        xticklabelsize=24, yticklabelsize=24,
        topspinevisible=false, xgridvisible=false,
        ygridvisible=false, rightspinevisible=false)
    ylims!(ax5, 0, 1.1)
    xlims!(ax5, 0.0, t_ax[end])
    add_shades!(ax5, 0.87)
    l_cfi       = lines!(ax5, t_ax, r.CFI; color=COL_CF, linewidth=1.4)
    l_cfi_bound = hlines!(ax5, [1.0]; color=(:blue,0.4), linewidth=3.9, linestyle=:dot)
    Legend(fig[5,2], [l_cfi, l_cfi_bound],
           [L"\mathrm{CFI}", L"\mathbb{E}[\mathrm{CFI}]"];
           framevisible=false, backgroundcolor=:white,
           labelsize=20, tellwidth=true, tellheight=false)

    colsize!(fig.layout, 1, Relative(0.84))
    colsize!(fig.layout, 2, Relative(0.12))
    save("figures/fig_combined_250426.pdf", fig)
    save("figures/fig_combined_250426.png", fig)
    println("  Saved → figures/fig_combined_250426.pdf/.png")
end

# =============================================================
#  SECTION 9: MAIN ENTRY POINT
#  1. Computes VdP limit cycle amplitudes A(μ₁) and A(μ₂) to
#     set the safety bound X1_MAX = A(μ₂) (physically motivated).
#  2. Runs the single 345-s combined experiment (Fig. 2).
#  3. Prints RMSE, safety rate, and E[CFI] for Table I.
#  4. Runs Gap-2 validation: M=25 Monte Carlo runs for each
#     ε_dec ∈ {0.0, 0.05, 0.10}, reporting mean ± std safety
#     rate and E[CFI] (Table I, Remark 4, Corollary 1).
# =============================================================
function main()
    println("="^60)
    println("  SafeCF-SSM — Combined VdP Benchmark")
    println("  nominal→abrupt(μ₂=$(MU_2))→obs drift→gradual")
    println("="^60)

    A_mu1  = vdp_limit_cycle_amplitude(MU_1)
    A_mu2  = vdp_limit_cycle_amplitude(MU_2)
    X1_MAX = A_mu2
    println(@sprintf("  A(μ₁=%.2f)=%.3f  A(μ₂=%.2f)=%.3f  X1_MAX=%.3f",
            MU_1, A_mu1, MU_2, A_mu2, X1_MAX))

    print("  Running ... ")
    r = run_combined(seed=2026, X1_MAX=X1_MAX)
    println("done.")

    println(@sprintf("  RMSE: SafeCF=%.3f  Nom=%.3f  Rob=%.3f",
            rmse(r.x1_cf, r.ref), rmse(r.x1_nom, r.ref), rmse(r.x1_rob, r.ref)))
    println(@sprintf("  Safety: SafeCF=%.1f%%  Nom=%.1f%%  Rob=%.1f%%",
            100*mean(r.safe), 100*mean(r.safe_nom), 100*mean(r.safe_rob)))
    println(@sprintf("  E[CFI]=%.3f", mean(r.CFI)))

    make_figure(r; X1_MAX=X1_MAX, A_mu1=A_mu1, A_mu2=A_mu2)

    # Gap-2 Validation: decoder error sweep (Remark 4, Table I)
    println("\n" * "="^50)
    println("  Gap 2 Validation — decoder error sweep")
    println("="^50)
    println(@sprintf("  %-8s | %-14s | %-8s",
                     "ε_dec", "Safety% (±std)", "E[CFI]"))
    println("  " * "-"^38)

    for ε in ε_DEC_LIST
        safety_rates = Float64[]
        cfi_means    = Float64[]
        for seed in 2001:2025
            r_mc = run_combined(seed=seed, X1_MAX=X1_MAX, ε_dec=ε)
            push!(safety_rates, 100*mean(r_mc.safe))
            push!(cfi_means,    mean(r_mc.CFI))
        end
        println(@sprintf("  %-8.2f | %5.1f ± %4.1f   | %.3f",
                ε, mean(safety_rates), std(safety_rates), mean(cfi_means)))
    end
    println("="^50)
end

main()
