# =============================================================
#  cf_deepsssm_vdp_v6[180426].jl
#  SafeCF-SSM — Van der Pol Benchmark
#  Coded by Thanana, copyright 2026
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
#  Three scenarios:
#    A: Abrupt μ shift   (μ₁→μ₂ at t_s)
#    B: Observation drift (additive bias after t_s)
#    C: Gradual μ drift  (exponential transition)
# =============================================================


using Random, Distributions, Statistics, LinearAlgebra
using CairoMakie
using Printf
using JuMP, OSQP

# =============================================================
#  SECTION 0: GLOBAL CONSTANTS
# =============================================================
const dt      = 0.05
const T       = 1600
const t_s     = 400
const MU_1    = 0.5
const MU_2    = 1.5
const σw      = 0.05
const σv      = 0.10
const U_MAX   = 3.0
const X1_MAX  = 2.5
const K1      = 12.0
const K2      = 6.0
const ETA_MAX = 0.15
const GRAD_CLIP = 2.0
const ε_CF    = ETA_MAX

# =============================================================
#  SECTION 1: TRUE SYSTEM (Van der Pol)
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
#  SECTION 3: SURPRISE AND CF ADAPTATION
# =============================================================
function surprise_and_update!(μ_hat::Vector{Float64},
                               z::Vector{Float64},
                               y_next::Vector{Float64},
                               u::Float64)
    z_pred = vdp_predict(z, u, μ_hat[1])
    err    = y_next - z_pred
    S_t    = 0.5 * sum(err.^2) / σw^2
    η_t    = ETA_MAX / (1.0 + sqrt(max(0.0, S_t)))
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

# ================================================================
#  SECTION 4: BMPC WITH ADAPTIVE TIGHTENING (SafeCF--SSM for MPC)
# ================================================================
function bmpc(z::Vector{Float64}, μ_hat::Float64,
              Σ::Matrix{Float64}, x_ref::Float64; N::Int=10)
    z1, z2 = z

    # Predictive uncertainty
    λ_max = max(0.0, maximum(real.(eigvals(Σ))))
    σ_t   = sqrt(λ_max)

    # Constraint tightening margin (Lemma 1)
    c_β = 0.4
    β_t = c_β * σ_t
    u_max_safe = max(0.1, U_MAX - β_t)

    # Linearized MPC: rollout nominal trajectory, then optimize perturbations
    # Nominal rollout with u=0
    z_nom_traj = Vector{Vector{Float64}}(undef, N+1)
    z_nom_traj[1] = copy(z)
    for k in 1:N
        z_nom_traj[k+1] = vdp_predict(z_nom_traj[k], 0.0, μ_hat)
    end

    # Linearized dynamics: δz_{k+1} = A_k*δz_k + B_k*u_k
    # B_k = [0; dt] (control enters z2 only)
    B_ctrl = [0.0; dt]

    model = Model(OSQP.Optimizer)
    set_silent(model)

    @variable(model, u_seq[1:N])
    @variable(model, dz[1:N+1, 1:2])  # perturbation from nominal

    # Initial perturbation = 0
    @constraint(model, dz[1,1] == 0.0)
    @constraint(model, dz[1,2] == 0.0)

    # Linearized dynamics constraints
    for k in 1:N
        Ak = vdp_jacobian(z_nom_traj[k], 0.0, μ_hat)
        @constraint(model, dz[k+1,1] == Ak[1,1]*dz[k,1] + Ak[1,2]*dz[k,2] + B_ctrl[1]*u_seq[k])
        @constraint(model, dz[k+1,2] == Ak[2,1]*dz[k,1] + Ak[2,2]*dz[k,2] + B_ctrl[2]*u_seq[k])
    end

    # Input constraints
    @constraint(model, [k=1:N], -u_max_safe <= u_seq[k] <= u_max_safe)

    # Quadratic objective over actual predicted state = nominal + perturbation
    @objective(model, Min,
    sum(20.0*(z_nom_traj[k][1] + dz[k,1] - x_ref)^2 +
        3.0*(z_nom_traj[k][2] + dz[k,2])^2 +
        0.05*u_seq[k]^2 for k in 1:N))

    optimize!(model)

    # Fallback to PD if infeasible
    u_opt = if termination_status(model) == MOI.OPTIMAL
        clamp(value(u_seq[1]), -u_max_safe, u_max_safe)
    else
        clamp(-K1*(z1-x_ref) - K2*z2, -u_max_safe, u_max_safe)
    end

    return u_opt, β_t, σ_t
end


# =============================================================
#  SECTION 5: ENCODER — EKF IMPLEMENTATION OF q_{φ_t}
#
#  Implements the latent belief update:
#    q_{φ_t}(z_t | H_t) = N(z_t; ẑ_t, Σ_t)
#  via Extended Kalman Filter (EKF), consistent with
#  Assumption ass:decoder (bounded linearization error).
#  Any encoder satisfying Assumption ass:decoder can replace this.
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
#  SECTION 6: SINGLE SIMULATION RUN
# =============================================================
function run_scenario(scenario::Symbol; seed::Int=2026)
    rng = MersenneTwister(seed)
    x_ref(t) = 1.2 * sin(0.05π * t * dt)+0.2 * cos(0.01π * t * dt)

    x     = [0.1, 0.0]
    z     = copy(x)
    Σ     = 0.1 * I(2) |> Matrix
    μ_hat = [MU_1]

    x_nom  = copy(x);  z_nom = copy(x);  Σ_nom = copy(Σ)
    x_rob  = copy(x);  z_rob = copy(x);  Σ_rob = copy(Σ)

    x1_cf   = zeros(T);  x1_nom = zeros(T);  x1_rob = zeros(T)
    u_cf    = zeros(T);  u_nom  = zeros(T)
    S_hist  = zeros(T);  η_hist = zeros(T)
    CFI_hist = zeros(T); μ_hist = zeros(T)
    safe_cf = ones(T);   β_hist = zeros(T)
    safe_nom = ones(T);  safe_rob = ones(T)
    obs_bias = zeros(2)

    for t in 1:T
        ref = x_ref(t)

        μ_true = if scenario == :abrupt
            t < t_s ? MU_1 : MU_2
        elseif scenario == :gradual
            κ = t < t_s ? 0.0 : 1.0 - exp(-(t - t_s)*dt / 6.0)
            MU_1 + κ*(MU_2 - MU_1)
        else
            MU_1
        end

        if scenario == :obs && t >= t_s
            κ = 1.0 - exp(-(t - t_s)*dt / 6.0)
           # obs_bias = [0.8*κ, 0.4*κ]
            obs_bias = κ * [0.8, 0.4]
        end

        u_t, β_t, σ_t = bmpc(z, μ_hat[1], Σ, ref)
        x = vdp_step(x, u_t, μ_true, rng)
        y_t = x + σv*randn(rng, 2) + obs_bias
        z, Σ, inn = ekf_update(z, Σ, y_t, u_t, μ_hat[1])
        S_t, η_t, CFI_t = surprise_and_update!(μ_hat, z, y_t, u_t)
        safe = abs(x[1]) ≤ X1_MAX && abs(u_t) ≤ U_MAX

        x1_cf[t]    = x[1];  u_cf[t]     = u_t
        S_hist[t]   = S_t;   η_hist[t]   = η_t
        CFI_hist[t] = CFI_t; μ_hist[t]   = μ_hat[1]
        safe_cf[t]  = safe ? 1.0 : 0.0
        β_hist[t]   = β_t

        u_nom_t, _, _ = bmpc(z_nom, MU_1, Σ_nom, ref)
        x_nom  = vdp_step(x_nom, u_nom_t, μ_true, rng)
        y_nom  = x_nom + σv*randn(rng, 2)
        z_nom, Σ_nom, _ = ekf_update(z_nom, Σ_nom, y_nom, u_nom_t, MU_1)
        x1_nom[t] = x_nom[1];  u_nom[t] = u_nom_t
        safe_nom[t] = (abs(x_nom[1]) ≤ X1_MAX && abs(u_nom_t) ≤ U_MAX) ? 1.0 : 0.0

        u_rob_t = clamp(u_nom_t, -(U_MAX-0.5), (U_MAX-0.5))
        x_rob  = vdp_step(x_rob, u_rob_t, μ_true, rng)
        y_rob  = x_rob + σv*randn(rng, 2)
        z_rob, Σ_rob, _ = ekf_update(z_rob, Σ_rob, y_rob, u_rob_t, MU_1)
        x1_rob[t] = x_rob[1]
        safe_rob[t] = (abs(x_rob[1]) ≤ X1_MAX && abs(u_rob_t) ≤ U_MAX) ? 1.0 : 0.0
    end

    ref_traj = [x_ref(t) for t in 1:T]
    return (x1_cf=x1_cf, x1_nom=x1_nom, x1_rob=x1_rob,
            u_cf=u_cf, u_nom=u_nom,
            S=S_hist, η=η_hist, CFI=CFI_hist,
            μ=μ_hist, safe=safe_cf, safe_nom=safe_nom, safe_rob=safe_rob, β=β_hist,
            ref=ref_traj)
end

# =============================================================
#  SECTION 7: METRICS
# =============================================================
rmse(x, ref)      = sqrt(mean((x .- ref).^2))
safety_pct(safe)  = 100.0 * mean(safe)
mean_cfi(CFI)     = mean(CFI[t_s:end])
mean_S(S)         = mean(S[t_s:end])

# =============================================================
#  SECTION 8: PLOTTING (unchanged)
# =============================================================
function make_figures(results)
    t_ax    = (1:T) .* dt
    t_s_sec = t_s * dt

    COL_CF  = RGBf(0/255,   114/255, 178/255)
    COL_NOM = RGBf(213/255,  94/255,   0/255)
    COL_ROB = RGBf(  0/255, 158/255, 115/255)
    COL_REF = RGBf(0/255, 0/255, 0/255)
    COL_MU  = RGBf(204/255, 121/255, 167/255)
    COL_ROBUST  = Makie.wong_colors()[4]

    mkpath("figures")

    # =================================================
    # ── Figure Scenario: A ───────────────────────────
    # =================================================
    
    r = results[:abrupt]
    fig = Figure(size=(790, 630), fontsize=13)

    ax1 = Axis(fig[1,1];
        ylabel=L"x_1",
        title=L"\textbf{(a)}\;\mathrm{Tracking}",
        xlabelsize=26, ylabelsize=30, titlesize=19,
        xticklabelsize=17, yticklabelsize=17)
    vlines!(ax1, [t_s_sec]; color=(:black,0.3), linestyle=:dash, linewidth=2.8)
    lines!(ax1, t_ax, r.x1_nom; color=COL_NOM, linewidth=3.9, linestyle=:dash,
           label=L"\mathrm{Nominal}")
    lines!(ax1, t_ax, r.x1_rob; color=COL_ROB, linewidth=2.8, linestyle=:dashdot,
           label=L"\mathrm{Robust}")
    lines!(ax1, t_ax, r.x1_cf;  color=COL_CF,  linewidth=3.4,
            label=L"\mathrm{SafeCF-SSM}")
    lines!(ax1, t_ax, r.ref;    color=COL_REF, linewidth=2.9, linestyle=:dash,
           label=L"\mathrm{ref}")
    hlines!(ax1, [X1_MAX, -X1_MAX]; color=(:red,0.25), linewidth=2.9, linestyle=:dot)
    axislegend(ax1, position=:rb, framevisible=true, labelsize=18)



    ax2 = Axis(fig[3,1];
        ylabel=L"\mathcal{S}_t",
        title=L"\textbf{(c)}\;\mathcal{S}_t\;\mathrm{and}\;\eta_t",
        xlabelsize=26, ylabelsize=30, titlesize=19,
        xticklabelsize=17, yticklabelsize=17)
    vlines!(ax2, [t_s_sec]; color=(:black,0.3), linestyle=:dash, linewidth=2.8)
    lines!(ax2, t_ax, r.S; color=COL_NOM, linewidth=1.2)

    ax2r = Axis(fig[3,1]; yaxisposition=:right,
                ylabel=L"\eta_t", ylabelsize=24, yticklabelsize=11)
    hidexdecorations!(ax2r; grid=false)
    ax2r.xgridvisible=false; ax2r.backgroundcolor=:transparent
    lines!(ax2r, t_ax, r.η; color=COL_CF, linewidth=1.0, linestyle=:dash)

    ax3 = Axis(fig[2,1];
        ylabel=L"\hat\mu_t",
        title=L"\textbf{(b)}\;\hat\mu_t\;\mathrm{adaptation}",
        xlabelsize=26, ylabelsize=30, titlesize=19,
        xticklabelsize=17, yticklabelsize=17)
    vlines!(ax3, [t_s_sec]; color=(:black,0.3), linestyle=:dash, linewidth=2.8)
    hlines!(ax3, [MU_1]; color=(:gray,0.5), linestyle=:dot, linewidth=3.0,
            label=L"\mu_1")
    hlines!(ax3, [MU_2]; color=(:red,0.4), linestyle=:dot, linewidth=3.0,
            label=L"\mu_2")
    lines!(ax3, t_ax, r.μ; color=COL_MU, linewidth=2.2,
           label=L"\hat\mu_t")
    axislegend(ax3, position=:rb, framevisible=true, labelsize=23)

    ax4 = Axis(fig[4,1];
        ylabel=L"\mathrm{CFI}_t", xlabel=L"t\;[\mathrm{s}]",
        title=L"\textbf{(d)}\;\mathrm{CFI}_t",
        xlabelsize=26, ylabelsize=30, titlesize=19,
        xticklabelsize=17, yticklabelsize=17)
    vlines!(ax4, [t_s_sec]; color=(:black,0.3), linestyle=:dash, linewidth=2.8)
    lines!(ax4, t_ax, r.CFI; color=COL_CF, linewidth=1.4)
    hlines!(ax4, [1.0]; color=(:red,0.4), linewidth=2.8, linestyle=:dot,
            label=L"\mathbb{E}[\mathrm{CFI}]\leq1")
    axislegend(ax4, position=:rt, framevisible=true, labelsize=22)

    save("figures/figA_scenario_abrupt_180426.pdf", fig)
    save("figures/figA_scenario_abrupt_180426.png", fig)

    # =================================================
    # ── Figure Scenario: B ───────────────────────────
    # =================================================
    r = results[:obs]
    fig2 = Figure(size=(790, 500), fontsize=13)

    ax1b = Axis(fig2[1,1];
        ylabel=L"x_1",
        title=L"\textbf{(a)}\;\mathrm{Tracking}",
        xlabelsize=26, ylabelsize=30, titlesize=20,
        xticklabelsize=17, yticklabelsize=17)
    vlines!(ax1b, [t_s_sec]; color=(:black,0.3), linestyle=:dash, linewidth=2.2)
    lines!(ax1b, t_ax, r.x1_nom; color=COL_NOM, linewidth=3.9, linestyle=:dash,
           label=L"\mathrm{Nominal}")
    lines!(ax1b, t_ax, r.x1_rob; color=COL_ROB, linewidth=2.8, linestyle=:dashdot,
           label=L"\mathrm{Robust}")
    lines!(ax1b, t_ax, r.x1_cf;  color=COL_CF,  linewidth=3.4,
           label=L"\mathrm{SafeCF-SSM}")
    lines!(ax1b, t_ax, r.ref;    color=COL_REF, linewidth=2.6, linestyle=:dash,
           label=L"\mathrm{ref}")
    hlines!(ax1b, [X1_MAX, -X1_MAX]; color=(:red,0.25), linewidth=2.9, linestyle=:dot)
    axislegend(ax1b, position=:rb, framevisible=true, labelsize=18)

    ax2b = Axis(fig2[2,1];
        ylabel=L"\mathcal{S}_t",
            # title=L"\textbf{(b)}\;\mathcal{S}_t\;\mathrm{and\;bounded\;learning}",
            title=L"\textbf{(b)}\;\mathcal{S}_t\;\mathrm{(Theorem~1)}",
        xlabelsize=26, ylabelsize=30, titlesize=20,
        xticklabelsize=17, yticklabelsize=17)
    vlines!(ax2b, [t_s_sec]; color=(:black,0.3), linestyle=:dash, linewidth=2.2)
    lines!(ax2b, t_ax, r.S; color=COL_NOM, linewidth=1.2)

    ax3b = Axis(fig2[3,1];
        xlabel=L"t\;[\mathrm{s}]",
        title=L"\textbf{(c)}\;\mathrm{Safety\;feasibility}",
        xlabelsize=26, ylabelsize=30, titlesize=19,
        xticklabelsize=17, yticklabelsize=20,
        yticks=([0,1], [L"\mathrm{Unsafe}",L"\mathrm{Safe}"]))
    vlines!(ax3b, [t_s_sec]; color=(:black,0.3), linestyle=:dash, linewidth=2.2)
    ylims!(ax3b, -0.1, 1.3)
    lines!(ax3b, t_ax, r.safe_rob; color=COL_ROB, linewidth=2.1,
           linestyle=:dashdot, label=L"\mathrm{Robust}")
    lines!(ax3b, t_ax, r.safe;     color=COL_CF,  linewidth=1.9,
           label=L"\mathrm{SafeCF-SSM}")
    lines!(ax3b, t_ax, r.safe_nom; color=COL_NOM, linewidth=2.9,
           linestyle=:dash, label=L"\mathrm{Nominal}")
    axislegend(ax3b, position=:rb, framevisible=true, labelsize=18)

    save("figures/figB_scenario_obs_180426.pdf", fig2)
    save("figures/figB_scenario_obs_180426.png", fig2)

    # =========================================================
    # ──────────── Figure Scenario: C ────────────────────────-
    # =========================================================
    r = results[:gradual]
    fig3 = Figure(size=(790, 400), fontsize=13)

    ax1c = Axis(fig3[1,1];
        ylabel=L"x_1",
        title=L"\textbf{(a)}\;\mathrm{Tracking}",
        xlabelsize=26, ylabelsize=30, titlesize=20,
        xticklabelsize=17, yticklabelsize=17)
    vlines!(ax1c, [t_s_sec]; color=(:black,0.3), linestyle=:dash, linewidth=2.2)
    lines!(ax1c, t_ax, r.x1_rob; color=COL_ROB, linewidth=2.9, linestyle=:dashdot,
           label=L"\mathrm{Robust}")
    lines!(ax1c, t_ax, r.x1_cf;  color=COL_CF,  linewidth=3.4,
           label=L"\mathrm{SafeCF-SSM}")
    lines!(ax1c, t_ax, r.ref;    color=COL_REF, linewidth=2.6, linestyle=:dash,
           label=L"\mathrm{ref}")
    lines!(ax1c, t_ax, r.x1_nom; color=COL_NOM, linewidth=3.9, linestyle=:dash,
           label=L"\mathrm{Nominal}")
           
    hlines!(ax1c, [X1_MAX, -X1_MAX]; color=(:red,0.25), linewidth=2.9, linestyle=:dot)
    axislegend(ax1c, position=:rt, framevisible=true, labelsize=18)

    ax2c = Axis(fig3[2,1];
        ylabel=L"\mathrm{CFI}_t", xlabel=L"t\;[\mathrm{s}]",
        title=L"\textbf{(b)}\;\mathrm{CFI}_t",
        xlabelsize=26, ylabelsize=30, titlesize=20,
        xticklabelsize=17, yticklabelsize=17)
    vlines!(ax2c, [t_s_sec]; color=(:black,0.3), linestyle=:dash, linewidth=2.2)
    lines!(ax2c, t_ax, r.CFI; color=COL_CF, linewidth=1.4)
    hlines!(ax2c, [1.0]; color=(:red,0.4), linewidth=2.9, linestyle=:dot,
            label=L"\mathbb{E}[\mathrm{CFI}]\leq1")
    axislegend(ax2c, position=:rt, framevisible=true, labelsize=22)

    save("figures/figC_scenario_gradual_180426.pdf", fig3)
    save("figures/figC_scenario_gradual_180426.png", fig3)

    println("  Saved all figures → ./figures/")
end

# =============================================================
#  SECTION 9: MAIN
# =============================================================
function main()
    println("="^60)
    println("  SafeCF-SSM — Van der Pol Benchmark (v6 with real MPC)")
    println("  μ₁=$MU_1 → μ₂=$MU_2  |  T=$T  t_s=$t_s  dt=$dt")
    println("="^60)

    scenarios = [:abrupt, :obs, :gradual]
    labels    = ["Scenario A (Abrupt shift)",
                 "Scenario B (Obs drift)",
                 "Scenario C (Gradual drift)"]

    results = Dict{Symbol, NamedTuple}()
    M_mc = 50
for (sc, lb) in zip(scenarios, labels)
    print("  Running $lb (M=$M_mc) ... ")
    r = run_scenario(sc; seed=2026)  # seed=2026 for figures
    results[sc] = r

    # Monte Carlo for RMSE statistics
    rmse_cf  = zeros(M_mc)
    rmse_nom = zeros(M_mc)
    rmse_rob = zeros(M_mc)
    for m in 1:M_mc
        rm = run_scenario(sc; seed=m)
        rmse_cf[m]  = rmse(rm.x1_cf,  rm.ref)
        rmse_nom[m] = rmse(rm.x1_nom, rm.ref)
        rmse_rob[m] = rmse(rm.x1_rob, rm.ref)
    end
    println(@sprintf("  RMSE: SafeCF=%.3f±%.3f  Nom=%.3f±%.3f  Rob=%.3f±%.3f",
        mean(rmse_cf), std(rmse_cf),
        mean(rmse_nom), std(rmse_nom),
        mean(rmse_rob), std(rmse_rob)))
end

    println("\n--- Performance Summary ---")
    println(rpad("Scenario",22), rpad("Method",18),
            lpad("RMSE",8), lpad("Safe%",8), lpad("Ē[CFI]",10))
    println("-"^68)
    for (sc, lb) in zip(scenarios, ["Abrupt","Obs drift","Gradual"])
        r = results[sc]
        for (x, name) in [(r.x1_cf,"SafeCF-SSM"),
                          (r.x1_nom,"Nominal MPC"),
                          (r.x1_rob,"Robust MPC")]
            s_pct = name == "SafeCF-SSM" ?
                    @sprintf("%.1f%%", safety_pct(r.safe)) : "  --"
            cfi   = name == "SafeCF-SSM" ?
                    @sprintf("%.3f", mean_cfi(r.CFI)) : "  --"
            println(rpad(name=="SafeCF-SSM" ? lb : "",22),
                    rpad(name,18),
                    lpad(@sprintf("%.3f", rmse(x, r.ref)),8),
                    lpad(s_pct,8), lpad(cfi,10))
        end
        println("-"^68)
    end

    println("\n  Generating figures ...")
    make_figures(results)
end

main()

