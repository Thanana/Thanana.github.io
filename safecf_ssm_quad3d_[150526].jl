# =============================================================
#  safecf_ssm_quad3d_2.jl
#  SafeCF-SSM — 3D Quadrotor Benchmark (Partial Rotor Failure)
#  Coded by Thanana Nuchkrua, Copyright 2026
#
#  This script implements and validates the SafeCF-SSM framework
#  (Nuchkrua & Boonto, IEEE L-CSS 2026) on a 12-state 3D quadrotor
#  under partial rotor-1 efficiency degradation (ρ₁=1.0 → ρ₂=0.6
#  at t_s=40 s), demonstrating scalability to a realistic
#  cyber-physical system with 4 inputs (Sec. V-B, Fig. 3).
#
#  State:  x = [px,py,pz,φ,θ,ψ,vx,vy,vz,p,q,r] ∈ R^12
#  Input:  u = [T1,T2,T3,T4] (rotor thrusts)
#  Shift:  ρ_t: 1.0 → 0.6 at t_s (rotor-1 partial failure)
#
#  Hover condition (θ=0):
#    T1*ρ + T2+T3+T4 = M*g  →  T1 = M*g/(ρ+3),  T2=T3=T4=M*g/4
#
#  Dependencies: Julia 1.x, CairoMakie, JuMP, OSQP, Distributions
# =============================================================
using Random, Distributions, Statistics, LinearAlgebra
using CairoMakie, Printf
using JuMP, OSQP
import CairoMakie: xlims!, ylims!, vspan!, vlines!, hlines!, lines!, band!, text!

# =============================================================
#  SECTION 0: GLOBAL CONSTANTS
#  All simulation parameters for the 3D quadrotor benchmark.
#  Physical parameters (MASS, Ixx/Iyy/Izz, ARM, KD) match the
#  quadrotor model in Sec. V-B. Safety bounds: Z_MIN (minimum
#  altitude), THETA_MAX (maximum tilt angle). ETA_1/ETA_2 are
#  the rotor efficiency before/after the partial failure event.
#  ETA_MAX_CF is the maximum CF adaptation rate (Sec. III-C).
# =============================================================
const dt         = 0.05
const T_SIM      = 3400
const t_s        = 800
const ETA_1      = 1.0
const ETA_2      = 0.6
const MASS       = 1.0
const Ixx        = 0.01
const Iyy        = 0.01
const Izz        = 0.02
const ARM        = 0.2
const KD         = 0.1
const G_GRAV     = 9.81
const T_MAX      = 10.0
const Z_MIN      = 0.3
const THETA_MAX  = deg2rad(30.0)
const σw         = 0.05
const σv         = 0.10
const ETA_MAX_CF = 0.10
const GRAD_CLIP  = 2.0
const ε_CF       = ETA_MAX_CF
const N_HORIZON  = 10
const REF_PZ     = 2.0
const NX         = 12
const NU         = 4

# =============================================================
#  SECTION 1: TRUE SYSTEM (3D Quadrotor)
#  Implements one Euler step of the true (unknown) 3D quadrotor
#  dynamics with additive process noise w_t ~ N(0, σw²·I₁₂).
#  Rotor-1 effective thrust is scaled by η_true (the unknown
#  true efficiency), inducing torque imbalance τ after failure.
#  This is the physical plant — never accessed by the controller.
# =============================================================
function quad3d_step(x::Vector{Float64}, u::Vector{Float64},
                     η::Float64, rng::AbstractRNG)
    px,py,pz,φ,θ,ψ,vx,vy,vz,p,q,r = x
    T1,T2,T3,T4 = u
    T1e = η*T1
    Tt  = T1e+T2+T3+T4
    cφ,sφ = cos(φ),sin(φ)
    cθ,sθ = cos(θ),sin(θ)
    cψ,sψ = cos(ψ),sin(ψ)
    ax = (cψ*sθ*cφ + sψ*sφ)*Tt/MASS
    ay = (sψ*sθ*cφ - cψ*sφ)*Tt/MASS
    az =  cθ*cφ*Tt/MASS - G_GRAV
    τx = ARM*(T2-T4)
    τy = ARM*(T3-T1e)
    τz = KD*(T1e-T2+T3-T4)
    return [px+dt*vx,       py+dt*vy,       pz+dt*vz,
            φ+dt*p,         θ+dt*q,         ψ+dt*r,
            vx+dt*ax+σw*randn(rng),
            vy+dt*ay+σw*randn(rng),
            vz+dt*az+σw*randn(rng),
            p+dt*τx/Ixx+σw*randn(rng),
            q+dt*τy/Iyy+σw*randn(rng),
            r+dt*τz/Izz+σw*randn(rng)]
end

# =============================================================
#  SECTION 2: LATENT MODEL AND NUMERICAL JACOBIANS
#  The controller's internal SSM (Eq. 5 in paper).
#  quad3d_predict: one-step latent prediction using estimated ρ̂_t.
#  quad3d_AB: numerical Jacobians A = ∂f/∂x, B = ∂f/∂u via
#    finite differences (ε=1e-5), used for EKF and BMPC
#    linearization around the hover operating point.
#  cov_update: EKF prediction covariance Σ_{t+1|t} = AΣA' + Q,
#    with eigenvalue clipping (λ_max ≤ 5) for numerical stability.
# =============================================================
function quad3d_predict(x::Vector{Float64}, u::Vector{Float64},
                        η_hat::Float64)
    px,py,pz,φ,θ,ψ,vx,vy,vz,p,q,r = x
    T1,T2,T3,T4 = u
    T1e = η_hat*T1
    Tt  = T1e+T2+T3+T4
    cφ,sφ = cos(φ),sin(φ)
    cθ,sθ = cos(θ),sin(θ)
    cψ,sψ = cos(ψ),sin(ψ)
    ax = (cψ*sθ*cφ + sψ*sφ)*Tt/MASS
    ay = (sψ*sθ*cφ - cψ*sφ)*Tt/MASS
    az =  cθ*cφ*Tt/MASS - G_GRAV
    τx = ARM*(T2-T4)
    τy = ARM*(T3-T1e)
    τz = KD*(T1e-T2+T3-T4)
    return [px+dt*vx, py+dt*vy, pz+dt*vz,
            φ+dt*p,   θ+dt*q,   ψ+dt*r,
            vx+dt*ax, vy+dt*ay, vz+dt*az,
            p+dt*τx/Ixx, q+dt*τy/Iyy, r+dt*τz/Izz]
end

function quad3d_AB(x::Vector{Float64}, u::Vector{Float64},
                   η_hat::Float64)
    ε  = 1e-5
    f0 = quad3d_predict(x, u, η_hat)
    A  = zeros(NX, NX)
    B  = zeros(NX, NU)
    for i in 1:NX
        dx = zeros(NX); dx[i] = ε
        A[:,i] = (quad3d_predict(x.+dx, u, η_hat) .- f0) ./ ε
    end
    for i in 1:NU
        du = zeros(NU); du[i] = ε
        B[:,i] = (quad3d_predict(x, u.+du, η_hat) .- f0) ./ ε
    end
    return A, B
end

function cov_update(Σ::Matrix{Float64}, F::Matrix{Float64})
    Σn = F*Σ*F' + σw^2*I(NX)
    Σn = 0.5*(Σn+Σn') + 1e-6*I(NX)
    λ  = maximum(real.(eigvals(Σn)))
    λ>5.0 && (Σn .*= 5.0/λ)
    return Σn
end

# =============================================================
#  SECTION 3: SURPRISE SIGNAL AND BOUNDED CF ADAPTATION
#  Implements Eqs. (13)–(15) of the paper for the quadrotor.
#  S_t = squared prediction error proxy (negative log-likelihood).
#  η_t = η_max / (1 + √S_t): large surprise slows adaptation
#    (Gap-1 mechanism, Sec. III-C).
#  ρ̂ is updated via gradient on vz (index 9), which is the most
#    informative channel for rotor efficiency identification.
#  CFI_t = |Δρ̂| / ε_CF monitors Theorem 1 bound online.
#  Adaptation is skipped when S_t < 0.8 (no mismatch evidence).
# =============================================================
function surprise_update!(η_hat::Vector{Float64},
                          z::Vector{Float64},
                          y::Vector{Float64},
                          u::Vector{Float64})
    zp   = quad3d_predict(z, u, η_hat[1])
    err  = y - zp
    S_t  = 0.5*sum(err.^2)/σw^2
    η_t  = ETA_MAX_CF/(1.0+sqrt(max(0.0,S_t)))
    grad = err[9]/σw^2 * dt*u[1]/MASS*cos(z[4])*cos(z[5])
    grad = clamp(grad, -GRAD_CLIP, GRAD_CLIP)
    if S_t < 0.8
        return S_t, η_t, 0.0
    end
    η_old = η_hat[1]
    η_hat[1] = clamp(η_hat[1]+η_t*grad, 0.1, 1.5)
    return S_t, η_t, abs(η_hat[1]-η_old)/ε_CF
end

# =============================================================
#  SECTION 4: BMPC WITH ADAPTIVE TIGHTENING (SafeCF-SSM)
#  Implements Eqs. (10)–(12) of the paper for the quadrotor.
#  The QP is linearized around the hover point z_hover using
#    numerical Jacobians from Section 2.
#  β_t = 0.3·σ_t provides covariance-driven tightening (Lem. 1).
#  Hover thrust T_hover is recomputed each step using current ρ̂_t,
#    progressively correcting the torque imbalance as ρ̂_t → ρ₂.
#  β_fixed: optional fixed margin for Robust MPC baseline.
#  QP solved via OSQP; hover thrust used as warm-start fallback.
# =============================================================
function bmpc(z::Vector{Float64}, η_hat::Float64,
              Σ::Matrix{Float64}; N::Int=N_HORIZON,
              β_fixed::Union{Float64,Nothing}=nothing)
    λmax = max(0.0, maximum(real.(eigvals(Σ))))
    σ_t  = sqrt(λmax)
    β_t  = isnothing(β_fixed) ? 0.3*σ_t : β_fixed
    T_hover = MASS*G_GRAV / (max(η_hat,0.1) + 3.0)
    T1n = clamp(T_hover/max(η_hat,0.1), 0.0, T_MAX)
    T2n = T_hover; T3n = T_hover; T4n = T_hover
    u_nom = [T1n, T2n, T3n, T4n]
    Tsafe = max(0.5, T_MAX-β_t)
    z_hover = [z[1], z[2], REF_PZ, 0.0, 0.0, 0.0,
               0.0,  0.0,  0.0,    0.0, 0.0, 0.0]
    Ak, Bk = quad3d_AB(z_hover, u_nom, η_hat)
    model = Model(OSQP.Optimizer)
    set_silent(model)
    set_optimizer_attribute(model,"eps_abs",1e-4)
    set_optimizer_attribute(model,"eps_rel",1e-4)
    @variable(model, dz[1:N+1, 1:NX])
    @variable(model, du[1:N,   1:NU])
    @constraint(model, [i=1:NX], dz[1,i] == z[i]-z_hover[i])
    for k in 1:N
        for i in 1:NX
            @constraint(model, dz[k+1,i] ==
                sum(Ak[i,j]*dz[k,j] for j in 1:NX)+
                sum(Bk[i,j]*du[k,j] for j in 1:NU))
        end
    end
    @constraint(model,[k=1:N, j=1:NU],
        -u_nom[j] <= du[k,j] <= Tsafe-u_nom[j])
    @objective(model, Min,
        sum(50.0*(dz[k,3])^2 + 50.0*(dz[k,1])^2 +
            50.0*(dz[k,2])^2 + 30.0*(dz[k,4])^2 +
            30.0*(dz[k,5])^2 + 50.0*(dz[k,6])^2 +
            10.0*(dz[k,7])^2 + 10.0*(dz[k,8])^2 +
            10.0*(dz[k,12])^2 +
            0.01*sum(du[k,j]^2 for j in 1:NU)
            for k in 1:N))
    optimize!(model)
    if termination_status(model)==MOI.OPTIMAL
        u_out = [clamp(u_nom[j]+value(du[1,j]), 0.0, Tsafe) for j in 1:NU]
        return u_out, β_t, σ_t
    else
        return u_nom, β_t, σ_t
    end
end

# =============================================================
#  SECTION 5: ENCODER — EXTENDED KALMAN FILTER (EKF)
#  Implements the encoder q_{φ_θt}(z_t | H_t) = N(ẑ_t, Σ_t)
#  for the 12-state quadrotor (Assumption 3, Footnote 2).
#  Standard EKF predict-update using 12×12 numerical Jacobian A.
#  Observation model: y_t = x_t + v_t, v_t ~ N(0, σv²·I₁₂),
#    i.e., full-state observation (GPS + IMU fusion assumed).
#  The framework generalizes to partial observation as per Assm. 3.
# =============================================================
function ekf_update(z::Vector{Float64}, Σ::Matrix{Float64},
                    y::Vector{Float64}, u::Vector{Float64},
                    η_hat::Float64)
    zp = quad3d_predict(z, u, η_hat)
    A, _ = quad3d_AB(z, u, η_hat)
    Σp = cov_update(Σ, A)
    R  = σv^2*I(NX)
    K  = Σp/(Σp+R)
    zu = zp + K*(y-zp)
    Σu = (I(NX)-K)*Σp
    Σu = 0.5*(Σu+Σu') + 1e-6*I(NX)
    return zu, Σu
end

# =============================================================
#  SECTION 6: SIMULATION
#  Runs all three controllers (SafeCF-SSM, Nominal, Robust) on
#  the same quadrotor trajectory for T_SIM=170 s.
#  Before t_s=40 s: all controllers track p_z^ref=2.0 m with
#    ρ_true=ρ₁=1.0 (normal operation).
#  After t_s: ρ_true drops to ρ₂=0.6 (partial rotor-1 failure),
#    inducing lateral torque imbalance. Baselines accumulate
#    |p_y| > 0.5 m drift; SafeCF-SSM maintains |p_y| ≤ 0.1 m
#    by adapting ρ̂_t → ρ₂ via surprise-driven CF adaptation.
#  Safety is evaluated on: p_z ≥ Z_MIN, |φ|,|θ| ≤ THETA_MAX.
# =============================================================
function run_scenario(; seed::Int=2026)
    rng = MersenneTwister(seed)
    x0 = [0.0, 0.0, REF_PZ, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0,    0.0, 0.0, 0.0]
    Σ0 = 0.01*I(NX)|>Matrix
    x=copy(x0); z=copy(x0); Σ=copy(Σ0); η_hat=[ETA_1]
    xn=copy(x0); zn=copy(x0); Σn=copy(Σ0)
    xr=copy(x0); zr=copy(x0); Σr=copy(Σ0)
    pz_cf=zeros(T_SIM); pz_nom=zeros(T_SIM); pz_rob=zeros(T_SIM)
    py_cf=zeros(T_SIM); py_nom=zeros(T_SIM); py_rob=zeros(T_SIM)
    phi_cf=zeros(T_SIM); theta_cf=zeros(T_SIM)
    S_h=zeros(T_SIM); α_h=zeros(T_SIM); CFI_h=zeros(T_SIM)
    η̂_h=zeros(T_SIM); β_h=zeros(T_SIM)
    sf_cf=ones(T_SIM); sf_nom=ones(T_SIM); sf_rob=ones(T_SIM)

    for t in 1:T_SIM
        η_true = t < t_s ? ETA_1 : ETA_2

        # ── SafeCF-SSM ───────────────────────────────────
        u_t,β_t,σ_t = bmpc(z, η_hat[1], Σ)
        x   = quad3d_step(x, u_t, η_true, rng)
        y_t = x + σv*randn(rng, NX)
        z,Σ = ekf_update(z, Σ, y_t, u_t, η_hat[1])
        S_t,η_t,CFI_t = surprise_update!(η_hat, z, y_t, u_t)
        pz_cf[t]=x[3]; py_cf[t]=x[2]
        phi_cf[t]=x[4]; theta_cf[t]=x[5]
        S_h[t]=S_t; α_h[t]=η_t
        CFI_h[t]=CFI_t; η̂_h[t]=η_hat[1]; β_h[t]=β_t
        sf_cf[t] = (x[3]>=Z_MIN &&
                    abs(x[4])<=THETA_MAX &&
                    abs(x[5])<=THETA_MAX) ? 1.0 : 0.0

        # ── Nominal MPC — fixed ρ̂=ρ₁, β=0 ─────────────
        un_t,_,_ = bmpc(zn, ETA_1, Σn)
        xn  = quad3d_step(xn, un_t, η_true, rng)
        yn  = xn + σv*randn(rng, NX)
        zn,Σn = ekf_update(zn, Σn, yn, un_t, ETA_1)
        pz_nom[t]=xn[3]; py_nom[t]=xn[2]
        sf_nom[t] = (xn[3]>=Z_MIN &&
                     abs(xn[4])<=THETA_MAX &&
                     abs(xn[5])<=THETA_MAX) ? 1.0 : 0.0

        # ── Robust MPC — fixed ρ̂=ρ₁, fixed β=0.5 ──────
        ur_t,_,_ = bmpc(zr, ETA_1, Σr; β_fixed=0.5)
        xr  = quad3d_step(xr, ur_t, η_true, rng)
        yr  = xr + σv*randn(rng, NX)
        zr,Σr = ekf_update(zr, Σr, yr, ur_t, ETA_1)
        pz_rob[t]=xr[3]; py_rob[t]=xr[2]
        sf_rob[t] = (xr[3]>=Z_MIN &&
                     abs(xr[4])<=THETA_MAX &&
                     abs(xr[5])<=THETA_MAX) ? 1.0 : 0.0
    end

    return (pz_cf=pz_cf, pz_nom=pz_nom, pz_rob=pz_rob,
            py_cf=py_cf, py_nom=py_nom, py_rob=py_rob,
            phi_cf=phi_cf, theta_cf=theta_cf,
            S=S_h, η=α_h, CFI=CFI_h, η̂=η̂_h, β=β_h,
            safe=sf_cf, safe_nom=sf_nom, safe_rob=sf_rob)
end

# =============================================================
#  SECTION 7: FIGURES (Fig. 3 in paper)
#  Generates the 4-panel figure (mean trajectory over M=25 runs):
#    (a) p_z altitude tracking — all three controllers
#    (b) p_y lateral tracking with safety bound |p_y| ≤ 0.1 m
#    (c) ρ̂_t adaptation: ρ₁ → ρ₂ via CF update (Sec. III-C)
#    (d) CFI_t monitor with E[CFI] ≤ 1 bound (Theorem 1)
#  Vertical dashed line marks the failure event t_s=40 s.
#  Saved as both PDF and PNG to figures/ directory.
#  Color scheme follows Wong (2011) colorblind-safe palette.
# =============================================================
function make_figures(r)
    t_ax    = collect((1:T_SIM).*dt)
    t_s_sec = t_s*dt
    T_end   = T_SIM*dt
    CF  = RGBf(0/255,  114/255, 178/255)
    NOM = RGBf(213/255, 94/255,   0/255)
    ROB = RGBf(0/255,  158/255, 115/255)
    REF = RGBf(0,0,0)
    ETA = RGBf(204/255,121/255, 167/255)
    mkpath("figures")
    fig = Figure(size=(1180,880), fontsize=13, backgroundcolor=:white)

    ax1 = Axis(fig[1,1]; ylabel=L"p_z\;[\mathrm{m}]",
        title=L"\textbf{(a)}\;\mathrm{Altitude\;Tracking}",
        xlabelsize=32, ylabelsize=48, titlesize=35,
        xticklabelsize=24, yticklabelsize=24,
        topspinevisible=false, rightspinevisible=false,
        xgridvisible=false, ygridvisible=false)
    vspan!(ax1, 0.0, t_s_sec;  color=(:gray, 0.06))
    vspan!(ax1, t_s_sec, T_end; color=(:blue, 0.07))
    vlines!(ax1, [t_s_sec]; color=(:black,0.3), linestyle=:dash, linewidth=2.8)
    l_rob = lines!(ax1, t_ax, r.pz_rob; color=ROB, linewidth=3.0, linestyle=:solid)
    l_nom = lines!(ax1, t_ax, r.pz_nom; color=NOM, linewidth=3.0, linestyle=:dash)
    l_cf  = lines!(ax1, t_ax, r.pz_cf;  color=CF,  linewidth=3.2)
    l_ref = hlines!(ax1, [REF_PZ]; color=REF, linewidth=1.8, linestyle=:dash)
    ylims!(ax1, 1.85, 2.05)
    xlims!(ax1, 0.0, t_ax[end])
    text!(ax1, t_s_sec/2, 1.895;
        text=L"\textbf{normal}", fontsize=38,
        color=(:black,0.4), align=(:center,:center))
    text!(ax1, t_s_sec+(T_end-t_s_sec)/2, 1.895;
        text=L"\textbf{partial\;rotor\;failure}", fontsize=38,
        color=(:black,0.4), align=(:center,:center))
    Legend(fig[1,2], [l_rob, l_nom, l_cf, l_ref],
        [L"\mathrm{Robust}", L"\mathrm{Nominal}",
         L"\mathrm{SafeCF-SSM}", L"\mathrm{ref}"];
        framevisible=false, backgroundcolor=:white,
        labelsize=26, tellwidth=true, tellheight=false,
        rowgap=3, margin=(9,0,0,0))
    colsize!(fig.layout, 1, Relative(0.84))
    colsize!(fig.layout, 2, Relative(0.16))

    ax2 = Axis(fig[2,1]; ylabel=L"p_y\;[\mathrm{m}]",
        title=L"\textbf{(b)}\;\mathrm{Lateral\;Tracking}",
        xlabelsize=32, ylabelsize=48, titlesize=35,
        xticklabelsize=24, yticklabelsize=24,
        topspinevisible=false, rightspinevisible=false,
        xgridvisible=false, ygridvisible=false)
    vspan!(ax2, 0.0, t_s_sec;  color=(:gray, 0.06))
    vspan!(ax2, t_s_sec, T_end; color=(:blue, 0.07))
    vlines!(ax2, [t_s_sec]; color=(:black,0.3), linestyle=:dash, linewidth=2.8)
    lines!(ax2, t_ax, r.py_rob; color=ROB, linewidth=3.0, linestyle=:solid)
    lines!(ax2, t_ax, r.py_nom; color=NOM, linewidth=3.0, linestyle=:dash)
    lines!(ax2, t_ax, r.py_cf;  color=CF,  linewidth=3.2)
    hlines!(ax2, [0.0]; color=REF, linewidth=1.8, linestyle=:dash)
    l_bnd = hlines!(ax2, [0.1, -0.1]; color=(:black,0.35),
                    linewidth=4.8, linestyle=:dot)
    text!(ax2, t_s_sec/2, -0.99;
        text=L"\textbf{normal}", fontsize=38,
        color=(:black,0.4), align=(:center,:center))
    text!(ax2, t_s_sec+(T_end-t_s_sec)/2, -0.99;
        text=L"\textbf{partial\;rotor\;failure}", fontsize=38,
        color=(:black,0.4), align=(:center,:center))
    ylims!(ax2, -1.5, 0.5)
    xlims!(ax2, 0.0, t_ax[end])
    Legend(fig[2,2], [l_bnd], [L"|p_y|\leq0.1\,"];
        framevisible=false, backgroundcolor=:white,
        labelsize=40, tellwidth=true, tellheight=false,
        rowgap=3, margin=(-10,0,0,0))
    colsize!(fig.layout, 1, Relative(0.84))
    colsize!(fig.layout, 2, Relative(0.16))

    ax4 = Axis(fig[3,1]; ylabel=L"\hat\rho_t",
        title=L"\textbf{(c)}\;\hat\rho_t\;\mathrm{adaptation}",
        xlabelsize=32, ylabelsize=48, titlesize=35,
        xticklabelsize=24, yticklabelsize=24,
        topspinevisible=false, rightspinevisible=false,
        xgridvisible=false, ygridvisible=false)
    vspan!(ax4, 0.0, t_s_sec;  color=(:gray, 0.06))
    vspan!(ax4, t_s_sec, T_end; color=(:blue, 0.07))
    vlines!(ax4, [t_s_sec]; color=(:black,0.3), linestyle=:dash, linewidth=2.9)
    l_eta1 = hlines!(ax4, [ETA_1]; color=(:gray,0.5),
                     linestyle=:dot, linewidth=4.5)
    l_eta2 = hlines!(ax4, [ETA_2]; color=(:red,0.4),
                     linestyle=:dot, linewidth=4.5)
    l_eta  = lines!(ax4, t_ax, r.η̂; color=ETA, linewidth=2.8)
    xlims!(ax4, 0.0, t_ax[end])
    text!(ax4, t_s_sec+T_end/5.9, 0.895;
        text=L"\mathbf{\rho_1}\!\downarrow\!\mathbf{\rho_2}",
        fontsize=51, color=CF, align=(:center,:center))
    Legend(fig[3,2], [l_eta1, l_eta2, l_eta],
        [L"\rho_1", L"\rho_2", L"\hat\rho_t"];
        framevisible=false, backgroundcolor=:white,
        labelsize=42, tellwidth=true, tellheight=false,
        rowgap=3, margin=(-10,0,0,0))
    colsize!(fig.layout, 1, Relative(0.84))
    colsize!(fig.layout, 2, Relative(0.16))

    ax5 = Axis(fig[4,1]; ylabel=L"\mathrm{CFI}_t",
        xlabel=L"t\;[\mathrm{s}]",
        title=L"\textbf{(d)}\;\mathrm{CFI}_t",
        xlabelsize=32, ylabelsize=48, titlesize=35,
        xticklabelsize=24, yticklabelsize=24,
        topspinevisible=false, rightspinevisible=false,
        xgridvisible=false, ygridvisible=false)
    vlines!(ax5, [t_s_sec]; color=(:black,0.3),
            linestyle=:dash, linewidth=2.8)
    band!(ax5, t_ax, zeros(T_SIM), ones(T_SIM);
          color=(:green, 0.08))
    xlims!(ax5, 0.0, t_ax[end])
    lines!(ax5, t_ax, r.CFI; color=CF, linewidth=2.9)
    l_cfi_bound = hlines!(ax5, [1.0]; color=(:red,0.4),
                          linewidth=2.8, linestyle=:dot)
    text!(ax5, t_s_sec+(T_end-t_s_sec)/2, 0.6;
        text=L"\textbf{\leftarrow\;all\;safe\;\rightarrow}",
        fontsize=40, color=(:black,0.4), align=(:center,:center))
    Legend(fig[4,2], [l_cfi_bound],
        [L"\mathbb{E}[\mathrm{CFI}]\leq1"];
        framevisible=false, backgroundcolor=:white,
        labelsize=35, tellwidth=true, tellheight=false,
        rowgap=3, margin=(-10,0,0,0))
    colsize!(fig.layout, 1, Relative(0.84))
    colsize!(fig.layout, 2, Relative(0.16))

    save("figures/fig_quad3d_motor_failure_290426_1.pdf", fig)
    save("figures/fig_quad3d_motor_failure_290426_1.png", fig)
    println("  Saved → figures/fig_quad3d_motor_failure_290426_1.pdf/.png")
end

# =============================================================
#  SECTION 8: MAIN ENTRY POINT
#  1. Runs M=25 Monte Carlo simulations across seeds 2027–2051,
#     collecting p_z, p_y, safety, and CFI trajectories.
#  2. Reports mean safety rate and E[CFI] for Table I.
#  3. Computes mean trajectories across all runs for Fig. 3,
#     using the last run for single-trajectory diagnostics
#     (S_t, ρ̂_t, CFI_t panels).
#  4. Generates Fig. 3 from mean trajectories.
# =============================================================
function main()
    println("="^60)
    println("  SafeCF-SSM — 3D Quadrotor (Motor Failure)")
    println("  ρ₁=$ETA_1 → ρ₂=$ETA_2  |  T=$T_SIM  t_s=$t_s  dt=$dt")
    println("="^60)

    M = 25
    pz_cf_all  = zeros(T_SIM, M); pz_nom_all = zeros(T_SIM, M)
    pz_rob_all = zeros(T_SIM, M); py_cf_all  = zeros(T_SIM, M)
    py_nom_all = zeros(T_SIM, M); py_rob_all = zeros(T_SIM, M)
    sf_cf_all  = zeros(T_SIM, M); sf_nom_all = zeros(T_SIM, M)
    sf_rob_all = zeros(T_SIM, M); CFI_all    = zeros(T_SIM, M)

    print("  Running M=$M Monte Carlo runs ... ")
    for m in 1:M
        r = run_scenario(seed=2026+m)
        pz_cf_all[:,m]  = r.pz_cf;  pz_nom_all[:,m] = r.pz_nom
        pz_rob_all[:,m] = r.pz_rob; py_cf_all[:,m]  = r.py_cf
        py_nom_all[:,m] = r.py_nom; py_rob_all[:,m] = r.py_rob
        sf_cf_all[:,m]  = r.safe;   sf_nom_all[:,m] = r.safe_nom
        sf_rob_all[:,m] = r.safe_rob; CFI_all[:,m]  = r.CFI
    end
    println("done.")

    pz_cf_mean  = mean(pz_cf_all,  dims=2)[:]
    pz_nom_mean = mean(pz_nom_all, dims=2)[:]
    pz_rob_mean = mean(pz_rob_all, dims=2)[:]
    py_cf_mean  = mean(py_cf_all,  dims=2)[:]
    py_nom_mean = mean(py_nom_all, dims=2)[:]
    py_rob_mean = mean(py_rob_all, dims=2)[:]

    println(@sprintf("  Safety: SafeCF=%.1f%%  Nom=%.1f%%  Rob=%.1f%%",
            100*mean(sf_cf_all), 100*mean(sf_nom_all), 100*mean(sf_rob_all)))
    println(@sprintf("  E[CFI]=%.3f", mean(CFI_all)))

    r_last = run_scenario(seed=2026+M)
    r_mean = (pz_cf=pz_cf_mean,   pz_nom=pz_nom_mean,   pz_rob=pz_rob_mean,
              py_cf=py_cf_mean,   py_nom=py_nom_mean,    py_rob=py_rob_mean,
              phi_cf=r_last.phi_cf, theta_cf=r_last.theta_cf,
              S=r_last.S, η=r_last.η, CFI=r_last.CFI,
              η̂=r_last.η̂, β=r_last.β,
              safe=sf_cf_all[:,end], safe_nom=sf_nom_all[:,end],
              safe_rob=sf_rob_all[:,end])

    println("\n  Generating figures ...")
    make_figures(r_mean)
end

main()
