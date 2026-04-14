# =============================================================
#  fig3_scaling.jl
#  L-CSS Paper 2 — Figure 3 (Fig. 4 in submitted version)
#  Nuchkrua, Boonto
#
#  PURPOSE:
#    Validates Theorem 2 (Finite Expected Switching) by showing
#    that E[N_T] scales as (delta - 2*eps_bar)^{-1}.
#
#  Panel (a): E[N_T] vs alpha = delta/eps_bar  (log scale)
#             Shows monotonic decay as margin ratio increases.
#             Empirical E[N_T] stays below Theorem 2 bound.
#
#  Panel (b): E[N_T] vs (delta - 2*eps_bar)^{-1}  (linear)
#             Confirms near-linear relationship predicted by
#             Theorem 2 bound equation (15) in paper.
#
#  Created and Coded by Thanana, copyright
# =============================================================

using Random, Distributions, StatsBase, Statistics
using CairoMakie, Colors, LinearAlgebra
using Printf, LaTeXStrings

# =============================================================
#  GLOBAL CONSTANTS
#  EPS_BAR : noise bound epsilon-bar (Assumption 1 in paper)
#  T_FIG3  : time horizon T
#  N_FIG3  : number of particles Np in bootstrap PF
#  M_FIG3  : number of Monte Carlo runs M
# =============================================================
const EPS_BAR = 1.5
const T_FIG3  = 200
const N_FIG3  = 500
const M_FIG3  = 100

# Alpha values to sweep: alpha = delta / eps_bar
# Range [2.1, 8.0] covers just above threshold (alpha > 2)
# to well-protected regime (alpha = 8)
ALPHA_VALS = [2.1, 2.2, 2.5, 2.8, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0]
ALPHA_EXP  = 2.5   # alpha used in Experiments IV-B to IV-D

# =============================================================
#  SYSTEM MODEL
#  f_true3 : true nonlinear state transition (eq. 19 in paper)
#            canonical stochastic growth model
#            [Gordon 1993, Arulampalam 2002]
#  obs3    : quadratic observation model (eq. 20 in paper)
# =============================================================
f_true3(z,t) = 0.5*z + 25*z/(1+z^2) + 8*cos(1.2*t)
obs3(z)      = z^2/20.0

# =============================================================
#  logsumexp(v)
#
#  PURPOSE:
#    Numerically stable log-sum-exp for particle filter
#    log-likelihood computation.
#    Avoids numerical underflow when computing log of
#    sum of exponentials of potentially large negative numbers.
#
#  INPUT:
#    v : vector of log-weights
#
#  OUTPUT:
#    scalar log(sum(exp(v)))
# =============================================================
function logsumexp(v::Vector{Float64})
    m = maximum(v)
    return m + log(sum(exp.(v .- m)))
end

# =============================================================
#  pf_step3!(p, y, t, structure, rng)
#
#  PURPOSE:
#    One step of bootstrap particle filter (Algorithm 1 in paper).
#    Propagates particles, computes log-likelihood of observation,
#    reweights, and resamples using systematic resampling.
#
#  INPUTS:
#    p         : particle vector (modified in-place)
#    y         : current observation y_{t+1}
#    t         : current time step
#    structure : :nl  → nonlinear model (true dynamics)
#                :lin → linear misspecified model (a=0.5)
#    rng       : random number generator
#
#  OUTPUT:
#    ll : log-likelihood log ℓ_{θ,s}(y_{t+1} | B_t, u_t)
#         used to compute innovation score Φ_t(s) = -ll
#
#  NOTE:
#    Uses systematic resampling for low variance.
#    Independent noise for each structure (no shared ε).
# =============================================================
function pf_step3!(p::Vector{Float64}, y::Float64, t::Int,
                   structure::Symbol, rng::AbstractRNG)
    n  = length(p)

    # Propagate particles through transition model
    # :nl  → true nonlinear dynamics f_true3
    # :lin → misspecified linear model z_{t+1} = 0.5*z_t
    pp = (structure == :nl ? f_true3.(p, t) : 0.5.*p) .+
         sqrt(10.0) .* randn(rng, n)

    # Compute log-weights from observation likelihood
    lw = logpdf.(Normal.(obs3.(pp), 1.0), y)

    # Log-likelihood of observation (innovation likelihood)
    ll = logsumexp(lw) - log(n)

    # Normalize weights for resampling
    w  = exp.(lw .- maximum(lw)); w ./= sum(w)

    # Systematic resampling
    cdf_w = cumsum(w)
    u0    = rand(rng)/n
    idx   = [searchsortedfirst(cdf_w, u0+(i-1)/n) for i in 1:n]
    clamp!(idx, 1, n)
    copyto!(p, pp[idx])

    return ll
end

# =============================================================
#  run_robust_cf(alpha, rng)
#
#  PURPOSE:
#    Simulates one Monte Carlo run of the Robust CF mechanism
#    (eq. 9 in paper) over horizon T_FIG3.
#    Counts total switches N_T and cumulative suboptimality
#    for Theorem 2 bound computation.
#
#  INPUTS:
#    alpha : margin ratio δ/ε̄  (determines δ = alpha * EPS_BAR)
#    rng   : random number generator (for reproducibility)
#
#  OUTPUTS:
#    sw  : total switch count N_T = Σ 1{s_{t+1} ≠ s_t}
#    sub : cumulative suboptimality Σ max(0, Φ_t(s_t) - min_s Φ_t(s))
#          used to evaluate Theorem 2 bound (eq. 15)
#
#  DETAILS:
#    - Both PFs (NL and LIN) initialized from N(0,5) prior
#    - Noisy scores: Ph = Φ + ε where ε ~ Uniform(-ε̄, ε̄)
#      independently for each structure (Assumption 1)
#    - Margin-based switching rule (eq. 9):
#      switch only if s_hat ≠ s_t AND (Φ_curr - Φ_cand) > δ
#    - All runs initialized at s_0 = s_nl (correct structure)
#      to isolate noise-induced spurious switching
# =============================================================
function run_robust_cf(alpha::Float64, rng::AbstractRNG)
    delta = alpha * EPS_BAR

    # Initialize particles for both structures from prior
    pnl = randn(rng, N_FIG3) .* sqrt(5.0)   # NL particles
    pli = randn(rng, N_FIG3) .* sqrt(5.0)   # LIN particles
    z   = randn(rng) * sqrt(5.0)             # true initial state

    s_t = :nl    # start at correct structure s* = s_nl
    sw  = 0      # switch counter N_T
    sub = 0.0    # cumulative suboptimality for Thm.2 bound

    for t in 1:T_FIG3
        # Generate true state and observation
        z = f_true3(z, t) + randn(rng)*sqrt(10.0)
        y = obs3(z) + randn(rng)

        # Compute innovation scores via PF log-likelihood
        ll_nl = pf_step3!(pnl, y, t, :nl,  rng)
        ll_li = pf_step3!(pli, y, t, :lin, rng)
        Phi_nl  = -ll_nl     # innovation score for NL
        Phi_lin = -ll_li     # innovation score for LIN

        # Add independent noise (Assumption 1: |ε| ≤ ε̄ a.s.)
        Ph_nl  = Phi_nl  + rand(rng, Uniform(-EPS_BAR, EPS_BAR))
        Ph_lin = Phi_lin + rand(rng, Uniform(-EPS_BAR, EPS_BAR))

        # Accumulate suboptimality for Theorem 2 bound
        phi_a = s_t == :nl ? Phi_nl : Phi_lin
        sub  += max(0.0, phi_a - min(Phi_nl, Phi_lin))

        # Margin-based CF switching rule (eq. 9 in paper)
        # Switch only if improvement exceeds margin δ
        if t < T_FIG3
            s_hat  = Ph_nl <= Ph_lin ? :nl : :lin   # greedy selector
            Phi_c  = s_t == :nl ? Ph_nl  : Ph_lin   # current score
            Phi_b  = s_t == :nl ? Ph_lin : Ph_nl    # challenger score
            if s_hat != s_t && (Phi_c - Phi_b) > delta
                s_t = s_hat
                sw += 1
            end
        end
    end

    return sw, sub
end

# =============================================================
#  main_fig3()
#
#  PURPOSE:
#    Main function that:
#    1. Sweeps alpha ∈ [2.1, 8.0] and runs M Monte Carlo
#       simulations at each alpha value
#    2. Computes empirical E[N_T] and Theorem 2 bound
#    3. Fits linear slope for Panel (b)
#    4. Generates Figure 3 (1-column, 2-row layout)
#    5. Saves as both .pdf (for LaTeX) and .png (for verification)
#
#  THEOREM 2 BOUND (eq. 15):
#    E[N_T] ≤ (1/(δ - 2ε̄)) * Σ E[Φ_t(s_t) - min_s Φ_t(s)]
#    Evaluated empirically using M=100 MC runs of Robust CF.
# =============================================================
function main_fig3()
    println("="^56)
    println("  Figure 3 — (delta-2*eps_bar)^{-1} scaling")
    println("  eps_bar=$EPS_BAR  M=$M_FIG3  T=$T_FIG3  N=$N_FIG3")
    println("="^56)

    NT_mean = Float64[]; NT_std  = Float64[]
    bounds  = Float64[]; inv_d   = Float64[]

    # ── Sweep over alpha values ────────────────────────────────
    for alpha in ALPHA_VALS
        delta = alpha * EPS_BAR
        denom = delta - 2*EPS_BAR   # (δ - 2ε̄) denominator

        NT = Float64[]; subs = Float64[]

        # Run M Monte Carlo simulations at this alpha
        for m in 1:M_FIG3
            rng_m = MersenneTwister(2025+m)   # reproducible seed
            sw, sub = run_robust_cf(alpha, rng_m)
            push!(NT, sw); push!(subs, sub)
        end

        # Store mean, std error, Theorem 2 bound, inverse denom
        push!(NT_mean, mean(NT))
        push!(NT_std,  std(NT)/sqrt(M_FIG3))
        push!(bounds,  mean(subs)/denom)   # Theorem 2 bound
        push!(inv_d,   1.0/denom)

        @printf("  α=%4.1f  E[N_T]=%5.2f±%4.2f  bound=%7.1f\n",
                alpha, mean(NT), std(NT)/sqrt(M_FIG3), mean(subs)/denom)
    end

    # ── Linear fit for Panel (b) ───────────────────────────────
    # Fit E[N_T] = slope * (δ - 2ε̄)^{-1} through origin
    # Consistent with Theorem 2: slope ≈ cumulative suboptimality
    slope = sum(inv_d .* NT_mean) / sum(inv_d .^ 2)
    println("\n  Linear slope (Panel B) = ", round(slope, digits=3))

    # ── Figure layout: 1-column, 2-row ────────────────────────
    fig = Figure(size=(500, 680), fontsize=13)

    # ── Panel (a): E[N_T] vs alpha (log scale) ────────────────
    # Validates monotonic decay of E[N_T] with increasing margin
    # Empirical (blue) always below Theorem 2 bound (red dashed)
    ax1 = Axis(fig[1,1];
        xlabel  = L"\alpha = \delta\,/\,\bar{\varepsilon}",
        ylabel  = L"\mathbb{E}[N_T]",
        title   = L"\textbf{(a)}\ \mathbb{E}[N_T]\ \mathrm{vs.\ margin\ ratio}\ \alpha",
        yscale  = log10,
        xlabelsize=20, ylabelsize=20, titlesize=18,
        xticklabelsize=18, yticklabelsize=18,
        ygridvisible       = true,
        yminorgridvisible  = true,
        yminorticksvisible = true,
        yminorticks        = IntervalsBetween(9),
        ygridcolor         = (:gray, 0.30),
        yminorgridcolor    = (:gray, 0.13),
        xgridvisible       = true,
        xgridcolor         = (:gray, 0.40))

    # Shaded band: ±1 standard error across M runs
    band!(ax1, ALPHA_VALS, max.(NT_mean .- NT_std, 1e-4), NT_mean .+ NT_std;
          color=(:royalblue, 0.18))
    l_emp = lines!(ax1, ALPHA_VALS, NT_mean;
                   color=:royalblue, linewidth=1.9)
    scatter!(ax1, ALPHA_VALS, NT_mean;
             color=:royalblue, markersize=11)
    l_bnd = lines!(ax1, ALPHA_VALS, bounds;
                   color=:tomato, linewidth=1.5, linestyle=:dash)
    scatter!(ax1, ALPHA_VALS, bounds;
             color=:tomato, marker=:rect, markersize=11)

    # Vertical dotted line at alpha=2.5 (used in Exp IV-B to IV-D)
    vlines!(ax1, [ALPHA_EXP];
            color=(:gray, 0.7), linestyle=:dot, linewidth=1.5)
    text!(ax1, ALPHA_EXP+0.22, 55.0;
          text=L"\alpha=2.5", fontsize=18, color=(:gray,0.85))

    axislegend(ax1,
        [l_emp, l_bnd],
        [L"\mathrm{Robust\ CF}\ (\mathrm{empirical}\ \mathbb{E}[N_T])",
         L"\mathrm{Theorem~2\ bound}"];
        framevisible=false, labelsize=16, position=:rt)
    xlims!(ax1, 2.05, 8.1)

    # ── Panel (b): E[N_T] vs (δ - 2ε̄)^{-1} (linear) ─────────
    # Validates linear scaling predicted by Theorem 2
    # Near-linear relationship confirms (δ - 2ε̄)^{-1} rate
    ax2 = Axis(fig[2,1];
        xlabel  = L"(\delta - 2\bar{\varepsilon})^{-1}",
        ylabel  = L"\mathbb{E}[N_T]\ (\mathrm{empirical})",
        title   = L"\textbf{(b)}\ \mathrm{Linear\ scaling\ (Theorem~2)}",
        xlabelsize=20, ylabelsize=20, titlesize=18,
        xticklabelsize=18, yticklabelsize=18)

    # Linear fit through origin: E[N_T] ≈ slope * (δ - 2ε̄)^{-1}
    x_line = range(0, stop=maximum(inv_d)*1.08, length=100)
    l_fit = lines!(ax2, collect(x_line), slope .* collect(x_line);
                   color=:tomato, linewidth=1.5, linestyle=:dash)

    # Error bars: ±1 standard error
    errorbars!(ax2, inv_d, NT_mean, NT_std;
               color=:royalblue, linewidth=1.5, whiskerwidth=7)
    l_pts = scatter!(ax2, inv_d, NT_mean;
                     color=:royalblue, markersize=11)

    axislegend(ax2,
        [l_fit, l_pts],
        [latexstring("\\mathrm{Linear\\ fit\\ (slope}\\approx$(round(slope,digits=2))\\mathrm{)}"),
         L"\mathrm{Robust\ CF}"];
        framevisible=false, labelsize=16, position=:lt)

    rowgap!(fig.layout, 40)

    # ── Save: both PDF (LaTeX) and PNG (visual verification) ──
    savebase = joinpath(@__DIR__, "..", "..", "figures", "fig_scaling_1")
    mkpath(dirname(savebase))
    save(savebase * ".pdf", fig)
    save(savebase * ".png", fig)

    println("\nSaved → ", savebase * ".pdf")
    println("Saved → ", savebase * ".png")
    display(fig)
end

main_fig3()
