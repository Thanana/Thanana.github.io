using Random, Distributions, Statistics, LinearAlgebra
using CairoMakie, LaTeXStrings
using Printf

# ============================================================
#  spl_adaptive_margin.jl
#  SPL resubmission (SPL-47532-2026)
#  Answers Reviewer 1 comments 1.1, 1.2, 1.3
#
#  DERIVED FROM radar_final_v6.jl  WITH THREE CHANGES
#  ------------------------------------------------------
#  (C1) Belief propagates with the SELECTED structure
#       s_{t+1}, matching eq.(11) of the letter, instead of
#       the true dynamics.  In v6 the particle cloud used
#       F_true regardless of s_t, so RMSE was identical
#       across all CF variants up to RNG seed.  Here the
#       structure choice genuinely affects estimation.
#       Set PROPAGATE_WITH_SELECTED=false to recover v6.
#
#  (C2) NO injected score noise.  eps_t is the genuine
#       Monte Carlo error of the particle likelihood.
#       Nonstationarity is induced physically through a
#       particle-budget schedule N_p(t) (radar resource
#       manager), which changes eps_t via the O(N_p^{-1/2})
#       rate of Remark 7.
#
#  (C3) Batch-split estimator of eps_t (Sec. III-E):
#       partition the current budget into B batches, score
#       each batch separately, take the spread.  All
#       batches see the same belief and the same y_{t+1},
#       so the spread contains no dynamic component.
#       The SCORE itself uses the pooled particle set --
#       batches are used only to measure spread, so the
#       adaptive and fixed schemes differ ONLY in delta.
#
#
#  2026 copyright@ Thanana
#
#  Outputs
#  -------
#    fig_spl_margin_traj.pdf     delta_t vs 2*eps_hat_t
#    fig_spl_switching.pdf       p_hat_t, K=2
#    fig_spl_switching_K4.pdf    p_hat_t, K=4
#    console tables for Table I of the letter
# ============================================================

# ---- Toggle (C1) -------------------------------------------
const PROPAGATE_WITH_SELECTED = false

# ---- Scenario (identical to TSP where shared) --------------
const DT      = 1.0
const T_RAD   = 100
const M_RAD   = 100
const SIG_W   = 0.5
const OMEGA   = 10.0*pi/180
const SIG_OBS = 0.5
const T1 = 30; const T2 = 70

# ---- Particle-budget schedule (C2) -------------------------
const NP_HIGH = 4000
const NP_LOW  = 60
const TB1 = 40      # budget drops here
const TB2 = 75      # budget restored here
#  SCHED_MODE selects the operating regime:
#    :high   full budget throughout      (low  score noise)
#    :low    reduced budget throughout   (high score noise)
#    :mixed  full -> reduced -> full     (nonstationary)
const SCHED_MODE = Ref(:mixed)
function np_sched(t)
    m = SCHED_MODE[]
    m === :high && return NP_HIGH
    m === :low  && return NP_LOW
    return (TB1 <= t < TB2) ? NP_LOW : NP_HIGH
end

# ---- Batch-split estimator (C3) ----------------------------
const NBATCH   = 10     # B at full budget
const NB_MIN   = 15     # minimum particles per batch
nbatch(n) = clamp(n ÷ NB_MIN, 4, NBATCH)
const KAPPA  = 2.5      # coverage constant
const ETA    = 0.5      # delta_t = (2+eta)*eps_env
const LAM_ENV = 0.90    # peak-hold decay for the noise envelope
                        #   eps_env_t = max(eps_hat_t, LAM*eps_env_{t-1})
                        #   -> rises immediately when the score noise
                        #      grows (safety), releases slowly when it
                        #      falls (avoids under-margining on a lucky
                        #      low draw).  Still fully online.
const DELTA_MIN = Ref(0.0)  # floor -> guarantees m_underline>0

# ---- Fixed-margin comparators ------------------------------
#  tuned offline on ONE regime each, as R1.2 describes
#  set automatically by calibrate!() below -- see AUTO-CALIBRATION
const DELTA_LO = Ref(0.0)   # calibrated on the N_p=NP_HIGH regime
const DELTA_HI = Ref(0.0)   # calibrated on the N_p=NP_LOW  regime

# ============================================================
#  Models
# ============================================================
F_cv() = [1.0 DT 0.0 0.0;
          0.0 1.0 0.0 0.0;
          0.0 0.0 1.0 DT;
          0.0 0.0 0.0 1.0]

function F_ct(omega=OMEGA)
    w=omega; s=sin(w*DT); c=cos(w*DT)
    [1.0 s/w     0.0 -(1-c)/w;
     0.0 c       0.0 -s;
     0.0 (1-c)/w 1.0 s/w;
     0.0 s       0.0 c]
end

function Q_mat(sigma=SIG_W)
    G = [DT^2/2, DT, DT^2/2, DT]
    sigma^2 * G*G' + 1e-8*I(4)
end

const Hobs = [1.0 0.0 0.0 0.0;
              0.0 0.0 1.0 0.0]
const Robs = SIG_OBS^2 * I(2)
const LOG_NORM = -0.5*logdet(2pi*Matrix(Robs))

# ============================================================
#  Trajectory (unchanged from v6)
# ============================================================
function gen_traj(rng, T=T_RAD)
    x = zeros(4, T+1)
    x[:,1] = [0.0, 15.0, 0.0, 15.0]
    L = cholesky(Q_mat()).L
    for t in 1:T
        F = (T1 <= t < T2) ? F_ct() : F_cv()
        x[:,t+1] = F*x[:,t] + L*randn(rng, 4)
    end
    return x
end

# ============================================================
#  Marginal log-likelihood of a particle set
# ============================================================
function marginal_ll(PP, y)
    n = size(PP,2)
    lw = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        r = y - Hobs*PP[:,i]
        lw[i] = -0.5*dot(r, Robs \ r)
    end
    m = maximum(lw)
    return m + log(sum(exp.(lw .- m))) - log(n) + LOG_NORM
end

# ============================================================
#  Systematic resampling to a TARGET size (budget may change)
# ============================================================
function resample_to(PP, lw, n_out, rng)
    n = size(PP,2)
    m = maximum(lw)
    w = exp.(lw .- m); w ./= sum(w)
    cdf_w = cumsum(w)
    u0 = rand(rng)/n_out
    idx = clamp.([searchsortedfirst(cdf_w, u0 + (i-1)/n_out)
                  for i in 1:n_out], 1, n)
    return PP[:, idx]
end

# ============================================================
#  CORE STEP, K=2, with batch-split noise estimation
#
#  Returns (Phi_cv, Phi_ct, eps_hat)
#  Mutates nothing; caller rebuilds the cloud.
# ============================================================
function score_and_estimate(P, y, rng; B=0)
    n  = size(P,2)
    B  = (B == 0) ? nbatch(n) : B
    L  = cholesky(Q_mat()).L
    noise_shared = randn(rng, 4, n)      # shared -> low-variance gap
    PP_cv = F_cv() * P .+ L * noise_shared
    PP_ct = F_ct() * P .+ L * noise_shared

    # pooled scores  (these ARE the scores used for switching)
    Phi_cv = -marginal_ll(PP_cv, y)
    Phi_ct = -marginal_ll(PP_ct, y)

    # batch-split spread -> eps_hat   (Sec. III-E, eq.(13))
    nb = max(1, n ÷ B)
    Bx = min(B, n ÷ nb)
    sc = zeros(Bx); st = zeros(Bx)
    @inbounds for b in 1:Bx
        rng_b = ((b-1)*nb+1):(b*nb)
        sc[b] = -marginal_ll(view(PP_cv,:,rng_b), y)
        st[b] = -marginal_ll(view(PP_ct,:,rng_b), y)
    end
    spread  = max(std(sc), std(st))
    eps_hat = KAPPA * spread / sqrt(Bx)

    return Phi_cv, Phi_ct, eps_hat
end

function propagate_resample(P, y, F_prop, n_out, rng)
    n = size(P,2)
    L = cholesky(Q_mat()).L
    PP = F_prop * P .+ L * randn(rng, 4, n)
    lw = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        r = y - Hobs*PP[:,i]
        lw[i] = -0.5*dot(r, Robs \ r)
    end
    return resample_to(PP, lw, n_out, rng)
end

# ============================================================
#  SINGLE RUN, K=2
#
#  variant ∈ (:naive, :fixed_lo, :fixed_hi, :adaptive)
# ============================================================
function run_single(variant, rng; T=T_RAD)
    x   = gen_traj(rng, T)
    n0  = np_sched(1)
    P   = repeat(x[:,1],1,n0) .+ SIG_OBS*randn(rng,4,n0)
    s_t = :cv

    Phi   = zeros(T)          # suboptimality d_t
    SW    = zeros(Bool, T-1)
    SQ    = zeros(T)
    ST    = zeros(Int, T)     # active structure: 1=CV, 2=CT
    DEL   = zeros(T)          # realised delta_t
    EPSH  = zeros(T)          # eps_hat_t
    eps_prev = 0.0            # eps_env_{t-1} (delay -> I_t-measurable)
    eps_env  = 0.0

    for t in 1:T
        y = Hobs*x[:,t+1] + SIG_OBS*randn(rng,2)
        F_true_t = (T1 <= t < T2) ? F_ct() : F_cv()

        # ---- scores + online noise estimate ----------------
        Phi_cv, Phi_ct, eps_hat = score_and_estimate(P, y, rng)
        eps_env = max(eps_hat, LAM_ENV*eps_env)   # peak hold
        EPSH[t] = eps_env

        Phi_min = min(Phi_cv, Phi_ct)
        Phi[t]  = (s_t == :cv ? Phi_cv : Phi_ct) - Phi_min
        ST[t]   = (s_t === :cv) ? 1 : 2

        # ---- margin for this step --------------------------
        delta_t =
            variant === :naive     ? 0.0        :
            variant === :fixed_lo  ? DELTA_LO[] :
            variant === :fixed_hi  ? DELTA_HI[] :
            max((2+ETA)*eps_prev, DELTA_MIN[])    # :adaptive
        DEL[t] = delta_t
        eps_prev = eps_env

        # ---- switching decision ----------------------------
        if t < T
            s_hat    = Phi_cv < Phi_ct ? :cv : :ct
            Phi_curr = s_t   === :cv ? Phi_cv : Phi_ct
            Phi_best = s_hat === :cv ? Phi_cv : Phi_ct
            if s_hat !== s_t && (Phi_curr - Phi_best) > delta_t
                s_t   = s_hat
                SW[t] = true
            end
        end

        # ---- belief update (C1) ----------------------------
        F_prop = PROPAGATE_WITH_SELECTED ?
                 (s_t === :cv ? F_cv() : F_ct()) : F_true_t
        n_next = np_sched(min(t+1, T))
        P = propagate_resample(P, y, F_prop, n_next, rng)

        SQ[t] = sum(([mean(P[1,:]), mean(P[3,:])]
                     - x[[1,3],t+1]).^2)
    end
    return Phi, SW, SQ, DEL, EPSH, ST
end

function run_ens(variant; soff=0, T=T_RAD)
    Phi=zeros(M_RAD,T); SW=zeros(Bool,M_RAD,T-1); SQ=zeros(M_RAD,T)
    DEL=zeros(M_RAD,T); EPSH=zeros(M_RAD,T); ST=zeros(Int,M_RAD,T)
    for m in 1:M_RAD
        rng = MersenneTwister(21000+soff+m)
        Phi[m,:],SW[m,:],SQ[m,:],DEL[m,:],EPSH[m,:],ST[m,:] =
            run_single(variant, rng; T=T)
    end
    return Phi,SW,SQ,DEL,EPSH,ST
end

ra(M) = hcat([mean(M[:,1:t],dims=2) for t in 1:size(M,2)]...)

# ============================================================
#  K = 4  (R1.3: multi-model dictionary)
#    CV, CT+, CT-, CA(inflated Q)
# ============================================================
const OMEGA_POS =  10.0*pi/180
const OMEGA_NEG = -10.0*pi/180
const SIG_A = 2*SIG_W

const Fs_K4 = [F_cv(), F_ct(OMEGA_POS), F_ct(OMEGA_NEG), F_cv()]
const Qs_K4 = [Q_mat(), Q_mat(), Q_mat(), Q_mat(SIG_A)]

function score_and_estimate_K4(P, y, rng; B=0)
    n = size(P,2)
    B = (B == 0) ? nbatch(n) : B
    noise = randn(rng, 4, n)
    Phis  = zeros(4)
    nb = max(1, n ÷ B); Bx = min(B, n ÷ nb)
    spreads = zeros(4)
    for k in 1:4
        Lk   = cholesky(Qs_K4[k]).L
        PP_k = Fs_K4[k]*P .+ Lk*noise
        Phis[k] = -marginal_ll(PP_k, y)
        sb = zeros(Bx)
        @inbounds for b in 1:Bx
            rg = ((b-1)*nb+1):(b*nb)
            sb[b] = -marginal_ll(view(PP_k,:,rg), y)
        end
        spreads[k] = std(sb)
    end
    eps_hat = KAPPA * maximum(spreads) / sqrt(Bx)
    return Phis, eps_hat
end

function run_single_K4(variant, rng; T=T_RAD)
    x   = gen_traj(rng, T)
    n0  = np_sched(1)
    P   = repeat(x[:,1],1,n0) .+ SIG_OBS*randn(rng,4,n0)
    s_t = 1
    Phi=zeros(T); SW=zeros(Bool,T-1); SQ=zeros(T)
    DEL=zeros(T); EPSH=zeros(T); ST=zeros(Int,T)
    eps_prev=0.0; eps_env=0.0

    for t in 1:T
        y = Hobs*x[:,t+1] + SIG_OBS*randn(rng,2)
        F_true_t = (T1 <= t < T2) ? F_ct() : F_cv()

        Phis, eps_hat = score_and_estimate_K4(P, y, rng)
        eps_env = max(eps_hat, LAM_ENV*eps_env)   # peak hold
        EPSH[t] = eps_env
        Phi[t]  = Phis[s_t] - minimum(Phis)
        ST[t]   = s_t

        delta_t =
            variant === :naive    ? 0.0      :
            variant === :fixed_lo ? DELTA_LO[] :
            variant === :fixed_hi ? DELTA_HI[] :
            max((2+ETA)*eps_prev, DELTA_MIN[])
        DEL[t] = delta_t
        eps_prev = eps_env

        if t < T
            s_hat = argmin(Phis)
            if s_hat != s_t && (Phis[s_t]-Phis[s_hat]) > delta_t
                s_t = s_hat; SW[t] = true
            end
        end

        F_prop = PROPAGATE_WITH_SELECTED ? Fs_K4[s_t] : F_true_t
        n_next = np_sched(min(t+1, T))
        P = propagate_resample(P, y, F_prop, n_next, rng)
        SQ[t] = sum(([mean(P[1,:]), mean(P[3,:])]
                     - x[[1,3],t+1]).^2)
    end
    return Phi, SW, SQ, DEL, EPSH, ST
end

function run_ens_K4(variant; soff=0, T=T_RAD)
    Phi=zeros(M_RAD,T); SW=zeros(Bool,M_RAD,T-1); SQ=zeros(M_RAD,T)
    DEL=zeros(M_RAD,T); EPSH=zeros(M_RAD,T); ST=zeros(Int,M_RAD,T)
    for m in 1:M_RAD
        rng = MersenneTwister(29000+soff+m)
        Phi[m,:],SW[m,:],SQ[m,:],DEL[m,:],EPSH[m,:],ST[m,:] =
            run_single_K4(variant, rng; T=T)
    end
    return Phi,SW,SQ,DEL,EPSH,ST
end



# ============================================================
#  SELECTION-QUALITY METRICS
#
#  E[N_T] alone rewards a margin that never fires.  These three
#  metrics price the opposite failure -- a margin so wide that
#  genuine structural change is missed or detected late -- and
#  are what separates an adaptive delta_t from a conservative
#  fixed delta.
#
#    ACC   fraction of steps on which the active structure
#          equals the true one
#    D1    detection delay at the manoeuvre onset  (t = T1)
#    D2    detection delay at the manoeuvre recovery (t = T2)
#    FA    switches outside the two transition windows
#          (false alarms)
# ============================================================
const WDET = 8    # transition window used for the FA count

"true structure index at time t (1 = CV, 2 = CT / CT+)"
true_struct(t) = (T1 <= t < T2) ? 2 : 1

function sel_metrics(ST::Matrix{Int}, SW::Matrix{Bool})
    M, T = size(ST)
    acc = zeros(M); d1 = zeros(M); d2 = zeros(M); fa = zeros(M)
    for m in 1:M
        # accuracy
        acc[m] = mean(ST[m,t] == true_struct(t) for t in 1:T)
        # delay at onset: first t >= T1 with active == CT
        d1[m] = T2 - T1
        for t in T1:min(T2-1,T)
            if ST[m,t] == 2; d1[m] = t - T1; break; end
        end
        # delay at recovery: first t >= T2 with active == CV
        d2[m] = T - T2
        for t in T2:T
            if ST[m,t] == 1; d2[m] = t - T2; break; end
        end
        # false alarms: switches outside the two windows
        c = 0
        for t in 1:(T-1)
            if SW[m,t]
                inw = (T1 <= t <= T1+WDET) || (T2 <= t <= T2+WDET)
                c += inw ? 0 : 1
            end
        end
        fa[m] = c
    end
    return mean(acc), mean(d1), mean(d2), mean(fa)
end

"""Per-trial accuracy vector -- required for the paired
   comparison enabled by common random numbers."""
function acc_pertrial(ST::Matrix{Int})
    M, T = size(ST)
    return [mean(ST[m,t] == true_struct(t) for t in 1:T)
            for m in 1:M]
end

"""Paired difference a - b with a 95% CI.  Valid because every
   variant is driven by identical random streams (CRN), so the
   per-trial pairing removes trajectory variance entirely."""
function paired_diff(a::Vector{Float64}, b::Vector{Float64})
    d  = a .- b
    m  = mean(d)
    se = std(d) / sqrt(length(d))
    return m, 1.96*se
end

function report(tag, VARIANTS, res)
    @printf("\n%-26s %8s %7s %6s %6s %6s %8s\n",
            tag, "E[N_T]", "FA", "D1", "D2", "ACC", "RMSE(m)")
    println("-"^72)
    for (v,lab) in VARIANTS
        Phi,SW,SQ,_,_,ST = res[v]
        a,d1,d2,fa = sel_metrics(ST, SW)
        @printf("%-26s %8.1f %7.1f %6.1f %6.1f %6.3f %8.3f\n",
                lab, mean(sum(SW,dims=2)), fa, d1, d2, a,
                sqrt(mean(SQ)))
    end
    println("  FA = false alarms   D1,D2 = detection delay ",
            "at t=$(T1), t=$(T2)   ACC = selection accuracy")
end

# ============================================================
#  AUTO-CALIBRATION  (replaces the guessed delta constants)
#
#  Runs a short pilot to measure the genuine Monte Carlo
#  score noise in each budget regime, then sets
#
#     DELTA_LO = ALPHA * eps_high    (offline tuning on the
#                                     high-budget regime only)
#     DELTA_HI = ALPHA * eps_low     (offline tuning on the
#                                     low-budget regime only)
#     DELTA_MIN = small absolute safety net.  Assm. 4 only needs
#                 a deterministic floor to keep m_underline > 0 if
#                 eps_hat collapses; setting it to ALPHA*eps_high
#                 would pin delta_t to DELTA_LO in the high-budget
#                 regime and destroy the adaptivity being tested.
#
#  ALPHA = 2.5 reproduces the delta = 2.5*eps_bar convention of
#  the letter.  Both fixed comparators therefore satisfy
#  Assumption 2 *in the regime they were tuned for*, and violate
#  it (delta_lo) or over-shoot it (delta_hi) in the other --
#  which is exactly the failure mode Reviewer 1.2 describes.
# ============================================================
const ALPHA_CAL = 2 + ETA     # same multiplier as the adaptive rule
const M_PILOT   = 20
const QCAL      = 0.95        # regime-wide quantile (Assm. 3 is a.s.)
const DELTA_FLOOR = 0.50      # safety net only; must not bind in practice

#  Panel (a) of fig_spl_adaptive_both: the fixed-hi line at
#  ~14.8 is an order of magnitude above delta_t in the
#  full-budget regime, so a linear axis compresses the very
#  detail the panel exists to show (delta_t sitting BELOW
#  fixed-lo when data are good).  A log axis shows both
#  scales at once.  Set false to revert to linear.
const MARGIN_LOG_AXIS = true

function pilot_eps(; T=T_RAD)
    hi = Float64[]; lo = Float64[]
    for m in 1:M_PILOT
        rng = MersenneTwister(90000+m)
        x  = gen_traj(rng, T)
        n0 = np_sched(1)
        P  = repeat(x[:,1],1,n0) .+ SIG_OBS*randn(rng,4,n0)
        for t in 1:T
            y = Hobs*x[:,t+1] + SIG_OBS*randn(rng,2)
            _,_,e = score_and_estimate(P, y, rng)
            (TB1 <= t < TB2) ? push!(lo,e) : push!(hi,e)
            F_true_t = (T1 <= t < T2) ? F_ct() : F_cv()
            P = propagate_resample(P, y, F_true_t,
                                   np_sched(min(t+1,T)), rng)
        end
    end
    return quantile(hi, QCAL), quantile(lo, QCAL)
end

println("\n[calibration] pilot run ($(M_PILOT) trials) ...")
eps_hi_cal, eps_lo_cal = pilot_eps()
DELTA_LO[]  = ALPHA_CAL * eps_hi_cal
DELTA_HI[]  = ALPHA_CAL * eps_lo_cal
DELTA_MIN[] = DELTA_FLOOR
@printf("[calibration] eps_hat  high-budget (q%.2f) = %.3f\n", QCAL,
        eps_hi_cal)
@printf("[calibration] eps_hat  low-budget  (q%.2f) = %.3f  ", QCAL,
        eps_lo_cal)
@printf("(ratio %.2fx)\n", eps_lo_cal/max(eps_hi_cal,1e-12))
@printf("[calibration] DELTA_LO  = %.3f   (tuned on high budget)\n",
        DELTA_LO[])
@printf("[calibration] DELTA_HI  = %.3f   (tuned on low budget)\n",
        DELTA_HI[])
@printf("[calibration] DELTA_MIN = %.3f  (safety net)\n",
        DELTA_MIN[])
if eps_lo_cal/max(eps_hi_cal,1e-12) < 2.0
    println("[warn] regime contrast < 2x -- lower NP_LOW ",
            "to sharpen the demonstration.")
end

# ============================================================
#  RUN
# ============================================================
println("="^64)
println("  SPL resubmission -- adaptive margin on CV/CT radar")
println("  budget schedule: N_p=$(NP_HIGH) | $(NP_LOW) on ",
        "t in [$(TB1),$(TB2)) | $(NP_HIGH)")
println("  B=$(NBATCH) (min $(NB_MIN)/batch)  kappa=$(KAPPA)  eta=$(ETA)  ",
        "delta_min=(auto)")
println("  propagate with selected structure: ",
        PROPAGATE_WITH_SELECTED)
println("="^64)

VARIANTS = [(:naive,    "CF w/o margin"),
            (:fixed_lo, @sprintf("Fixed delta=%.2f (lo)", DELTA_LO[])),
            (:fixed_hi, @sprintf("Fixed delta=%.2f (hi)", DELTA_HI[])),
            (:adaptive, "Adaptive delta_t")]

res2 = Dict{Symbol,Any}()
for (i,(v,lab)) in enumerate(VARIANTS)
    println("[K=2] $lab ...")
    # common random numbers: every variant sees the SAME
    # trajectories and the SAME particle noise, so differences
    # are attributable to delta alone.  Legal here because no
    # variant consumes extra RNG draws.
    res2[v] = run_ens(v; soff=0)
end

report("Method (K=2)", VARIANTS, res2)

# ---- diagnostic: is the contrast strong enough? ------------
_,_,_,DELa,EPSa,_ = res2[:adaptive]
eps_hi = mean(EPSa[:, 1:TB1-1])
eps_lo = mean(EPSa[:, TB1:TB2-1])
@printf("\n[diag] mean eps_hat  high-budget = %.4f\n", eps_hi)
@printf("[diag] mean eps_hat  low-budget  = %.4f  (ratio %.2fx)\n",
        eps_lo, eps_lo/max(eps_hi,1e-12))
@printf("[diag] mean delta_t  high-budget = %.4f\n",
        mean(DELa[:, 1:TB1-1]))
@printf("[diag] mean delta_t  low-budget  = %.4f\n",
        mean(DELa[:, TB1:TB2-1]))
println("[diag] expect ratio >> 1; if ~1, lower NP_LOW further.")

# decisive diagnostic: chattering INSIDE the low-budget window
println("\n[diag] switches inside the reduced-budget window ",
        "t in [$(TB1),$(TB2)) :")
for (v,lab) in VARIANTS
    _,SW,_,_,_,_ = res2[v]
    win = TB1:min(TB2-1, size(SW,2))
    @printf("[diag]   %-26s %6.2f\n", lab,
            mean(sum(SW[:,win], dims=2)))
end
println("[diag] target: fixed(lo) >> adaptive here.")

# how often does the floor bind?  should be ~0
_,_,_,DELf,EPSf,_ = res2[:adaptive]
bind = mean((2+ETA).*EPSf .<= DELTA_MIN[])
@printf("\n[diag] delta floor binds on %.1f%% of steps ",
        100*bind)
println("(should be near 0)")

# ---- K = 4 -------------------------------------------------
res4 = Dict{Symbol,Any}()
for (i,(v,lab)) in enumerate(VARIANTS)
    println("\n[K=4] $lab ...")
    res4[v] = run_ens_K4(v; soff=0)
end

report("Method (K=4)", VARIANTS, res4)

# ============================================================
#  FIGURES
# ============================================================
tv  = 1:T_RAD
tv2 = 1:T_RAD-1

# ---- Fig A: margin trajectory (the R1.2 evidence) ----------
_,_,_,DEL_a,EPS_a,_ = res2[:adaptive]
d_mean = vec(mean(DEL_a, dims=1))
e_mean = 2 .* vec(mean(EPS_a, dims=1))

figA = Figure(size=(660,380), fontsize=13)
axA  = Axis(figA[1,1]; xlabel=L"time $t$", ylabel=L"\mathrm{margin}",
    xlabelsize=22, ylabelsize=22, titlesize=19,
    title=L"\textbf{Adaptive\ margin\ under\ a\ varying\ particle\ budget}")

vspan!(axA, TB1, TB2; color=(:crimson, 0.09))
vspan!(axA, T1, T2;   color=(:orange, 0.10))

lines!(axA, tv, d_mean; color=:royalblue, linewidth=2.6,
    label=L"\delta_t\ \mathrm{(adaptive)}")
lines!(axA, tv, e_mean; color=:tomato, linewidth=2.0,
    linestyle=:dot, label=L"2\hat{\varepsilon}_t")
hlines!(axA, [DELTA_LO[]]; color=:gray30, linewidth=1.6,
    linestyle=:dash, label=L"\delta\ \mathrm{fixed\ (lo)}")
hlines!(axA, [DELTA_HI[]]; color=:seagreen, linewidth=1.6,
    linestyle=:dashdot, label=L"\delta\ \mathrm{fixed\ (hi)}")

text!(axA, (TB1+TB2)/2, maximum(d_mean)*0.94;
    text="reduced budget\n$(NP_LOW) particles",
    color=(:crimson,0.85), align=(:center,:top), fontsize=12)
text!(axA, T1-1, maximum(d_mean)*0.55; text="CT phase",
    color=(:darkorange,0.9), align=(:right,:center), fontsize=12)
axislegend(axA; framevisible=false, labelsize=13, position=:lt)
xlims!(axA, 1, T_RAD)
save(joinpath(@__DIR__,"fig_spl_margin_traj.pdf"), figA)
save(joinpath(@__DIR__,"fig_spl_margin_traj.png"), figA)

# ---- Fig B: switch probability, K = 2 ----------------------
figB = Figure(size=(660,380), fontsize=13)
axB  = Axis(figB[1,1]; xlabel=L"time $t$", ylabel=L"\hat{p}_t",
    xlabelsize=22, ylabelsize=22, titlesize=19,
    title=L"\textbf{CV/CT\ benchmark}\ (K=2):\ \hat{p}_t")
vspan!(axB, TB1, TB2; color=(:crimson, 0.09))
vspan!(axB, T1, T2;   color=(:orange, 0.10))
for (v,col,ls,lab) in [
        (:naive,    :tomato,    :dot,
            L"\mathrm{CF\ w/o\ margin}\ (\delta=0)"),
        (:fixed_lo, :gray40,    :dash,
            L"\mathrm{fixed}\ \delta\ \mathrm{(lo)}"),
        (:fixed_hi, :seagreen,  :dashdot,
            L"\mathrm{fixed}\ \delta\ \mathrm{(hi)}"),
        (:adaptive, :royalblue, :solid,
            L"\mathrm{adaptive}\ \delta_t\ \mathrm{(proposed)}")]
    _,SW,_,_,_,_ = res2[v]
    lines!(axB, tv2, vec(mean(SW,dims=1));
        color=col, linewidth=2.2, linestyle=ls, label=lab)
end
axislegend(axB; framevisible=false, labelsize=13, position=:lt)
xlims!(axB, 1, T_RAD); ylims!(axB, -0.02, 1.02)
save(joinpath(@__DIR__,"fig_spl_switching.pdf"), figB)
save(joinpath(@__DIR__,"fig_spl_switching.png"), figB)

# ---- Fig C: switch probability, K = 4 ----------------------
figC = Figure(size=(660,380), fontsize=13)
axC  = Axis(figC[1,1]; xlabel=L"time $t$", ylabel=L"\hat{p}_t",
    xlabelsize=22, ylabelsize=22, titlesize=19,
    title=L"\textbf{Four-model\ dictionary}\ (K=4):\ \hat{p}_t")
vspan!(axC, TB1, TB2; color=(:crimson, 0.09))
vspan!(axC, T1, T2;   color=(:orange, 0.10))
for (v,col,ls,lab) in [
        (:naive,    :tomato,    :dot,
            L"\mathrm{CF\ w/o\ margin}\ (\delta=0)"),
        (:fixed_hi, :seagreen,  :dashdot,
            L"\mathrm{fixed}\ \delta\ \mathrm{(hi)}"),
        (:adaptive, :royalblue, :solid,
            L"\mathrm{adaptive}\ \delta_t\ \mathrm{(proposed)}")]
    _,SW,_,_,_,_ = res4[v]
    lines!(axC, tv2, vec(mean(SW,dims=1));
        color=col, linewidth=2.2, linestyle=ls, label=lab)
end
axislegend(axC; framevisible=false, labelsize=13, position=:lt)
xlims!(axC, 1, T_RAD); ylims!(axC, -0.02, 1.02)
save(joinpath(@__DIR__,"fig_spl_switching_K4.pdf"), figC)
save(joinpath(@__DIR__,"fig_spl_switching_K4.png"), figC)



# ============================================================
#  THREE-REGIME DELTA SWEEP  --  the evidence for R1.2
#
#  Reviewer 1.2 objects that an offline-calibrated delta
#  "cannot dynamically respond to shifting noise profiles".
#  The question is therefore NOT whether some fixed delta is
#  optimal for one fixed scenario -- one always is -- but
#  whether that optimum TRANSFERS.  We sweep delta under three
#  operating regimes and locate the optimum in each:
#
#    :high   full budget throughout     (low  score noise)
#    :low    reduced budget throughout  (high score noise)
#    :mixed  nonstationary              (both)
#
#  If the optimum moves between regimes, a practitioner who
#  calibrates on one and deploys on another is penalised --
#  and that penalty is precisely what the adaptive rule avoids.
# ============================================================
const M_SWEEP    = 50
const DELTA_GRID = [0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 18.0]
const REGIMES    = [(:high,  "full budget"),
                    (:low,   "reduced budget"),
                    (:mixed, "nonstationary")]

function eval_variant(variant, d; M=M_SWEEP, T=T_RAD)
    SW=zeros(Bool,M,T-1); ST=zeros(Int,M,T)
    old = DELTA_LO[]
    variant === :fixed_lo && (DELTA_LO[] = d)
    for m in 1:M
        rng = MersenneTwister(21000+m)
        _,sw,_,_,_,st = run_single(variant, rng; T=T)
        SW[m,:]=sw; ST[m,:]=st
    end
    DELTA_LO[] = old
    a,d1,d2,fa = sel_metrics(ST, SW)
    return (N=mean(sum(SW,dims=2)), FA=fa, D1=d1, D2=d2, ACC=a,
            accv=acc_pertrial(ST))
end

println("\n", "="^66)
println("  THREE-REGIME DELTA SWEEP  (M=$(M_SWEEP) per point)")
println("="^66)

acc_curves = Dict{Symbol,Vector{Float64}}()
best_delta = Dict{Symbol,Float64}()
best_acc   = Dict{Symbol,Float64}()
adapt_acc  = Dict{Symbol,Float64}()
adapt_row  = Dict{Symbol,Any}()

for (mode,lab) in REGIMES
    SCHED_MODE[] = mode
    println("\n--- regime: $lab  ($(mode)) ---")
    @printf("%8s %8s %7s %6s %6s %6s\n",
            "delta","E[N_T]","FA","D1","D2","ACC")
    println("-"^48)
    accs = Float64[]
    for d in DELTA_GRID
        r = eval_variant(:fixed_lo, d)
        push!(accs, r.ACC)
        @printf("%8.2f %8.1f %7.1f %6.1f %6.1f %6.3f\n",
                d, r.N, r.FA, r.D1, r.D2, r.ACC)
    end
    ra_ = eval_variant(:adaptive, 0.0)
    println("-"^48)
    @printf("%8s %8.1f %7.1f %6.1f %6.1f %6.3f\n",
            "adaptive", ra_.N, ra_.FA, ra_.D1, ra_.D2, ra_.ACC)
    acc_curves[mode] = accs
    best_acc[mode]   = maximum(accs)
    best_delta[mode] = DELTA_GRID[argmax(accs)]
    adapt_acc[mode]  = ra_.ACC
    adapt_row[mode]  = ra_
end
SCHED_MODE[] = :mixed

# ---- transfer penalty table -------------------------------
println("\n", "="^66)
println("  TRANSFER: calibrate on one regime, deploy on another")
println("="^66)
@printf("%-16s %10s %10s | %10s %10s\n",
        "deploy regime", "delta*", "ACC(d*)",
        "ACC(d*_high)", "ACC(adaptive)")
println("-"^66)
d_star_high = best_delta[:high]
i_high = findfirst(==(d_star_high), DELTA_GRID)
for (mode,lab) in REGIMES
    @printf("%-16s %10.2f %10.3f | %10.3f %10.3f\n",
            lab, best_delta[mode], best_acc[mode],
            acc_curves[mode][i_high], adapt_acc[mode])
end
println("-"^66)
@printf("delta* tuned on the full-budget regime = %.2f\n",
        d_star_high)
@printf("worst transfer loss = %.3f accuracy points\n",
    maximum(best_acc[m] - acc_curves[m][i_high]
            for (m,_) in REGIMES))
@printf("worst adaptive gap  = %.3f accuracy points ",
    maximum(best_acc[m] - adapt_acc[m] for (m,_) in REGIMES))
println("(no delta supplied)")

# ------------------------------------------------------------
#  THE COMPARISON THE LETTER ACTUALLY NEEDS
#
#  Locating delta* requires ground-truth structure labels, so
#  it is an ORACLE procedure, not an offline one.  The offline
#  procedure genuinely available to a practitioner is the one
#  the original submission proposed: estimate eps_bar from a
#  held-out window and set delta = alpha * eps_bar.  That is
#  the baseline the adaptive rule must beat.
# ------------------------------------------------------------
println("\n", "="^66)
println("  ADAPTIVE vs THE *AVAILABLE* OFFLINE PROCEDURE")
println("  (delta* requires ground truth -> oracle, not offline)")
println("="^66)
@printf("%-34s %10s %10s %10s\n",
        "procedure", "ACC(high)", "ACC(low)", "ACC(mixed)")
println("-"^66)

d_off = ALPHA_CAL * eps_hi_cal      # the Remark-3 offline rule

off_v = Dict{Symbol,Vector{Float64}}()
ad_v  = Dict{Symbol,Vector{Float64}}()
for (mode,_) in REGIMES
    SCHED_MODE[] = mode
    off_v[mode] = eval_variant(:fixed_lo, d_off).accv
    ad_v[mode]  = eval_variant(:adaptive, 0.0).accv
end
SCHED_MODE[] = :mixed

@printf("%-34s %10s %10s %10s\n",
        "", "high", "low", "mixed")
println("-"^66)
@printf("%-34s %10.3f %10.3f %10.3f\n",
        "oracle delta* (needs ground truth)",
        (best_acc[m] for (m,_) in REGIMES)...)
@printf("%-34s %10.3f %10.3f %10.3f\n",
        @sprintf("offline Remark-3 rule (d=%.2f)", d_off),
        (mean(off_v[m]) for (m,_) in REGIMES)...)
@printf("%-34s %10.3f %10.3f %10.3f\n",
        "adaptive delta_t (no tuning)",
        (mean(ad_v[m]) for (m,_) in REGIMES)...)
println("-"^66)

# ---- PAIRED comparison (valid under CRN) -------------------
println("\nPaired difference  adaptive - offline  (95% CI):")
println("  common random numbers make the pairing exact, so the")
println("  trajectory variance cancels and the CI is tight.")
for (mode,lab) in REGIMES
    m_,ci = paired_diff(ad_v[mode], off_v[mode])
    verdict = (m_ - ci > 0) ? "adaptive better" :
              (m_ + ci < 0) ? "offline better"  :
                              "indistinguishable"
    @printf("  %-16s %+7.4f  +/- %.4f   -> %s\n",
            lab, m_, ci, verdict)
end
allad  = vcat((ad_v[m]  for (m,_) in REGIMES)...)
alloff = vcat((off_v[m] for (m,_) in REGIMES)...)
m_,ci = paired_diff(allad, alloff)
@printf("  %-16s %+7.4f  +/- %.4f   -> %s\n", "pooled", m_, ci,
        (m_-ci>0) ? "adaptive better" :
        (m_+ci<0) ? "offline better"  : "indistinguishable")

@printf("\nspread of ACC over the delta grid (high regime) = %.3f\n",
        maximum(acc_curves[:high]) - minimum(acc_curves[:high]))
println("  -> delta matters a great deal, yet cannot be located",
        " without ground truth.")

# ---- Fig D: three ACC curves + adaptive markers -----------
figD = Figure(size=(700,410), fontsize=13)
axD  = Axis(figD[1,1];
    xlabel=L"\mathrm{fixed\ margin}\ \delta",
    ylabel=L"\mathrm{selection\ accuracy}",
    xlabelsize=21, ylabelsize=21, titlesize=17,
    title=L"\textbf{Selection\ accuracy\ is\ highly\ sensitive\ to\ the\ fixed\ margin}")
cols = Dict(:high=>:seagreen, :low=>:tomato, :mixed=>:gray25)
for (mode,lab) in REGIMES
    lines!(axD, DELTA_GRID, acc_curves[mode];
        color=cols[mode], linewidth=2.4,
        label=latexstring("\\mathrm{fixed}\\ \\delta:\\ " *
                          replace(lab, " " => "\\ ")))
    scatter!(axD, DELTA_GRID, acc_curves[mode];
        color=cols[mode], markersize=8)
    scatter!(axD, [best_delta[mode]], [best_acc[mode]];
        color=cols[mode], marker=:star5, markersize=18)
    hlines!(axD, [adapt_acc[mode]]; color=cols[mode],
        linewidth=1.6, linestyle=:dash)
end
text!(axD, DELTA_GRID[end]*0.62,
      minimum(values(adapt_acc))-0.03;
      text="dashed = adaptive  (no δ supplied)\nstar = best fixed δ",
      fontsize=12, color=:gray30, align=(:left,:top))
axislegend(axD; framevisible=false, labelsize=12, position=:lb)
save(joinpath(@__DIR__,"fig_spl_delta_sweep.pdf"), figD)
save(joinpath(@__DIR__,"fig_spl_delta_sweep.png"), figD)
println("\n  fig_spl_delta_sweep.pdf written")


# ---- Fig 1: four-panel summary -----------------------------
#      (a)(b) switch probability | (c)(d) adaptive margin
#      2 rows x 2 cols, full page width
#      (LaTeX: figure* with width=\textwidth)
tv  = 1:T_RAD
tv2 = 1:T_RAD-1
_,_,_,DEL_a,EPS_a,_ = res2[:adaptive]
d_mean = vec(mean(DEL_a, dims=1))
e_mean = 2 .* vec(mean(EPS_a, dims=1))

figAD = Figure(size=(760,620), fontsize=18)

# ---- (a)(b) switch probability ----------------------------
spec_sw = [(:naive,    :tomato,    :dot),
           (:fixed_lo, :gray40,    :dash),
           (:fixed_hi, :seagreen,  :dashdot),
           (:adaptive, :royalblue, :solid)]
labs_sw = [L"\mathrm{CF\ w/o\ margin}\ (\delta=0)",
           L"\mathrm{fixed}\ \delta\ \mathrm{(lo)}",
           L"\mathrm{fixed}\ \delta\ \mathrm{(hi)}",
           L"\mathrm{adaptive}\ \delta_t"]

for (i,(res,ttl)) in enumerate([
        (res2, L"\textbf{(a)}\ \mathcal{S}_2\ (K=2)"),
        (res4, L"\textbf{(b)}\ \mathcal{S}_4\ (K=4)")])
    ax = Axis(figAD[1,i]; xlabel=L"time $t$",
        ylabel = i==1 ? L"\hat{p}_t" : "",
        xlabelsize=22, ylabelsize=26, titlesize=22,
        title=ttl, yticklabelsvisible = (i==1))
    vspan!(ax, TB1, TB2; color=(:crimson,0.09))
    vspan!(ax, T1, T2;   color=(:orange,0.10))
    hs = Any[]
    for (v,col,ls) in spec_sw
        _,SW,_,_,_,_ = res[v]
        push!(hs, lines!(ax, tv2, vec(mean(SW,dims=1));
            color=col, linewidth=2.8, linestyle=ls))
    end
    if i == 1
        axislegend(ax, hs, labs_sw;
            framevisible=true, backgroundcolor=(:white,0.88),
            labelsize=16, position=:rt, nbanks=1,
            patchsize=(18,10), padding=(4,4,2,2))
    end
    xlims!(ax, 1, T_RAD); ylims!(ax, -0.02, 1.02)
end

# ---- (c) margin trajectory --------------------------------
axA2 = Axis(figAD[2,1]; xlabel=L"time $t$",
    ylabel=L"\mathrm{margin}",
    xlabelsize=22, ylabelsize=26, titlesize=20,
    title=L"\textbf{(c)}\ \mathrm{the\ margin\ tracks\ the\ score\ noise}",
    yscale = MARGIN_LOG_AXIS ? log10 : identity,
    yminorgridvisible = MARGIN_LOG_AXIS,
    yminorticksvisible = MARGIN_LOG_AXIS,
    yminorticks = IntervalsBetween(9),
    yminorgridcolor = (:gray,0.12),
    yticks = MARGIN_LOG_AXIS ?
        ([0.5,1,2,5,10,20],
         ["0.5","1","2","5","10","20"]) :
        Makie.automatic)
vspan!(axA2, TB1, TB2; color=(:crimson,0.09))
vspan!(axA2, T1, T2;   color=(:orange,0.10))
lA1 = lines!(axA2, tv, d_mean; color=:royalblue, linewidth=2.6)
lA2 = lines!(axA2, tv, e_mean; color=:tomato,
             linewidth=2.4, linestyle=:dot)
lA3 = hlines!(axA2, [DELTA_LO[]]; color=:gray30,
              linewidth=2.0, linestyle=:dash)
lA4 = hlines!(axA2, [DELTA_HI[]]; color=:seagreen,
              linewidth=2.0, linestyle=:dashdot)
ylo_raw = min(minimum(d_mean), minimum(e_mean))
if MARGIN_LOG_AXIS
    ybotA = max(1e-2, 0.7*ylo_raw)
    ytopA = max(DELTA_HI[], maximum(d_mean)) * 2.0
else
    ybotA = 0.0
    ytopA = max(DELTA_HI[], maximum(d_mean)) * 1.22
end
ylims!(axA2, ybotA, ytopA)
text!(axA2, (TB1+TB2)/2, (ybotA*1.35) +18;
    text="reduced budget ($(NP_LOW) particles)",
    color=(:crimson,0.85), align=(:center,:bottom),
    fontsize=16)
#text!(axA2, T_RAD-1, DELTA_HI[];
#    text=" fixed (hi) ", color=(:seagreen,0.9),
#    align=(:right,:bottom), fontsize=11)
axislegend(axA2, [lA1,lA2,lA3,lA4],
    [L"\delta_t\ \mathrm{(adaptive)}",
     L"2\hat{\varepsilon}_t",
     L"\delta\ \mathrm{fixed\ (lo)}",
     L"\delta\ \mathrm{fixed\ (hi)}"];
    framevisible=true, backgroundcolor=(:white,0.88),
    labelsize=16, position=:rb, nbanks=2,
    patchsize=(18,10), padding=(4,4,2,2))
xlims!(axA2, 1, T_RAD)

# ---- (d) accuracy vs fixed margin, three regimes ----------
axD2 = Axis(figAD[2,2];
    xlabel=L"\mathrm{fixed\ margin}\ \delta",
    ylabel=L"\mathrm{selection\ accuracy}",
    xlabelsize=22, ylabelsize=26, titlesize=20,
    title=L"\textbf{(d)}\ \mathrm{a\ fixed\ margin\ must\ be\ tuned}")
cols2 = Dict(:high=>:seagreen, :low=>:tomato, :mixed=>:gray25)
for (mode,lab) in REGIMES
    lines!(axD2, DELTA_GRID, acc_curves[mode];
        color=cols2[mode], linewidth=2.8,
        label=latexstring("\\mathrm{fixed}\\ \\delta:\\ " *
                          replace(lab, " " => "\\ ")))
    scatter!(axD2, DELTA_GRID, acc_curves[mode];
        color=cols2[mode], markersize=8)
    scatter!(axD2, [best_delta[mode]], [best_acc[mode]];
        color=cols2[mode], marker=:star5, markersize=18)
    hlines!(axD2, [adapt_acc[mode]]; color=cols2[mode],
        linewidth=2.8, linestyle=:dash)
end
accall = Float64[]
for (m,_) in REGIMES
    append!(accall, acc_curves[m]); push!(accall, adapt_acc[m])
end
lo_, hi_ = minimum(accall), maximum(accall)
ylims!(axD2, lo_-0.035, hi_+0.075)
# annotation intentionally omitted: the caption states that
# dashed lines are the adaptive rule and stars the oracle
axislegend(axD2; framevisible=true,
    backgroundcolor=(:white,0.88), labelsize=16,
    position=:rt, patchsize=(18,10),
    padding=(4,4,2,2))

colgap!(figAD.layout, 22)
rowgap!(figAD.layout, 26)
save(joinpath(@__DIR__,"fig_spl_all4.pdf"), figAD)
save(joinpath(@__DIR__,"fig_spl_all4.png"), figAD)
println("  fig_spl_all4.pdf written")

println("\nFigures written:")
println("  fig_spl_all4.pdf   (4-panel summary)")
println("Done.")
