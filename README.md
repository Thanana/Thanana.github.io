# Robust CF Filtering — IEEE L-CSS 2026

> **Robust Cognitive-Flexible Filtering under Noisy Innovation Scores**  
> T. Nuchkrua and S. Boonto  
> *IEEE Control Systems Letters (L-CSS)*, 2026  
> Status: Under Review

---

## Overview

This repository contains all Julia source code to reproduce the numerical
experiments in the paper. The proposed **margin-based switching rule**
suppresses spurious structure transitions under bounded score noise,
restoring all three stability properties of noiseless CF theory.

---

## Requirements

Julia 1.9+ with the following packages:

```julia
] add Distributions Statistics Random Printf
```

---

## Files

| File | Description |
|------|-------------|
| `lcss_experiments_documented.jl` | Figures 2–3, Table I (Theorems 1–3) |
| `fig3_scaling.jl` | Figure 4, scaling validation (Theorem 2) |

---

## Run

```julia
julia lcss_experiments_documented.jl   # Figs 2–3, Table I
julia fig_scaling_documented.jl                  # Fig 4
```

Figures are saved to `figures/` automatically.

---

## Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `Np` | 500 | Number of particles |
| `M` | 100 | Monte Carlo runs |
| `T` | 200 | Horizon length |
| `α` | 2.5 | Margin multiplier (δ = α·ε̄) |
| `ε̄` | 0.5, 1.5, 3.0 | Score noise bounds (Table I) |

---

## Key Result

| Method | E[N_T] (ε̄=0.5) | E[N_T] (ε̄=1.5) | E[N_T] (ε̄=3.0) |
|--------|----------------|----------------|----------------|
| Exact CF (oracle) | 0.3 | 0.3 | 0.3 |
| CF without margin (δ=0) | 83.7 | 81.1 | 79.2 |
| **Robust CF (proposed)** | **0.3** | **1.3** | **7.9** |
| Thm. 2 bound | 8.2 | 11.4 | 17.6 |

Robust CF empirical count stays well below Theorem 2 bound ✓

---

## Citation

```bibtex
@article{nuchkrua2026robustcf,
  author  = {Nuchkrua, T. and Boonto, S.},
  title   = {Robust Cognitive-Flexible Filtering under
             Noisy Innovation Scores},
  journal = {IEEE Control Systems Letters},
  year    = {2026},
  note    = {Under review}
}
```

---

## Webpage

[https://thanana.github.io/LCSS.html](https://thanana.github.io/RobustCF.html)

---

&copy; 2026 Thanana Nuchkrua | Control & Robotics Research Group
