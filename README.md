# Robust CF Filtering — IEEE SPL 2026 (under review)



## Overview

This repository contains the Julia source code to reproduce the
numerical experiments in the paper. The proposed **adaptive
margin-based switching rule** suppresses spurious structure
transitions under nonstationary score noise, restoring all
noiseless-style stability guarantees while removing offline
calibration entirely. The noise envelope is estimated online via a
zero-overhead **batch-split estimator** with a peak-hold envelope.

---

## Requirements

Julia 1.9+ with the following packages:

```julia
] add Distributions Statistics Random Printf CairoMakie LaTeXStrings
```

---

## Files

| File                                    | Description                                          |
|------------------------------------------|-------------------------------------------------------|
| `radar_final_v6b_SPLrevision_2_1.jl`     | Li–Jilkov CV/CT benchmark, batch-split estimator, adaptive margin, three-regime sweep (Fig. 1, Tables I–II) |

---

## Run

```julia
julia radar_final_v6b_SPLrevision_2_1.jl
```

Figures are saved to `figures/` automatically.

---

## Parameters

| Parameter       | Value             | Description                            |
|-----------------|-------------------|-----------------------------------------|
| `T`             | 100               | Horizon length                          |
| `M`             | 100               | Monte Carlo runs                        |
| `N_p` (full)    | 4000              | Particle budget outside the reduced window |
| `N_p` (reduced) | 60                | Particle budget on $t\in[40,75)$        |
| `B`             | 10                | Number of particle batches              |
| `κ`             | 2.5               | Batch-spread coverage constant          |
| `η`             | 0.5               | Adaptive margin multiplier              |
| `λ`             | 0.9               | Peak-hold decay rate                    |
| `δ_min`         | 0.5               | Margin floor (safety net)               |

---

## Key Result

Nonstationary regime, binary candidate set $\mathcal{S}_2$
($T=100$, $M=100$):

| Method                    | E[N_T] | FA   | D₁   | ACC       |
|---------------------------|--------|------|------|-----------|
| CF without margin         | 17.5   | 12.9 | 1.0  | 0.840     |
| Fixed δ = 4.83            | 2.5    | 0.7  | 3.2  | **0.873** |
| Fixed δ = 14.81           | 0.6    | 0.4  | 32.8 | 0.591     |
| **Adaptive δ_t (proposed)** | 3.9  | 1.9  | 1.4  | 0.849     |

The margin suppresses false alarms tenfold relative to unprotected
switching; the adaptive rule matches offline calibration without
being tuned. ✓

---


## Webpage

[https://thanana.github.io/RobustCF.html](https://thanana.github.io/RobustCF.html)

---

&copy; 2026 Control & Robotics Research Group
