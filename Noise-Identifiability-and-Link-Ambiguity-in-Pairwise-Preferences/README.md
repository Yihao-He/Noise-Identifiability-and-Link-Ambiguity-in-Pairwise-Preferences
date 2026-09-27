# Noise Identifiability and Link Ambiguity in Pairwise Preferences
Yihao He and Quanyi Li. Corresponding author: Yihao He (heyh@mails.neu.edu.cn).

This repository contains the synthetic data, graph lists, fits, plotting code, and
numeric summaries underlying the manuscript and supplement. The raw data and
all uncertainty estimates are retained. Run commands from the archive root.

## Quick start

```bash
python -m pip install -r requirements.txt
python experiments/plot_results.py
python experiments/validate_revision.py
```

These commands use the saved results and reproduce the main figures and numeric
checks without rerunning the Monte Carlo experiments. Figures are written to
`manuscript/figures/`, tables to `supplementary/tables/`, and verification results
to `experiments/results/revision_checks.json`.

## Repository contents

| Directory | Contents |
|---|---|
| `experiments/` | General-graph and finite-sample simulation, fitting, diagnostics, and plotting |
| `experiments/results/` | Graphs, raw binomial counts, estimates, summaries, and runtime records |
| `data/` | Saved K=3,4 experiments and numerical judge-audit summaries |
| `code/` | Additional verification, plotting, and historical experiment routines |
| `manuscript/figures/` | Vector PDF figures reproduced from numeric results |
| `supplementary/tables/` | Generated numeric table rows |

`DATA_README.md` explains the numeric data files. `MANIFEST.sha256` records
checksums of the initial repository contents.

## Environment

Tested on Windows, Python 3.13.5, AMD Ryzen 5 7640HS (6 physical cores, 12 logical processors), CPU only. Exact library versions are in `requirements.txt`: NumPy 2.1.3, SciPy 1.15.3, pandas 2.2.3, mpmath 1.3.0, SymPy 1.13.3, Matplotlib 3.10.0, threadpoolctl 3.5.0, PyMuPDF 1.28.2. PyMuPDF is used only for PDF checks. No GPU, deep model, or remote compute is required.

```text
python -m pip install -r requirements.txt
```

## Run all new experiments

Run from the package root, in this order:

```text
python experiments/graph_generation.py
python experiments/scaling_experiment.py
python experiments/finite_sample.py --replicates 200 --workers 8
python experiments/refine_profiles.py
python experiments/boundary_sensitivity.py
python experiments/profile_control.py
python experiments/plot_results.py
python experiments/validate_revision.py
```

`refine_profiles.py` completes the declared numerical protocol on the same saved counts. It refits every n=100 dataset and any other flagged fit using a noise grid, multiple utility starts, scalar refinement, and a wider utility domain. Do not present the initial finite-sample CSV as the final results before this refinement. The initial diagnostics are retained in `initial_*` columns. `boundary_sensitivity.py` checks all 60 final saturation-boundary fits with a doubled cap; it does not replace or exclude observations. Eight worker processes are used for these audits; the correct-link control uses four. BLAS threads are limited to one per worker.

The finite-sample profile uses all K−1 utilities and noise in [0,0.5]. The interval is the hull of detected likelihood-ratio components. It is a nominal profile interval, not a finite-sample coverage theorem. The separate Hoeffding interval `C_H` has a genuine link-robust coverage guarantee. Its midpoint bias is explicitly labeled and is not an MLE bias. `C_L` is the older logistic-only Hoeffding interval.

To redraw from the shipped results without simulation:

```text
python experiments/plot_results.py
python experiments/validate_revision.py
```

To redraw only the finite-sample figure, use `python experiments/plot_results.py --figure finite`. This figure separates the three delta values into columns, with coverage above and mean width below. Method colors, markers, and line styles are shared across panels; the legend sits outside the plotting area. All original estimates and uncertainty intervals are retained.

### Measured runtime

| Completed operation | Wall-clock time on the tested machine |
|---|---:|
| 104 general-graph population cells | 65.6 s |
| 2,400 initial finite-sample fits, eight workers | 299.4 s |
| 605 weak-identification profile refinements, eight workers | 125.3 s |
| 60 utility-bound sensitivity refits, eight workers | 39.3 s |
| 200 correctly specified logistic controls, four workers | 27.9 s |
| Total of these stages | about 9.3 min |

Runtime JSON files retain the measured values. Plotting and compilation take additional seconds; first-time TeX package downloads can take longer. The total is a sum of measured stage runtimes, not a new end-to-end benchmark. It excludes exploratory debugging and the earlier inherited simulations.

### Seeds and design

- Complete graphs: K=5,10,20, no graph randomness.
- Connected ER: K=10,20,50; p=0.2,0.5. Seed `20260920 + 100*K + int(100*p)`. Rejection sampling conditions only on connectivity.
- Sparse connected graphs: path plus K distinct random chords at K=10,20,50; seed `20260920 + 10*K`.
- Tree control: K=10 path. Cycle count means cycle-space dimension M−K+1, not the number of all simple cycles.
- Every graph uses unnormalized `w_i=K-i`, eta=0.2, and delta=0.2,0.1,0.05,0.025. The additional, separately labeled near-tie window is 0.001,0.0005,0.00025,0.000125.
- Finite samples: complete K=10, delta=0.05,0.1,0.2, n=100,1000,10000,100000 **per edge**, 200 replicates. Counts use `SeedSequence([20260921,10,di,ni])` for ordered zero-based grid indices. Total comparisons are 45n. Counts are generated before parallel fitting, so worker scheduling does not alter them.
- Correct-link control: delta=0.2, n=100000, 200 datasets, seed 20260922, true logistic noise 0.1707572295.
- The code computes actual numerical KL values with 60-digit residual evaluation and double-precision least-squares updates; it never inserts the theoretical leading term as a measured result. The noise fitting domain for population KL is b in [0.3,0.95].

The requested larger scales do **not** all have slopes six and ten. The table retains that finding. These powers emerge in the additional near-tie window, with fixed graph and utilities. One ER realization per design is reported; no random-graph frequency claim is made.


## Additional figure reproduction

    python code/reproduce_tables.py
    python code/verify_key_claims.py
    python code/plot_figures.py

These commands use the saved K=3,4 experiments in data/. See code/README.md for
their numerical settings. The historical judge analysis supplies numeric summaries;
raw third-party text, model tensors, and the original model calls are not distributed.
The manuscript and supplement are submitted separately to the journal.

## Historical simulation regeneration

```bash
python code/regenerate_simulations.py --output historical_rerun
```

This opt-in command recomputes the historical high-precision population results
in a new directory. Add `--include-fits` to regenerate the 3,200 historical
Monte Carlo datasets and fits; this is substantially more expensive. The output
directory must not exist. The published saved data are retained unchanged.

## Contact and citation

Corresponding author: Yihao He, heyh@mails.neu.edu.cn.
See `CITATION.cff` for the paper title and authors. 
