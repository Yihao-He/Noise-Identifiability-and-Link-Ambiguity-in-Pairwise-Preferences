# Numeric data inventory

The repository contains saved synthetic comparison counts, fitted estimates,
population calculations, and numerical summaries. It does not contain API
credentials, model weights, or raw third-party prompt/response text.

## File inventory

| File | Rows or format | Fields / description |
|---|---:|---|
| `data/cache_edges.csv` | 192 | id, i, j, forward_p_i, reversed_mapped_p_i, forward_cached_pA, reverse_cached_pA, logit_forward, logit_reverse_mapped (additional fields are in the header) |
| `data/cache_fit_medians.csv` | 8 | aggregation, link, eta_reference, residual_rmse, cycle_signal, mismatch |
| `data/cache_fits.csv` | 256 | id, aggregation, link, eta_reference, residual_rmse, cycle_signal, mismatch, success, boundary |
| `data/cache_prompts.csv` | 32 | id, position_rms, mean_beta, beta_edge_sd, cycle_logit_rms, common_beta_logit_rmse, common_beta_probability_rmse, mean_aggregation_change, max_aggregation_change (additional fields are in the header) |
| `data/cache_summary.json` | JSON | prompts, cached_forwards, new_judge_calls, cache_sha256, max_probability_reconstruction_difference, max_old_mapping_difference, medians, mismatch_counts |
| `data/conservative_intervals.csv` | 3200 | k, delta, n, world, rep, eta_true, lo, hi, width (additional fields are in the header) |
| `data/finite.csv` | 3200 | k, delta, n, world, rep, eta_true, etaQ, n_delta6, n_delta10 (additional fields are in the header) |
| `data/finite_summary.csv` | 16 | k, delta, n, world, L_eta, P_eta, L_width, P_width, L_covers_true (additional fields are in the header) |
| `data/high_precision_summary.json` | JSON | eta_limit, slopes, k4_fifth_residual, k4_fifth_residual_norm2, k4_optimized_kl_leading_coefficient, k4_optimized_kl_leading_coefficient_float, precision_decimal_digits |
| `data/information_check.csv` | 8 | delta, known_information, unknown_information |
| `data/integrity.json` | JSON | old_files_checked, changed_old_files, finite_rows, expected_finite_rows, cache_forwards, new_judge_calls, cache_fits, cache_nonconverged_fits |
| `data/mechanisms.json` | JSON | 8 records |
| `data/numerical_checks.json` | JSON | kl_relative_errors, population_fits |
| `data/paper_summary.csv` | 16 | k, delta, n, world, repeats, L_eta, L_width, L_width_se, L_cover_count (additional fields are in the header) |
| `data/population.csv` | 28 | k, delta, etaL, explicit_max_probability_difference, optimized_max_probability_difference, explicit_kl, optimized_kl, stationarity |
| `data/selected_high_precision_checks.json` | JSON | selection, precision, checks, max_eta_difference, max_deviance_difference, max_profile_cutoff_error |
| `experiments/results/boundary_sensitivity.csv` | 60 | delta, n, replicate, eta_160, eta_320, lo_160, lo_320, hi_160, hi_320 (additional fields are in the header) |
| `experiments/results/boundary_sensitivity.json` | JSON | checked, max_eta_change, max_lower_change, max_upper_change, max_deviance_change, coverage_decisions_changed, profile_failures, seconds |
| `experiments/results/finite_runtime.json` | JSON | seconds, seed, replicates_per_cell, workers, failures, profile_failures, utility_boundary, widened_fits |
| `experiments/results/finite_sample_raw.csv` | 2400 | k, delta, n, replicate, seed, counts, eta_hat, profile_lo, profile_hi (additional fields are in the header) |
| `experiments/results/finite_sample_summary.csv` | 36 | k, delta, n, total_comparisons, method, replicates, coverage, coverage_count, coverage_lo (additional fields are in the header) |
| `experiments/results/graph_summary.csv` | 13 | graph, family, k, M, cycle_rank, density, G3, G5, seed (additional fields are in the header) |
| `experiments/results/graphs.json` | JSON | 13 records |
| `experiments/results/profile_control.json` | JSON | eta_true, coverage, coverage_count, replicates, mean_width, bias, failed_fits, failed_profiles |
| `experiments/results/profile_control_raw.csv` | 200 | k, delta, n, replicate, seed, counts, eta_hat, profile_lo, profile_hi (additional fields are in the header) |
| `experiments/results/refinement_runtime.json` | JSON | seconds, refined_cells, failed_profile_evaluations, final_failed_fits, boundary_mle, maximum_profile_gradient |
| `experiments/results/revision_checks.json` | JSON | graphs, population_cells, finite_datasets, exact_sparse_G3, exact_sparse_G5, all_population_solvers_converged, max_information_coefficient_relative_error, max_kl_coefficient_relative_error |
| `experiments/results/scaling.csv` | 104 | graph, k, M, family, window, delta, max_gap, G3, G5 (additional fields are in the header) |
| `experiments/results/scaling_runtime.json` | JSON | seconds, precision, noise_domain_b, graphs |

## Conventions

- `delta`: utility separation scale; `n`: independent comparisons per edge.
- `eta`: binary flip probability. The generative probit noise is 0.2 in the main finite-sample experiment.
- `L` and `P`: logistic and probit models in the historical files.
- Main finite-sample data contain 2,400 datasets, 200 per design cell.
- `initial_*` columns retain the pre-refinement numerical diagnostics; final estimates use the refined protocol.
- The 200-dataset correctly specified logistic control is stored separately.
- Numeric judge-audit IDs are hashes; the original prompts/responses are not included.
- CSV column names and experiment scripts define the complete schema.

For environment, seeds, runtime, reproduction commands, and contact information, see `README.md`.
