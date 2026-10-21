# Baseline A — Classical Conformance Summary

**Run**: `run_20260505_000244`
**Trace source**: `HARDER/run_20260504_221347` (same as APM-5cur, comparable directly)

## Macro F1 per (pair × dataset) vs APM

| Pair / Dataset | hiring | hospital | lending | renting | macro |
|---|---|---|---|---|---|
| petri_net_inductive__alignments | 0.965 | 0.980 | 0.970 | 0.980 | **0.974** |
| petri_net_inductive__token_replay | 0.945 | 0.960 | 0.949 | 0.960 | **0.953** |
| petri_net_heuristics__alignments | 0.274 | 0.000 | 0.896 | 0.842 | **0.503** |
| petri_net_ilp__alignments | 0.911 | 0.970 | 0.980 | 0.980 | **0.960** |
| petri_net_alpha__alignments | 0.000 | 0.666 | 0.408 | 0.000 | **0.268** |
| process_tree__alignments | 0.965 | 0.980 | 0.970 | 0.980 | **0.974** |
| dfg__footprints | 0.800 | 0.800 | 0.800 | 0.800 | **0.800** |
| declare__conformance | 0.695 | 0.807 | 0.658 | 0.758 | **0.729** |
| log_skeleton__conformance | 0.943 | 0.965 | 0.968 | 0.967 | **0.961** |
| APM-5cur (5 reps + LLM agent) | 0.939 | 0.959 | 0.980 | 0.973 | **0.963** |

## Per-perturbation TN-rate vs APM-5cur (mean across 4 datasets)

Higher TN-rate = better detection of that perturbation type.

| Type | petri_net_inductive__alignments | petri_net_inductive__token_replay | petri_net_heuristics__alignments | petri_net_ilp__alignments | petri_net_alpha__alignments | process_tree__alignments | dfg__footprints | declare__conformance | log_skeleton__conformance | APM-5cur |
|---|---|---|---|---|---|---|---|---|---|---|
| A1_random_swap | 0.995 | 0.995 | 1.000 | 0.855 | 1.000 | 0.995 | 0.000 | 0.895 | 0.905 | 0.955 |
| A3_dependency_aware_swap | 0.995 | 0.995 | 1.000 | 0.865 | 1.000 | 0.995 | 0.000 | 0.915 | 0.885 | 0.950 |
| B1_insert_out_of_alphabet | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.655 | 1.000 | 0.930 |
| B2_insert_misplaced_existing | 0.975 | 0.975 | 1.000 | 0.810 | 1.000 | 0.975 | 0.000 | 0.845 | 0.900 | 0.925 |
| B3_insert_duplicate_unique | 1.000 | 1.000 | 1.000 | 0.850 | 1.000 | 1.000 | 0.000 | 0.830 | 1.000 | 0.940 |
| C1_delete_mandatory | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.905 |
| C2_delete_start_or_end | 0.760 | 0.760 | 1.000 | 0.855 | 1.000 | 0.760 | 0.000 | 0.750 | 0.510 | 0.675 |
| D1_cross_role | 0.000 | 0.000 | 0.555 | 0.000 | 0.785 | 0.000 | 0.000 | 0.345 | 0.000 | 0.000 |
| E1_two_swaps | 1.000 | 1.000 | 1.000 | 0.910 | 1.000 | 1.000 | 0.000 | 0.940 | 0.970 | 0.985 |
| E2_swap_and_insert | 1.000 | 0.980 | 1.000 | 0.965 | 1.000 | 1.000 | 0.000 | 0.930 | 0.995 | 0.985 |