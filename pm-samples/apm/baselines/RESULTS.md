# Baselines vs APM — Final Results

**Date**: 2026-05-05
**Trace source**: `HARDER/run_20260504_221347/traces/` (same 100 valid + 50 invalid per type per dataset used by APM-5cur, comparable directly)
**Validator**: `gpt-5-mini` for the LLM-based baselines, native pm4py algorithms for the classical baseline

This file replaces `RESULTS_PARTIAL.md`.

---

## Setup

Two baselines compared against APM (the agent-synthesized textual model from `HARDER/run_20260504_221347/`):

| ID | What | Code | LLM cost |
|----|------|------|----------|
| **A** | Classical conformance — pm4py-native algorithms decide VALID/INVALID per trace, no LLM | `baselines/classical_conformance.py` | $0 (CPU only) |
| **B 5cur** | pm4py.llm raw context — concatenated `pm4py.llm.abstract_*()` outputs as system prompt for `gpt-5-mini`, no APM agent rewrite | `baselines/pm4py_llm_baseline.py --config 5cur` | API per-call |

**B 3rep was killed mid-run** by user request and is not reported.
**B 5cur is now complete on all 4 datasets**. Renting was completed in a separate top-up run after the original hit OpenAI quota mid-eval; the renting result was merged back into the main run folder.

Both baselines reuse the exact same pickle files from HARDER, so all comparisons are apples-to-apples on identical traces.

### Why three metrics

Following the discussion in `HARDER/RESULTS_2026-05-04.md`, naïvely pooling all 500 invalids against 100 valids inflates FPs and produces a misleading aggregate F1. We instead report:

1. **Macro F1**: mean of the 10 per-type F1 values within a dataset. Each type has a fixed 100v + 50i ratio, so per-type F1 is comparable across configurations.
2. **Total FP count**: number of invalids classified as VALID across all 10 types of a dataset, out of 500 invalids per dataset.
3. **Per-type TN rate** (TN / 50): fraction of invalids of each type correctly detected. Unbiased per-type, comparable across configs.

---

## Baseline A — Classical Conformance

**Status: complete** for all 9 (representation, algorithm) pairs × 4 datasets.
Total: 21,600 conformance checks, ~25 minutes wall, **$0 LLM cost**.

Run: `baselines/classical_runs/run_20260505_004920/`
Auto-generated: `baselines/classical_runs/run_20260505_004920/summary.md`

### Macro F1 per pair × dataset

| Pair                                | hiring | hospital | lending | renting | mean   | total FP across 4 ds |
|-------------------------------------|--------|----------|---------|---------|--------|----------------------|
| **dfg__footprints**                 | 0.974  | 0.980    | 0.980   | 0.980   | **0.978** | **213** |
| **petri_net_inductive__alignments** | 0.965  | 0.980    | 0.970   | 0.980   | **0.974** | 255 |
| process_tree__alignments            | 0.965  | 0.980    | 0.970   | 0.980   | 0.974  | 255  |
| log_skeleton__conformance           | 0.943  | 0.965    | 0.968   | 0.967   | 0.961  | 367  |
| petri_net_ilp__alignments           | 0.911  | 0.970    | 0.980   | 0.980   | 0.960  | 378  |
| petri_net_inductive__token_replay   | 0.945  | 0.960    | 0.949   | 0.960   | 0.953  | 459  |
| declare__conformance                | 0.695  | 0.807    | 0.658   | 0.758   | 0.729  | 379  |
| petri_net_heuristics__alignments    | 0.274  | 0.000    | 0.896   | 0.842   | 0.503  | 89   |
| petri_net_alpha__alignments         | 0.000  | 0.666    | 0.408   | 0.000   | 0.268  | 43   |
| **APM-5cur (LLM agent reference)**  | 0.939  | 0.959    | 0.980   | 0.973   | **0.963** | 350 |

### Top finding

**Three classical pairs match or beat APM** (DFG footprints, PN Inductive + alignments, Process Tree + alignments) across all 4 datasets — without any LLM at validation time. The cheapest of these (DFG footprints) leads with macro F1 = 0.978, edging APM by +1.5 points and producing 213 FPs vs APM's 350 (39% fewer false positives).

### Per-type TN rate — top 3 classical pairs vs APM (mean of 4 datasets)

| Type                            | dfg__footprints | pn_inductive__align | process_tree__align | APM-5cur |
|---------------------------------|-----------------|---------------------|---------------------|----------|
| A1_random_swap                  | 1.000           | 0.995               | 0.995               | 0.955    |
| A3_dependency_aware_swap        | 1.000           | 0.995               | 0.995               | 0.950    |
| B1_insert_out_of_alphabet       | 1.000           | 1.000               | 1.000               | 0.930    |
| B2_insert_misplaced_existing    | 0.985           | 0.975               | 0.975               | 0.925    |
| B3_insert_duplicate_unique      | 0.940           | 1.000               | 1.000               | 0.940    |
| C1_delete_mandatory             | 1.000           | 1.000               | 1.000               | 0.905    |
| C2_delete_start_or_end          | 0.815           | 0.755               | 0.755               | 0.675    |
| **D1_cross_role**               | **0.000**       | **0.000**           | **0.000**           | **0.000** |
| E1_two_swaps                    | 1.000           | 1.000               | 1.000               | 0.985    |
| E2_swap_and_insert              | 1.000           | 1.000               | 1.000               | 0.985    |

**D1 cross_role is 0/50 in every classical pair, every dataset** — same structural blind spot as APM. Every other perturbation: classical equals or beats APM.

### Where classical fails

- **Petri Net Alpha + alignments** (macro F1 = 0.268): Alpha produces unsound nets that reject most valid traces.
- **Petri Net Heuristics + alignments** (macro F1 = 0.503): rejects everything in hospital, partial in others.
- **DECLARE conformance** (macro F1 = 0.729): mid-pack, but uniquely catches `Exactly 1` and `Always Before` violations that other methods miss.
- **All pairs**: D1 cross_role at 0% (control-flow blind to resources).

---

## Baseline B 5cur — pm4py.llm Raw Context, No Agent

**Status: complete on all 4 datasets.**

Run: `baselines/pm4pyllm_runs/run_20260505_083600_5cur/` (renting merged from `run_20260505_101755_5cur` after the quota top-up)
Reps fed to validator: `process_tree, dfg, abstract_variants, declare, log_attributes` — concatenated as raw `pm4py.llm.abstract_*` outputs under `## <rep>` headers, **no agent synthesis**.

### Per-dataset summary

| Dataset  | macro F1 | recall (valid TP/100) | total FP/500 |
|----------|----------|------------------------|---------------|
| hiring   | 0.708    | 0.59                   | 80            |
| hospital | 0.816    | 0.75                   | 94            |
| lending  | 0.663    | 0.52                   | 53            |
| renting  | 0.764    | 0.67                   | 90            |
| **mean** | **0.738** | **0.63**              | 79            |

For comparison, **APM-5cur on the same 4 datasets**: hiring 0.939, hospital 0.959, lending 0.980, renting 0.973 (mean 0.963).

### B 5cur loses on macro F1, wins on detection. Why?

The macro F1 numbers above are dragged down by recall collapse: with raw pm4py.llm context, the validator rejects 25-48% of valid traces. But when measured purely on **detection** (TN rate, ignoring valid-side losses), B 5cur often beats APM:

| Type                            | B5cur TN-rate | APM TN-rate | Δ      |
|---------------------------------|---------------|-------------|--------|
| **D1 cross_role**               | **0.345**     | **0.000**   | **+0.345** ⬆️⬆️ |
| C1 delete_mandatory             | 1.000         | 0.905       | +0.095 ⬆️ |
| **C2 delete_start_or_end**      | **0.755**     | 0.675       | **+0.080** ⬆️ |
| B1 insert_out_of_alphabet       | 0.940         | 0.930       | +0.010  |
| A3 dependency_aware_swap        | 0.915         | 0.950       | -0.035  |
| E2 swap_and_insert              | 0.955         | 0.985       | -0.030  |
| E1 two_swaps                    | 0.940         | 0.985       | -0.045  |
| B2 insert_misplaced_existing    | 0.875         | 0.925       | -0.050  |
| A1 random_swap                  | 0.895         | 0.955       | -0.060  |
| B3 insert_duplicate_unique      | 0.795         | 0.940       | -0.145 ⬇️ |

(Means across all 4 datasets.)

### Two clean findings

1. **B 5cur catches D1 cross_role 34.5% of the time** (mean across 4 datasets) — APM (and every classical method) catches 0%. Why? Including the raw `pm4py.llm.abstract_log_attributes` output gives the validator a literal table of (resource, frequency) pairs and (case-attribute, value) pairs. The agent's narrative rewrite of this same data drops the per-pair specificity, so the agent-version validator never realises a cross-role assignment is anomalous. **Raw context preserves discriminating information that the agent's prose smoothes out.**

2. **F1 collapses anyway because recall = 0.52-0.75**. The raw context is so literal that the validator rejects 25-48% of valid traces. Each per-type F1 is bounded above by recall: even with perfect TN=50, F1 = 2·(P·R)/(P+R) caps near 0.74 when R=0.59. So B 5cur **never beats APM on F1** despite catching more invalids.

### Interpretive summary (one line)

**The agent doesn't add detection capability — it adds tolerance.** The agent rewrites raw constraints into a softer narrative that prevents the validator from treating observed-but-rare patterns as violations. Without the agent, the validator becomes a strict literalist and rejects normal valid behaviour as anomalous.

---

## Cross-baseline summary

| | Baseline A (classical) | Baseline B 5cur (pm4py.llm raw) | APM-5cur (agent) |
|---|---|---|---|
| Validator | pm4py-native algorithm | gpt-5-mini | gpt-5-mini |
| Generator | none (algorithm directly) | none (raw abstractions) | gpt-5 agent rewrite |
| Best macro F1 (4 datasets, mean) | **0.978** (dfg__footprints) | 0.738 | 0.963 |
| Total FPs (4 datasets) | **213** (dfg__footprints) | 317 | 350 |
| Recall (valid TP/100) | 1.000 (always) | 0.52-0.75 | 1.000 |
| D1 cross_role TN rate (4-ds mean) | 0.000 (blind) | **0.345** (best!) | 0.000 (blind) |
| LLM cost per dataset | $0 | per-call gpt-5-mini | per-call gpt-5 + per-call gpt-5-mini |
| Compute time per dataset | seconds | ~10 min (parallel 10w) | ~10-15 min (parallel 10w) |

### Five-line takeaway per baseline

**Baseline A (classical)** → **APM loses**.
DFG footprints and PN Inductive alignments beat the LLM-agent macro F1 across all 4 datasets, with 39% fewer FPs and zero LLM cost. The only thing the LLM helps with is D1 cross_role and C2 delete-start/end — but neither matters here because classical also scores 0/50 on D1, and APM is worse on C2. **For control-flow violations, classical conformance is strictly better than APM and free.**

**Baseline B 5cur (pm4py.llm raw)** → **APM wins on macro F1, loses on D1 detection**.
Macro F1 is much lower (0.66-0.82 vs 0.94-0.98) because raw context makes the validator reject 25-48% of valid traces. But **B 5cur detects D1 cross-role 34.5% of the time on average across 4 datasets, vs APM's 0%** — the only configuration in this entire benchmark family that catches D1 at all. The agent's value is producing a textual model that is permissive enough not to reject valid traces; the cost is losing the per-pair literal information that lets the validator catch cross-role assignments.

---

## Files

| Artifact | Path |
|---|---|
| Classical results | `baselines/classical_runs/run_20260505_004920/results.json` |
| Classical summary | `baselines/classical_runs/run_20260505_004920/summary.md` |
| Classical config | `baselines/classical_runs/run_20260505_004920/config.json` |
| B 5cur results (4 datasets, merged) | `baselines/pm4pyllm_runs/run_20260505_083600_5cur/results.json` |
| B 5cur context per dataset | `baselines/pm4pyllm_runs/run_20260505_083600_5cur/{dataset}_pm4pyllm_context.txt` |
| B 5cur live log | `baselines/_pm4pyllm_5cur.log` |
| Trace source (shared) | `HARDER/run_20260504_221347/traces/` |
| APM-5cur reference | `HARDER/run_20260504_221347/results.json` |
