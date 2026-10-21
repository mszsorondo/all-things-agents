#!/usr/bin/env python3
"""
Baseline A: classical conformance checking.

For each dataset, discover formal representations and classify all 600 pre-pickled
traces (100 valid + 10×50 invalid) using pm4py's native conformance algorithms —
NO LLM in the validation step.

Pairs (representation, algorithm):
  Petri Net Inductive   → alignments       (cost == 0  ⇒ VALID)
  Petri Net Inductive   → token replay     (fitness == 1.0 ⇒ VALID)
  Petri Net Heuristics  → alignments
  Petri Net ILP         → alignments
  Petri Net Alpha       → alignments
  Process Tree          → convert to PN + alignments
  DFG                   → footprint comparison
  DECLARE               → conformance_declare
  Log Skeleton          → conformance_log_skeleton

Output: baselines/classical_runs/run_<ts>/{config.json, results.json, summary.md}
"""

import os
import sys
import json
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
import pm4py

# Silence pm4py noise that doesn't affect results
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from baselines._common import (
    DATASETS, PERTURBATION_TYPES, HARDER_TRACE_RUN, xes_path_for,
    load_traces, load_full_log, trace_to_event_log_df,
    compute_metrics, empty_tokens, write_json, synthesize_timestamps,
)


# =============================================================================
# REP DISCOVERY
# =============================================================================

def _ensure_log_columns(log_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure full log has the columns pm4py discoverers expect."""
    df = log_df.copy()
    if "concept:name" not in df.columns and "activity" in df.columns:
        df["concept:name"] = df["activity"]
    if "case:concept:name" not in df.columns and "case:case" in df.columns:
        df["case:concept:name"] = df["case:case"]
    return df


def discover_petri_net_inductive(log_df):
    return pm4py.discover_petri_net_inductive(log_df)


def discover_petri_net_heuristics(log_df):
    return pm4py.discover_petri_net_heuristics(log_df)


def discover_petri_net_ilp(log_df):
    return pm4py.discover_petri_net_ilp(log_df)


def discover_petri_net_alpha(log_df):
    return pm4py.discover_petri_net_alpha(log_df)


def discover_process_tree_petri_net(log_df):
    """Discover process tree, then convert to Petri net."""
    tree = pm4py.discover_process_tree_inductive(log_df)
    net, im, fm = pm4py.convert_to_petri_net(tree)
    return net, im, fm


def discover_dfg_full(log_df):
    """Discover DFG and the full log df itself (used for footprint comparison)."""
    dfg, start, end = pm4py.discover_dfg(log_df)
    return dfg, start, end, log_df


def discover_declare(log_df):
    return pm4py.discover_declare(log_df)


def discover_log_skeleton(log_df):
    return pm4py.discover_log_skeleton(log_df)


# =============================================================================
# CLASSIFIERS — return "VALID" / "INVALID"
# =============================================================================

def classify_pn_alignments(trace_df: pd.DataFrame, model) -> str:
    """Use fitness (not raw cost) because tau moves contribute to cost
    even for trace-perfect fits. Fitness==1.0 ⇔ no real deviations."""
    net, im, fm = model
    log = trace_to_event_log_df(trace_df)
    try:
        diag = pm4py.conformance_diagnostics_alignments(log, net, im, fm)
        if not diag:
            return "INVALID"
        fitness = diag[0].get("fitness", 0.0)
        return "VALID" if fitness >= 0.9999 else "INVALID"
    except Exception:
        return "INVALID"


def classify_pn_token_replay(trace_df: pd.DataFrame, model) -> str:
    net, im, fm = model
    log = trace_to_event_log_df(trace_df)
    try:
        diag = pm4py.conformance_diagnostics_token_based_replay(log, net, im, fm)
        if not diag:
            return "INVALID"
        # trace_fitness == 1.0 means no missing/remaining tokens.
        # Strict: produced+consumed match exactly.
        info = diag[0]
        fitness = info.get("trace_fitness", 0.0)
        is_fit = info.get("trace_is_fit", False)
        if isinstance(is_fit, bool) and is_fit:
            return "VALID"
        return "VALID" if fitness >= 0.9999 else "INVALID"
    except Exception:
        return "INVALID"


def classify_dfg_footprints(trace_df: pd.DataFrame, model) -> str:
    """Compare trace footprint against the full-log footprint via pm4py.
    The diagnostic dict has 'is_footprints_fit' (bool) plus per-dimension
    deviation sets ('footprints', 'start_activities', 'end_activities')."""
    dfg, start, end, full_log = model
    log = trace_to_event_log_df(trace_df)
    try:
        diag = pm4py.conformance_diagnostics_footprints(log, full_log)
        if isinstance(diag, list):
            diag = diag[0] if diag else {}
        if not diag:
            return "INVALID"
        if "is_footprints_fit" in diag:
            fit = diag["is_footprints_fit"]
            min_len = diag.get("min_length_fit", True)
            return "VALID" if (fit and min_len) else "INVALID"
        # Fallback: any deviation set non-empty ⇒ invalid
        for k in ("footprints", "start_activities", "end_activities"):
            v = diag.get(k)
            if v:
                return "INVALID"
        return "VALID"
    except Exception:
        return "INVALID"


def classify_declare(trace_df: pd.DataFrame, model) -> str:
    """pm4py.conformance_declare returns dict per case with keys including
    'no_dev_total' (count of violations), 'dev_fitness' (0..1), 'is_fit' (bool)."""
    log = trace_to_event_log_df(trace_df)
    try:
        diag = pm4py.conformance_declare(log, model)
        if isinstance(diag, dict):
            diag = [diag]
        if not diag:
            return "INVALID"
        info = diag[0]
        if "is_fit" in info:
            return "VALID" if info["is_fit"] else "INVALID"
        no_dev = info.get("no_dev_total", 0)
        return "VALID" if no_dev == 0 else "INVALID"
    except Exception:
        return "INVALID"


def classify_log_skeleton(trace_df: pd.DataFrame, model) -> str:
    """pm4py.conformance_log_skeleton diagnostics: 'is_fit' bool or 'no_dev_total' count."""
    log = trace_to_event_log_df(trace_df)
    try:
        diag = pm4py.conformance_log_skeleton(log, model)
        if isinstance(diag, dict):
            diag = [diag]
        if not diag:
            return "INVALID"
        info = diag[0]
        if "is_fit" in info:
            return "VALID" if info["is_fit"] else "INVALID"
        no_dev = info.get("no_dev_total", 0)
        return "VALID" if no_dev == 0 else "INVALID"
    except Exception:
        return "INVALID"


# =============================================================================
# REGISTRY OF (rep_name, algo_name) → (discover_fn, classifier_fn)
# =============================================================================

PAIRS: Dict[str, Tuple[Callable, Callable]] = {
    "petri_net_inductive__alignments":
        (discover_petri_net_inductive, classify_pn_alignments),
    "petri_net_inductive__token_replay":
        (discover_petri_net_inductive, classify_pn_token_replay),
    "petri_net_heuristics__alignments":
        (discover_petri_net_heuristics, classify_pn_alignments),
    "petri_net_ilp__alignments":
        (discover_petri_net_ilp, classify_pn_alignments),
    "petri_net_alpha__alignments":
        (discover_petri_net_alpha, classify_pn_alignments),
    "process_tree__alignments":
        (discover_process_tree_petri_net, classify_pn_alignments),
    "dfg__footprints":
        (discover_dfg_full, classify_dfg_footprints),
    "declare__conformance":
        (discover_declare, classify_declare),
    "log_skeleton__conformance":
        (discover_log_skeleton, classify_log_skeleton),
}


# =============================================================================
# RUNNER
# =============================================================================

def evaluate_pair_on_dataset(pair_name: str, dataset: str, full_log_df: pd.DataFrame,
                              valid_traces: List[pd.DataFrame],
                              invalid_by_type: Dict[str, List[dict]]) -> dict:
    """Run a single (rep, algo) pair on one dataset."""
    discover_fn, classify_fn = PAIRS[pair_name]
    print(f"  [{pair_name}] discovering...", flush=True)
    t0 = time.time()
    try:
        model = discover_fn(full_log_df)
    except Exception as e:
        print(f"    DISCOVERY FAILED: {type(e).__name__}: {str(e)[:120]}", flush=True)
        return {
            "status": "discovery_failed",
            "discovery_error": f"{type(e).__name__}: {str(e)[:200]}",
            "discover_time": time.time() - t0,
        }
    discover_time = time.time() - t0
    print(f"    discovered in {discover_time:.2f}s", flush=True)

    # Valid traces
    print(f"  [{pair_name}] classifying 100 valid traces...", flush=True)
    t0 = time.time()
    valid_predictions = []
    for trace in valid_traces:
        pred = classify_fn(trace, model)
        valid_predictions.append(pred == "VALID")
    valid_time = time.time() - t0
    TP = sum(valid_predictions)
    FN = len(valid_predictions) - TP
    print(f"    valid done in {valid_time:.2f}s: TP={TP} FN={FN}", flush=True)

    # Invalid by type
    per_type = {}
    for ptype, records in invalid_by_type.items():
        t0 = time.time()
        invalid_predictions = []
        for record in records:
            trace = record["trace"]
            pred = classify_fn(trace, model)
            invalid_predictions.append(pred == "INVALID")
        type_time = time.time() - t0
        m = compute_metrics(valid_predictions, invalid_predictions)
        per_type[ptype] = {
            **m,
            "compute_time_seconds": type_time,
            "tokens": empty_tokens(),
        }
        print(f"    [{ptype}] TN={m['TN']:2d} FP={m['FP']:2d} F1={m['f1']:.3f} "
              f"({type_time:.2f}s)", flush=True)

    # Aggregate
    all_invalid_preds = []
    for info in per_type.values():
        all_invalid_preds.extend([True] * info["TN"] + [False] * info["FP"])
    agg = compute_metrics(valid_predictions, all_invalid_preds)
    agg["compute_time_seconds"] = (
        valid_time + sum(info["compute_time_seconds"] for info in per_type.values())
    )
    agg["tokens"] = empty_tokens()
    agg["n_valid"] = len(valid_traces)
    agg["n_invalid_total"] = sum(len(r) for r in invalid_by_type.values())

    return {
        "status": "completed",
        "discovery_time_seconds": discover_time,
        "valid_evaluation": {
            "compute_time_seconds": valid_time,
            "tokens": empty_tokens(),
            "TP": TP, "FN": FN,
        },
        "per_type": per_type,
        "aggregate": agg,
    }


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("baselines/classical_runs") / f"run_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "validator_kind": "classical_conformance",
        "validator_model": "pm4py-native",
        "k_valid": 100,
        "k_invalid_per_type": 50,
        "perturbation_types": PERTURBATION_TYPES,
        "datasets": DATASETS,
        "trace_source": str(HARDER_TRACE_RUN),
        "pairs": list(PAIRS.keys()),
    }
    write_json(out_dir / "config.json", config)

    results = {
        "_meta": {
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "config": config,
            "run_folder": str(out_dir),
            "status": "running",
        }
    }
    write_json(out_dir / "results.json", results)

    # Pre-load full logs once per dataset (discovery is per-pair)
    print(f"\n{'='*70}")
    print(f"BASELINE A — CLASSICAL CONFORMANCE  →  {out_dir}")
    print(f"{'='*70}")
    print(f"Pairs: {len(PAIRS)}  Datasets: {len(DATASETS)}")

    full_logs = {}
    for ds in DATASETS:
        print(f"  Loading log: {ds}...", flush=True)
        log = load_full_log(ds)
        full_logs[ds] = _ensure_log_columns(synthesize_timestamps(log))

    for pair_name in PAIRS:
        print(f"\n{'-'*70}\nPAIR: {pair_name}\n{'-'*70}")
        results[pair_name] = {}
        for ds in DATASETS:
            ds_key = f"{ds}_log_high"
            print(f"\n  Dataset: {ds}", flush=True)
            valid, invalid = load_traces(ds)
            ds_result = evaluate_pair_on_dataset(
                pair_name, ds, full_logs[ds], valid, invalid
            )
            ds_result["xes_path"] = xes_path_for(ds)
            results[pair_name][ds_key] = ds_result
            write_json(out_dir / "results.json", results)

    results["_meta"]["end_time"] = datetime.now().isoformat()
    results["_meta"]["status"] = "completed"
    write_json(out_dir / "results.json", results)

    # Summary
    write_summary(out_dir, results)
    print(f"\n{'='*70}\nDONE → {out_dir}\n{'='*70}")
    return 0


def write_summary(out_dir: Path, results: dict):
    """Write summary.md comparing each (pair, dataset) vs APM-3rep & APM-5cur."""
    apm_3rep_runs = {
        "hiring":   "HARDER/run_20260503_221317",
        "hospital": "HARDER/run_20260504_113728",
        "lending":  "HARDER/run_20260504_133408",
        "renting":  "HARDER/run_20260504_211708",
    }
    apm_5cur_run = "HARDER/run_20260504_221347"
    apm_5cur = json.loads(Path(f"{apm_5cur_run}/results.json").read_text())

    lines = [
        f"# Baseline A — Classical Conformance Summary\n",
        f"**Run**: `{out_dir.name}`",
        f"**Trace source**: `{HARDER_TRACE_RUN}` (same as APM-5cur, comparable directly)\n",
        f"## Macro F1 per (pair × dataset) vs APM\n",
        "| Pair / Dataset | hiring | hospital | lending | renting | macro |",
        "|---|---|---|---|---|---|",
    ]
    # For each pair, compute macro F1 per dataset
    for pair_name, by_ds in results.items():
        if pair_name == "_meta":
            continue
        row = [pair_name]
        macros = []
        for ds in DATASETS:
            ds_key = f"{ds}_log_high"
            if ds_key not in by_ds:
                row.append("-"); continue
            r = by_ds[ds_key]
            if r.get("status") != "completed":
                row.append(f"`{r.get('status')}`")
                macros.append(None)
                continue
            f1s = [info["f1"] for info in r["per_type"].values()]
            macro = sum(f1s) / len(f1s) if f1s else 0.0
            row.append(f"{macro:.3f}")
            macros.append(macro)
        valid_macros = [m for m in macros if m is not None]
        macro_mean = sum(valid_macros) / len(valid_macros) if valid_macros else 0.0
        row.append(f"**{macro_mean:.3f}**")
        lines.append("| " + " | ".join(row) + " |")

    # APM rows
    lines.append("| APM-5cur (5 reps + LLM agent) | "
                  + " | ".join(
                      f"{sum(t['f1'] for t in apm_5cur[f'{ds}_log_high']['per_type'].values())/10:.3f}"
                      for ds in DATASETS
                  ) + " | **{:.3f}** |".format(
                      sum(sum(t['f1'] for t in apm_5cur[f'{ds}_log_high']['per_type'].values())/10
                          for ds in DATASETS) / 4
                  ))

    lines += [
        "",
        "## Per-perturbation TN-rate vs APM-5cur (mean across 4 datasets)\n",
        "Higher TN-rate = better detection of that perturbation type.\n",
        "| Type | " + " | ".join(p for p in PAIRS.keys()) + " | APM-5cur |",
        "|---|" + "---|" * (len(PAIRS) + 1),
    ]
    for ptype in PERTURBATION_TYPES:
        row = [ptype]
        for pair_name in PAIRS:
            tns = []
            for ds in DATASETS:
                ds_key = f"{ds}_log_high"
                r = results[pair_name].get(ds_key, {})
                if r.get("status") == "completed":
                    tn = r["per_type"][ptype]["TN"]
                    tns.append(tn / 50)
            row.append(f"{sum(tns)/len(tns):.3f}" if tns else "-")
        # APM-5cur
        apm_tns = [apm_5cur[f"{ds}_log_high"]["per_type"][ptype]["TN"] / 50 for ds in DATASETS]
        row.append(f"{sum(apm_tns)/4:.3f}")
        lines.append("| " + " | ".join(row) + " |")

    (out_dir / "summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
