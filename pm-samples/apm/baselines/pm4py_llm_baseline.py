#!/usr/bin/env python3
"""
Baseline B: pm4py.llm raw abstractions as validator prompt.

Replaces the APM agent synthesis (which gpt-5 turns into a polished textual model)
with the *raw* pm4py.llm.abstract_*() outputs, concatenated under per-rep headers.
Same validator (gpt-5-mini) classifies the same pre-pickled traces from HARDER.

Two configs (mirroring HARDER):
  3-rep:  abstract_petri_net (inductive) + abstract_dfg + abstract_variants
  5-cur:  abstract_process_tree + abstract_dfg + abstract_variants
          + abstract_declare + abstract_log_attributes

Output: baselines/pm4pyllm_runs/run_<ts>_<config>/
  config.json, results.json (HARDER schema), {dataset}_pm4pyllm_context.txt, summary.md
"""

import os
import sys
import json
import time
import argparse
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
import pm4py
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from baselines._common import (
    DATASETS, PERTURBATION_TYPES, HARDER_TRACE_RUN, xes_path_for,
    load_traces, load_full_log, compute_metrics, empty_tokens,
    write_json, synthesize_timestamps,
)


# pm4py traces are sent to the LLM via key-value rows formatted same as HARDER.
def df_to_key_value_rows(df, max_rows=50):
    df_short = df.head(max_rows).reset_index(drop=True)
    lines = []
    for i, row in df_short.iterrows():
        parts = [f"{col}={row[col]}" for col in df_short.columns]
        lines.append(f"Event number {i}: " + "; ".join(parts))
    return "\n".join(lines)


# =============================================================================
# REPRESENTATION ABSTRACTIONS — using pm4py.llm.abstract_* directly, no synthesis
# =============================================================================

def _ensure(log_df: pd.DataFrame) -> pd.DataFrame:
    df = log_df.copy()
    if "concept:name" not in df.columns and "activity" in df.columns:
        df["concept:name"] = df["activity"]
    if "case:concept:name" not in df.columns and "case:case" in df.columns:
        df["case:concept:name"] = df["case:case"]
    return synthesize_timestamps(df)


def abstract_petri_net_inductive(log_df: pd.DataFrame) -> str:
    log = _ensure(log_df)
    net, im, fm = pm4py.discover_petri_net_inductive(log)
    return pm4py.llm.abstract_petri_net(net, im, fm)


def abstract_dfg(log_df: pd.DataFrame) -> str:
    return pm4py.llm.abstract_dfg(_ensure(log_df))


def abstract_variants(log_df: pd.DataFrame) -> str:
    return pm4py.llm.abstract_variants(_ensure(log_df), relative_frequency=True)


def abstract_process_tree(log_df: pd.DataFrame) -> str:
    log = _ensure(log_df)
    tree = pm4py.discover_process_tree_inductive(log)
    return str(tree)


def abstract_declare(log_df: pd.DataFrame) -> str:
    log = _ensure(log_df)
    declare = pm4py.discover_declare(log)
    return pm4py.llm.abstract_declare(declare)


def abstract_log_attributes(log_df: pd.DataFrame) -> str:
    return pm4py.llm.abstract_log_attributes(_ensure(log_df))


CONFIGS: Dict[str, List[Tuple[str, Callable]]] = {
    "3rep": [
        ("petri_net_inductive", abstract_petri_net_inductive),
        ("dfg", abstract_dfg),
        ("abstract_variants", abstract_variants),
    ],
    "5cur": [
        ("process_tree", abstract_process_tree),
        ("dfg", abstract_dfg),
        ("abstract_variants", abstract_variants),
        ("declare", abstract_declare),
        ("log_attributes", abstract_log_attributes),
    ],
}


def build_pm4pyllm_context(log_df: pd.DataFrame, config: str) -> str:
    """Concatenate raw pm4py.llm abstractions under per-rep headers."""
    parts = []
    for rep_name, fn in CONFIGS[config]:
        try:
            text = fn(log_df)
        except Exception as e:
            text = f"[Discovery failed: {type(e).__name__}: {e}]"
        parts.append(f"## {rep_name}\n{text}\n")
    return "\n".join(parts)


# =============================================================================
# VALIDATOR — same shape as HARDER's resilient validator
# =============================================================================

def make_validator(context: str, model_name: str = "gpt-5-mini",
                    request_timeout: int = 90, max_retries: int = 5):
    if ":" in model_name:
        provider, model_id = model_name.split(":", 1)
    else:
        provider, model_id = "openai", model_name

    if "grok" in provider.lower() or "xai" in provider.lower():
        llm = ChatOpenAI(model=model_id, api_key=os.getenv("XAI_API_KEY"),
                          base_url="https://api.x.ai/v1",
                          timeout=request_timeout, max_retries=max_retries)
    elif "deepseek" in provider.lower():
        llm = ChatOpenAI(model=model_id, api_key=os.getenv("DEEPSEEK_API_KEY"),
                          base_url="https://api.deepseek.com",
                          timeout=request_timeout, max_retries=max_retries)
    else:
        llm = ChatOpenAI(model=model_id, api_key=os.getenv("OPENAI_API_KEY"),
                          timeout=request_timeout, max_retries=max_retries)

    system_prompt = f"""You are an expert business process analyst. You will be given a business process description (raw discovered process abstractions) and a trace (sequence of activities).

Business Process Description (raw pm4py.llm abstractions, no human synthesis):
{context}

Your task: Determine if the given trace is VALID (conforms to the observed business process) or INVALID (contains anomalies).

IMPORTANT: The business model describes processes as they are observed, which may include inefficiencies or non-ideal behaviors that still occur in practice.

Respond with ONLY one word: "VALID" or "INVALID"."""

    def validate(trace_df):
        trace_text = df_to_key_value_rows(trace_df)
        messages = [("system", system_prompt), ("human", trace_text)]
        last_err = None
        for attempt in range(max_retries):
            try:
                start = time.time()
                response = llm.invoke(messages)
                ct = time.time() - start
                content = (response.content or "").strip().upper()
                if "VALID" not in content and "INVALID" not in content:
                    raise ValueError(f"Non-classification: {content[:120]!r}")
                pred = "INVALID" if "INVALID" in content else "VALID"
                tokens = response.response_metadata.get("token_usage", {})
                return pred, ct, tokens
            except Exception as e:
                last_err = e
                wait = min(2 ** attempt, 30)
                print(f"  [validator retry {attempt+1}/{max_retries} after {type(e).__name__}: "
                      f"{str(e)[:120]}; sleeping {wait}s]", flush=True)
                time.sleep(wait)
        print(f"  [validator gave up: {last_err}]", flush=True)
        return "INVALID", 0.0, empty_tokens()

    return validate, system_prompt


def parallel_eval(traces, validator, max_workers, label):
    results = [None] * len(traces)
    completed = 0
    total = len(traces)
    start_wall = time.time()

    def _safe(idx_trace):
        idx, t = idx_trace
        try:
            return idx, validator(t)
        except Exception as e:
            print(f"  [{label}] worker error idx={idx}: {type(e).__name__}: {str(e)[:100]}",
                   flush=True)
            return idx, ("INVALID", 0.0, empty_tokens())

    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = [exe.submit(_safe, (i, t)) for i, t in enumerate(traces)]
        for fut in as_completed(futures):
            idx, res = fut.result()
            results[idx] = res
            completed += 1
            if completed % max(1, total // 10) == 0 or completed == total:
                elapsed = time.time() - start_wall
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (total - completed) / rate if rate > 0 else 0
                print(f"    [{label}] {completed}/{total} | wall={elapsed:.0f}s "
                      f"rate={rate:.1f}/s ETA={eta:.0f}s", flush=True)
    return results


# =============================================================================
# MAIN
# =============================================================================

def evaluate_dataset(dataset: str, run_folder: Path, config: str,
                     validator_model: str, max_workers: int, results: dict):
    ds_key = f"{dataset}_log_high"
    print(f"\n{'='*70}\n  Dataset: {dataset}\n{'='*70}", flush=True)
    results[ds_key] = {"status": "in_progress", "xes_path": xes_path_for(dataset)}
    write_json(run_folder / "results.json", results)

    print(f"  Loading log...", flush=True)
    log = load_full_log(dataset)

    print(f"  Building pm4py.llm context ({config})...", flush=True)
    t0 = time.time()
    context = build_pm4pyllm_context(log, config)
    discover_time = time.time() - t0
    ctx_path = run_folder / f"{dataset}_pm4pyllm_context.txt"
    ctx_path.write_text(context)
    print(f"    {len(context)} chars  ({discover_time:.1f}s)", flush=True)
    results[ds_key]["pm4pyllm_context_path"] = str(ctx_path)
    results[ds_key]["discover_time_seconds"] = discover_time
    results[ds_key]["context_length_chars"] = len(context)

    print(f"  Loading traces (HARDER cache)...", flush=True)
    valid, invalid_by_type = load_traces(dataset)

    validator, system_prompt = make_validator(context, model_name=validator_model)
    (run_folder / f"{dataset}_system_prompt.txt").write_text(system_prompt)

    # Valid eval
    print(f"  Eval 100 valid traces (workers={max_workers})...", flush=True)
    t0 = time.time()
    res_v = parallel_eval(valid, validator, max_workers, "valid")
    valid_time = time.time() - t0
    valid_predictions = []
    valid_tokens = empty_tokens()
    for pred, ct, tok in res_v:
        valid_predictions.append(pred == "VALID")
        for k in valid_tokens: valid_tokens[k] += tok.get(k, 0)
    TP = sum(valid_predictions); FN = len(valid_predictions) - TP
    print(f"    valid done: TP={TP} FN={FN} ({valid_time:.0f}s)", flush=True)
    results[ds_key]["valid_evaluation"] = {
        "compute_time_seconds": valid_time,
        "tokens": valid_tokens, "TP": TP, "FN": FN,
    }

    # Per-type
    per_type = {}
    for ptype, records in invalid_by_type.items():
        print(f"  Eval [{ptype}] (50 invalids, workers={max_workers})...", flush=True)
        t0 = time.time()
        traces_only = [r["trace"] for r in records]
        res_i = parallel_eval(traces_only, validator, max_workers, ptype)
        type_time = time.time() - t0
        invalid_predictions = []
        type_tokens = empty_tokens()
        for pred, ct, tok in res_i:
            invalid_predictions.append(pred == "INVALID")
            for k in type_tokens: type_tokens[k] += tok.get(k, 0)
        m = compute_metrics(valid_predictions, invalid_predictions)
        per_type[ptype] = {**m, "compute_time_seconds": type_time, "tokens": type_tokens}
        print(f"    [{ptype}] TN={m['TN']} FP={m['FP']} F1={m['f1']:.3f} ({type_time:.0f}s)",
               flush=True)
        results[ds_key]["per_type"] = per_type
        write_json(run_folder / "results.json", results)

    # Aggregate
    all_invalid_preds = []
    for info in per_type.values():
        all_invalid_preds.extend([True] * info["TN"] + [False] * info["FP"])
    agg = compute_metrics(valid_predictions, all_invalid_preds)
    agg_tokens = {k: valid_tokens[k] + sum(info["tokens"].get(k, 0)
                   for info in per_type.values()) for k in valid_tokens}
    agg_time = valid_time + sum(info["compute_time_seconds"] for info in per_type.values())
    agg.update({"compute_time_seconds": agg_time, "tokens": agg_tokens,
                 "n_valid": len(valid), "n_invalid_total": sum(len(r) for r in invalid_by_type.values())})
    results[ds_key]["aggregate"] = agg
    results[ds_key]["status"] = "completed"
    write_json(run_folder / "results.json", results)
    print(f"  AGG: P={agg['precision']:.3f} R={agg['recall']:.3f} F1={agg['f1']:.3f}",
           flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", choices=list(CONFIGS.keys()), required=True,
                    help="3rep or 5cur (mirrors HARDER configs)")
    p.add_argument("--validator-llm", default="gpt-5-mini")
    p.add_argument("--max-workers", type=int, default=10)
    p.add_argument("--dataset", default=None,
                    help="Single dataset (default: all 4)")
    args = p.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("baselines/pm4pyllm_runs") / f"run_{timestamp}_{args.config}"
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = [args.dataset] if args.dataset else DATASETS

    config_dict = {
        "validator_kind": "pm4py_llm_raw",
        "config": args.config,
        "rep_set": [r for r, _ in CONFIGS[args.config]],
        "validator_model": args.validator_llm,
        "k_valid": 100,
        "k_invalid_per_type": 50,
        "perturbation_types": PERTURBATION_TYPES,
        "datasets": datasets,
        "trace_source": str(HARDER_TRACE_RUN),
        "max_workers": args.max_workers,
    }
    write_json(out_dir / "config.json", config_dict)

    results = {
        "_meta": {
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "config": config_dict,
            "run_folder": str(out_dir),
            "status": "running",
        }
    }
    write_json(out_dir / "results.json", results)

    print(f"\n{'='*70}")
    print(f"BASELINE B — pm4py.llm raw context  →  {out_dir}")
    print(f"  config: {args.config}  reps: {[r for r,_ in CONFIGS[args.config]]}")
    print(f"  validator: {args.validator_llm}  workers: {args.max_workers}")
    print(f"{'='*70}")

    for ds in datasets:
        evaluate_dataset(ds, out_dir, args.config, args.validator_llm,
                          args.max_workers, results)

    results["_meta"]["end_time"] = datetime.now().isoformat()
    results["_meta"]["status"] = "completed"
    write_json(out_dir / "results.json", results)
    print(f"\nDONE → {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
