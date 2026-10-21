#!/usr/bin/env python3
"""
Phase 5 HARDER Benchmark — Multi-perturbation trace classification.

Generates the same textual AS-IS model as phase5_benchmark.py (PETRI_NET + DFG +
ABSTRACT_VARIANTS), but evaluates against 10 distinct perturbation types instead
of just A1 (random_swap_events).

Per-type perturbations (50 invalid traces each, all derived from the same pool of
100 valid sampled traces):

    A1  random_swap                     - swap two events at random positions
    A3  dependency_aware_swap           - swap a pair (a,b) where a always_before b
    B1  insert_out_of_alphabet          - insert an activity not in the log alphabet
    B2  insert_misplaced_existing       - insert an existing activity at random position
    B3  insert_duplicate_unique         - duplicate an activity that always appears once
    C1  delete_mandatory                - delete an activity present in 100% of traces
    C2  delete_start_or_end             - delete the first or last event
    D1  cross_role                      - reassign an event to a resource that never does that activity
    E1  two_swaps                       - apply A1 twice
    E2  swap_and_insert                 - apply A1 + (B1 or B2)

Each invalid trace is verified to be "effectively invalid": its activity
sequence must NOT match any variant observed in the log (D1 is invalid by
construction since it changes resources, not activities).

Usage:
    # Run perturbation tests only (no LLM calls)
    python phase5_benchmark_harder.py --test --dataset hiring

    # Run benchmark on one dataset
    python phase5_benchmark_harder.py --dataset hiring

    # Run benchmark on all 4 datasets
    python phase5_benchmark_harder.py --all
"""

import os
import sys
import json
import time
import random
import pickle
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
import pm4py
from dotenv import load_dotenv

load_dotenv()

# Import existing helpers WITHOUT modifying them. We use ablation_study (which has
# no deepagents dependency) instead of phase5_benchmark for textual model generation
# — same representations.
from ablation_study import (
    RepresentationDiscoverer,
    RepresentationType,
    AblationMiner,
    AblationMiningConfig,
    load_interpretation_guide,
    df_to_key_value_rows,
)
from langchain_openai import ChatOpenAI


def create_validator(as_is_model: str, model_name: str = "gpt-5-mini",
                      request_timeout: int = 90, max_retries: int = 5):
    """Validator wrapper that adds explicit request timeout and retry-with-backoff.

    The ablation_study.create_validator builds a ChatOpenAI client with default
    timeout (None = unlimited) which has caused multi-hour hangs in long runs.
    This version sets request_timeout, max_retries on the client itself plus
    a python-level retry loop to handle transient APIConnectionError and the
    occasional empty/garbage response.
    """
    if ":" in model_name:
        provider, model_id = model_name.split(":", 1)
    else:
        provider, model_id = "openai", model_name

    if "grok" in provider.lower() or "xai" in provider.lower():
        llm = ChatOpenAI(
            model=model_id, api_key=os.getenv("XAI_API_KEY"),
            base_url="https://api.x.ai/v1",
            timeout=request_timeout, max_retries=max_retries,
        )
    elif "deepseek" in provider.lower():
        llm = ChatOpenAI(
            model=model_id, api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
            timeout=request_timeout, max_retries=max_retries,
        )
    else:
        llm = ChatOpenAI(
            model=model_id, api_key=os.getenv("OPENAI_API_KEY"),
            timeout=request_timeout, max_retries=max_retries,
        )

    system_prompt = f"""You are an expert business process analyst. You will be given a business process description and a trace (sequence of activities).

Business Process Description:
{as_is_model}

Your task: Determine if the given trace is VALID (conforms to the observed business process) or INVALID (contains anomalies).

IMPORTANT: The business model describes processes as they are observed, which may include inefficiencies or non-ideal behaviors that still occur in practice.

Respond with ONLY one word: "VALID" or "INVALID"."""

    def validate(trace_df):
        trace_text = df_to_key_value_rows(trace_df)
        messages = [("system", system_prompt), ("human", trace_text)]
        # Outer python-level retry loop on top of the client's max_retries.
        # Catches APIConnectionError, Timeout, and odd empty responses.
        last_err = None
        for attempt in range(max_retries):
            try:
                start = time.time()
                response = llm.invoke(messages)
                compute_time = time.time() - start
                content = (response.content or "").strip().upper()
                if "VALID" not in content and "INVALID" not in content:
                    raise ValueError(f"Validator returned non-classification: {content[:200]!r}")
                pred = "INVALID" if "INVALID" in content else "VALID"
                tokens = response.response_metadata.get("token_usage", {})
                return pred, compute_time, tokens
            except Exception as e:
                last_err = e
                wait = min(2 ** attempt, 30)
                print(f"      [validator retry {attempt+1}/{max_retries} after {type(e).__name__}: {str(e)[:120]}; sleeping {wait}s]",
                       flush=True)
                time.sleep(wait)
        # All retries exhausted: surface the failure but return INVALID with zero
        # cost so the loop continues. Caller will see suspicious 0-time entries.
        print(f"      [validator giving up after {max_retries} retries; last error: {last_err}]",
               flush=True)
        return "INVALID", 0.0, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    return validate, system_prompt

# Curated 5-representation set chosen for MAXIMUM DIVERSITY of perspective with
# MINIMAL REDUNDANCY. Selection based on the single-rep isolation study and
# leave-one-out ablation in ablation_runs/ and ablation_results/:
#
#   - PROCESS_TREE: best single-rep mean F1 (0.976), hierarchical, explicit
#     operators (sequence/parallel/choice/loop/silent). Most LLM-interpretable.
#   - DFG: local activity-to-activity transitions with frequencies. Complements
#     PT's global view. Single-rep F1 = 0.962.
#   - ABSTRACT_VARIANTS: enumerates observed trace sequences. Unique perspective
#     (no other rep gives ground-truth log shapes).
#   - DECLARE: temporal logic constraints (Existence, Exactly 1, Always Before,
#     Always After, Never Together, Response, Precedence...). Strictly richer
#     than Log Skeleton. Different perspective than all control-flow reps.
#     Particularly relevant for B3 (insert duplicate of always-singular activity
#     → violates `Exactly 1`) and A3 (dependency swap → violates `Always Before`).
#   - LOG_ATTRIBUTES: only source of non-control-flow info (resource lists,
#     attribute statistics). Does not give resource→activity bindings (so will
#     not fix D1) but provides global resource context.
#
# Dropped from the 8-rep attempt (and from the full 11):
#   - All other Petri Net variants (Inductive, Heuristics, ILP, Alpha):
#     redundant with Process Tree (same control-flow info, less interpretable).
#     Leave-one-out showed REMOVING PN-Inductive improved F1 by +0.059 in hiring.
#   - Heuristics Net: subsumed by DFG + DECLARE.
#   - Log Skeleton: strict subset of DECLARE (per its own interpretation guide).
HARDER_REPRESENTATIONS = [
    RepresentationType.PROCESS_TREE,
    RepresentationType.DFG,
    RepresentationType.ABSTRACT_VARIANTS,
    RepresentationType.DECLARE,
    RepresentationType.LOG_ATTRIBUTES,
]


# =============================================================================
# COLUMN DETECTION
# =============================================================================

ACTIVITY_COL_CANDIDATES = ["concept:name", "activity", "Activity"]
RESOURCE_COL_CANDIDATES = ["org:resource", "resource", "org:role", "Resource"]
CASE_COL_CANDIDATES = ["case:case", "case:concept:name", "case_id", "case"]


def detect_columns(df: pd.DataFrame) -> Tuple[str, Optional[str], str]:
    activity_col = next((c for c in ACTIVITY_COL_CANDIDATES if c in df.columns), None)
    resource_col = next((c for c in RESOURCE_COL_CANDIDATES if c in df.columns), None)
    case_col = next((c for c in CASE_COL_CANDIDATES if c in df.columns), None)
    if activity_col is None:
        raise ValueError(f"No activity column found. Tried: {ACTIVITY_COL_CANDIDATES}")
    if case_col is None:
        raise ValueError(f"No case column found. Tried: {CASE_COL_CANDIDATES}")
    return activity_col, resource_col, case_col


# =============================================================================
# LOG CONTEXT
# =============================================================================

IGNORED_COLS_FOR_LLM = ["time", "@@index", "time:timestamp"]


def _strip_for_llm(df: pd.DataFrame) -> pd.DataFrame:
    """Drop noisy columns (timestamps, indices) before sending traces to the LLM."""
    return df.drop(columns=[c for c in IGNORED_COLS_FOR_LLM if c in df.columns])


def build_log_context(xes_path: str) -> Tuple[pd.DataFrame, dict]:
    """Load XES log (keeping ALL columns so discovery still works) and compute
    all stats needed by perturbations.

    Note: timestamps are kept in the returned log_df because pm4py discovery
    algorithms (Petri net, DFG, etc.) require them. They are stripped later
    via _strip_for_llm() before sampling traces for the validator.
    """
    log_df = pm4py.read_xes(xes_path)

    activity_col, resource_col, case_col = detect_columns(log_df)
    cases = log_df.groupby(case_col)
    case_ids = list(cases.groups.keys())
    total_cases = len(case_ids)

    variant_set = set()
    alphabet = set()
    activity_in_trace_count: Dict[str, int] = {}
    activity_count_per_trace: Dict[str, List[int]] = {}
    resource_to_activities: Dict[str, set] = {}
    activity_to_resources: Dict[str, set] = {}
    pair_counts: Dict[Tuple[str, str], Dict[str, int]] = {}

    for cid in case_ids:
        case_df = cases.get_group(cid)
        activities = case_df[activity_col].tolist()
        variant_set.add(tuple(activities))
        alphabet.update(activities)

        for act in set(activities):
            activity_in_trace_count[act] = activity_in_trace_count.get(act, 0) + 1
            activity_count_per_trace.setdefault(act, []).append(activities.count(act))

        if resource_col:
            for _, row in case_df.iterrows():
                a = row[activity_col]
                r = row[resource_col]
                resource_to_activities.setdefault(r, set()).add(a)
                activity_to_resources.setdefault(a, set()).add(r)

        # Always_before: track first occurrence position for each activity in trace
        first_pos: Dict[str, int] = {}
        for i, act in enumerate(activities):
            if act not in first_pos:
                first_pos[act] = i
        distinct = list(first_pos.keys())
        for a in distinct:
            for b in distinct:
                if a == b:
                    continue
                key = (a, b)  # "does a precede b in this trace?"
                pair_counts.setdefault(key, {"both": 0, "a_before_b": 0})
                pair_counts[key]["both"] += 1
                if first_pos[a] < first_pos[b]:
                    pair_counts[key]["a_before_b"] += 1

    # always_before: in EVERY trace where both appear, a precedes b
    # Require at least 2 co-occurrences to avoid singleton noise
    always_before = set()
    for (a, b), counts in pair_counts.items():
        if counts["both"] >= 2 and counts["a_before_b"] == counts["both"]:
            always_before.add((a, b))

    mandatory = {act for act, c in activity_in_trace_count.items() if c == total_cases}
    always_singular = {
        act for act, counts in activity_count_per_trace.items()
        if all(c == 1 for c in counts)
    }

    ctx = {
        "activity_col": activity_col,
        "resource_col": resource_col,
        "case_col": case_col,
        "alphabet": alphabet,
        "variant_set": variant_set,
        "mandatory": mandatory,
        "always_singular": always_singular,
        "resource_to_activities": resource_to_activities,
        "activity_to_resources": activity_to_resources,
        "always_before": always_before,
        "total_cases": total_cases,
    }
    return log_df, ctx


def sample_valid_traces_from_loaded(log_df: pd.DataFrame, case_col: str, k: int) -> List[pd.DataFrame]:
    """Sample k traces and strip noisy columns before perturbations/LLM use."""
    cases = log_df.groupby(case_col)
    case_ids = list(cases.groups.keys())
    selected = random.sample(case_ids, min(k, len(case_ids)))
    return [_strip_for_llm(cases.get_group(cid).reset_index(drop=True)) for cid in selected]


# =============================================================================
# PERTURBATION HELPERS
# =============================================================================

def _swap_rows(df: pd.DataFrame, i: int, j: int) -> pd.DataFrame:
    out = df.reset_index(drop=True).copy()
    tmp = out.iloc[i].copy()
    out.iloc[i] = out.iloc[j].copy()
    out.iloc[j] = tmp
    return out


def _insert_row(df: pd.DataFrame, position: int, template_idx: int,
                overrides: Dict[str, object]) -> pd.DataFrame:
    out = df.reset_index(drop=True).copy()
    new_row = out.iloc[template_idx].copy()
    for col, val in overrides.items():
        new_row[col] = val
    new_row_df = pd.DataFrame([new_row])
    top = out.iloc[:position]
    bottom = out.iloc[position:]
    return pd.concat([top, new_row_df, bottom], ignore_index=True)


def _delete_row(df: pd.DataFrame, position: int) -> pd.DataFrame:
    out = df.reset_index(drop=True).copy()
    return out.drop(index=position).reset_index(drop=True)


# =============================================================================
# PERTURBATIONS
# =============================================================================

def perturb_A1_random_swap(trace: pd.DataFrame, ctx: dict):
    if len(trace) < 2:
        return None, False, {"reason": "trace too short"}
    i, j = random.sample(range(len(trace)), 2)
    return _swap_rows(trace, i, j), True, {"swap_indices": [i, j]}


def perturb_A3_dependency_aware_swap(trace: pd.DataFrame, ctx: dict):
    activity_col = ctx["activity_col"]
    activities = trace[activity_col].tolist()
    candidates = []
    for i in range(len(activities)):
        for j in range(i + 1, len(activities)):
            if (activities[i], activities[j]) in ctx["always_before"]:
                candidates.append((i, j))
    if not candidates:
        return None, False, {"reason": "no always_before pair in trace"}
    i, j = random.choice(candidates)
    perturbed = _swap_rows(trace, i, j)
    return perturbed, True, {
        "swap_indices": [i, j],
        "broken_dependency": [str(activities[i]), str(activities[j])],
    }


def perturb_B1_insert_out_of_alphabet(trace: pd.DataFrame, ctx: dict):
    fakes = ["__INVALID_ACTIVITY__", "Unauthorized Step", "Bypass Approval",
             "External Hack", "Synthetic Garbage Event"]
    fake = next((f for f in fakes if f not in ctx["alphabet"]), None)
    if fake is None:
        # Construct a unique one
        fake = "__INVALID_" + str(random.randint(0, 1_000_000)) + "__"
        while fake in ctx["alphabet"]:
            fake = "__INVALID_" + str(random.randint(0, 1_000_000)) + "__"

    pos = random.randint(0, len(trace))
    template_idx = random.randint(0, max(0, len(trace) - 1))
    overrides = {ctx["activity_col"]: fake}
    if "concept:name" in trace.columns and ctx["activity_col"] != "concept:name":
        overrides["concept:name"] = fake
    perturbed = _insert_row(trace, pos, template_idx, overrides)
    return perturbed, True, {"inserted_activity": fake, "position": pos}


def perturb_B2_insert_misplaced_existing(trace: pd.DataFrame, ctx: dict):
    if not ctx["alphabet"]:
        return None, False, {"reason": "empty alphabet"}
    chosen = random.choice(list(ctx["alphabet"]))
    pos = random.randint(0, len(trace))
    template_idx = random.randint(0, max(0, len(trace) - 1))
    overrides = {ctx["activity_col"]: chosen}
    if "concept:name" in trace.columns and ctx["activity_col"] != "concept:name":
        overrides["concept:name"] = chosen
    perturbed = _insert_row(trace, pos, template_idx, overrides)
    return perturbed, True, {"inserted_activity": str(chosen), "position": pos}


def perturb_B3_insert_duplicate_unique(trace: pd.DataFrame, ctx: dict):
    activity_col = ctx["activity_col"]
    activities_in_trace = set(trace[activity_col].tolist())
    candidates = list(activities_in_trace & ctx["always_singular"])
    if not candidates:
        return None, False, {"reason": "no always-singular activity in trace"}
    chosen = random.choice(candidates)
    src_positions = trace.index[trace[activity_col] == chosen].tolist()
    template_idx = src_positions[0]
    pos = random.randint(0, len(trace))
    perturbed = _insert_row(trace, pos, template_idx, overrides={})
    return perturbed, True, {"duplicated_activity": str(chosen), "position": pos}


def perturb_C1_delete_mandatory(trace: pd.DataFrame, ctx: dict):
    activity_col = ctx["activity_col"]
    activities = trace[activity_col].tolist()
    candidates = [(i, a) for i, a in enumerate(activities) if a in ctx["mandatory"]]
    if not candidates:
        return None, False, {"reason": "no mandatory activity in trace"}
    i, chosen = random.choice(candidates)
    perturbed = _delete_row(trace, i)
    return perturbed, True, {"deleted_activity": str(chosen), "position": i}


def perturb_C2_delete_start_or_end(trace: pd.DataFrame, ctx: dict):
    if len(trace) < 2:
        return None, False, {"reason": "trace too short"}
    which = random.choice(["start", "end"])
    activity_col = ctx["activity_col"]
    if which == "start":
        deleted = trace.iloc[0][activity_col]
        perturbed = _delete_row(trace, 0)
    else:
        deleted = trace.iloc[-1][activity_col]
        perturbed = _delete_row(trace, len(trace) - 1)
    return perturbed, True, {"deleted_position": which, "deleted_activity": str(deleted)}


def perturb_D1_cross_role(trace: pd.DataFrame, ctx: dict):
    resource_col = ctx["resource_col"]
    if not resource_col:
        return None, False, {"reason": "no resource column in log"}

    activity_col = ctx["activity_col"]
    all_resources = set(ctx["resource_to_activities"].keys())
    candidates = []
    for i in range(len(trace)):
        act = trace.iloc[i][activity_col]
        legit = ctx["activity_to_resources"].get(act, set())
        illegitimate = all_resources - legit
        if illegitimate:
            candidates.append((i, act, list(illegitimate)))
    if not candidates:
        return None, False, {"reason": "no cross-role assignment possible"}

    i, act, options = random.choice(candidates)
    fake_res = random.choice(options)
    perturbed = trace.reset_index(drop=True).copy()
    current = perturbed.iloc[i][resource_col]
    perturbed.iloc[i, perturbed.columns.get_loc(resource_col)] = fake_res
    return perturbed, True, {
        "position": i,
        "activity": str(act),
        "from_resource": str(current),
        "to_resource": str(fake_res),
    }


def perturb_E1_two_swaps(trace: pd.DataFrame, ctx: dict):
    if len(trace) < 3:
        return None, False, {"reason": "trace too short for two swaps"}
    perturbed, ok1, m1 = perturb_A1_random_swap(trace, ctx)
    if not ok1:
        return None, False, {"reason": "first swap failed"}
    perturbed2, ok2, m2 = perturb_A1_random_swap(perturbed, ctx)
    if not ok2:
        return None, False, {"reason": "second swap failed"}
    return perturbed2, True, {"first_swap": m1, "second_swap": m2}


def perturb_E2_swap_and_insert(trace: pd.DataFrame, ctx: dict):
    perturbed, ok1, m1 = perturb_A1_random_swap(trace, ctx)
    if not ok1:
        return None, False, {"reason": "swap failed"}
    insert_fn = random.choice([perturb_B1_insert_out_of_alphabet,
                                perturb_B2_insert_misplaced_existing])
    perturbed2, ok2, m2 = insert_fn(perturbed, ctx)
    if not ok2:
        return None, False, {"reason": f"insert failed: {m2.get('reason')}"}
    return perturbed2, True, {"swap": m1, "insert_kind": insert_fn.__name__, "insert": m2}


# Registry: name → (function, skip_variant_check_flag)
# skip_variant_check=True for D1 because it does not change the activity
# sequence; invalidity comes from the resource attribute.
PERTURBATION_REGISTRY: Dict[str, Tuple[Callable, bool]] = {
    "A1_random_swap":              (perturb_A1_random_swap, False),
    "A3_dependency_aware_swap":    (perturb_A3_dependency_aware_swap, False),
    "B1_insert_out_of_alphabet":   (perturb_B1_insert_out_of_alphabet, False),
    "B2_insert_misplaced_existing":(perturb_B2_insert_misplaced_existing, False),
    "B3_insert_duplicate_unique":  (perturb_B3_insert_duplicate_unique, False),
    "C1_delete_mandatory":         (perturb_C1_delete_mandatory, False),
    "C2_delete_start_or_end":      (perturb_C2_delete_start_or_end, False),
    "D1_cross_role":               (perturb_D1_cross_role, True),
    "E1_two_swaps":                (perturb_E1_two_swaps, False),
    "E2_swap_and_insert":          (perturb_E2_swap_and_insert, False),
}


# =============================================================================
# VALIDITY CHECK + GENERATION
# =============================================================================

def is_effectively_invalid(perturbed: pd.DataFrame, ctx: dict) -> bool:
    seq = tuple(perturbed[ctx["activity_col"]].tolist())
    return seq not in ctx["variant_set"]


def generate_invalid_for_type(perturb_name: str, valid_traces: List[pd.DataFrame],
                               ctx: dict, k: int = 50,
                               max_attempts_factor: int = 30) -> Tuple[List[dict], dict]:
    """Generate k invalid traces for a single perturbation type.

    Returns: (invalid_records, stats)
        invalid_records: list of {"trace": df, "seed_idx": int, "metadata": {...}}
        stats: {"attempts": int, "rejected_coincidentally_valid": int,
                "rejected_perturbation_failed": int, "rejected_duplicate": int}
    """
    perturb_fn, skip_variant_check = PERTURBATION_REGISTRY[perturb_name]
    invalid: List[dict] = []
    stats = {"attempts": 0, "rejected_coincidentally_valid": 0,
             "rejected_perturbation_failed": 0, "rejected_duplicate": 0,
             "failure_reasons": {}}

    max_attempts = k * max_attempts_factor
    while len(invalid) < k and stats["attempts"] < max_attempts:
        stats["attempts"] += 1
        seed_idx = random.randrange(len(valid_traces))
        seed = valid_traces[seed_idx]
        perturbed, success, meta = perturb_fn(seed, ctx)
        if not success:
            stats["rejected_perturbation_failed"] += 1
            reason = meta.get("reason", "unknown")
            stats["failure_reasons"][reason] = stats["failure_reasons"].get(reason, 0) + 1
            continue
        if not skip_variant_check and not is_effectively_invalid(perturbed, ctx):
            stats["rejected_coincidentally_valid"] += 1
            continue
        # De-duplication against previously collected
        is_dup = False
        for prev in invalid:
            if perturbed.shape == prev["trace"].shape and perturbed.equals(prev["trace"]):
                is_dup = True
                break
        if is_dup:
            stats["rejected_duplicate"] += 1
            continue
        invalid.append({"trace": perturbed, "seed_idx": seed_idx, "metadata": meta})

    return invalid, stats


# =============================================================================
# PERTURBATION TESTS
# =============================================================================

def run_perturbation_tests(xes_path: str, n_per_type: int = 50, verbose: bool = True) -> dict:
    """Test each perturbation N times on random seeds. Return a structured report.

    Per perturbation, verifies:
      1. The perturbed trace is structurally different from the seed.
      2. Validity check is correct (perturbed sequence not in variant_set,
         except for D1 which is invalid by construction).
      3. Type-specific assertions hold.
    Reports retry rates for each type so we can detect silently broken perturbations.
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"PERTURBATION TESTS: {xes_path}")
        print(f"{'='*70}")

    log_df, ctx = build_log_context(xes_path)
    if verbose:
        print(f"  Loaded log: {len(log_df)} events, {ctx['total_cases']} cases")
        print(f"  Alphabet size: {len(ctx['alphabet'])}")
        print(f"  Mandatory activities: {len(ctx['mandatory'])} -> {sorted(list(ctx['mandatory']))[:5]}")
        print(f"  Always-singular: {len(ctx['always_singular'])} -> {sorted(list(ctx['always_singular']))[:5]}")
        print(f"  Always_before pairs: {len(ctx['always_before'])}")
        print(f"  Resource column: {ctx['resource_col']}")
        if ctx["resource_col"]:
            print(f"  Resources: {len(ctx['resource_to_activities'])}")

    # Sample 100 seeds (matches the benchmark scenario)
    valid_traces = sample_valid_traces_from_loaded(log_df, ctx["case_col"], 100)

    report = {"xes_path": xes_path, "n_per_type": n_per_type,
              "log_stats": {
                  "total_events": int(len(log_df)),
                  "total_cases": ctx["total_cases"],
                  "alphabet_size": len(ctx["alphabet"]),
                  "mandatory_count": len(ctx["mandatory"]),
                  "always_singular_count": len(ctx["always_singular"]),
                  "always_before_pair_count": len(ctx["always_before"]),
                  "resource_col": ctx["resource_col"],
                  "resource_count": len(ctx.get("resource_to_activities", {})),
              },
              "results": {}}

    all_passed = True

    for perturb_name in PERTURBATION_REGISTRY.keys():
        if verbose:
            print(f"\n  Testing {perturb_name}...")
        invalid, stats = generate_invalid_for_type(
            perturb_name, valid_traces, ctx, k=n_per_type, max_attempts_factor=30
        )

        type_report = {
            "produced": len(invalid),
            "target": n_per_type,
            "stats": stats,
            "structural_assertions": {"checked": 0, "passed": 0, "failures": []},
        }

        # Structural assertions per type
        for record in invalid[:30]:  # check first 30 for speed
            type_report["structural_assertions"]["checked"] += 1
            seed = valid_traces[record["seed_idx"]]
            perturbed = record["trace"]
            meta = record["metadata"]
            ok, fail_msg = _assert_perturbation_structure(
                perturb_name, seed, perturbed, meta, ctx
            )
            if ok:
                type_report["structural_assertions"]["passed"] += 1
            else:
                type_report["structural_assertions"]["failures"].append(fail_msg)
                if len(type_report["structural_assertions"]["failures"]) > 5:
                    break

        # Pass criteria:
        #   - produced exactly n_per_type
        #   - all structural assertions passed
        produced_ok = len(invalid) == n_per_type
        struct_ok = len(type_report["structural_assertions"]["failures"]) == 0
        type_report["passed"] = bool(produced_ok and struct_ok)
        if not type_report["passed"]:
            all_passed = False

        report["results"][perturb_name] = type_report

        if verbose:
            tag = "OK" if type_report["passed"] else "FAIL"
            attempts = stats["attempts"]
            yield_rate = len(invalid) / attempts if attempts > 0 else 0.0
            print(f"    [{tag}] produced={len(invalid)}/{n_per_type} "
                  f"attempts={attempts} yield={yield_rate:.1%} "
                  f"perturb_failed={stats['rejected_perturbation_failed']} "
                  f"coincidentally_valid={stats['rejected_coincidentally_valid']} "
                  f"dups={stats['rejected_duplicate']}")
            if not struct_ok:
                for f in type_report["structural_assertions"]["failures"][:3]:
                    print(f"      structural failure: {f}")
            if stats["failure_reasons"]:
                top = sorted(stats["failure_reasons"].items(), key=lambda x: -x[1])[:3]
                print(f"      top failure reasons: {top}")

    report["all_passed"] = all_passed
    if verbose:
        print(f"\n{'='*70}")
        print(f"OVERALL: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
        print(f"{'='*70}")
    return report


def _assert_perturbation_structure(name: str, seed: pd.DataFrame,
                                    perturbed: pd.DataFrame, meta: dict,
                                    ctx: dict) -> Tuple[bool, str]:
    """Verify type-specific structural properties of a single perturbation."""
    activity_col = ctx["activity_col"]
    seed_acts = seed[activity_col].tolist()
    pert_acts = perturbed[activity_col].tolist()

    # Universal: D1 is the only type where activity sequence is unchanged
    if name != "D1_cross_role":
        if seed_acts == pert_acts:
            # Also check that the row contents differ (e.g., D1-like attribute change is acceptable for some)
            if seed.shape == perturbed.shape and seed.reset_index(drop=True).equals(perturbed.reset_index(drop=True)):
                return False, f"{name}: perturbed equals seed"

    if name == "A1_random_swap":
        if len(perturbed) != len(seed):
            return False, "A1: length changed"
        if sorted(seed_acts) != sorted(pert_acts):
            return False, "A1: multiset of activities changed"

    elif name == "A3_dependency_aware_swap":
        if len(perturbed) != len(seed):
            return False, "A3: length changed"
        if sorted(seed_acts) != sorted(pert_acts):
            return False, "A3: multiset changed"
        broken = meta.get("broken_dependency")
        if broken and tuple(broken) not in ctx["always_before"]:
            return False, f"A3: claimed broken dependency {broken} not in always_before"

    elif name == "B1_insert_out_of_alphabet":
        if len(perturbed) != len(seed) + 1:
            return False, "B1: length not incremented by 1"
        inserted = meta.get("inserted_activity")
        if inserted in ctx["alphabet"]:
            return False, f"B1: inserted activity '{inserted}' is in alphabet"
        if inserted not in pert_acts:
            return False, f"B1: inserted activity '{inserted}' not in perturbed trace"

    elif name == "B2_insert_misplaced_existing":
        if len(perturbed) != len(seed) + 1:
            return False, "B2: length not incremented by 1"
        inserted = meta.get("inserted_activity")
        if inserted not in ctx["alphabet"]:
            return False, f"B2: inserted activity '{inserted}' not in alphabet"

    elif name == "B3_insert_duplicate_unique":
        if len(perturbed) != len(seed) + 1:
            return False, "B3: length not incremented by 1"
        dup = meta.get("duplicated_activity")
        if dup not in ctx["always_singular"]:
            return False, f"B3: duplicated '{dup}' not in always_singular"
        seed_count = seed_acts.count(dup)
        pert_count = pert_acts.count(dup)
        if pert_count != seed_count + 1:
            return False, f"B3: count of '{dup}' didn't increase by 1 ({seed_count}→{pert_count})"

    elif name == "C1_delete_mandatory":
        if len(perturbed) != len(seed) - 1:
            return False, "C1: length not decremented by 1"
        deleted = meta.get("deleted_activity")
        if deleted not in ctx["mandatory"]:
            return False, f"C1: deleted '{deleted}' not in mandatory"

    elif name == "C2_delete_start_or_end":
        if len(perturbed) != len(seed) - 1:
            return False, "C2: length not decremented by 1"
        which = meta.get("deleted_position")
        if which not in ("start", "end"):
            return False, f"C2: invalid deleted_position '{which}'"

    elif name == "D1_cross_role":
        if seed_acts != pert_acts:
            return False, "D1: activity sequence changed (should not)"
        if not ctx["resource_col"]:
            return False, "D1: no resource column"
        i = meta.get("position")
        new_res = meta.get("to_resource")
        act = meta.get("activity")
        if str(perturbed.iloc[i][ctx["resource_col"]]) != str(new_res):
            return False, f"D1: resource at pos {i} != claimed new resource"
        # Verify the new resource never does this activity in the log
        legit = ctx["activity_to_resources"].get(act, set())
        legit_str = {str(r) for r in legit}
        if str(new_res) in legit_str:
            return False, f"D1: '{new_res}' actually does '{act}' in log"

    elif name == "E1_two_swaps":
        if len(perturbed) != len(seed):
            return False, "E1: length changed"
        if sorted(seed_acts) != sorted(pert_acts):
            return False, "E1: multiset changed"

    elif name == "E2_swap_and_insert":
        if len(perturbed) != len(seed) + 1:
            return False, "E2: length not incremented by 1"

    return True, ""


# =============================================================================
# BENCHMARK
# =============================================================================

def _parallel_evaluate(traces: List[pd.DataFrame], validator, max_workers: int,
                        label: str = "") -> List[Tuple[str, float, dict]]:
    """Run validator over all traces in parallel. Returns results in input order.

    Each entry is (pred, compute_time, tokens). Errors are surfaced via the
    validator's own retry loop; if a worker still raises, we record INVALID +
    zero metrics so the loop continues.
    """
    results: List[Optional[Tuple[str, float, dict]]] = [None] * len(traces)
    completed = 0
    total = len(traces)
    start_wall = time.time()

    def _safe_validate(idx_trace):
        idx, trace = idx_trace
        try:
            return idx, validator(trace)
        except Exception as e:
            print(f"      [{label}] worker error idx={idx}: {type(e).__name__}: {str(e)[:120]}",
                   flush=True)
            return idx, ("INVALID", 0.0,
                          {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})

    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = [exe.submit(_safe_validate, (i, t)) for i, t in enumerate(traces)]
        for fut in as_completed(futures):
            idx, res = fut.result()
            results[idx] = res
            completed += 1
            if completed % max(1, total // 20) == 0 or completed == total:
                elapsed = time.time() - start_wall
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (total - completed) / rate if rate > 0 else 0
                print(f"    [{label}] {completed}/{total} done | wall={elapsed:.0f}s "
                      f"rate={rate:.1f}/s ETA={eta:.0f}s", flush=True)
    return results  # type: ignore[return-value]


def evaluate_traces_with_per_type_metrics(
    valid_traces: List[pd.DataFrame],
    invalid_by_type: Dict[str, List[dict]],
    validator,
    misclassified_root: Path,
    resume_state_path: Path = None,
    max_workers: int = 10,
):
    """
    1. Evaluate every valid trace once (shared TP/FN across types).
    2. For each perturbation type, evaluate its 50 invalid traces.
    Save misclassified into misclassified_root/{TYPE}/{fn,fp}_XXXX.pkl.
    If resume_state_path is provided, skip valid eval and per-type eval that
    are already marked complete in the resume state.
    """
    # ---------- Resume state ----------
    state = _load_resume_state(resume_state_path) if resume_state_path else None
    if state is None:
        state = {"valid_eval_complete": False, "valid_TP": 0, "valid_FN": 0,
                 "valid_compute_time": 0.0, "valid_tokens":
                     {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                 "completed_types": {}}

    # ---------- Valid traces (once) ----------
    valid_misclassified_dir = misclassified_root / "_valid_FN"
    valid_misclassified_dir.mkdir(parents=True, exist_ok=True)

    if state.get("valid_eval_complete"):
        TP = state["valid_TP"]
        FN = state["valid_FN"]
        valid_predictions = [True] * TP + [False] * FN
        valid_compute_time = state.get("valid_compute_time", 0.0)
        valid_tokens = state.get("valid_tokens",
                                  {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        print(f"\n  [RESUME] Valid eval already complete: TP={TP} FN={FN} (recall={TP/(TP+FN):.3f})", flush=True)
    else:
        print(f"\n  Evaluating {len(valid_traces)} valid traces in parallel (workers={max_workers})...",
               flush=True)
        results = _parallel_evaluate(valid_traces, validator, max_workers, label="valid")

        valid_predictions: List[bool] = []
        valid_compute_time = 0.0
        valid_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        for vidx, (pred, ct, tok) in enumerate(results):
            is_correct = pred == "VALID"
            valid_predictions.append(is_correct)
            valid_compute_time += ct
            for k in valid_tokens:
                valid_tokens[k] += tok.get(k, 0)
            if not is_correct:
                with open(valid_misclassified_dir / f"fn_{vidx:04d}.pkl", "wb") as f:
                    pickle.dump(valid_traces[vidx], f)

        TP = sum(valid_predictions)
        FN = len(valid_predictions) - TP
        print(f"  Valid done: TP={TP} FN={FN} (recall={TP/(TP+FN):.3f}) "
              f"compute={valid_compute_time:.0f}s tokens={valid_tokens['total_tokens']}",
               flush=True)
        # Persist valid eval state
        state["valid_eval_complete"] = True
        state["valid_TP"] = TP
        state["valid_FN"] = FN
        state["valid_compute_time"] = valid_compute_time
        state["valid_tokens"] = valid_tokens
        if resume_state_path:
            _save_resume_state(resume_state_path, state)

    # ---------- Invalid by type ----------
    by_type = {}
    grand_invalid_total = 0
    grand_invalid_compute = 0.0
    grand_invalid_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for ptype, records in invalid_by_type.items():
        # Skip if already complete in resume state
        if ptype in state.get("completed_types", {}):
            done = state["completed_types"][ptype]
            prec = TP / (TP + done["FP"]) if (TP + done["FP"]) > 0 else 0.0
            rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            print(f"\n  [RESUME-SKIP] {ptype}: TN={done['TN']} FP={done['FP']} "
                  f"P={prec:.3f} F1={f1:.3f}", flush=True)
            by_type[ptype] = {
                "invalid_predictions": [True] * done["TN"] + [False] * done["FP"],
                "compute_time_seconds": done.get("compute_time_seconds", 0.0),
                "tokens": done.get("tokens",
                                    {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}),
                "TP": TP, "FN": FN, "TN": done["TN"], "FP": done["FP"],
                "precision": prec, "recall": rec, "f1": f1,
                "resumed_from_existing": True,
            }
            grand_invalid_total += done["TN"] + done["FP"]
            grand_invalid_compute += by_type[ptype]["compute_time_seconds"]
            for k in grand_invalid_tokens:
                grand_invalid_tokens[k] += by_type[ptype]["tokens"].get(k, 0)
            continue

        print(f"\n  Evaluating {len(records)} invalid for type [{ptype}] in parallel "
              f"(workers={max_workers})...", flush=True)
        type_dir = misclassified_root / ptype
        type_dir.mkdir(parents=True, exist_ok=True)

        traces_only = [r["trace"] for r in records]
        results = _parallel_evaluate(traces_only, validator, max_workers, label=ptype)

        invalid_predictions: List[bool] = []
        type_compute = 0.0
        type_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        for iidx, (pred, ct, tok) in enumerate(results):
            is_correct = pred == "INVALID"
            invalid_predictions.append(is_correct)
            type_compute += ct
            for k in type_tokens:
                type_tokens[k] += tok.get(k, 0)
            if not is_correct:
                with open(type_dir / f"fp_{iidx:04d}.pkl", "wb") as f:
                    pickle.dump(traces_only[iidx], f)

        TN = sum(invalid_predictions)
        FP = len(invalid_predictions) - TN
        prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        by_type[ptype] = {
            "invalid_predictions": invalid_predictions,
            "compute_time_seconds": type_compute,
            "tokens": type_tokens,
            "TP": TP, "FN": FN, "TN": TN, "FP": FP,
            "precision": prec, "recall": rec, "f1": f1,
        }
        grand_invalid_total += len(invalid_predictions)
        grand_invalid_compute += type_compute
        for k in grand_invalid_tokens:
            grand_invalid_tokens[k] += type_tokens[k]
        print(f"  [{ptype}] done: P={prec:.3f} R={rec:.3f} F1={f1:.3f} (TN={TN} FP={FP})", flush=True)
        # Persist per-type completion to resume state
        state.setdefault("completed_types", {})[ptype] = {
            "TN": TN, "FP": FP,
            "compute_time_seconds": type_compute,
            "tokens": type_tokens,
        }
        if resume_state_path:
            _save_resume_state(resume_state_path, state)

    # ---------- Aggregate ----------
    agg_TN = sum(by_type[t]["TN"] for t in by_type)
    agg_FP = sum(by_type[t]["FP"] for t in by_type)
    agg_prec = TP / (TP + agg_FP) if (TP + agg_FP) > 0 else 0.0
    agg_rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    agg_f1 = 2 * agg_prec * agg_rec / (agg_prec + agg_rec) if (agg_prec + agg_rec) > 0 else 0.0
    aggregate = {
        "TP": TP, "FN": FN, "TN": agg_TN, "FP": agg_FP,
        "precision": agg_prec, "recall": agg_rec, "f1": agg_f1,
        "compute_time_seconds": valid_compute_time + grand_invalid_compute,
        "tokens": {
            k: valid_tokens[k] + grand_invalid_tokens[k]
            for k in valid_tokens
        },
        "n_valid": len(valid_traces),
        "n_invalid_total": grand_invalid_total,
    }
    return {
        "valid_predictions": valid_predictions,
        "valid_compute_time_seconds": valid_compute_time,
        "valid_tokens": valid_tokens,
        "by_type": by_type,
        "aggregate": aggregate,
    }


def save_invalid_traces_to_run(run_folder: Path, problem_name: str,
                                valid_traces: List[pd.DataFrame],
                                invalid_by_type: Dict[str, List[dict]],
                                generation_stats: Dict[str, dict]) -> dict:
    traces_folder = run_folder / "traces" / problem_name
    traces_folder.mkdir(parents=True, exist_ok=True)
    with open(traces_folder / "valid_traces.pkl", "wb") as f:
        pickle.dump(valid_traces, f)

    inv_folder = traces_folder / "invalid_by_type"
    inv_folder.mkdir(exist_ok=True)
    metadata_summary = {}
    for ptype, records in invalid_by_type.items():
        with open(inv_folder / f"{ptype}.pkl", "wb") as f:
            pickle.dump(records, f)
        metadata_summary[ptype] = {
            "count": len(records),
            "seed_indices": [r["seed_idx"] for r in records],
            "generation_stats": generation_stats[ptype],
        }
    (traces_folder / "invalid_metadata.json").write_text(
        json.dumps(metadata_summary, indent=2, default=str)
    )
    return {
        "valid_traces_path": str(traces_folder / "valid_traces.pkl"),
        "invalid_folder": str(inv_folder),
        "n_valid": len(valid_traces),
        "n_invalid_per_type": {t: len(r) for t, r in invalid_by_type.items()},
    }


def run_benchmark_for_dataset(xes_path: str, run_folder: Path,
                               llm_model: str, validator_model: str,
                               k_valid: int = 100, k_invalid_per_type: int = 50,
                               results: dict = None, resume: bool = False,
                               max_workers: int = 10) -> dict:
    problem_name = Path(xes_path).stem
    print(f"\n{'='*70}", flush=True)
    print(f"PROCESSING: {problem_name}{' [RESUME]' if resume else ''}", flush=True)
    print(f"{'='*70}", flush=True)

    if problem_name not in results:
        results[problem_name] = {}
    results[problem_name]["status"] = "in_progress"
    results[problem_name]["xes_path"] = xes_path
    save_results(run_folder, results)

    # Load log + context (always needed)
    print(f"  Loading log and computing context...", flush=True)
    log_df, ctx = build_log_context(xes_path)
    print(f"    {ctx['total_cases']} cases, {len(ctx['alphabet'])} activities, "
          f"{len(ctx['mandatory'])} mandatory, {len(ctx['always_singular'])} always-singular, "
          f"{len(ctx['always_before'])} always_before pairs", flush=True)

    # Persist log context summary for transparency
    ctx_summary = {
        "alphabet_size": len(ctx["alphabet"]),
        "alphabet": sorted(map(str, ctx["alphabet"])),
        "mandatory": sorted(map(str, ctx["mandatory"])),
        "always_singular": sorted(map(str, ctx["always_singular"])),
        "always_before_pair_count": len(ctx["always_before"]),
        "resource_col": ctx["resource_col"],
        "n_resources": len(ctx.get("resource_to_activities", {})),
        "total_cases": ctx["total_cases"],
        "n_unique_variants": len(ctx["variant_set"]),
    }
    (run_folder / f"{problem_name}_log_context.json").write_text(
        json.dumps(ctx_summary, indent=2, default=str)
    )

    # Sample 100 valid (or load existing if resuming)
    traces_folder = run_folder / "traces" / problem_name
    valid_pkl = traces_folder / "valid_traces.pkl"
    invalid_folder = traces_folder / "invalid_by_type"
    if resume and valid_pkl.exists() and invalid_folder.exists():
        print(f"  [RESUME] Loading cached traces from {traces_folder}", flush=True)
        with open(valid_pkl, "rb") as f:
            valid_traces = pickle.load(f)
        invalid_by_type: Dict[str, List[dict]] = {}
        for ptype in PERTURBATION_REGISTRY.keys():
            ipath = invalid_folder / f"{ptype}.pkl"
            if ipath.exists():
                with open(ipath, "rb") as f:
                    invalid_by_type[ptype] = pickle.load(f)
            else:
                # Type missing on disk → generate it now
                print(f"    [WARN] No cached invalid for {ptype}, generating...", flush=True)
                invalid, stats = generate_invalid_for_type(
                    ptype, valid_traces, ctx, k=k_invalid_per_type, max_attempts_factor=30
                )
                invalid_by_type[ptype] = invalid
                with open(ipath, "wb") as f:
                    pickle.dump(invalid, f)
        print(f"    {len(valid_traces)} valid + {sum(len(r) for r in invalid_by_type.values())} invalid loaded",
               flush=True)
    else:
        valid_traces = sample_valid_traces_from_loaded(log_df, ctx["case_col"], k_valid)
        print(f"  Sampled {len(valid_traces)} valid traces", flush=True)

        # Generate invalid by type
        print(f"\n  Generating invalid traces ({k_invalid_per_type} per type)...", flush=True)
        invalid_by_type: Dict[str, List[dict]] = {}
        generation_stats: Dict[str, dict] = {}
        for ptype in PERTURBATION_REGISTRY.keys():
            invalid, stats = generate_invalid_for_type(
                ptype, valid_traces, ctx, k=k_invalid_per_type, max_attempts_factor=30
            )
            invalid_by_type[ptype] = invalid
            generation_stats[ptype] = stats
            yield_rate = len(invalid) / stats["attempts"] if stats["attempts"] > 0 else 0.0
            tag = "OK" if len(invalid) == k_invalid_per_type else "PARTIAL"
            print(f"    [{tag}] {ptype}: produced={len(invalid)}/{k_invalid_per_type} "
                  f"attempts={stats['attempts']} yield={yield_rate:.1%}", flush=True)

        # Save traces
        trace_info = save_invalid_traces_to_run(run_folder, problem_name, valid_traces,
                                                  invalid_by_type, generation_stats)
        results[problem_name]["traces"] = trace_info
        save_results(run_folder, results)

    # Generate textual model with the SAME 3 reps as phase5_benchmark.py
    # (or load existing if resuming)
    textual_model_path = run_folder / f"{problem_name}_textual_model.txt"
    if resume and textual_model_path.exists():
        as_is_model = textual_model_path.read_text()
        print(f"\n  [RESUME] Loaded existing textual model ({len(as_is_model)} chars)", flush=True)
        results[problem_name]["textual_model_path"] = str(textual_model_path)
    else:
        print(f"\n  Generating textual AS-IS model from {len(HARDER_REPRESENTATIONS)} representations: "
              f"{[r.value for r in HARDER_REPRESENTATIONS]}", flush=True)
        discoverer = RepresentationDiscoverer(log_df, xes_path)
        for rep_type in HARDER_REPRESENTATIONS:
            try:
                print(f"    discovering {rep_type.value}...", end=" ", flush=True)
                discoverer._discover_one(rep_type)
                discoverer.interpretation_guides[rep_type] = load_interpretation_guide(rep_type)
                print("OK", flush=True)
            except Exception as e:
                print(f"FAILED: {e}", flush=True)
                discoverer.representations[rep_type] = None
                discoverer.string_representations[rep_type] = f"[Discovery failed: {e}]"
        mining_config = AblationMiningConfig(
            model_name=llm_model,
            prompt_file="GENERATE_AS_IS_MODEL.txt",
        )
        miner_output_dir = run_folder / "model_generation"
        miner_output_dir.mkdir(exist_ok=True)
        miner = AblationMiner(discoverer, mining_config, miner_output_dir)
        as_is_model, model_folder = miner.generate_textual_model(
            HARDER_REPRESENTATIONS, problem_name
        )
        textual_model_path.write_text(as_is_model)
        results[problem_name]["textual_model_path"] = str(textual_model_path)
        results[problem_name]["mining_config"] = {
            "representations": [r.value for r in HARDER_REPRESENTATIONS],
            "model_name": llm_model,
            "prompt_file": "GENERATE_AS_IS_MODEL.txt",
            "model_folder": str(model_folder),
        }
        save_results(run_folder, results)
        print(f"  Textual model saved ({len(as_is_model)} chars)", flush=True)

    # Validator
    validator, system_prompt = create_validator(as_is_model, model_name=validator_model)
    (run_folder / f"{problem_name}_system_prompt.txt").write_text(system_prompt)

    # Evaluate (with resume support via _resume_state.json in run folder)
    misclassified_root = run_folder / "misclassified" / problem_name
    misclassified_root.mkdir(parents=True, exist_ok=True)
    resume_state_path = run_folder / f"{problem_name}_resume_state.json"

    eval_result = evaluate_traces_with_per_type_metrics(
        valid_traces, invalid_by_type, validator, misclassified_root,
        resume_state_path=resume_state_path, max_workers=max_workers,
    )

    # Persist all metrics
    results[problem_name]["valid_evaluation"] = {
        "compute_time_seconds": eval_result["valid_compute_time_seconds"],
        "tokens": eval_result["valid_tokens"],
        "TP": eval_result["aggregate"]["TP"],
        "FN": eval_result["aggregate"]["FN"],
    }
    results[problem_name]["per_type"] = {
        ptype: {k: v for k, v in info.items() if k != "invalid_predictions"}
        for ptype, info in eval_result["by_type"].items()
    }
    results[problem_name]["aggregate"] = eval_result["aggregate"]
    results[problem_name]["status"] = "completed"
    save_results(run_folder, results)

    print(f"\n  AGGREGATE: P={eval_result['aggregate']['precision']:.3f} "
          f"R={eval_result['aggregate']['recall']:.3f} "
          f"F1={eval_result['aggregate']['f1']:.3f}")
    print(f"  PER-TYPE F1:")
    for ptype, info in eval_result["by_type"].items():
        print(f"    {ptype:32s} P={info['precision']:.3f} F1={info['f1']:.3f} (FP={info['FP']}/{info['FP']+info['TN']})")
    return results


# =============================================================================
# I/O HELPERS
# =============================================================================

def save_results(run_folder: Path, results: dict):
    (run_folder / "results.json").write_text(json.dumps(results, indent=2, default=str))


def create_run_folder() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = Path("HARDER") / f"run_{timestamp}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _load_resume_state(path: Path) -> Optional[dict]:
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _save_resume_state(path: Path, state: dict):
    path.write_text(json.dumps(state, indent=2, default=str))


def _bootstrap_resume_state_from_folder(run_folder: Path, problem_name: str,
                                         k_invalid_per_type: int) -> dict:
    """When --resume is used on a folder created by an older script (no _resume_state.json),
    infer completed types from the misclassified folder. We mark valid eval as complete
    only if _valid_FN exists (indicating the eval started); otherwise unknown.

    A type is considered "complete" if its folder exists AND the user explicitly listed
    it via --skip-types. We never auto-assume completion from FP files alone (since a
    type might have been interrupted with the same FP count as a complete one).
    """
    state = {"valid_eval_complete": False, "valid_TP": 0, "valid_FN": 0,
             "valid_compute_time": 0.0,
             "valid_tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
             "completed_types": {}}
    misc_root = run_folder / "misclassified" / problem_name
    valid_fn_dir = misc_root / "_valid_FN"
    if valid_fn_dir.exists():
        # Count FN files (almost always 0 for these datasets)
        fn_files = list(valid_fn_dir.glob("fn_*.pkl"))
        FN = len(fn_files)
        TP = 100 - FN  # assumes k_valid=100
        state["valid_eval_complete"] = True
        state["valid_TP"] = TP
        state["valid_FN"] = FN
    return state


def _seed_completed_types(state: dict, run_folder: Path, problem_name: str,
                           skip_types: List[str], k_invalid_per_type: int) -> dict:
    """For each type the user explicitly says is complete, count its FP files
    and add to completed_types in the state."""
    misc_root = run_folder / "misclassified" / problem_name
    for ptype in skip_types:
        type_dir = misc_root / ptype
        if not type_dir.exists():
            print(f"  [WARN] --skip-types listed '{ptype}' but folder doesn't exist; ignoring",
                   flush=True)
            continue
        fp_count = len(list(type_dir.glob("fp_*.pkl")))
        TN = k_invalid_per_type - fp_count
        state.setdefault("completed_types", {})[ptype] = {
            "TN": TN, "FP": fp_count,
            "compute_time_seconds": 0.0,  # unknown from previous run
            "tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "seeded_from_folder": True,
        }
    return state


# =============================================================================
# MAIN
# =============================================================================

DEFAULT_PROBLEMS = [
    "benchmark/hiring/hiring_log_high.xes",
    "benchmark/hospital/hospital_log_high.xes",
    "benchmark/lending/lending_log_high.xes",
    "benchmark/renting/renting_log_high.xes",
]


def parse_args():
    p = argparse.ArgumentParser(description="Phase 5 HARDER Benchmark — multi-perturbation trace classification")
    p.add_argument("--dataset", default=None,
                    help="Dataset name (hiring, hospital, lending, renting). If omitted, uses --xes or --all.")
    p.add_argument("--xes", default=None, help="Direct path to XES file (overrides --dataset)")
    p.add_argument("--all", action="store_true", help="Run all 4 datasets")
    p.add_argument("--llm", default="openai:gpt-5-mini", help="LLM for textual model generation")
    p.add_argument("--validator-llm", default="gpt-5-mini", help="LLM for trace validation")
    p.add_argument("--k-valid", type=int, default=100)
    p.add_argument("--k-invalid-per-type", type=int, default=50)
    p.add_argument("--test", action="store_true",
                    help="Run perturbation tests only (no LLM calls). Requires --dataset or --xes.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    p.add_argument("--resume", default=None, metavar="RUN_FOLDER",
                    help="Resume an existing run by reusing its textual model, traces, and "
                         "completed perturbation evaluations. Reads/writes _resume_state.json "
                         "in the run folder.")
    p.add_argument("--skip-types", default=None,
                    help="Comma-separated list of perturbation types to mark complete from "
                         "the existing misclassified folder (counts FP files as the source of "
                         "truth). Used together with --resume on runs created by an older "
                         "version of this script that did not write _resume_state.json.")
    p.add_argument("--max-workers", type=int, default=10,
                    help="Number of concurrent LLM calls during evaluation. Default: 10.")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    # Resolve target dataset(s)
    # If --resume, read datasets from saved config.json (set later); allow empty here
    if args.resume:
        problems = []  # filled in below from saved config
    elif args.all:
        problems = DEFAULT_PROBLEMS
    elif args.xes:
        problems = [args.xes]
    elif args.dataset:
        problems = [f"benchmark/{args.dataset}/{args.dataset}_log_high.xes"]
    else:
        print("Error: must specify --dataset, --xes, --all, or --resume")
        return 1

    for p in problems:
        if not Path(p).exists():
            print(f"Error: XES file not found: {p}")
            return 1

    # Test mode (no LLM calls)
    if args.test:
        out_dir = Path("HARDER") / f"perturbation_tests_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        out_dir.mkdir(parents=True, exist_ok=True)
        all_passed = True
        for xes_path in problems:
            report = run_perturbation_tests(xes_path, n_per_type=args.k_invalid_per_type)
            stem = Path(xes_path).stem
            (out_dir / f"{stem}_test_report.json").write_text(
                json.dumps(report, indent=2, default=str)
            )
            if not report["all_passed"]:
                all_passed = False
        print(f"\nTest reports saved to: {out_dir}")
        return 0 if all_passed else 2

    # Benchmark mode — either fresh or resume
    if args.resume:
        run_folder = Path(args.resume)
        if not run_folder.exists():
            print(f"Error: --resume folder not found: {run_folder}", flush=True)
            return 1
        # Load existing config (else use CLI args as fallback)
        config_path = run_folder / "config.json"
        if config_path.exists():
            saved_config = json.loads(config_path.read_text())
            args.llm = saved_config.get("llm_model", args.llm)
            args.validator_llm = saved_config.get("validator_model", args.validator_llm)
            args.k_valid = saved_config.get("k_valid", args.k_valid)
            args.k_invalid_per_type = saved_config.get("k_invalid_per_type", args.k_invalid_per_type)
            problems = saved_config.get("datasets", problems)
        config = saved_config if config_path.exists() else None
    else:
        run_folder = create_run_folder()
        config = None
    if config is None:
        config = {
            "llm_model": args.llm,
            "validator_model": args.validator_llm,
            "k_valid": args.k_valid,
            "k_invalid_per_type": args.k_invalid_per_type,
            "perturbation_types": list(PERTURBATION_REGISTRY.keys()),
            "seed": args.seed,
            "datasets": problems,
        }
        (run_folder / "config.json").write_text(json.dumps(config, indent=2))

    # When --skip-types is provided, seed _resume_state.json from existing FP files
    skip_types_list = []
    if args.skip_types:
        skip_types_list = [t.strip() for t in args.skip_types.split(",") if t.strip()]
        for xes_path in problems:
            problem_name = Path(xes_path).stem
            state_path = run_folder / f"{problem_name}_resume_state.json"
            state = _load_resume_state(state_path) or _bootstrap_resume_state_from_folder(
                run_folder, problem_name, args.k_invalid_per_type
            )
            state = _seed_completed_types(
                state, run_folder, problem_name, skip_types_list, args.k_invalid_per_type
            )
            _save_resume_state(state_path, state)
            print(f"  [SEED] {problem_name}: marked {len(skip_types_list)} types as complete in "
                  f"resume state ({state_path.name})", flush=True)

    print(f"\n{'='*70}", flush=True)
    print(f"HARDER BENCHMARK{' [RESUME]' if args.resume else ''}", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"Run folder:    {run_folder}", flush=True)
    print(f"LLM (gen):     {args.llm}", flush=True)
    print(f"LLM (valid):   {args.validator_llm}", flush=True)
    print(f"k_valid:       {args.k_valid}", flush=True)
    print(f"k_invalid/typ: {args.k_invalid_per_type} (× {len(PERTURBATION_REGISTRY)} types)", flush=True)
    print(f"Datasets:      {[Path(p).stem for p in problems]}", flush=True)
    if skip_types_list:
        print(f"Skipping types (assumed complete from FP files): {skip_types_list}", flush=True)

    # Load or initialize results
    results_path = run_folder / "results.json"
    if args.resume and results_path.exists():
        results = json.loads(results_path.read_text())
        results["_meta"]["status"] = "running"
        results["_meta"]["resume_at"] = datetime.now().isoformat()
    else:
        results = {
            "_meta": {
                "start_time": datetime.now().isoformat(),
                "end_time": None,
                "config": config,
                "run_folder": str(run_folder),
                "status": "running",
            }
        }
    save_results(run_folder, results)

    for xes_path in problems:
        run_benchmark_for_dataset(
            xes_path, run_folder, args.llm, args.validator_llm,
            k_valid=args.k_valid, k_invalid_per_type=args.k_invalid_per_type,
            results=results, resume=bool(args.resume),
            max_workers=args.max_workers,
        )

    results["_meta"]["end_time"] = datetime.now().isoformat()
    results["_meta"]["status"] = "completed"
    save_results(run_folder, results)

    # Final summary
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY")
    print(f"{'='*70}")
    for xes_path in problems:
        name = Path(xes_path).stem
        if name not in results or "aggregate" not in results[name]:
            continue
        agg = results[name]["aggregate"]
        print(f"\n{name}:")
        print(f"  Aggregate: P={agg['precision']:.3f} R={agg['recall']:.3f} F1={agg['f1']:.3f}")
        print(f"  Per-type F1:")
        for ptype, info in results[name]["per_type"].items():
            print(f"    {ptype:32s} {info['f1']:.3f}")
    print(f"\nResults saved to: {run_folder}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
