"""Common helpers shared by baselines/* — trace loading, schema, metric computation."""

import json
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import pandas as pd
import pm4py


# Source of truth for traces and the "ground truth" APM run we compare against.
# Using the 5-cur run because it's the only May 4 run with all 4 datasets.
HARDER_TRACE_RUN = Path("HARDER/run_20260504_221347")

DATASETS = ["hiring", "hospital", "lending", "renting"]

PERTURBATION_TYPES = [
    "A1_random_swap", "A3_dependency_aware_swap",
    "B1_insert_out_of_alphabet", "B2_insert_misplaced_existing",
    "B3_insert_duplicate_unique",
    "C1_delete_mandatory", "C2_delete_start_or_end",
    "D1_cross_role",
    "E1_two_swaps", "E2_swap_and_insert",
]


def xes_path_for(dataset: str) -> str:
    return f"benchmark/{dataset}/{dataset}_log_high.xes"


def load_traces(dataset: str) -> Tuple[List[pd.DataFrame], Dict[str, List[dict]]]:
    """Load 100 valid + 50 invalid per type for a dataset from the HARDER trace cache."""
    traces_dir = HARDER_TRACE_RUN / "traces" / f"{dataset}_log_high"
    with open(traces_dir / "valid_traces.pkl", "rb") as f:
        valid = pickle.load(f)
    invalid_by_type = {}
    for ptype in PERTURBATION_TYPES:
        with open(traces_dir / "invalid_by_type" / f"{ptype}.pkl", "rb") as f:
            invalid_by_type[ptype] = pickle.load(f)
    return valid, invalid_by_type


def synthesize_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Add a sequential time:timestamp column if missing. Required by some pm4py
    conformance algorithms even when control-flow only matters."""
    if "time:timestamp" in df.columns:
        return df
    out = df.copy()
    base = datetime(2020, 1, 1)
    out["time:timestamp"] = [base + timedelta(minutes=i) for i in range(len(out))]
    return out


def trace_to_event_log_df(trace_df: pd.DataFrame, case_id: str = "synthetic_case") -> pd.DataFrame:
    """Convert a single-trace DataFrame into a pm4py-compatible event log DataFrame."""
    df = trace_df.copy()
    if "concept:name" not in df.columns and "activity" in df.columns:
        df["concept:name"] = df["activity"]
    df["case:concept:name"] = case_id
    df = synthesize_timestamps(df)
    df = pm4py.format_dataframe(
        df, case_id="case:concept:name",
        activity_key="concept:name", timestamp_key="time:timestamp",
    )
    return df


def load_full_log(dataset: str) -> pd.DataFrame:
    """Load the full XES log. Used to discover representations."""
    return pm4py.read_xes(xes_path_for(dataset))


def compute_metrics(predictions_valid: List[bool], predictions_invalid: List[bool]) -> Dict:
    """predictions_valid[i]=True means trace i was correctly classified VALID.
       predictions_invalid[i]=True means trace i was correctly classified INVALID."""
    TP = sum(predictions_valid)
    FN = len(predictions_valid) - TP
    TN = sum(predictions_invalid)
    FP = len(predictions_invalid) - TN
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "TP": TP, "FN": FN, "TN": TN, "FP": FP,
        "precision": precision, "recall": recall, "f1": f1,
    }


def empty_tokens() -> Dict[str, int]:
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
