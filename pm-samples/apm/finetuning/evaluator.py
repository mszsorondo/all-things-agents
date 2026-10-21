"""Trace classification evaluation via OpenAI API.

Uses create_validator from phase5_benchmark.py for prompt consistency
with the main benchmark pipeline. Traces are classified in parallel
via ThreadPoolExecutor.

Reuses trace sampling/generation and metrics from the parent codebase.
"""

import sys
import time
from pathlib import Path
from typing import Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from phase5_benchmark import (
    create_validator,
    sample_valid_traces,
    generate_invalid_traces,
    calculate_metrics,
    get_trace_cache_key,
    save_traces_to_cache,
    load_traces_from_cache,
)
from finetuning.config import EvalConfig


def get_or_generate_traces(xes_path: str, config: EvalConfig):
    """Sample traces from the event log.

    By default (resample_traces_each_step=True), draws a fresh random subsample
    of size k from the full log each time. This means every training step
    evaluates on different traces, giving the generator diverse reward signal
    across epochs.

    Returns:
        (valid_traces, invalid_traces): Lists of trace DataFrames
    """
    if not config.resample_traces_each_step and config.use_trace_cache:
        cache_key = get_trace_cache_key(
            xes_path, config.k_valid, config.k_invalid, ["random_swap_events"]
        )
        valid, invalid = load_traces_from_cache(cache_key)
        if valid is not None:
            return valid, invalid

    valid = sample_valid_traces(xes_path, k=config.k_valid)
    invalid = generate_invalid_traces(valid, k=config.k_invalid)

    if not config.resample_traces_each_step and config.use_trace_cache:
        save_traces_to_cache(cache_key, valid, invalid)

    return valid, invalid


def evaluate_textual_model(
    textual_model: str,
    xes_path: str,
    config: EvalConfig,
) -> Dict:
    """Full evaluation: sample traces, classify via API, return metrics.

    Uses create_validator from phase5_benchmark.py so the validator prompt
    is identical to the main benchmark pipeline.

    Returns:
        dict with keys: precision, recall, f1, TP, FP, TN, FN, ...
    """
    valid_traces, invalid_traces = get_or_generate_traces(xes_path, config)

    validator_fn, _ = create_validator(textual_model, model_name=config.validator_model)

    all_traces = valid_traces + invalid_traces
    results = [None] * len(all_traces)
    total_time = 0.0
    total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    with ThreadPoolExecutor(max_workers=config.k_parallel) as executor:
        future_to_idx = {}
        for idx, trace in enumerate(all_traces):
            future = executor.submit(validator_fn, trace)
            future_to_idx[future] = idx

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            pred, compute_time, tokens_used = future.result()
            results[idx] = pred
            total_time += compute_time
            for key in total_tokens:
                total_tokens[key] += tokens_used.get(key, 0)

    valid_preds = [results[i] == "VALID" for i in range(len(valid_traces))]
    invalid_preds = [results[i] == "INVALID" for i in range(len(valid_traces), len(all_traces))]

    return calculate_metrics(valid_preds, invalid_preds, total_time, total_tokens)
