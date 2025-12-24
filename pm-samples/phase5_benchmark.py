#!/usr/bin/env python3
"""Phase 5 Testing Benchmark - Minimal Implementation"""

import os
import json
import random
import time
import pickle
import hashlib
import argparse
import pm4py
from pathlib import Path
from datetime import datetime
from copy import deepcopy
from dataclasses import asdict
from langchain_openai import ChatOpenAI
from enhanced_agentic_miner import AgenticProcessMiner, MiningConfiguration, RepresentationType
from benchmark_template import random_swap_events, skip_mandatory_activity, df_to_key_value_rows

def get_as_is_model(xes_path, precomputed_model_path=None, llm_model="openai:gpt-5-mini"):
    """
    Generate or load the AS-IS process model.

    Returns:
        tuple: (model_text, mining_config_info)
            - model_text: The textual model content
            - mining_config_info: Dict with mining configuration details
    """
    if precomputed_model_path:
        model_text = Path(precomputed_model_path).read_text()
        mining_config_info = {
            "source": "precomputed",
            "precomputed_model_path": str(precomputed_model_path),
            "mining_config": None,
            "run_folder": None
        }
        return model_text, mining_config_info

    config = MiningConfiguration(
        representations=[RepresentationType.PETRI_NET, RepresentationType.DFG, RepresentationType.ABSTRACT_VARIANTS],
        model_name=llm_model,
        prompt_file="ONLY_GENERATE_TEXTUAL_MODEL.txt",
        use_cache=True
    )
    miner = AgenticProcessMiner(config)
    miner.log_file = xes_path
    miner.xes_path = xes_path
    miner.dataset_name = Path(xes_path).stem
    miner.discover_representations()
    miner.run_agent()

    model_text = (Path(miner.run_folder) / "textual_model.txt").read_text()

    # Serialize MiningConfiguration for storage
    config_dict = asdict(config)
    config_dict["representations"] = [r.value for r in config.representations]

    mining_config_info = {
        "source": "generated",
        "precomputed_model_path": None,
        "mining_config": config_dict,
        "run_folder": str(miner.run_folder),
        "generation_timestamp": datetime.now().isoformat()
    }

    return model_text, mining_config_info


def get_as_is_model_no_tools(xes_path, precomputed_model_path=None, llm_model="openai:gpt-4o-mini",
                              prompt_file_as_is="GENERATE_AS_IS_MODEL.txt",
                              prompt_file_ideal="GENERATE_IDEAL_MODEL.txt"):
    """
    Generate or load the AS-IS process model using direct LLM calls (no tool calling).
    Uses two-step conversation: first generates AS-IS, then IDEAL with shared context.

    Args:
        xes_path: Path to the XES event log file
        precomputed_model_path: Optional path to a precomputed textual model
        llm_model: LLM model to use (format: "provider:model", e.g., "openai:gpt-4o-mini")
        prompt_file_as_is: Path to the AS-IS generation prompt file
        prompt_file_ideal: Path to the IDEAL generation prompt file

    Returns:
        tuple: (as_is_model_text, ideal_model_text, mining_config_info)
            - as_is_model_text: The AS-IS textual model content
            - ideal_model_text: The IDEAL textual model content
            - mining_config_info: Dict with mining configuration details
    """
    if precomputed_model_path:
        model_text = Path(precomputed_model_path).read_text()
        # For precomputed, we don't have IDEAL model stored separately
        # Try to find ideal model in same folder
        precomputed_folder = Path(precomputed_model_path).parent
        ideal_path = precomputed_folder / "ideal_textual_model.txt"
        ideal_model_text = ideal_path.read_text() if ideal_path.exists() else None

        mining_config_info = {
            "source": "precomputed",
            "precomputed_model_path": str(precomputed_model_path),
            "mining_config": None,
            "run_folder": str(precomputed_folder),
            "method": "precomputed"
        }
        return model_text, ideal_model_text, mining_config_info

    config = MiningConfiguration(
        representations=[RepresentationType.PETRI_NET, RepresentationType.DFG, RepresentationType.ABSTRACT_VARIANTS],
        model_name=llm_model,
        prompt_file=prompt_file_as_is,  # Not used directly, but stored for reference
        use_cache=True
    )
    miner = AgenticProcessMiner(config)
    miner.log_file = xes_path
    miner.xes_path = xes_path
    miner.dataset_name = Path(xes_path).stem

    # Discover process representations
    print(f"  Discovering process representations...")
    miner.discover_representations()

    # Run two-step generation (AS-IS then IDEAL with shared context)
    print(f"  Running two-step LLM generation (no tools)...")
    as_is_model_text, ideal_model_text = miner.run_without_tools(
        prompt_file_as_is=prompt_file_as_is,
        prompt_file_ideal=prompt_file_ideal
    )

    # Serialize MiningConfiguration for storage
    config_dict = asdict(config)
    config_dict["representations"] = [r.value for r in config.representations]

    mining_config_info = {
        "source": "generated",
        "precomputed_model_path": None,
        "mining_config": config_dict,
        "run_folder": str(miner.run_folder),
        "generation_timestamp": datetime.now().isoformat(),
        "method": "no-tools",
        "prompt_file_as_is": prompt_file_as_is,
        "prompt_file_ideal": prompt_file_ideal
    }

    return as_is_model_text, ideal_model_text, mining_config_info


def sample_valid_traces(xes_path, k):
    log = pm4py.read_xes(xes_path)
    ignored_columns = ["time", "@@index", "time:timestamp"]
    log = log.drop([col for col in ignored_columns if col in log.columns], axis=1)
    traces = log.groupby('case:case')
    valid = [traces.get_group(tid) for tid in traces.groups.keys()]
    return random.sample(valid, min(k, len(valid)))

def generate_invalid_traces(valid_traces, k):
    invalid = []
    perturbations = [random_swap_events]#
    attempts = 0
    max_attempts = k * 10
    while len(invalid) < k and attempts < max_attempts:
        attempts += 1
        valid_trace = random.choice(valid_traces)
        perturbation = random.choice(perturbations)
        invalid_trace = perturbation(valid_trace)
        is_duplicate = any(invalid_trace.equals(t) for t in invalid)
        if not is_duplicate:
            invalid.append(invalid_trace)
    return invalid

def create_validator(as_is_model, model_name="gpt-5-nano"):
    llm = ChatOpenAI(model=model_name, api_key=os.getenv("OPENAI_API_KEY"))
    system_prompt = f"""You are an expert business process analyst. You will be given a business process description and a trace (sequence of activities).

Business Process Description:
{as_is_model}

Your task: Determine if the given trace is VALID (conforms to the observed business process) or INVALID (contains anomalies).

IMPORTANT: The business model describes processes as they are observed, which may include inefficiencies or non-ideal behaviors that still occur in practice.

Respond with ONLY one word: "VALID" or "INVALID"."""

    def validate(trace_df):
        trace_text = df_to_key_value_rows(trace_df)
        messages = [
            ("system", system_prompt),
            ("human", trace_text)
        ]
        start_time = time.time()
        response = llm.invoke(messages)
        compute_time = time.time() - start_time

        tokens_used = response.response_metadata.get('token_usage', {})

        return response.content.strip(), compute_time, tokens_used

    return validate, system_prompt

def load_misclassified_trace(file_path: str):
    """
    Load a misclassified trace from a pickle file.

    Usage:
        trace_df = load_misclassified_trace(misclassified_ref["file"])
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def save_misclassified_trace(trace_df, folder: Path, error_type: str, index: int, expected: str, predicted: str):
    """
    Save a misclassified trace to a pickle file and return reference info.

    Args:
        trace_df: The trace DataFrame
        folder: Folder to save the trace
        error_type: 'fn' for false negative, 'fp' for false positive
        index: Original index of the trace
        expected: Expected label
        predicted: Predicted label

    Returns:
        dict with file path and summary info
    """
    filename = f"{error_type}_{index:04d}.pkl"
    filepath = folder / filename

    with open(filepath, 'wb') as f:
        pickle.dump(trace_df, f)

    # Extract activity summary for quick reference
    if 'concept:name' in trace_df.columns:
        activities = trace_df['concept:name'].tolist()
    elif 'activity' in trace_df.columns:
        activities = trace_df['activity'].tolist()
    else:
        activity_col = [c for c in trace_df.columns if 'case' not in c.lower()][0]
        activities = trace_df[activity_col].tolist()

    case_id = None
    if 'case:case' in trace_df.columns:
        case_id = str(trace_df['case:case'].iloc[0])
    elif 'case:concept:name' in trace_df.columns:
        case_id = str(trace_df['case:concept:name'].iloc[0])

    return {
        "file": str(filepath),
        "index": index,
        "case_id": case_id,
        "activities": activities,
        "num_events": len(trace_df),
        "expected": expected,
        "predicted": predicted
    }


def evaluate_traces(valid_traces, invalid_traces, validator, misclassified_folder: Path = None):
    """
    Evaluate traces and optionally save misclassified ones to files.

    Args:
        valid_traces: List of valid trace DataFrames
        invalid_traces: List of invalid trace DataFrames
        validator: Validation function
        misclassified_folder: If provided, save misclassified traces here
    """
    valid_predictions = []
    invalid_predictions = []
    total_compute_time = 0
    total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    # Track misclassified traces
    misclassified = {
        "false_negatives": [],  # Valid traces classified as INVALID
        "false_positives": []   # Invalid traces classified as VALID
    }

    for i, trace in enumerate(valid_traces):
        pred, compute_time, tokens_used = validator(trace)
        is_correct = pred == "VALID"
        valid_predictions.append(is_correct)

        # Track false negatives (valid traces misclassified as invalid)
        if not is_correct and misclassified_folder:
            trace_ref = save_misclassified_trace(
                trace, misclassified_folder, "fn", i, "VALID", pred
            )
            misclassified["false_negatives"].append(trace_ref)

        total_compute_time += compute_time
        for key in total_tokens:
            total_tokens[key] += tokens_used.get(key, 0)

        TP = sum(valid_predictions)
        FN = len(valid_predictions) - TP
        TN = sum(invalid_predictions)
        FP = len(invalid_predictions) - TN

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"Valid trace {i+1}/{len(valid_traces)}: {pred} ({'✓' if is_correct else '✗'}) | P={precision:.3f} R={recall:.3f} F1={f1:.3f} | TP={TP} FP={FP} TN={TN} FN={FN} | Time={compute_time:.2f}s Tokens={tokens_used.get('total_tokens', 0)}")
        time.sleep(1)

    for i, trace in enumerate(invalid_traces):
        pred, compute_time, tokens_used = validator(trace)
        is_correct = pred == "INVALID"
        invalid_predictions.append(is_correct)

        # Track false positives (invalid traces misclassified as valid)
        if not is_correct and misclassified_folder:
            trace_ref = save_misclassified_trace(
                trace, misclassified_folder, "fp", i, "INVALID", pred
            )
            misclassified["false_positives"].append(trace_ref)

        total_compute_time += compute_time
        for key in total_tokens:
            total_tokens[key] += tokens_used.get(key, 0)

        TP = sum(valid_predictions)
        FN = len(valid_predictions) - TP
        TN = sum(invalid_predictions)
        FP = len(invalid_predictions) - TN

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"Invalid trace {i+1}/{len(invalid_traces)}: {pred} ({'✓' if is_correct else '✗'}) | P={precision:.3f} R={recall:.3f} F1={f1:.3f} | TP={TP} FP={FP} TN={TN} FN={FN} | Time={compute_time:.2f}s Tokens={tokens_used.get('total_tokens', 0)}")
        time.sleep(1)

    return valid_predictions, invalid_predictions, total_compute_time, total_tokens, misclassified

def calculate_metrics(valid_preds, invalid_preds, total_compute_time, total_tokens):
    TP = sum(valid_preds)
    TN = sum(invalid_preds)
    FN = len(valid_preds) - TP
    FP = len(invalid_preds) - TN

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "total_compute_time_seconds": total_compute_time,
        "total_tokens": total_tokens["total_tokens"],
        "prompt_tokens": total_tokens["prompt_tokens"],
        "completion_tokens": total_tokens["completion_tokens"]
    }

def get_trace_cache_key(xes_path, k_valid, k_invalid, perturbations):
    xes_stat = os.stat(xes_path)
    key_parts = [xes_path, str(xes_stat.st_size), str(xes_stat.st_mtime), str(k_valid), str(k_invalid), str(perturbations)]
    key_string = "|".join(key_parts)
    return hashlib.md5(key_string.encode()).hexdigest()

def save_traces_to_cache(cache_key, valid_traces, invalid_traces, cache_dir="cache/traces"):
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    cache_file = cache_path / f"{cache_key}.pkl"
    with open(cache_file, "wb") as f:
        pickle.dump({"valid": valid_traces, "invalid": invalid_traces}, f)

def load_traces_from_cache(cache_key, cache_dir="cache/traces"):
    cache_file = Path(cache_dir) / f"{cache_key}.pkl"
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
            return data["valid"], data["invalid"]
    return None, None

def create_random_baseline_validator():
    def validate(trace_df):
        pred = "VALID" if random.random() > 0.5 else "INVALID"
        return pred, 0.0, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return validate

def create_run_folder():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = Path(f"benchmark_runs/run_{timestamp}")
    run_folder.mkdir(parents=True, exist_ok=True)
    return run_folder


def get_default_mining_config(llm_model="openai:gpt-5-mini"):
    """Get the default MiningConfiguration as a serializable dict."""
    config = MiningConfiguration(
        representations=[RepresentationType.PETRI_NET, RepresentationType.DFG, RepresentationType.ABSTRACT_VARIANTS],
        model_name=llm_model,
        prompt_file="ONLY_GENERATE_TEXTUAL_MODEL.txt",
        use_cache=True
    )
    config_dict = asdict(config)
    config_dict["representations"] = [r.value for r in config.representations]
    return config_dict


def save_results(run_folder: Path, results: dict):
    """Save results to JSON file."""
    (run_folder / "results.json").write_text(json.dumps(results, indent=2))


def create_empty_metrics():
    """Create placeholder for unfilled metrics."""
    return {
        "precision": None,
        "recall": None,
        "f1": None,
        "TP": None,
        "TN": None,
        "FP": None,
        "FN": None,
        "total_compute_time_seconds": None,
        "total_tokens": None,
        "prompt_tokens": None,
        "completion_tokens": None
    }


def initialize_results_structure(problems, config, precomputed_models, run_folder):
    """
    Initialize the complete results structure with experiment info and empty metrics.
    This is saved BEFORE any computation begins.
    """
    results = {
        "_meta": {
            "benchmark_start_time": datetime.now().isoformat(),
            "benchmark_end_time": None,
            "run_folder": str(run_folder),
            "status": "initialized",
            "method": config.get("method", "tools"),
            "llm_model": config.get("llm_model", "openai:gpt-4o-mini"),
            "validator_model": config.get("validator_model", config.get("model", "gpt-4o-mini"))
        }
    }

    for xes_path in problems:
        problem_name = Path(xes_path).stem
        precomputed_path = precomputed_models.get(problem_name)
        cache_key = get_trace_cache_key(xes_path, config["k_valid"], config["k_invalid"], config["perturbations"])

        # Determine mining config info structure
        if precomputed_path:
            mining_config_info = {
                "source": "precomputed",
                "precomputed_model_path": str(precomputed_path),
                "mining_config": None,
                "run_folder": None,
                "generation_timestamp": None
            }
        else:
            mining_config_info = {
                "source": "to_be_generated",
                "precomputed_model_path": None,
                "mining_config": get_default_mining_config(),
                "run_folder": None,
                "generation_timestamp": None
            }

        results[problem_name] = {
            "status": "pending",
            "xes_path": xes_path,
            "experiment_info": {
                "mining_configuration": mining_config_info,
                "trace_classification": {
                    "validator_model": config.get("validator_model", config.get("model", "gpt-4o-mini")),
                    "system_prompt_path": str(run_folder / f"{problem_name}_system_prompt.txt")
                },
                "trace_generation": {
                    "cache_key": cache_key,
                    "k_valid": config["k_valid"],
                    "k_invalid": config["k_invalid"],
                    "perturbations": config["perturbations"],
                    "traces_generated_at": None
                }
            },
            "textual_model_path": str(run_folder / f"{problem_name}_textual_model.txt"),
            "ideal_textual_model_path": str(run_folder / f"{problem_name}_ideal_textual_model.txt") if config.get("method") == "no-tools" else None,
            "llm": create_empty_metrics(),
            "misclassified_traces": {
                "folder": str(run_folder / "misclassified" / problem_name),
                "false_negatives": [],  # Valid traces classified as INVALID (file references)
                "false_positives": []   # Invalid traces classified as VALID (file references)
            },
            "random_baseline": create_empty_metrics() if config["run_baseline"] else None
        }

    return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Phase 5 Benchmark - LLM-based Process Trace Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (tools method, gpt-4o-mini)
  python phase5_benchmark.py

  # Run with no-tools method (two-step conversation)
  python phase5_benchmark.py --method no-tools

  # Specify LLM for textual model generation
  python phase5_benchmark.py --llm openai:gpt-4o

  # Specify LLM for trace validation
  python phase5_benchmark.py --validator-llm gpt-4o-mini

  # Full example with all options
  python phase5_benchmark.py --method no-tools --llm openai:gpt-4o --validator-llm gpt-4o-mini

  # Use Anthropic model
  python phase5_benchmark.py --llm anthropic:claude-sonnet-4-20250514
"""
    )

    parser.add_argument(
        "--method",
        type=str,
        choices=["tools", "no-tools"],
        default="tools",
        help="Method for textual model generation: 'tools' (agent with tool calling) or 'no-tools' (direct two-step LLM calls). Default: tools"
    )

    parser.add_argument(
        "--llm",
        type=str,
        default="openai:gpt-4o-mini",
        help="LLM for textual model generation (format: provider:model). Default: openai:gpt-4o-mini"
    )

    parser.add_argument(
        "--validator-llm",
        type=str,
        default="gpt-4o-mini",
        help="LLM model for trace validation (OpenAI model name). Default: gpt-4o-mini"
    )

    parser.add_argument(
        "--k-valid",
        type=int,
        default=100,
        help="Number of valid traces to sample. Default: 100"
    )

    parser.add_argument(
        "--k-invalid",
        type=int,
        default=100,
        help="Number of invalid traces to generate. Default: 100"
    )

    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip random baseline evaluation"
    )

    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable trace caching"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    run_folder = create_run_folder()

    config = {
        "method": args.method,
        "llm_model": args.llm,
        "validator_model": args.validator_llm,
        "k_valid": args.k_valid,
        "k_invalid": args.k_invalid,
        "perturbations": ["random_swap_events"],
        "use_cache": not args.no_cache,
        "run_baseline": not args.no_baseline,
        # Legacy field for backward compatibility
        "model": args.validator_llm
    }

    problems = [
        "benchmark/hiring/hiring_log_high.xes",
        "benchmark/hospital/hospital_log_high.xes",
        "benchmark/lending/lending_log_high.xes",
        "benchmark/renting/renting_log_high.xes"
    ]

    precomputed_models = {
        #"hiring_log_high": "runs/run_hiring_log_high_20251116_210442/textual_model.txt",
    }

    # Save benchmark config
    (run_folder / "config.json").write_text(json.dumps(config, indent=2))

    # Initialize complete results structure with empty metrics BEFORE any computation
    print(f"\n{'='*80}")
    print(f"INITIALIZING BENCHMARK")
    print(f"{'='*80}")
    print(f"Run folder: {run_folder}")
    print(f"Method: {config['method']}")
    print(f"LLM (textual model): {config['llm_model']}")
    print(f"LLM (validator): {config['validator_model']}")
    print(f"Problems: {[Path(p).stem for p in problems]}")

    results = initialize_results_structure(problems, config, precomputed_models, run_folder)
    save_results(run_folder, results)
    print(f"Initial experiment structure saved to {run_folder / 'results.json'}")

    # Update meta status
    results["_meta"]["status"] = "running"
    save_results(run_folder, results)

    # Process each problem
    for xes_path in problems:
        problem_name = Path(xes_path).stem
        precomputed_path = precomputed_models.get(problem_name)

        print(f"\n{'='*80}")
        print(f"PROCESSING: {problem_name}")
        print(f"{'='*80}")

        # Update status to in_progress
        results[problem_name]["status"] = "in_progress"
        save_results(run_folder, results)

        # Step 1: Generate/load traces
        cache_key = results[problem_name]["experiment_info"]["trace_generation"]["cache_key"]
        valid_traces, invalid_traces = load_traces_from_cache(cache_key)

        if valid_traces is None or invalid_traces is None:
            print(f"[{problem_name}] Generating traces...")
            valid_traces = sample_valid_traces(xes_path, k=config["k_valid"])
            invalid_traces = generate_invalid_traces(valid_traces, k=config["k_invalid"])
            if config["use_cache"]:
                save_traces_to_cache(cache_key, valid_traces, invalid_traces)
            results[problem_name]["experiment_info"]["trace_generation"]["traces_generated_at"] = datetime.now().isoformat()
        else:
            print(f"[{problem_name}] Loaded {len(valid_traces)} valid + {len(invalid_traces)} invalid traces from cache")
            results[problem_name]["experiment_info"]["trace_generation"]["traces_generated_at"] = "loaded_from_cache"

        save_results(run_folder, results)

        # Step 2: Generate/load AS-IS model (and IDEAL model for no-tools method)
        print(f"[{problem_name}] Generating textual model using '{config['method']}' method...")

        if config["method"] == "no-tools":
            as_is_model, ideal_model, mining_config_info = get_as_is_model_no_tools(
                xes_path,
                precomputed_model_path=precomputed_path,
                llm_model=config["llm_model"]
            )
            # Save both models to benchmark run folder
            textual_model_path = run_folder / f"{problem_name}_textual_model.txt"
            textual_model_path.write_text(as_is_model)

            ideal_model_path = None
            if ideal_model:
                ideal_model_path = run_folder / f"{problem_name}_ideal_textual_model.txt"
                ideal_model_path.write_text(ideal_model)

            # Update paths in results
            results[problem_name]["textual_model_path"] = str(textual_model_path)
            results[problem_name]["ideal_textual_model_path"] = str(ideal_model_path) if ideal_model_path else None
        else:
            as_is_model, mining_config_info = get_as_is_model(
                xes_path,
                precomputed_model_path=precomputed_path,
                llm_model=config["llm_model"]
            )
            textual_model_path = run_folder / f"{problem_name}_textual_model.txt"
            textual_model_path.write_text(as_is_model)
            results[problem_name]["textual_model_path"] = str(textual_model_path)

        # Update mining configuration with actual results
        results[problem_name]["experiment_info"]["mining_configuration"] = mining_config_info
        save_results(run_folder, results)
        print(f"[{problem_name}] Textual model saved to {textual_model_path}")

        # Step 3: Create validator and save system prompt
        validator, system_prompt = create_validator(as_is_model, model_name=config["validator_model"])
        (run_folder / f"{problem_name}_system_prompt.txt").write_text(system_prompt)
        save_results(run_folder, results)

        # Step 4: Evaluate traces with LLM
        print(f"[{problem_name}] Evaluating traces with LLM validator...")

        # Create folder for misclassified traces
        misclassified_folder = run_folder / "misclassified" / problem_name
        misclassified_folder.mkdir(parents=True, exist_ok=True)

        valid_preds, invalid_preds, total_compute_time, total_tokens, misclassified = evaluate_traces(
            valid_traces, invalid_traces, validator, misclassified_folder=misclassified_folder
        )
        llm_metrics = calculate_metrics(valid_preds, invalid_preds, total_compute_time, total_tokens)

        # Update LLM metrics and misclassified traces
        results[problem_name]["llm"] = llm_metrics
        results[problem_name]["misclassified_traces"] = {
            "folder": str(misclassified_folder),
            "false_negatives": misclassified["false_negatives"],
            "false_positives": misclassified["false_positives"]
        }
        save_results(run_folder, results)
        print(f"[{problem_name}] LLM evaluation complete - F1={llm_metrics['f1']:.3f}")
        print(f"[{problem_name}] Misclassified: {len(misclassified['false_negatives'])} FN, {len(misclassified['false_positives'])} FP")

        # Step 5: Run baseline if configured
        if config["run_baseline"]:
            print(f"[{problem_name}] Running random baseline...")
            baseline_validator = create_random_baseline_validator()
            baseline_valid_preds, baseline_invalid_preds, baseline_time, baseline_tokens, _ = evaluate_traces(valid_traces, invalid_traces, baseline_validator)
            baseline_metrics = calculate_metrics(baseline_valid_preds, baseline_invalid_preds, baseline_time, baseline_tokens)
            results[problem_name]["random_baseline"] = baseline_metrics
            save_results(run_folder, results)
            print(f"[{problem_name}] Baseline complete - F1={baseline_metrics['f1']:.3f}")

        # Mark problem as completed
        results[problem_name]["status"] = "completed"
        save_results(run_folder, results)
        print(f"[{problem_name}] COMPLETED")

    # Update meta with completion info
    results["_meta"]["benchmark_end_time"] = datetime.now().isoformat()
    results["_meta"]["status"] = "completed"
    save_results(run_folder, results)

    # Print summary
    print(f"\n{'='*80}")
    print(f"BENCHMARK RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"Method: {config['method']}")
    print(f"Textual Model LLM: {config['llm_model']}")
    print(f"Validator LLM: {config['validator_model']}")
    for problem_name in problems:
        pname = Path(problem_name).stem
        problem_results = results[pname]
        print(f"\n{pname}:")
        print(f"  LLM ({config['validator_model']}):")
        print(f"    F1={problem_results['llm']['f1']:.3f} | P={problem_results['llm']['precision']:.3f} | R={problem_results['llm']['recall']:.3f}")
        print(f"    Tokens={problem_results['llm']['total_tokens']} | Time={problem_results['llm']['total_compute_time_seconds']:.1f}s")
        if problem_results.get("random_baseline"):
            print(f"  Random Baseline:")
            print(f"    F1={problem_results['random_baseline']['f1']:.3f} | P={problem_results['random_baseline']['precision']:.3f} | R={problem_results['random_baseline']['recall']:.3f}")
    print(f"\n{'='*80}")
    print(f"Results saved to {run_folder}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
