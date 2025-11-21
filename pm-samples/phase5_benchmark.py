#!/usr/bin/env python3
"""Phase 5 Testing Benchmark - Minimal Implementation"""

import os
import json
import random
import time
import pickle
import hashlib
import pm4py
from pathlib import Path
from datetime import datetime
from copy import deepcopy
from langchain_openai import ChatOpenAI
from enhanced_agentic_miner import AgenticProcessMiner, MiningConfiguration, RepresentationType
from benchmark_template import random_swap_events, skip_mandatory_activity, df_to_key_value_rows

def get_as_is_model(xes_path, precomputed_model_path=None):
    if precomputed_model_path:
        return Path(precomputed_model_path).read_text()

    config = MiningConfiguration(
        representations=[RepresentationType.PETRI_NET, RepresentationType.DFG, RepresentationType.ABSTRACT_VARIANTS],
        model_name="anthropic:claude-sonnet-4-5",
        prompt_file="ONLY_GENERATE_TEXTUAL_MODEL.txt",
        use_cache=True
    )
    miner = AgenticProcessMiner(config)
    miner.log_file = xes_path
    miner.xes_path = xes_path
    miner.dataset_name = Path(xes_path).stem
    #breakpoint()
    miner.discover_representations()
    miner.run_agent()
    return (Path(miner.run_folder) / "textual_model.txt").read_text()

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
    #breakpoint()
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

def evaluate_traces(valid_traces, invalid_traces, validator):
    valid_predictions = []
    invalid_predictions = []
    total_compute_time = 0
    total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for i, trace in enumerate(valid_traces):
        pred, compute_time, tokens_used = validator(trace)
        is_correct = pred == "VALID"
        valid_predictions.append(is_correct)

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

    return valid_predictions, invalid_predictions, total_compute_time, total_tokens

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

def main():
    run_folder = create_run_folder()

    config = {
        "model": "gpt-5-nano",
        "k_valid": 100,
        "k_invalid": 100,
        "perturbations": ["random_swap_events"],
        "use_cache": True,
        "run_baseline": True
    }

    problems = [
        "benchmark/hiring/hiring_log_high.xes",
        "benchmark/hospital/hospital_log_high.xes",
        "benchmark/lending/lending_log_high.xes",
        "benchmark/renting/renting_log_high.xes"
    ]

    precomputed_models = {
        "hiring_log_high": "runs/run_hiring_log_high_20251116_210442/textual_model.txt",
    }

    (run_folder / "config.json").write_text(json.dumps(config, indent=2))

    results = {}

    for xes_path in problems:
        problem_name = Path(xes_path).stem
        precomputed_path = precomputed_models.get(problem_name)

        cache_key = get_trace_cache_key(xes_path, config["k_valid"], config["k_invalid"], config["perturbations"])
        valid_traces, invalid_traces = load_traces_from_cache(cache_key)

        if valid_traces is None or invalid_traces is None:
            print(f"\n[{problem_name}] Generating traces...")
            valid_traces = sample_valid_traces(xes_path, k=config["k_valid"])
            invalid_traces = generate_invalid_traces(valid_traces, k=config["k_invalid"])
            if config["use_cache"]:
                save_traces_to_cache(cache_key, valid_traces, invalid_traces)
        else:
            print(f"\n[{problem_name}] Loaded {len(valid_traces)} valid + {len(invalid_traces)} invalid traces from cache")

        as_is_model = get_as_is_model(xes_path, precomputed_model_path=precomputed_path)
        (run_folder / f"{problem_name}_textual_model.txt").write_text(as_is_model)

        validator, system_prompt = create_validator(as_is_model, model_name=config["model"])
        (run_folder / f"{problem_name}_system_prompt.txt").write_text(system_prompt)

        valid_preds, invalid_preds, total_compute_time, total_tokens = evaluate_traces(valid_traces, invalid_traces, validator)
        llm_metrics = calculate_metrics(valid_preds, invalid_preds, total_compute_time, total_tokens)

        problem_results = {"llm": llm_metrics}

        if config["run_baseline"]:
            baseline_validator = create_random_baseline_validator()
            baseline_valid_preds, baseline_invalid_preds, baseline_time, baseline_tokens = evaluate_traces(valid_traces, invalid_traces, baseline_validator)
            baseline_metrics = calculate_metrics(baseline_valid_preds, baseline_invalid_preds, baseline_time, baseline_tokens)
            problem_results["random_baseline"] = baseline_metrics

        results[problem_name] = problem_results

        (run_folder / "results.json").write_text(json.dumps(results, indent=2))
        print(f"\n[{problem_name}] Results saved incrementally to {run_folder / 'results.json'}")

    (run_folder / "results.json").write_text(json.dumps(results, indent=2))

    print(f"\n{'='*80}")
    print(f"BENCHMARK RESULTS SUMMARY")
    print(f"{'='*80}")
    for problem_name, problem_results in results.items():
        print(f"\n{problem_name}:")
        print(f"  LLM ({config['model']}):")
        print(f"    F1={problem_results['llm']['f1']:.3f} | P={problem_results['llm']['precision']:.3f} | R={problem_results['llm']['recall']:.3f}")
        print(f"    Tokens={problem_results['llm']['total_tokens']} | Time={problem_results['llm']['total_compute_time_seconds']:.1f}s")
        if "random_baseline" in problem_results:
            print(f"  Random Baseline:")
            print(f"    F1={problem_results['random_baseline']['f1']:.3f} | P={problem_results['random_baseline']['precision']:.3f} | R={problem_results['random_baseline']['recall']:.3f}")
    print(f"\n{'='*80}")
    print(f"Results saved to {run_folder}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
