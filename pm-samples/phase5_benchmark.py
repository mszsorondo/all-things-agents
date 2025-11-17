#!/usr/bin/env python3
"""Phase 5 Testing Benchmark - Minimal Implementation"""

import os
import json
import random
import time
import pm4py
from pathlib import Path
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

def create_validator(as_is_model):
    #breakpoint()
    llm = ChatOpenAI(model="gpt-5-nano", api_key=os.getenv("OPENAI_API_KEY"))
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
        response = llm.invoke(messages)
        return response.content.strip()

    return validate

def evaluate_traces(valid_traces, invalid_traces, validator):
    valid_predictions = []
    invalid_predictions = []

    for i, trace in enumerate(valid_traces):
        pred = validator(trace)
        is_correct = pred == "VALID"
        valid_predictions.append(is_correct)

        TP = sum(valid_predictions)
        FN = len(valid_predictions) - TP
        TN = sum(invalid_predictions)
        FP = len(invalid_predictions) - TN

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"Valid trace {i+1}/{len(valid_traces)}: {pred} ({'✓' if is_correct else '✗'}) | P={precision:.3f} R={recall:.3f} F1={f1:.3f} | TP={TP} FP={FP} TN={TN} FN={FN}")
        time.sleep(1)

    for i, trace in enumerate(invalid_traces):
        pred = validator(trace)
        is_correct = pred == "INVALID"
        invalid_predictions.append(is_correct)

        TP = sum(valid_predictions)
        FN = len(valid_predictions) - TP
        TN = sum(invalid_predictions)
        FP = len(invalid_predictions) - TN

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"Invalid trace {i+1}/{len(invalid_traces)}: {pred} ({'✓' if is_correct else '✗'}) | P={precision:.3f} R={recall:.3f} F1={f1:.3f} | TP={TP} FP={FP} TN={TN} FN={FN}")
        time.sleep(1)

    return valid_predictions, invalid_predictions

def calculate_metrics(valid_preds, invalid_preds):
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
        "FN": FN
    }

def main():
    problems = [
        "benchmark/hiring/hiring_log_high.xes",
        "benchmark/hospital/hospital_log_high.xes",
        "benchmark/lending/lending_log_high.xes",
        "benchmark/renting/renting_log_high.xes"
    ]

    precomputed_models = {
        "hiring_log_high": "runs/run_hiring_log_high_20251116_210442/textual_model.txt",
    }

    results = {}

    for xes_path in problems:
        problem_name = Path(xes_path).stem
        precomputed_path = precomputed_models.get(problem_name)

        as_is_model = get_as_is_model(xes_path, precomputed_model_path=precomputed_path)
        valid_traces = sample_valid_traces(xes_path, k=100)
        invalid_traces = generate_invalid_traces(valid_traces, k=100)
        #breakpoint()
        validator = create_validator(as_is_model)
        valid_preds, invalid_preds = evaluate_traces(valid_traces, invalid_traces, validator)
        metrics = calculate_metrics(valid_preds, invalid_preds)

        results[problem_name] = metrics

    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
