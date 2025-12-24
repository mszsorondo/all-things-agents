#!/usr/bin/env python3
"""
Refine benchmark textual models based on misclassified traces.

Usage:
    python refine_benchmark.py benchmark_runs/run_20251222_004259
"""

import os
import json
import pickle
import argparse
import time
from pathlib import Path
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from benchmark_template import df_to_key_value_rows
from phase5_benchmark import (
    create_validator,
    evaluate_traces,
    calculate_metrics,
    load_traces_from_cache,
    save_results
)


def load_textual_model(run_folder: Path, problem_name: str) -> str:
    """Load the AS-IS textual model for a problem."""
    model_path = run_folder / f"{problem_name}_textual_model.txt"
    if not model_path.exists():
        raise FileNotFoundError(f"Textual model not found: {model_path}")
    return model_path.read_text()


def load_misclassified_traces(run_folder: Path, problem_name: str, problem_data: dict) -> list:
    """Load all misclassified traces (FN and FP) with their ground truth."""
    traces = []
    misc_data = problem_data.get("misclassified_traces", {})

    for fn in misc_data.get("false_negatives", []):
        file_path = fn["file"]
        if not Path(file_path).exists():
            print(f"  Warning: FN trace file not found: {file_path}")
            continue
        with open(file_path, "rb") as f:
            df = pickle.load(f)
        traces.append({
            "trace_text": df_to_key_value_rows(df),
            "expected": "VALID",
            "predicted": "INVALID",
            "activities": fn.get("activities", []),
            "type": "FN"
        })

    for fp in misc_data.get("false_positives", []):
        file_path = fp["file"]
        if not Path(file_path).exists():
            print(f"  Warning: FP trace file not found: {file_path}")
            continue
        with open(file_path, "rb") as f:
            df = pickle.load(f)
        traces.append({
            "trace_text": df_to_key_value_rows(df),
            "expected": "INVALID",
            "predicted": "VALID",
            "activities": fp.get("activities", []),
            "type": "FP"
        })

    return traces


def load_representations(problem_data: dict) -> str:
    """Load process representations from the original mining run."""
    mining_config = problem_data.get("experiment_info", {}).get("mining_configuration", {})
    run_folder = mining_config.get("run_folder")

    if run_folder:
        user_prompt_path = Path(run_folder) / "user_prompt.txt"
        if user_prompt_path.exists():
            return user_prompt_path.read_text()

    return None


def create_llm(model_spec: str):
    """Create LLM client from model specification (provider:model)."""
    if ":" in model_spec:
        provider, model_id = model_spec.split(":", 1)
    else:
        provider, model_id = "openai", model_spec

    if "anthropic" in provider.lower():
        return ChatAnthropic(
            model=model_id,
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
    else:
        return ChatOpenAI(
            model=model_id,
            api_key=os.getenv("OPENAI_API_KEY")
        )


def build_refinement_prompt(as_is_model: str, misclassified: list, representations: str) -> str:
    """Build the prompt for LLM to refine the model."""
    # Count FN and FP
    fn_count = sum(1 for t in misclassified if t["type"] == "FN")
    fp_count = sum(1 for t in misclassified if t["type"] == "FP")

    # Format misclassified traces
    traces_text = ""
    for i, t in enumerate(misclassified):
        traces_text += f"\n--- Trace {i+1} ({t['type']}: Expected {t['expected']}, Got {t['predicted']}) ---\n"
        traces_text += f"Activities: {' -> '.join(t['activities'])}\n"

    prompt = f"""You are an expert process analyst helping to improve a textual process model.

## Context: What is an AS-IS Textual Model?

An AS-IS textual model is a natural language description of a business process as it is ACTUALLY observed in practice. It is generated from process mining outputs (Petri nets, DFGs, process variants) extracted from event logs. The model describes:
- The sequence of activities that occur in the process
- Valid orderings and paths through the process
- Optional and mandatory activities
- Parallel activities and decision points
- All observed variations, including edge cases and exceptions

The purpose of this model is to serve as a reference for an LLM-based trace validator that determines whether a given execution trace (sequence of activities) conforms to the observed business process.

## Context: Trace Classification Task

A **trace** is a sequence of activities representing one execution instance of the business process (e.g., one job application, one loan request, one patient visit).

- **VALID trace**: A trace that was actually observed in the event log. It represents a real execution path that occurred in practice. The AS-IS model should recognize and accept these traces.

- **INVALID trace**: A trace that was artificially created by perturbing a valid trace (e.g., swapping the order of activities). These traces contain anomalies that violate the observed process behavior. The AS-IS model should reject these traces.

The classification task tests whether the textual model accurately captures the process behavior: it should accept all valid traces (high recall) while rejecting invalid/anomalous traces (high precision).

## Current AS-IS Model (that made classification mistakes):
{as_is_model}

## Process Representations (ground truth from process mining):
{representations if representations else "[Not available]"}

## Misclassified Traces ({len(misclassified)} total: {fn_count} False Negatives, {fp_count} False Positives):
{traces_text}

## Understanding the Errors:
- **False Negatives (FN)**: {fn_count} traces were VALID (actually observed) but the model caused them to be classified as INVALID.
  This probably means the model is TOO RESTRICTIVE - it doesn't describe some valid process paths that actually occur.

- **False Positives (FP)**: {fp_count} traces were INVALID (artificially perturbed) but the model caused them to be classified as VALID.
  This probably means the model is TOO PERMISSIVE - it doesn't specify constraints that would reject anomalous sequences.

## Your Task:
1. Analyze WHY each trace was misclassified based on the current model's description
2. Identify what's missing, incorrect, or ambiguous in the AS-IS model
3. Write a REFINED AS-IS textual model that would correctly classify ALL these traces

IMPORTANT GUIDELINES:
- For FN traces: The model failed to describe valid paths. Add or clarify the valid orderings/paths that were missed.
- For FP traces: The model failed to constrain invalid paths. Add explicit rules about required orderings or dependencies.
- Be EXPLICIT about which activity orderings are valid vs invalid
- Clearly state which activities are optional, mandatory, or conditional
- Reference the Process Representations (Petri net, DFG, variants) as the ground truth for what actually occurs

Output ONLY the refined AS-IS model (no explanations or preamble). Start directly with "# AS-IS Process Model"."""

    return prompt


def refine_model(llm, prompt: str) -> str:
    """Call LLM to generate refined model."""
    messages = [("human", prompt)]
    response = llm.invoke(messages)
    return response.content.strip()


def save_refined_model(refined_folder: Path, problem_name: str, refined_model: str):
    """Save the refined textual model."""
    model_path = refined_folder / f"{problem_name}_textual_model.txt"
    model_path.write_text(refined_model)
    return model_path


def run_benchmark_on_refined(refined_folder: Path, config: dict, original_results: dict, run_folder: Path):
    """Run trace classification benchmark on refined models."""
    refined_results = {
        "_meta": {
            "refinement_of": str(run_folder),
            "refinement_time": datetime.now().isoformat(),
            "validator_model": config.get("validator_model", config.get("model", "gpt-4o-mini"))
        }
    }

    for problem_name, problem_data in original_results.items():
        if problem_name.startswith("_"):
            continue

        # Check if refined model exists
        refined_model_path = refined_folder / f"{problem_name}_textual_model.txt"
        if not refined_model_path.exists():
            print(f"  Skipping {problem_name}: no refined model found")
            continue

        print(f"\n  Evaluating refined model for {problem_name}...")
        refined_model = refined_model_path.read_text()

        # Create validator with refined model
        validator_model = config.get("validator_model", config.get("model", "gpt-4o-mini"))
        validator, system_prompt = create_validator(refined_model, model_name=validator_model)

        # Save system prompt
        (refined_folder / f"{problem_name}_system_prompt.txt").write_text(system_prompt)

        # Load traces from cache using original cache key
        cache_key = problem_data.get("experiment_info", {}).get("trace_generation", {}).get("cache_key")
        if not cache_key:
            print(f"  Skipping {problem_name}: no trace cache key found")
            continue

        valid_traces, invalid_traces = load_traces_from_cache(cache_key)
        if valid_traces is None or invalid_traces is None:
            print(f"  Skipping {problem_name}: traces not found in cache")
            continue

        print(f"  Loaded {len(valid_traces)} valid + {len(invalid_traces)} invalid traces from cache")

        # Create misclassified folder for refined run
        misclassified_folder = refined_folder / "misclassified" / problem_name
        misclassified_folder.mkdir(parents=True, exist_ok=True)

        # Evaluate
        valid_preds, invalid_preds, total_compute_time, total_tokens, misclassified = evaluate_traces(
            valid_traces, invalid_traces, validator, misclassified_folder=misclassified_folder
        )
        metrics = calculate_metrics(valid_preds, invalid_preds, total_compute_time, total_tokens)

        # Store results
        refined_results[problem_name] = {
            "metrics": metrics,
            "misclassified_traces": {
                "folder": str(misclassified_folder),
                "false_negatives": misclassified["false_negatives"],
                "false_positives": misclassified["false_positives"]
            },
            "original_metrics": problem_data.get("llm", {}),
            "improvement": {
                "f1_delta": metrics["f1"] - problem_data.get("llm", {}).get("f1", 0),
                "fn_delta": metrics["FN"] - problem_data.get("llm", {}).get("FN", 0),
                "fp_delta": metrics["FP"] - problem_data.get("llm", {}).get("FP", 0)
            }
        }

        print(f"  {problem_name}: F1={metrics['f1']:.3f} (was {problem_data.get('llm', {}).get('f1', 0):.3f})")

    # Save results
    with open(refined_folder / "results.json", "w") as f:
        json.dump(refined_results, f, indent=2)

    return refined_results


def diagnose_misclassifications(run_folder: Path, problem_name: str, problem_data: dict,
                                  misclassified: list, diagnose_model: str) -> dict:
    """
    Re-evaluate misclassified traces using a more capable model to determine if errors
    are due to the textual model or the classifier LLM's capability.

    Args:
        run_folder: Path to benchmark run folder
        problem_name: Name of the problem
        problem_data: Problem data from results.json
        misclassified: List of misclassified traces
        diagnose_model: Model to use for diagnosis (e.g., "anthropic:claude-opus-4-5-20251101")

    Returns dict with diagnosis results.
    """
    # Load system prompt (same as original)
    system_prompt_path = run_folder / f"{problem_name}_system_prompt.txt"
    if not system_prompt_path.exists():
        print(f"  Warning: System prompt not found: {system_prompt_path}")
        return None

    system_prompt = system_prompt_path.read_text()

    # Create LLM for diagnosis
    llm = create_llm(diagnose_model)

    results = {
        "diagnose_model": diagnose_model,
        "model_issues": [],      # Diagnose model also misclassified -> textual model is the problem
        "classifier_issues": [], # Diagnose model got it right -> original classifier was the problem
        "total_verified": 0
    }

    for i, trace in enumerate(misclassified):
        print(f"    Trace {i+1}/{len(misclassified)} ({trace['type']})...", end=" ")

        messages = [
            ("system", system_prompt),
            ("human", trace["trace_text"])
        ]

        start_time = time.time()
        response = llm.invoke(messages)
        elapsed = time.time() - start_time

        diagnose_prediction = response.content.strip()
        expected = trace["expected"]
        original_prediction = trace["predicted"]

        # Check if diagnose model got it right
        diagnose_correct = diagnose_prediction == expected

        trace_info = {
            "index": i,
            "type": trace["type"],
            "expected": expected,
            "original_prediction": original_prediction,
            "diagnose_prediction": diagnose_prediction,
            "activities": trace["activities"]
        }

        if diagnose_correct:
            # Diagnose model got it right, original classifier was wrong -> classifier issue
            results["classifier_issues"].append(trace_info)
            print(f"{diagnose_prediction} ✓ (classifier issue) [{elapsed:.1f}s]")
        else:
            # Diagnose model also got it wrong -> model issue
            results["model_issues"].append(trace_info)
            print(f"{diagnose_prediction} ✗ (model issue) [{elapsed:.1f}s]")

        results["total_verified"] += 1
        time.sleep(0.5)  # Rate limiting

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Refine benchmark textual models based on misclassified traces",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Standard refinement
    python refine_benchmark.py benchmark_runs/run_20251222_004259

    # Diagnose first to identify model vs classifier issues (uses Claude Opus by default)
    python refine_benchmark.py benchmark_runs/run_20251222_004259 --diagnose

    # Diagnose with a different model
    python refine_benchmark.py benchmark_runs/run_20251222_004259 --diagnose --diagnose-model openai:gpt-4o

    # Only diagnose, don't refine
    python refine_benchmark.py benchmark_runs/run_20251222_004259 --diagnose --no-refine
"""
    )
    parser.add_argument(
        "benchmark_run",
        type=str,
        help="Path to benchmark run folder (e.g., benchmark_runs/run_20251222_004259)"
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Re-evaluate misclassified traces with a more capable model to identify model vs classifier issues"
    )
    parser.add_argument(
        "--diagnose-model",
        type=str,
        default="anthropic:claude-opus-4-5-20251101",
        help="Model to use for diagnosis (default: anthropic:claude-opus-4-5-20251101)"
    )
    parser.add_argument(
        "--no-refine",
        action="store_true",
        help="Skip refinement step (useful with --diagnose to only diagnose)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_folder = Path(args.benchmark_run)

    if not run_folder.exists():
        print(f"Error: Benchmark run folder not found: {run_folder}")
        return

    # Load configs
    config_path = run_folder / "config.json"
    results_path = run_folder / "results.json"

    if not config_path.exists() or not results_path.exists():
        print(f"Error: config.json or results.json not found in {run_folder}")
        return

    config = json.loads(config_path.read_text())
    results = json.loads(results_path.read_text())

    print(f"Benchmark run: {run_folder}")
    if args.diagnose:
        print(f"Diagnose model: {args.diagnose_model}")
    if not args.no_refine:
        print(f"Refinement model: {config.get('llm_model', 'openai:gpt-5-mini')}")

    # Create refined folder
    refined_folder = run_folder / "refined"
    refined_folder.mkdir(exist_ok=True)

    # Create LLM for refinement (only if needed)
    llm = None
    if not args.no_refine:
        llm = create_llm(config.get("llm_model", "openai:gpt-5-mini"))

    # Store diagnosis results for summary
    all_diagnosis_results = {}

    # Process each problem
    for problem_name, problem_data in results.items():
        if problem_name.startswith("_"):
            continue

        print(f"\n{'='*60}")
        print(f"Processing: {problem_name}")
        print(f"{'='*60}")

        # Check if there are misclassified traces
        misc_data = problem_data.get("misclassified_traces", {})
        fn_count = len(misc_data.get("false_negatives", []))
        fp_count = len(misc_data.get("false_positives", []))

        if fn_count == 0 and fp_count == 0:
            print(f"  No misclassified traces - skipping")
            continue

        print(f"  Misclassified: {fn_count} FN, {fp_count} FP")

        # Load data
        try:
            as_is_model = load_textual_model(run_folder, problem_name)
        except FileNotFoundError as e:
            print(f"  Error: {e}")
            continue

        misclassified = load_misclassified_traces(run_folder, problem_name, problem_data)
        representations = load_representations(problem_data)

        print(f"  Loaded {len(misclassified)} misclassified traces")
        print(f"  Representations: {'Found' if representations else 'Not found'}")

        # Step 1: Diagnose (optional)
        if args.diagnose:
            print(f"\n  --- Diagnosis with {args.diagnose_model} ---")
            diagnosis = diagnose_misclassifications(
                run_folder, problem_name, problem_data, misclassified, args.diagnose_model
            )
            if diagnosis:
                all_diagnosis_results[problem_name] = diagnosis
                model_issues = len(diagnosis["model_issues"])
                classifier_issues = len(diagnosis["classifier_issues"])
                print(f"  Diagnosis: {model_issues} textual model issues, {classifier_issues} classifier issues")

        # Step 2: Refine (unless --no-refine)
        if not args.no_refine:
            print(f"\n  --- Refinement ---")
            # Build refinement prompt
            refinement_prompt = build_refinement_prompt(as_is_model, misclassified, representations)

            # Call LLM for refinement
            print(f"  Calling LLM for refinement...")
            start_time = time.time()
            refined_model = refine_model(llm, refinement_prompt)
            elapsed = time.time() - start_time
            print(f"  Refinement completed in {elapsed:.1f}s")

            # Save refined model
            model_path = save_refined_model(refined_folder, problem_name, refined_model)
            print(f"  Saved refined model to {model_path}")

    # Print diagnosis summary if applicable
    if args.diagnose and all_diagnosis_results:
        print(f"\n{'='*60}")
        print(f"DIAGNOSIS SUMMARY ({args.diagnose_model})")
        print(f"{'='*60}")

        total_model_issues = 0
        total_classifier_issues = 0

        for problem_name, diagnosis in all_diagnosis_results.items():
            model_issues = len(diagnosis["model_issues"])
            classifier_issues = len(diagnosis["classifier_issues"])
            total = diagnosis["total_verified"]
            total_model_issues += model_issues
            total_classifier_issues += classifier_issues

            print(f"\n{problem_name}: ({total} traces)")
            print(f"  Textual Model issues:      {model_issues:3d} ({100*model_issues/total:.0f}%) - textual model needs improvement")
            print(f"  Classifier issues: {classifier_issues:3d} ({100*classifier_issues/total:.0f}%) - original validator LLM was insufficient")

        if len(all_diagnosis_results) > 1:
            total = total_model_issues + total_classifier_issues
            print(f"\nOVERALL:")
            print(f"  Textual Model issues:      {total_model_issues:3d} ({100*total_model_issues/total:.0f}%)")
            print(f"  Classifier issues: {total_classifier_issues:3d} ({100*total_classifier_issues/total:.0f}%)")

        # Save diagnosis results
        diagnosis_path = refined_folder / "diagnosis_results.json"
        with open(diagnosis_path, "w") as f:
            json.dump(all_diagnosis_results, f, indent=2)
        print(f"\nDiagnosis saved to {diagnosis_path}")

    # Re-run benchmark on refined models (unless --no-refine)
    if not args.no_refine:
        print(f"\n{'='*60}")
        print(f"Re-running benchmark on refined models")
        print(f"{'='*60}")

        refined_results = run_benchmark_on_refined(refined_folder, config, results, run_folder)

        # Print refinement summary
        print(f"\n{'='*60}")
        print(f"REFINEMENT SUMMARY")
        print(f"{'='*60}")

        for problem_name, data in refined_results.items():
            if problem_name.startswith("_"):
                continue

            metrics = data.get("metrics", {})
            original = data.get("original_metrics", {})
            improvement = data.get("improvement", {})

            print(f"\n{problem_name}:")
            print(f"  Original: F1={original.get('f1', 0):.3f} | FN={original.get('FN', 0)} | FP={original.get('FP', 0)}")
            print(f"  Refined:  F1={metrics.get('f1', 0):.3f} | FN={metrics.get('FN', 0)} | FP={metrics.get('FP', 0)}")
            print(f"  Delta:    F1={improvement.get('f1_delta', 0):+.3f} | FN={improvement.get('fn_delta', 0):+d} | FP={improvement.get('fp_delta', 0):+d}")

        print(f"\nResults saved to {refined_folder / 'results.json'}")


if __name__ == "__main__":
    main()
