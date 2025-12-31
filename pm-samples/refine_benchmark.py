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
    save_results,
    load_traces_from_run_folder,
    save_traces_to_run_folder,
    sample_valid_traces,
    generate_invalid_traces
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


def load_representations(problem_data: dict, run_folder: Path = None) -> str:
    """
    Load process representations from the original mining run.
    Walks up the parent chain for refined folders to find the original representations.
    """
    # Try direct path first
    mining_config = problem_data.get("experiment_info", {}).get("mining_configuration", {})
    mining_run_folder = mining_config.get("run_folder")

    if mining_run_folder:
        user_prompt_path = Path(mining_run_folder) / "user_prompt.txt"
        if user_prompt_path.exists():
            return user_prompt_path.read_text()

    # For refined folders, walk up the parent chain to find original representations
    if run_folder:
        current = run_folder
        max_depth = 10  # Prevent infinite loops
        for _ in range(max_depth):
            # Check if this folder has a config.json with parent_run
            config_path = current / "config.json"
            if config_path.exists():
                try:
                    config = json.loads(config_path.read_text())
                    parent_run = config.get("parent_run")
                    if parent_run:
                        parent_path = Path(parent_run)
                        # Check parent's results.json for experiment_info
                        parent_results_path = parent_path / "results.json"
                        if parent_results_path.exists():
                            parent_results = json.loads(parent_results_path.read_text())
                            # Find the problem in parent results
                            for key, val in parent_results.items():
                                if isinstance(val, dict) and "experiment_info" in val:
                                    parent_mining = val.get("experiment_info", {}).get("mining_configuration", {})
                                    parent_mining_folder = parent_mining.get("run_folder")
                                    if parent_mining_folder:
                                        user_prompt_path = Path(parent_mining_folder) / "user_prompt.txt"
                                        if user_prompt_path.exists():
                                            return user_prompt_path.read_text()
                        current = parent_path
                        continue
                except Exception:
                    pass
            break

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
    elif "grok" in provider.lower() or "xai" in provider.lower():
        return ChatOpenAI(
            model=model_id,
            api_key=os.getenv("GROK_API_KEY"),
            base_url="https://api.x.ai/v1"
        )
    elif "deepseek" in provider.lower():
        return ChatOpenAI(
            model=model_id,
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
    else:
        return ChatOpenAI(
            model=model_id,
            api_key=os.getenv("OPENAI_API_KEY")
        )


def build_refinement_prompt(as_is_model: str, misclassified: list, representations: str, incremental: bool = False) -> str:
    """
    Build the prompt for LLM to refine the model.

    Args:
        as_is_model: The current textual model
        misclassified: List of misclassified traces
        representations: Process mining representations (DFG, Petri net, etc.)
        incremental: If True, use incremental refinement mode (minimal targeted edits)
    """
    # Count FN and FP
    fn_count = sum(1 for t in misclassified if t["type"] == "FN")
    fp_count = sum(1 for t in misclassified if t["type"] == "FP")

    # Format misclassified traces
    traces_text = ""
    for i, t in enumerate(misclassified):
        traces_text += f"\n--- Trace {i+1} ({t['type']}: Expected {t['expected']}, Got {t['predicted']}) ---\n"
        traces_text += f"Activities: {' -> '.join(t['activities'])}\n"

    # Common context sections
    context = f"""## Context: What is an AS-IS Textual Model?

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
  This probably means the model is TOO PERMISSIVE - it doesn't specify constraints that would reject anomalous sequences."""

    if incremental:
        # Incremental refinement prompt - minimal targeted edits
        prompt = f"""You are an expert process analyst helping to make MINIMAL, TARGETED improvements to a textual process model.

{context}

## Your Task: INCREMENTAL REFINEMENT

You must make the SMALLEST possible changes to fix the misclassifications. Your goal is to PRESERVE the existing model structure while making surgical edits.

STRICT RULES FOR INCREMENTAL REFINEMENT:

1. **PRESERVE STRUCTURE**: Keep all existing section headers, formatting, and overall organization exactly as they are.

2. **MINIMAL EDITS ONLY**: For each misclassified trace, identify the SPECIFIC sentence or phrase that caused the error, and modify ONLY that part.

3. **EXTEND, DON'T REWRITE**:
   - Add clarifying phrases within existing sentences
   - Insert new bullet points within existing lists
   - Append additional conditions to existing rules
   - DO NOT delete or restructure existing content unless absolutely necessary

4. **TARGETED FIXES**:
   - For each FN: Find the specific constraint that's too restrictive and relax it minimally
   - For each FP: Find the specific rule that's too permissive and tighten it minimally
   - Add parenthetical clarifications like "(including cases where X)" or "(except when Y)"

5. **PRESERVE WORKING PARTS**: If a section of the model correctly handles some traces, DO NOT modify it.

EXAMPLES OF INCREMENTAL EDITS:

**Example 1: Relaxing a constraint (for FN)**
- BEFORE: "Activity B must follow Activity A"
- AFTER: "Activity B must follow Activity A (or may occur after Activity C in alternative paths)"

**Example 2: Adding a clarification (for FN)**
- BEFORE: "The process ends with Activity Z"
- AFTER: "The process ends with Activity Z (note: Activity Y may optionally occur before Z)"

**Example 3: Tightening a rule (for FP)**
- BEFORE: "Activities are executed in sequence"
- AFTER: "Activities are executed in sequence, with the constraint that Activity X must always precede Activity Y"

**Example 4: Adding a new bullet to an existing list**
- BEFORE:
  "## Valid Activity Sequences
   - A → B → C
   - A → C → B"
- AFTER:
  "## Valid Activity Sequences
   - A → B → C
   - A → C → B
   - A → B → D → C (when condition X applies)"

**Example 5: Adding a new subsection within an existing section**
- BEFORE:
  "## Process Flow
   The main flow consists of..."
- AFTER:
  "## Process Flow
   The main flow consists of...

   ### Edge Cases
   In rare cases, the following alternative paths are valid:
   - ..."

**Example 6: Adding a completely new section (when needed)**
- Add at the end of the model:
  "## Additional Constraints
   The following ordering constraints must be respected:
   - Activity X must never directly follow Activity Y
   - ..."

Output the refined AS-IS model with your minimal changes. Start directly with "# AS-IS Process Model"."""
    else:
        # Standard refinement prompt - full rewrite allowed
        prompt = f"""You are an expert process analyst helping to improve a textual process model.

{context}

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


def load_misclassified_traces_for_eval(run_folder: Path, problem_name: str, problem_data: dict) -> list:
    """Load misclassified traces with their DataFrames for re-evaluation."""
    traces = []
    misc_data = problem_data.get("misclassified_traces", {})

    for fn in misc_data.get("false_negatives", []):
        file_path = fn["file"]
        if not Path(file_path).exists():
            continue
        with open(file_path, "rb") as f:
            df = pickle.load(f)
        traces.append({
            "trace_df": df,
            "expected": "VALID",
            "type": "FN",
            "activities": fn.get("activities", [])
        })

    for fp in misc_data.get("false_positives", []):
        file_path = fp["file"]
        if not Path(file_path).exists():
            continue
        with open(file_path, "rb") as f:
            df = pickle.load(f)
        traces.append({
            "trace_df": df,
            "expected": "INVALID",
            "type": "FP",
            "activities": fp.get("activities", [])
        })

    return traces


def evaluate_on_misclassified(refined_model: str, misclassified_traces: list, validator_model: str) -> dict:
    """
    Test refined model ONLY on the previously misclassified traces.
    Returns dict with results and improvement metrics.
    """
    validator, _ = create_validator(refined_model, model_name=validator_model)

    fixed = 0
    still_wrong = 0
    results_by_type = {"FN": {"fixed": 0, "still_wrong": 0}, "FP": {"fixed": 0, "still_wrong": 0}}

    for i, trace in enumerate(misclassified_traces):
        # Re-evaluate with refined model
        pred, compute_time, tokens = validator(trace["trace_df"])
        is_correct = (pred == trace["expected"])

        if is_correct:
            fixed += 1
            results_by_type[trace["type"]]["fixed"] += 1
        else:
            still_wrong += 1
            results_by_type[trace["type"]]["still_wrong"] += 1

        status = "✓ FIXED" if is_correct else "✗ still wrong"
        print(f"    Trace {i+1}/{len(misclassified_traces)} ({trace['type']}): {pred} {status}")
        time.sleep(0.5)

    return {
        "total": len(misclassified_traces),
        "fixed": fixed,
        "still_wrong": still_wrong,
        "fix_rate": fixed / len(misclassified_traces) if misclassified_traces else 0,
        "by_type": results_by_type
    }


def run_benchmark_on_refined(refined_folder: Path, config: dict, original_results: dict, run_folder: Path, new_sampling: bool = False):
    """Run trace classification benchmark on refined models.

    Args:
        refined_folder: Path to save refined results
        config: Benchmark configuration
        original_results: Results from parent benchmark
        run_folder: Parent run folder
        new_sampling: If True, generate new traces instead of using parent's
    """
    refined_results = {
        "_meta": {
            "refinement_of": str(run_folder),
            "refinement_time": datetime.now().isoformat(),
            "validator_model": config.get("validator_model", config.get("model", "gpt-4o-mini")),
            "new_sampling": new_sampling
        }
    }

    k_valid = config.get("k_valid", 100)
    k_invalid = config.get("k_invalid", 100)

    for problem_name, problem_data in original_results.items():
        if problem_name.startswith("_"):
            continue

        # Check if original model already had perfect F1 - copy results instead of re-evaluating
        original_llm = problem_data.get("llm", {})
        original_f1 = original_llm.get("f1", 0)
        if original_f1 == 1.0:
            print(f"\n  {problem_name}: F1=1.000 (perfect) - copying results, no evaluation needed")
            refined_results[problem_name] = {
                "status": "copied_optimal",
                "xes_path": problem_data.get("xes_path"),
                "textual_model_path": str(refined_folder / f"{problem_name}_textual_model.txt"),
                "llm": original_llm,
                "metrics": original_llm,
                "original_metrics": original_llm,
                "improvement": {"f1_delta": 0, "fn_delta": 0, "fp_delta": 0},
                "note": "Model already optimal (F1=1.0), copied from parent"
            }
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

        # Get traces - either generate new or load from parent
        valid_traces = None
        invalid_traces = None

        if new_sampling:
            # Generate new traces from XES file
            xes_path = problem_data.get("xes_path")

            # If xes_path not in current results, try to find it from parent chain
            if not xes_path:
                # Try standard benchmark path based on problem name
                # e.g., "hiring_log_high" -> "benchmark/hiring/hiring_log_high.xes"
                category = problem_name.split("_")[0]  # e.g., "hiring" from "hiring_log_high"
                standard_path = f"benchmark/{category}/{problem_name}.xes"
                if Path(standard_path).exists():
                    xes_path = standard_path
                    print(f"  Found XES at standard path: {xes_path}")

            if xes_path and Path(xes_path).exists():
                print(f"  Generating new traces from {xes_path}...")
                valid_traces = sample_valid_traces(xes_path, k=k_valid)
                invalid_traces = generate_invalid_traces(valid_traces, k=k_invalid)
                print(f"  Generated {len(valid_traces)} valid + {len(invalid_traces)} invalid traces")
            else:
                print(f"  Warning: XES path not found ({xes_path}), falling back to parent traces")

        # Fall back to loading from parent if new_sampling failed or wasn't requested
        if valid_traces is None:
            # Try to load traces from run folder first (for reproducibility)
            valid_traces, invalid_traces = load_traces_from_run_folder(run_folder, problem_name)

            if valid_traces is not None:
                print(f"  Loaded {len(valid_traces)} valid + {len(invalid_traces)} invalid traces from run folder")
            else:
                # Fall back to cache for backward compatibility
                cache_key = problem_data.get("experiment_info", {}).get("trace_generation", {}).get("cache_key")
                if cache_key:
                    valid_traces, invalid_traces = load_traces_from_cache(cache_key)
                    if valid_traces is not None:
                        print(f"  Loaded {len(valid_traces)} valid + {len(invalid_traces)} invalid traces from cache")

        if valid_traces is None or invalid_traces is None:
            print(f"  Skipping {problem_name}: traces not found in run folder or cache")
            continue

        # Save traces to refined folder for self-containment
        save_traces_to_run_folder(refined_folder, problem_name, valid_traces, invalid_traces)

        # Create misclassified folder for refined run
        misclassified_folder = refined_folder / "misclassified" / problem_name
        misclassified_folder.mkdir(parents=True, exist_ok=True)

        # Evaluate
        valid_preds, invalid_preds, total_compute_time, total_tokens, misclassified = evaluate_traces(
            valid_traces, invalid_traces, validator, misclassified_folder=misclassified_folder
        )
        metrics = calculate_metrics(valid_preds, invalid_preds, total_compute_time, total_tokens)

        # Get original metrics (handle both parent and refined folder structures)
        original_llm = problem_data.get("llm") or problem_data.get("metrics", {})
        original_f1 = original_llm.get("f1", 0)
        original_fn = original_llm.get("FN", 0)
        original_fp = original_llm.get("FP", 0)

        # Store results in a structure compatible with parent (enables recursive refinement)
        refined_results[problem_name] = {
            "status": "completed",
            "xes_path": problem_data.get("xes_path"),  # Inherit from parent
            "textual_model_path": str(refined_folder / f"{problem_name}_textual_model.txt"),
            "llm": metrics,  # Use same key as parent for compatibility
            "metrics": metrics,  # Keep for backward compatibility
            "misclassified_traces": {
                "folder": str(misclassified_folder),
                "false_negatives": misclassified["false_negatives"],
                "false_positives": misclassified["false_positives"]
            },
            "original_metrics": original_llm,
            "improvement": {
                "f1_delta": metrics["f1"] - original_f1,
                "fn_delta": metrics["FN"] - original_fn,
                "fp_delta": metrics["FP"] - original_fp
            }
        }

        print(f"  {problem_name}: F1={metrics['f1']:.3f} (was {original_f1:.3f})")

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
    parser.add_argument(
        "--refine-model",
        type=str,
        default=None,
        help="LLM for refinement (format: provider:model). Default: same as parent benchmark"
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Use incremental refinement mode: minimal targeted edits instead of full rewrite"
    )
    parser.add_argument(
        "--use-parent-traces",
        action="store_true",
        help="Reuse parent's traces instead of generating new ones for full benchmark"
    )
    parser.add_argument(
        "--skip-full-benchmark",
        action="store_true",
        help="Only run misclassified-traces test, skip full benchmark"
    )
    parser.add_argument(
        "--force-full-benchmark",
        action="store_true",
        help="Run full benchmark even if misclassified test shows no improvement"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Custom output folder name (default: auto-increment refined, refined_1, refined_2, etc.)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="REFINED_FOLDER",
        help="Resume an interrupted experiment by running Stage 2 on an existing refined folder"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_folder = Path(args.benchmark_run)

    if not run_folder.exists():
        print(f"Error: Benchmark run folder not found: {run_folder}")
        return

    # Handle --resume: skip Stage 1 and run Stage 2 on existing refined folder
    if args.resume:
        resume_folder = Path(args.resume)
        if not resume_folder.exists():
            print(f"Error: Resume folder not found: {resume_folder}")
            return

        resume_config_path = resume_folder / "config.json"
        if not resume_config_path.exists():
            print(f"Error: config.json not found in resume folder: {resume_folder}")
            return

        resume_config = json.loads(resume_config_path.read_text())

        # Load parent results for comparison
        parent_run = Path(resume_config.get("parent_run", run_folder))
        parent_results_path = parent_run / "results.json"
        if not parent_results_path.exists():
            print(f"Error: Parent results.json not found: {parent_results_path}")
            return

        parent_results = json.loads(parent_results_path.read_text())

        print(f"{'='*60}")
        print(f"RESUMING EXPERIMENT")
        print(f"{'='*60}")
        print(f"Refined folder: {resume_folder}")
        print(f"Parent run: {parent_run}")
        print(f"Validator model: {resume_config.get('validator_model', 'gpt-4o-mini')}")
        print(f"Trace source: {resume_config.get('trace_source', 'new_sampling')}")
        print()

        # Determine if we should use new sampling based on resume config
        use_new_sampling = resume_config.get("trace_source", "new_sampling") == "new_sampling"
        if args.use_parent_traces:
            use_new_sampling = False

        # Run Stage 2 (full benchmark) on the existing refined models
        print(f"--- Running Stage 2: Full Benchmark ---")
        refined_results = run_benchmark_on_refined(
            resume_folder, resume_config, parent_results, parent_run,
            new_sampling=use_new_sampling
        )

        # Print summary
        print(f"\n{'='*60}")
        print(f"RESUME COMPLETE")
        print(f"{'='*60}")

        for problem_name, data in refined_results.items():
            if problem_name.startswith("_"):
                continue
            if "llm" in data:
                metrics = data["llm"]
                original_f1 = parent_results.get(problem_name, {}).get("llm", {}).get("f1", 0)
                print(f"  {problem_name}: F1={metrics.get('f1', 0):.3f} (was {original_f1:.3f})")

        print(f"\nResults saved to {resume_folder}/results.json")
        return

    # Load configs
    config_path = run_folder / "config.json"
    results_path = run_folder / "results.json"

    if not config_path.exists() or not results_path.exists():
        print(f"Error: config.json or results.json not found in {run_folder}")
        return

    config = json.loads(config_path.read_text())
    results = json.loads(results_path.read_text())

    # Determine refinement model
    refine_model_spec = args.refine_model or config.get("llm_model", "openai:gpt-4o-mini")

    print(f"Benchmark run: {run_folder}")
    if args.diagnose:
        print(f"Diagnose model: {args.diagnose_model}")
    if not args.no_refine:
        print(f"Refinement model: {refine_model_spec}")
    print(f"Validator model: {config.get('validator_model', config.get('model', 'gpt-4o-mini'))}")

    # Determine refined folder name (auto-increment if exists)
    if args.output:
        # User specified custom output folder name
        refined_folder = run_folder / args.output
    else:
        # Auto-increment: refined, refined_1, refined_2, etc.
        refined_folder = run_folder / "refined"
        if refined_folder.exists():
            # Find next available number
            i = 1
            while True:
                refined_folder = run_folder / f"refined_{i}"
                if not refined_folder.exists():
                    break
                i += 1

    refined_folder.mkdir(exist_ok=True)
    print(f"Output folder: {refined_folder}")

    # Determine refinement prompt type
    refinement_prompt_type = "incremental" if args.incremental else "default"

    # Save config for refined folder (enables recursive refinement)
    refined_config = {
        "parent_run": str(run_folder),
        "refine_model": refine_model_spec,
        "validator_model": config.get("validator_model", config.get("model", "gpt-4o-mini")),
        "llm_model": refine_model_spec,  # For recursive refinement
        "k_valid": config.get("k_valid", 100),
        "k_invalid": config.get("k_invalid", 100),
        "trace_source": "inherited" if args.use_parent_traces else "new_sampling",
        "refinement_prompt_type": refinement_prompt_type,
        "refinement_timestamp": datetime.now().isoformat(),
        "perturbations": config.get("perturbations", ["random_swap_events"])
    }
    (refined_folder / "config.json").write_text(json.dumps(refined_config, indent=2))

    # Create LLM for refinement (only if needed)
    llm = None
    if not args.no_refine:
        llm = create_llm(refine_model_spec)

    # Store diagnosis results for summary
    all_diagnosis_results = {}

    # Store Stage 1 results for summary
    all_stage1_results = {}
    problems_with_improvement = []

    # Process each problem
    for problem_name, problem_data in results.items():
        if problem_name.startswith("_"):
            continue

        print(f"\n{'='*60}")
        print(f"Processing: {problem_name}")
        print(f"{'='*60}")

        # Check if model already has perfect F1 score
        llm_metrics = problem_data.get("llm", {})
        current_f1 = llm_metrics.get("f1", 0)
        if current_f1 == 1.0:
            print(f"  F1=1.000 (perfect) - copying optimal model, no refinement needed")
            # Copy the textual model to refined folder
            try:
                original_model = load_textual_model(run_folder, problem_name)
                model_path = save_refined_model(refined_folder, problem_name, original_model)
                print(f"  Copied model to {model_path}")
            except FileNotFoundError:
                print(f"  Warning: Could not copy textual model")
            # Copy results for this problem
            all_stage1_results[problem_name] = {
                "total": 0,
                "fixed": 0,
                "still_wrong": 0,
                "fix_rate": 1.0,
                "skipped": True,
                "reason": "F1=1.0 (optimal)"
            }
            continue

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
        representations = load_representations(problem_data, run_folder)

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
            mode_str = "Incremental Refinement" if args.incremental else "Refinement"
            print(f"\n  --- {mode_str} ---")
            # Build refinement prompt
            refinement_prompt = build_refinement_prompt(as_is_model, misclassified, representations, incremental=args.incremental)

            # Save refinement prompt for reproducibility
            prompt_path = refined_folder / f"{problem_name}_refinement_prompt_{refinement_prompt_type}.txt"
            prompt_path.write_text(refinement_prompt)
            print(f"  Saved refinement prompt to {prompt_path.name}")

            # Call LLM for refinement
            print(f"  Calling LLM for refinement{' (incremental mode)' if args.incremental else ''}...")
            start_time = time.time()
            refined_model = refine_model(llm, refinement_prompt)
            elapsed = time.time() - start_time
            print(f"  Refinement completed in {elapsed:.1f}s")

            # Save refined model
            model_path = save_refined_model(refined_folder, problem_name, refined_model)
            print(f"  Saved refined model to {model_path}")

            # --- Stage 1: Misclassified Traces Test ---
            print(f"\n  --- Stage 1: Misclassified Traces Test ---")
            print(f"  Testing refined model on {len(misclassified)} previously misclassified traces...")

            # Load actual trace DataFrames for evaluation
            misclassified_for_eval = load_misclassified_traces_for_eval(run_folder, problem_name, problem_data)

            validator_model = config.get("validator_model", config.get("model", "gpt-4o-mini"))
            stage1_results = evaluate_on_misclassified(refined_model, misclassified_for_eval, validator_model)

            fix_rate_pct = stage1_results['fix_rate'] * 100
            print(f"\n  Stage 1 Result: {stage1_results['fixed']}/{stage1_results['total']} now correct ({fix_rate_pct:.0f}% fixed)")

            # Store stage 1 results for this problem
            all_stage1_results[problem_name] = stage1_results

            # Check if improvement was made
            improved = stage1_results["fixed"] > 0
            if improved:
                problems_with_improvement.append(problem_name)
            else:
                print(f"  No traces fixed by refinement")

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

    # Print Stage 1 Summary
    if not args.no_refine and all_stage1_results:
        print(f"\n{'='*60}")
        print(f"STAGE 1 SUMMARY (Misclassified Traces Test)")
        print(f"{'='*60}")

        for problem_name, s1_results in all_stage1_results.items():
            status = "✓ improved" if s1_results["fixed"] > 0 else "✗ no improvement"
            print(f"\n{problem_name}: {status}")
            print(f"  Fixed: {s1_results['fixed']}/{s1_results['total']} ({s1_results['fix_rate']*100:.0f}%)")
            print(f"  FN: {s1_results['by_type']['FN']['fixed']} fixed, {s1_results['by_type']['FN']['still_wrong']} still wrong")
            print(f"  FP: {s1_results['by_type']['FP']['fixed']} fixed, {s1_results['by_type']['FP']['still_wrong']} still wrong")

    # Check if we should proceed to Stage 2
    if not args.no_refine and not args.skip_full_benchmark:
        should_run_full = len(problems_with_improvement) > 0 or args.force_full_benchmark

        if not should_run_full:
            print(f"\n{'='*60}")
            print(f"SKIPPING FULL BENCHMARK")
            print(f"{'='*60}")
            print(f"No problems showed improvement in Stage 1.")
            print(f"Use --force-full-benchmark to run full benchmark anyway.")
        else:
            # Re-run benchmark on refined models
            print(f"\n{'='*60}")
            print(f"STAGE 2: Full Benchmark on Refined Models")
            print(f"{'='*60}")

            if args.force_full_benchmark and len(problems_with_improvement) == 0:
                print(f"(Running due to --force-full-benchmark)")
            else:
                print(f"Problems with improvement: {', '.join(problems_with_improvement)}")

            refined_results = run_benchmark_on_refined(refined_folder, config, results, run_folder, new_sampling=not args.use_parent_traces)

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

                s1 = all_stage1_results.get(problem_name, {})
                s1_info = f" | Stage1: {s1.get('fixed', 0)}/{s1.get('total', 0)} fixed" if s1 else ""

                print(f"\n{problem_name}:{s1_info}")
                print(f"  Original: F1={original.get('f1', 0):.3f} | FN={original.get('FN', 0)} | FP={original.get('FP', 0)}")
                print(f"  Refined:  F1={metrics.get('f1', 0):.3f} | FN={metrics.get('FN', 0)} | FP={metrics.get('FP', 0)}")
                print(f"  Delta:    F1={improvement.get('f1_delta', 0):+.3f} | FN={improvement.get('fn_delta', 0):+d} | FP={improvement.get('fp_delta', 0):+d}")

            print(f"\nResults saved to {refined_folder / 'results.json'}")
    elif args.skip_full_benchmark:
        print(f"\n(Skipping full benchmark due to --skip-full-benchmark)")


if __name__ == "__main__":
    main()
