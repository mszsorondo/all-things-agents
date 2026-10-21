#!/usr/bin/env python3
"""
Run full benchmark with new interpretation guides.

Usage:
  # Default: All representations with new guides
  python run_with_new_guides.py

  # With separate generator and validator LLMs
  python run_with_new_guides.py --generator-llm openai:gpt-5 --validator-llm gpt-5-mini

  # Without interpretation guides
  python run_with_new_guides.py --no-guides

  # Ablation study: Run separate benchmark for EACH of 11 representations
  python run_with_new_guides.py --one-rep
  python run_with_new_guides.py --one-rep --generator-llm openai:gpt-5-mini --k-valid 50 --k-invalid 50

  # Single representation only
  python run_with_new_guides.py --one-rep petri_net_inductive
  python run_with_new_guides.py --one-rep dfg --generator-llm openai:gpt-4o

  # Resume from folder
  python run_with_new_guides.py --resume benchmark_runs/new_guides_run_20260126_194400

Available representations (for --one-rep <name>) - from ablation_study.py:
  - petri_net_inductive (Petri Net - Inductive Miner)
  - petri_net_alpha (Petri Net - Alpha Miner)
  - petri_net_heuristics (Petri Net - Heuristics Miner)
  - petri_net_ilp (Petri Net - ILP Miner)
  - dfg (Directly-Follows Graph)
  - heuristics_net (Heuristics Net)
  - process_tree (Process Tree)
  - abstract_variants (Abstract Process Variants)
  - log_skeleton (Log Skeleton)
  - declare (DECLARE Constraints)
  - log_attributes (Log Attributes)

Excluded: bpmn (excessive tokens), prefix_tree (no string format)
Backward compatible: "petri_net" maps to "petri_net_inductive"
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from enhanced_agentic_miner import AgenticProcessMiner, MiningConfiguration, RepresentationType
from phase5_benchmark import evaluate_traces, calculate_metrics, sample_valid_traces, generate_invalid_traces, create_validator

# Import ablation study components for full representation support
from ablation_study import (
    RepresentationType as AblationRepresentationType,
    RepresentationDiscoverer,
    AblationMiner,
    AblationMiningConfig,
    EXCLUDED_FROM_ABLATION
)
import pm4py

def run_problem(xes_path: str, generator_llm: str, validator_llm: str, k_valid: int, k_invalid: int, run_folder: Path, use_guides: bool = True, one_rep: str = None):
    """Run benchmark for a single problem."""
    problem_name = Path(xes_path).stem

    print(f"\n{'='*80}")
    print(f"Processing: {problem_name}")
    print(f"{'='*80}")

    if one_rep:
        print(f"Mode: Single representation only ({one_rep})")
    else:
        print(f"Mode: All representations (petri_net, dfg, abstract_variants)")

    # Check for existing textual model
    model_file = run_folder / f"{problem_name}_textual_model.txt"

    if model_file.exists():
        print(f"Found existing textual model: {model_file}")
        model_text = model_file.read_text()
        miner_run_folder = "reused_precomputed"
    else:
        # Determine which representations to use
        if one_rep:
            representations = [RepresentationType(one_rep)]
        else:
            representations = [
                RepresentationType.PETRI_NET,
                RepresentationType.DFG,
                RepresentationType.ABSTRACT_VARIANTS
            ]

        # Configure miner
        config = MiningConfiguration(
            representations=representations,
            model_name=generator_llm,
            prompt_file="ONLY_GENERATE_TEXTUAL_MODEL.txt",
            use_cache=True
        )

        miner = AgenticProcessMiner(config)
        miner.log_file = xes_path
        miner.xes_path = xes_path
        miner.dataset_name = problem_name

        print(f"Discovering representations using: {generator_llm}")
        miner.discover_representations()

        if use_guides:
            print("Loading NEW interpretation guides from: representation_interpretation_guides/")
            miner.modify_interpretation_guides("representation_interpretation_guides")
        else:
            print("Running WITHOUT interpretation guides (using hardcoded defaults)")

        print("Generating textual model...")
        miner.run_agent()

        model_text = (Path(miner.run_folder) / "textual_model.txt").read_text()

        # Save to run folder for reuse
        model_file.write_text(model_text)
        print(f"Saved textual model to: {model_file}")

        miner_run_folder = str(miner.run_folder)

    # Run validation
    print(f"\nRunning trace validation using: {validator_llm}")
    valid_traces = sample_valid_traces(xes_path, k=k_valid)
    invalid_traces = generate_invalid_traces(valid_traces, k=k_invalid)

    validator, system_prompt = create_validator(model_text, validator_llm)

    valid_preds, invalid_preds, total_time, total_tokens, misclassified = evaluate_traces(
        valid_traces, invalid_traces, validator
    )

    metrics = calculate_metrics(valid_preds, invalid_preds, total_time, total_tokens)

    print(f"\n{problem_name} Results:")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall: {metrics['recall']:.3f}")
    print(f"  F1: {metrics['f1']:.3f}")

    return {
        "xes_path": xes_path,
        "textual_model_path": str(model_file),
        "llm": metrics,
        "miner_run_folder": miner_run_folder
    }


def run_problem_ablation_style(xes_path: str, generator_llm: str, validator_llm: str, k_valid: int, k_invalid: int,
                                 run_folder: Path, use_guides: bool, rep_type: AblationRepresentationType):
    """Run benchmark for a single problem using ablation_study.py logic (supports all 11 representations)."""
    problem_name = Path(xes_path).stem

    print(f"\n{'='*80}")
    print(f"Processing: {problem_name}")
    print(f"{'='*80}")
    print(f"Mode: Single representation only ({rep_type.value}) - using ablation_study discovery")

    # Check for existing textual model
    model_file = run_folder / f"{problem_name}_textual_model.txt"

    if model_file.exists():
        print(f"Found existing textual model: {model_file}")
        model_text = model_file.read_text()
        miner_run_folder = "reused_precomputed"
    else:
        # Load event log
        print(f"Loading event log: {xes_path}")
        log = pm4py.read_xes(xes_path)

        # Initialize discoverer (from ablation_study.py)
        discoverer = RepresentationDiscoverer(log, xes_path)

        print(f"Discovering all representations (ablation_study mode) using: {generator_llm}")
        discoverer.discover_all()

        # Configure ablation miner
        mining_config = AblationMiningConfig(
            model_name=generator_llm,
            prompt_file="ONLY_GENERATE_TEXTUAL_MODEL.txt",
            use_interpretations=use_guides
        )

        miner = AblationMiner(discoverer, mining_config, run_folder)

        # Generate textual model using ONLY the specified representation
        print(f"Generating textual model using only: {rep_type.value}")
        model_text, miner_run_folder = miner.generate_textual_model(
            representations=[rep_type],
            run_name=f"{problem_name}_{rep_type.value}_generation"
        )

        # Save to run folder for reuse
        model_file.write_text(model_text)
        print(f"Saved textual model to: {model_file}")

    # Run validation
    print(f"\nRunning trace validation using: {validator_llm}")
    valid_traces = sample_valid_traces(xes_path, k=k_valid)
    invalid_traces = generate_invalid_traces(valid_traces, k=k_invalid)

    validator, system_prompt = create_validator(model_text, validator_llm)

    valid_preds, invalid_preds, total_time, total_tokens, misclassified = evaluate_traces(
        valid_traces, invalid_traces, validator
    )

    metrics = calculate_metrics(valid_preds, invalid_preds, total_time, total_tokens)

    print(f"\n{problem_name} Results:")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall: {metrics['recall']:.3f}")
    print(f"  F1: {metrics['f1']:.3f}")

    return {
        "xes_path": xes_path,
        "textual_model_path": str(model_file),
        "llm": metrics,
        "miner_run_folder": str(miner_run_folder)
    }


def main():
    parser = argparse.ArgumentParser(
        description='Run process mining benchmark with interpretation guides',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: All representations with new guides
  python run_with_new_guides.py

  # With separate generator and validator LLMs
  python run_with_new_guides.py --generator-llm openai:gpt-5 --validator-llm gpt-5-mini

  # Without interpretation guides
  python run_with_new_guides.py --no-guides

  # Single representation
  python run_with_new_guides.py --one-rep dfg

  # Ablation study (all representations separately)
  python run_with_new_guides.py --one-rep

  # Custom trace counts
  python run_with_new_guides.py --k-valid 50 --k-invalid 50

  # Resume from folder
  python run_with_new_guides.py --resume benchmark_runs/new_guides_run_20260126_194400
        """
    )

    # LLM configuration
    parser.add_argument(
        '--generator-llm',
        type=str,
        default='openai:gpt-5-mini',
        help='LLM for textual model generation (default: openai:gpt-5-mini)'
    )
    parser.add_argument(
        '--validator-llm',
        type=str,
        default='gpt-5-mini',
        help='LLM for trace validation (default: gpt-5-mini)'
    )

    # Trace configuration
    parser.add_argument(
        '--k-valid',
        type=int,
        default=100,
        help='Number of valid traces to test (default: 100)'
    )
    parser.add_argument(
        '--k-invalid',
        type=int,
        default=100,
        help='Number of invalid traces to generate (default: 100)'
    )

    # Interpretation guides
    parser.add_argument(
        '--no-guides',
        action='store_true',
        help='Run without interpretation guides (use hardcoded defaults)'
    )

    # Representation configuration
    parser.add_argument(
        '--one-rep',
        nargs='?',
        const='__ABLATION__',
        default=None,
        metavar='REPRESENTATION',
        help='Run with single representation, or run ablation study if no name provided'
    )

    # Resume
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        metavar='FOLDER',
        help='Resume from existing run folder'
    )

    # Backward compatibility: support old positional argument format
    # python run_with_new_guides.py openai:gpt-5-mini 100 100
    args, unknown = parser.parse_known_args()

    # If there are unknown positional arguments, treat them as old format
    if unknown and not unknown[0].startswith('--'):
        print("Note: Detected legacy positional argument format")
        print("  Converting to new format: use --generator-llm, --validator-llm, --k-valid, --k-invalid")
        if len(unknown) >= 1:
            args.generator_llm = unknown[0]
            args.validator_llm = unknown[0]  # Same LLM for both in legacy mode
        if len(unknown) >= 2:
            args.k_valid = int(unknown[1])
        if len(unknown) >= 3:
            args.k_invalid = int(unknown[2])
        if len(unknown) >= 4:
            args.resume = unknown[3]

    use_guides = not args.no_guides
    one_rep = None
    ablation_study = False

    # All representations available for ablation study (from ablation_study.py)
    # Get all representations except those excluded
    valid_reps = [
        rep_type.value for rep_type in AblationRepresentationType
        if rep_type not in EXCLUDED_FROM_ABLATION
    ]

    # Backward compatibility: map old names to new names
    rep_name_mapping = {
        "petri_net": "petri_net_inductive",  # Old enhanced_agentic_miner name -> ablation name
    }

    # Parse --one-rep argument
    if args.one_rep is not None:
        if args.one_rep == '__ABLATION__':
            # --one-rep without argument: ablation study
            ablation_study = True
        else:
            # --one-rep with argument: single representation
            rep_arg = args.one_rep

            # Apply backward compatibility mapping
            if rep_arg in rep_name_mapping:
                mapped_arg = rep_name_mapping[rep_arg]
                print(f"Note: Mapping '{rep_arg}' to '{mapped_arg}' for compatibility")
                rep_arg = mapped_arg

            # Validate representation
            if rep_arg in valid_reps:
                one_rep = rep_arg
            else:
                print(f"Error: Invalid representation type: {rep_arg}")
                print(f"Available: {', '.join(sorted(valid_reps))}")
                print(f"Backward compatible: {', '.join(rep_name_mapping.keys())}")
                sys.exit(1)

    generator_llm = args.generator_llm
    validator_llm = args.validator_llm
    k_valid = args.k_valid
    k_invalid = args.k_invalid
    resume_folder = args.resume

    problems = [
        "benchmark/hiring/hiring_log_high.xes",
        "benchmark/hospital/hospital_log_high.xes",
        "benchmark/lending/lending_log_high.xes",
        "benchmark/renting/renting_log_high.xes"
    ]

    # Determine which representations to test
    if ablation_study:
        # Ablation study: test each representation separately
        representations_to_test = valid_reps
        print(f"\n{'='*80}")
        print(f"ABLATION STUDY MODE")
        print(f"{'='*80}")
        print(f"Will run {len(representations_to_test)} separate benchmarks, one for each representation:")
        for i, rep in enumerate(representations_to_test, 1):
            print(f"  {i:2d}. {rep}")
        print(f"{'='*80}\n")
    elif one_rep:
        # Single representation mode
        representations_to_test = [one_rep]
    else:
        # Default mode: no --one-rep flag, use all default representations together
        representations_to_test = [None]  # None means use default representations

    # Create base run folder
    if resume_folder:
        base_run_folder = Path(resume_folder)
        print(f"Resuming from: {base_run_folder}")
        if not base_run_folder.exists():
            print(f"Error: Run folder not found: {base_run_folder}")
            sys.exit(1)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if ablation_study:
            base_run_folder = Path(f"ablation_runs/ablation_study_{timestamp}")
        else:
            base_run_folder = Path(f"benchmark_runs/new_guides_run_{timestamp}")
        base_run_folder.mkdir(parents=True, exist_ok=True)

    # Aggregate results across all ablation runs
    ablation_results = {}

    # Run benchmark for each representation configuration
    for rep_idx, current_rep in enumerate(representations_to_test):
        # Create subfolder for ablation study runs
        if ablation_study:
            run_folder = base_run_folder / f"{current_rep}_only"
            run_folder.mkdir(parents=True, exist_ok=True)
            print(f"\n{'='*80}")
            print(f"ABLATION RUN {rep_idx + 1}/{len(representations_to_test)}: {current_rep}")
            print(f"{'='*80}")
        else:
            run_folder = base_run_folder

        # Load or initialize results for this run
        if resume_folder and not ablation_study:
            results_file = run_folder / "results.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    results = json.load(f)
                print(f"Loaded existing results with {len([k for k in results.keys() if k != '_meta'])} problems")
            else:
                results = {}
        else:
            results = {}

        print(f"\n{'='*80}")
        if use_guides:
            print(f"FULL BENCHMARK WITH NEW INTERPRETATION GUIDES")
        else:
            print(f"FULL BENCHMARK WITHOUT INTERPRETATION GUIDES (DEFAULT HARDCODED)")
        print(f"{'='*80}")
        print(f"Generator LLM: {generator_llm}")
        print(f"Validator LLM: {validator_llm}")
        print(f"Valid traces: {k_valid}")
        print(f"Invalid traces: {k_invalid}")
        print(f"Interpretation guides: {'representation_interpretation_guides/' if use_guides else 'NONE (hardcoded defaults)'}")
        if ablation_study:
            print(f"Representation: {current_rep} (ablation study mode)")
        else:
            print(f"Representations: {current_rep if current_rep else 'DEFAULT (petri_net, dfg, abstract_variants)'}")
        print(f"Run folder: {run_folder}")
        print(f"{'='*80}")

        # Initialize or update metadata
        if "_meta" not in results:
            results["_meta"] = {
                "benchmark_start_time": datetime.now().isoformat(),
                "generator_llm": generator_llm,
                "validator_llm": validator_llm,
                "k_valid": k_valid,
                "k_invalid": k_invalid,
                "interpretation_guides": "representation_interpretation_guides" if use_guides else "none_hardcoded_defaults",
                "use_guides": use_guides,
                "one_representation": current_rep if current_rep else None,
                "representations": [current_rep] if current_rep else ["petri_net", "dfg", "abstract_variants"],
                "ablation_study": ablation_study,
                "ablation_index": rep_idx if ablation_study else None
            }

        for xes_path in problems:
            problem_name = Path(xes_path).stem

            # Skip if already completed
            if problem_name in results and results[problem_name].get("status") == "completed":
                print(f"\nSkipping {problem_name} (already completed)")
                continue

            # Run problem - use ablation style if current_rep is an ablation representation
            if current_rep and current_rep in valid_reps:
                # Use ablation_study.py logic for full representation support
                rep_enum = AblationRepresentationType(current_rep)
                results[problem_name] = run_problem_ablation_style(
                    xes_path, generator_llm, validator_llm, k_valid, k_invalid, run_folder, use_guides, rep_enum
                )
            else:
                # Use enhanced_agentic_miner.py for default mode
                results[problem_name] = run_problem(
                    xes_path, generator_llm, validator_llm, k_valid, k_invalid, run_folder, use_guides, current_rep
                )
            results[problem_name]["status"] = "completed"

            # Save after each problem
            results["_meta"]["benchmark_end_time"] = datetime.now().isoformat()
            results_file = run_folder / "results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Progress saved to: {results_file}")

        results["_meta"]["benchmark_end_time"] = datetime.now().isoformat()

        # Save results for this representation
        results_file = run_folder / "results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        # Store results for ablation summary
        if ablation_study:
            ablation_results[current_rep] = {
                "results": results,
                "run_folder": str(run_folder)
            }

        # Print summary for this run
        print(f"\n{'='*80}")
        if ablation_study:
            print(f"BENCHMARK SUMMARY - {current_rep}")
        else:
            print("BENCHMARK SUMMARY")
        print(f"{'='*80}")
        print(f"{'Problem':<20} {'Precision':<12} {'Recall':<12} {'F1':<12}")
        print("-" * 56)

        for problem in problems:
            problem_name = Path(problem).stem
            if problem_name in results and "llm" in results[problem_name]:
                m = results[problem_name]["llm"]
                print(f"{problem_name:<20} {m['precision']:<12.3f} {m['recall']:<12.3f} {m['f1']:<12.3f}")
            else:
                print(f"{problem_name:<20} {'FAILED':<12}")

        print(f"\nResults saved to: {results_file}")
        print(f"{'='*80}")

    # Print ablation study summary
    if ablation_study:
        print(f"\n{'='*80}")
        print("ABLATION STUDY SUMMARY - ALL REPRESENTATIONS")
        print(f"{'='*80}\n")

        # Save aggregated ablation results
        ablation_summary_file = base_run_folder / "ablation_summary.json"
        with open(ablation_summary_file, 'w') as f:
            json.dump(ablation_results, f, indent=2)

        # Print comparison table for each problem
        for problem in problems:
            problem_name = Path(problem).stem
            print(f"\n{problem_name}:")
            print(f"{'Representation':<25} {'Precision':<12} {'Recall':<12} {'F1':<12}")
            print("-" * 61)

            for rep in valid_reps:
                if rep in ablation_results:
                    res = ablation_results[rep]["results"]
                    if problem_name in res and "llm" in res[problem_name]:
                        m = res[problem_name]["llm"]
                        print(f"{rep:<25} {m['precision']:<12.3f} {m['recall']:<12.3f} {m['f1']:<12.3f}")
                    else:
                        print(f"{rep:<25} {'FAILED':<12}")

        print(f"\n{'='*80}")
        print(f"Ablation study complete!")
        print(f"Summary saved to: {ablation_summary_file}")
        print(f"Individual results in: {base_run_folder}/")
        print(f"{'='*80}")


if __name__ == "__main__":
    main()
