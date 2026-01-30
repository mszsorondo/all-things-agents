#!/usr/bin/env python3
"""
Adaptive Process Model Explanation System - Validation Benchmark

Tests adapted explanations preserve semantic fidelity via trace classification.
Compares original textual model performance against all audience adaptations.
"""

import os
import json
import pickle
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from dotenv import load_dotenv

load_dotenv()

# Import from existing benchmark infrastructure
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from phase5_benchmark import (
    create_validator,
    evaluate_traces,
    calculate_metrics,
    sample_valid_traces,
    generate_invalid_traces,
    save_misclassified_trace
)
from benchmark_template import df_to_key_value_rows, random_swap_events

# Import local adapters module
from adapters import adapt_textual_model


def benchmark_single_audience(
    audience: str,
    original_model: str,
    valid_traces: list,
    invalid_traces: list,
    adapter_llm: str,
    validator_llm: str,
    output_dir: Path,
    skip_adaptation: bool = False,
    adapted_model_path: str = None
) -> Tuple[str, dict, dict]:
    """
    Benchmark one audience adaptation.

    Args:
        audience: Target audience name
        original_model: Original textual model text
        valid_traces: List of valid trace DataFrames
        invalid_traces: List of invalid trace DataFrames
        adapter_llm: LLM for adaptation (e.g., "openai:gpt-5")
        validator_llm: LLM for validation (e.g., "gpt-5-mini")
        output_dir: Directory to save results
        skip_adaptation: If True, load pre-adapted model instead of generating
        adapted_model_path: Path to pre-adapted model file (if skip_adaptation=True)

    Returns:
        Tuple of (adapted_model_text, metrics_dict, misclassified_dict)
    """
    print(f"\n{'='*80}")
    print(f"BENCHMARKING AUDIENCE: {audience.upper()}")
    print(f"{'='*80}\n")

    # Step 1: Get adapted model (either adapt or load)
    if skip_adaptation and adapted_model_path:
        print(f"Step 1: Loading pre-adapted model from {adapted_model_path}...")
        try:
            adapted_model = Path(adapted_model_path).read_text()
            print(f"✓ Loaded pre-adapted model")
            print(f"  Original length: {len(original_model)} chars")
            print(f"  Adapted length: {len(adapted_model)} chars")
        except Exception as e:
            print(f"✗ Failed to load pre-adapted model: {e}")
            raise

        # Save copy to output dir
        output_adapted_path = output_dir / f"adapted_{audience}.txt"
        output_adapted_path.write_text(adapted_model)
        print(f"  Saved copy to: {output_adapted_path}")
    else:
        print(f"Step 1: Adapting model for {audience} using {adapter_llm}...")
        try:
            adapted_model = adapt_textual_model(original_model, audience, adapter_llm)
            print(f"✓ Adaptation complete")
            print(f"  Original length: {len(original_model)} chars")
            print(f"  Adapted length: {len(adapted_model)} chars")
        except Exception as e:
            print(f"✗ Adaptation failed: {e}")
            raise

        # Save adapted model
        output_adapted_path = output_dir / f"adapted_{audience}.txt"
        output_adapted_path.write_text(adapted_model)
        print(f"  Saved to: {output_adapted_path}")

    # Step 2: Create validator with adapted model
    print(f"\nStep 2: Creating validator using {validator_llm}...")
    validator, system_prompt = create_validator(adapted_model, validator_llm)
    print(f"✓ Validator created")

    # Step 3: Evaluate traces
    print(f"\nStep 3: Evaluating {len(valid_traces)} valid + {len(invalid_traces)} invalid traces...")
    print(f"Note: This benchmark will run to completion regardless of performance metrics\n")

    misclassified_folder = output_dir / "misclassified" / audience
    misclassified_folder.mkdir(parents=True, exist_ok=True)

    valid_preds, invalid_preds, total_time, total_tokens, misclassified = evaluate_traces(
        valid_traces, invalid_traces, validator, misclassified_folder
    )

    # Step 4: Calculate metrics
    metrics = calculate_metrics(valid_preds, invalid_preds, total_time, total_tokens)

    print(f"\n{'-'*80}")
    print(f"RESULTS FOR {audience.upper()}:")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall:    {metrics['recall']:.3f}")
    print(f"  F1 Score:  {metrics['f1']:.3f}")
    print(f"  TP={metrics['TP']}, FP={metrics['FP']}, TN={metrics['TN']}, FN={metrics['FN']}")
    print(f"  Total time: {metrics['total_compute_time_seconds']:.1f}s")
    print(f"  Total tokens: {metrics['total_tokens']}")
    print(f"{'-'*80}\n")

    return adapted_model, metrics, misclassified


def compare_metrics(baseline_metrics: dict, adapted_metrics: dict) -> dict:
    """
    Calculate comparison metrics between baseline and adapted models.

    Args:
        baseline_metrics: Metrics from original model (can be None)
        adapted_metrics: Dict of {audience: metrics} for adapted models

    Returns:
        Comparison dictionary with deltas and rankings
    """
    comparison = {
        "baseline_f1": baseline_metrics["f1"] if baseline_metrics else None,
        "audiences": {}
    }

    for audience, metrics in adapted_metrics.items():
        if baseline_metrics:
            f1_delta = metrics["f1"] - baseline_metrics["f1"]
        else:
            f1_delta = None

        comparison["audiences"][audience] = {
            "f1": metrics["f1"],
            "f1_delta": f1_delta,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "TP": metrics["TP"],
            "FP": metrics["FP"],
            "TN": metrics["TN"],
            "FN": metrics["FN"]
        }

    # Rank audiences by F1 preservation (closest to baseline) or just by F1 if no baseline
    if baseline_metrics:
        ranked = sorted(
            comparison["audiences"].items(),
            key=lambda x: abs(x[1]["f1_delta"]) if x[1]["f1_delta"] is not None else float('inf')
        )
    else:
        ranked = sorted(
            comparison["audiences"].items(),
            key=lambda x: x[1]["f1"],
            reverse=True
        )
    comparison["ranking"] = [audience for audience, _ in ranked]

    return comparison


def generate_report(
    results: dict,
    output_dir: Path,
    config: dict
):
    """
    Generate comprehensive markdown report.

    Args:
        results: Complete results dictionary
        output_dir: Directory to save report
        config: Benchmark configuration
    """
    report_path = output_dir / "comparison_report.md"

    baseline = results.get("baseline")
    audiences = results["audiences"]
    comparison = results["comparison"]

    # Format audiences list
    tested_audiences = list(audiences.keys())
    audiences_str = ', '.join(tested_audiences)

    report = f"""# Adaptive Process Model Benchmark Report

**Model Identifier:** {config['model_identifier']}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Configuration

- **Model Identifier:** {config['model_identifier']}
- **Textual Model File:** {config['textual_model_filename']}
- **XES File:** {config['xes_filename']}
- **Adapter LLM:** {config['adapter_llm']}
- **Validator LLM:** {config['validator_llm']}
- **Valid Traces:** {config['k_valid']}
- **Invalid Traces:** {config['k_invalid']}
- **Tested Audiences:** {audiences_str}
- **Baseline Mode:** {'Skipped' if config['skip_baseline'] else ('Last' if config['baseline_last'] else 'First')}

## Executive Summary

This benchmark evaluates how well audience-adapted process models preserve semantic fidelity for trace validation. {len(tested_audiences)} audience(s) completed successfully.

"""

    if baseline and baseline['metrics']:
        report += "**Baseline Performance (Original Model):**\n"
        report += "- F1 Score: {:.3f}\n".format(baseline['metrics']['f1'])
        report += "- Precision: {:.3f}\n".format(baseline['metrics']['precision'])
        report += "- Recall: {:.3f}\n".format(baseline['metrics']['recall'])
    else:
        report += "**Note:** Baseline (original model) was not tested.\n"

    report += "\n## Results by Audience\n\n"
    report += "### Performance Comparison Table\n\n"
    report += "| Audience | F1 Score | Δ F1 | Precision | Recall | TP | FP | TN | FN |\n"
    report += "|----------|----------|------|-----------|--------|----|----|----|----||\n"

    if baseline and baseline['metrics']:
        report += f"| **Baseline** | {baseline['metrics']['f1']:.3f} | - | {baseline['metrics']['precision']:.3f} | {baseline['metrics']['recall']:.3f} | {baseline['metrics']['TP']} | {baseline['metrics']['FP']} | {baseline['metrics']['TN']} | {baseline['metrics']['FN']} |\n"

    for audience in tested_audiences:
        if audience in comparison['audiences']:
            m = comparison['audiences'][audience]
            delta_str = "N/A" if 'f1_delta' not in m or m.get('f1_delta') is None else f"{'+' if m['f1_delta'] >= 0 else ''}{m['f1_delta']:.3f}"
            report += f"| {audience.replace('_', ' ').title()} | {m['f1']:.3f} | {delta_str} | {m['precision']:.3f} | {m['recall']:.3f} | {m['TP']} | {m['FP']} | {m['TN']} | {m['FN']} |\n"

    report += f"""
### Audience Ranking

"""

    if baseline and baseline['metrics']:
        report += "Ranked from best to worst F1 preservation:\n\n"
        for rank, audience in enumerate(comparison['ranking'], 1):
            m = comparison['audiences'][audience]
            delta_sign = '+' if m.get('f1_delta', 0) >= 0 else ''
            delta_val = m.get('f1_delta', 0)
            report += f"{rank}. **{audience.replace('_', ' ').title()}**: F1={m['f1']:.3f} (Δ {delta_sign}{delta_val:.3f})\n"
    else:
        report += "Ranked by F1 score:\n\n"
        sorted_audiences = sorted(tested_audiences, key=lambda a: comparison['audiences'][a]['f1'], reverse=True)
        for rank, audience in enumerate(sorted_audiences, 1):
            m = comparison['audiences'][audience]
            report += f"{rank}. **{audience.replace('_', ' ').title()}**: F1={m['f1']:.3f}\n"

    report += "\n## Detailed Results\n\n"

    for audience in tested_audiences:
        aud_data = audiences[audience]
        m = aud_data['metrics']

        if baseline and baseline['metrics']:
            delta_sign = '+' if m['f1'] - baseline['metrics']['f1'] >= 0 else ''
            delta_val = m['f1'] - baseline['metrics']['f1']
            f1_line = "- F1 Score: {:.3f} ({}{:.3f} vs baseline)\n".format(m['f1'], delta_sign, delta_val)
        else:
            f1_line = "- F1 Score: {:.3f}\n".format(m['f1'])

        audience_title = audience.replace('_', ' ').title()

        audience_section = "### {}\n\n**Performance:**\n".format(audience_title)
        audience_section += f1_line
        audience_section += "- Precision: {:.3f}\n".format(m['precision'])
        audience_section += "- Recall: {:.3f}\n".format(m['recall'])
        audience_section += "- True Positives: {}\n".format(m['TP'])
        audience_section += "- False Positives: {}\n".format(m['FP'])
        audience_section += "- True Negatives: {}\n".format(m['TN'])
        audience_section += "- False Negatives: {}\n\n".format(m['FN'])
        audience_section += "**Resources:**\n"
        audience_section += "- Compute Time: {:.1f} seconds\n".format(m['total_compute_time_seconds'])
        audience_section += "- Total Tokens: {:,}\n\n".format(m['total_tokens'])
        audience_section += "**Misclassified Traces:**\n"
        audience_section += "- False Negatives: {}\n".format(len(aud_data['misclassified']['false_negatives']))
        audience_section += "- False Positives: {}\n".format(len(aud_data['misclassified']['false_positives']))
        audience_section += "- Details: See `misclassified/{}/`\n\n".format(audience)
        audience_section += "**Adapted Model:** `adapted_{}.txt`\n\n---\n\n".format(audience)

        report += audience_section

    report += f"""## Observations

All {len(tested_audiences)} audience adaptation(s) completed successfully.

"""

    if baseline and baseline['metrics']:
        # Note any significant degradations
        significant_drops = [
            (aud, comparison['audiences'][aud].get('f1_delta', 0))
            for aud in comparison['audiences']
            if comparison['audiences'][aud].get('f1_delta', 0) < -0.1
        ]

        if significant_drops:
            report += "**Note:** The following audiences showed F1 degradation > 0.1:\n"
            for aud, delta in significant_drops:
                report += f"- {aud.replace('_', ' ').title()}: Δ {delta:.3f}\n"
            report += "\nConsider refining adapter prompts for these audiences.\n"

    report += """
## Files Generated

- `config.json` - Benchmark configuration
- `original_textual_model.txt` - Baseline model
- `adapted_<audience>.txt` - 5 adapted versions
- `results.json` - Complete metrics data
- `comparison_report.md` - This report
- `traces/valid_traces.pkl` - Test valid traces
- `traces/invalid_traces.pkl` - Test invalid traces
- `misclassified/<audience>/` - Misclassified traces per audience

## Methodology

1. **Baseline Establishment:** Original textual model validated against test traces
2. **Adaptation:** Each audience receives high-quality adaptation using GPT-5
3. **Validation:** Adapted models validated using cost-effective GPT-5-mini
4. **Comparison:** F1 scores compared to measure semantic preservation
5. **Completion:** All benchmarks run to completion regardless of performance

The goal is zero information loss - adapted models should preserve all behavioral information for trace validation.
"""

    report_path.write_text(report)
    print(f"✓ Report saved: {report_path}")


def benchmark_all_audiences(
    xes_path: str,
    textual_model_path: str,
    adapter_llm: str = 'openai:gpt-5',
    validator_llm: str = 'gpt-5-mini',
    k_valid: int = 100,
    k_invalid: int = 100,
    audiences: list = None,
    skip_adaptation: bool = False,
    adapted_models_dir: str = None,
    skip_baseline: bool = False,
    baseline_last: bool = False
) -> dict:
    """
    Run complete benchmark comparing original vs all adapted versions.
    ALWAYS runs all benchmarks to completion - never interrupts due to poor metrics.

    Args:
        xes_path: Path to XES event log
        textual_model_path: Path to original textual model
        adapter_llm: LLM for model adaptation (high quality)
        validator_llm: LLM for trace classification (cost effective)
        k_valid: Number of valid traces to test
        k_invalid: Number of invalid traces to test

    Returns:
        Complete results dictionary
    """
    print("\n" + "="*80)
    print("ADAPTIVE PROCESS MODEL BENCHMARK")
    print("="*80 + "\n")

    # Use textual model filename (without extension) as identifier
    textual_model_file = Path(textual_model_path)
    model_identifier = textual_model_file.stem  # e.g., "hiring_log_high_textual_model"

    # Setup output directory with model identifier
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(__file__).parent / 'adapters_benchmark_results' / f'{model_identifier}_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Model Identifier: {model_identifier}")
    print(f"Output directory: {output_dir}\n")

    # Load original model
    print("Loading original textual model...")
    original_model = Path(textual_model_path).read_text()
    print(f"✓ Loaded: {len(original_model)} characters\n")

    # Save original model to output
    (output_dir / "original_textual_model.txt").write_text(original_model)

    # Generate test traces
    print(f"Generating test traces from: {xes_path}")
    print(f"  Sampling {k_valid} valid traces...")
    valid_traces = sample_valid_traces(xes_path, k_valid)
    print(f"  ✓ Got {len(valid_traces)} valid traces")

    print(f"  Generating {k_invalid} invalid traces...")
    invalid_traces = generate_invalid_traces(valid_traces, k_invalid)
    print(f"  ✓ Generated {len(invalid_traces)} invalid traces\n")

    # Save traces
    traces_dir = output_dir / "traces"
    traces_dir.mkdir(exist_ok=True)
    with open(traces_dir / "valid_traces.pkl", 'wb') as f:
        pickle.dump(valid_traces, f)
    with open(traces_dir / "invalid_traces.pkl", 'wb') as f:
        pickle.dump(invalid_traces, f)
    print(f"✓ Traces saved to {traces_dir}\n")

    # Save experiment configuration with all variables
    experiment_config = {
        "experiment_identifier": f"{model_identifier}_{timestamp}",
        "model_identifier": model_identifier,
        "timestamp": timestamp,
        "datetime": datetime.now().isoformat(),

        # Input files
        "xes_path": str(xes_path),
        "xes_filename": Path(xes_path).name,
        "textual_model_path": str(textual_model_path),
        "textual_model_filename": textual_model_file.name,

        # LLM configuration
        "adapter_llm": adapter_llm,
        "validator_llm": validator_llm,

        # Test configuration
        "k_valid": k_valid,
        "k_invalid": k_invalid,
        "audiences_tested": audiences if audiences else ["executives", "process_analysts", "developers", "auditors", "children"],
        "skip_adaptation": skip_adaptation,
        "adapted_models_dir": str(adapted_models_dir) if adapted_models_dir else None,
        "skip_baseline": skip_baseline,
        "baseline_last": baseline_last,

        # Output
        "output_dir": str(output_dir)
    }

    # Save as config.json (for backwards compatibility)
    with open(output_dir / "config.json", 'w') as f:
        json.dump(experiment_config, f, indent=2)

    # Also save as experiment_details.json with additional metadata
    experiment_details = {
        **experiment_config,
        "command_line_args": {
            "xes_path": str(xes_path),
            "textual_model_path": str(textual_model_path),
            "adapter_llm": adapter_llm,
            "validator_llm": validator_llm,
            "k_valid": k_valid,
            "k_invalid": k_invalid,
            "audiences": audiences,
            "skip_adaptation": skip_adaptation,
            "adapted_models_dir": str(adapted_models_dir) if adapted_models_dir else None,
            "skip_baseline": skip_baseline,
            "baseline_last": baseline_last
        },
        "execution_metadata": {
            "started_at": datetime.now().isoformat(),
            "python_version": None,  # Will be filled later if needed
            "working_directory": str(Path.cwd())
        }
    }

    with open(output_dir / "experiment_details.json", 'w') as f:
        json.dump(experiment_details, f, indent=2)

    # Keep config for backwards compatibility with existing code
    config = experiment_config

    # Set default audiences if not specified
    if audiences is None:
        audiences = ['executives', 'process_analysts', 'developers', 'auditors', 'children']

    # Function to run baseline
    def run_baseline():
        print("="*80)
        print("BASELINE: ORIGINAL MODEL")
        print("="*80 + "\n")

        print(f"Creating baseline validator using {validator_llm}...")
        baseline_validator, _ = create_validator(original_model, validator_llm)
        print(f"✓ Validator created\n")

        print(f"Evaluating baseline with {len(valid_traces)} valid + {len(invalid_traces)} invalid traces...\n")

        baseline_misclassified_folder = output_dir / "misclassified" / "original"
        baseline_misclassified_folder.mkdir(parents=True, exist_ok=True)

        baseline_valid_preds, baseline_invalid_preds, baseline_time, baseline_tokens, baseline_misclassified = evaluate_traces(
            valid_traces, invalid_traces, baseline_validator, baseline_misclassified_folder
        )

        baseline_metrics = calculate_metrics(
            baseline_valid_preds, baseline_invalid_preds, baseline_time, baseline_tokens
        )

        print(f"\n{'-'*80}")
        print(f"BASELINE RESULTS:")
        print(f"  Precision: {baseline_metrics['precision']:.3f}")
        print(f"  Recall:    {baseline_metrics['recall']:.3f}")
        print(f"  F1 Score:  {baseline_metrics['f1']:.3f}")
        print(f"  TP={baseline_metrics['TP']}, FP={baseline_metrics['FP']}, TN={baseline_metrics['TN']}, FN={baseline_metrics['FN']}")
        print(f"  Total time: {baseline_metrics['total_compute_time_seconds']:.1f}s")
        print(f"  Total tokens: {baseline_metrics['total_tokens']}")
        print(f"{'-'*80}\n")

        return baseline_metrics, baseline_misclassified

    # Run baseline first if not skipped and not deferred
    baseline_metrics = None
    baseline_misclassified = None

    if not skip_baseline and not baseline_last:
        baseline_metrics, baseline_misclassified = run_baseline()

    # Benchmark selected audiences
    audience_results = {}

    print(f"Beginning audience adaptations (testing {len(audiences)} audience(s))...\n")

    for i, audience in enumerate(audiences, 1):
        print(f"\n{'#'*80}")
        print(f"AUDIENCE {i}/{len(audiences)}: {audience.upper()}")
        print(f"{'#'*80}")

        try:
            # Determine adapted model path if using pre-adapted models
            adapted_model_path = None
            if skip_adaptation and adapted_models_dir:
                adapted_model_path = Path(adapted_models_dir) / f"adapted_{audience}.txt"
                if not adapted_model_path.exists():
                    print(f"⚠ Warning: Pre-adapted model not found: {adapted_model_path}")
                    print(f"  Will adapt on-the-fly instead...\n")
                    adapted_model_path = None

            adapted_model, metrics, misclassified = benchmark_single_audience(
                audience=audience,
                original_model=original_model,
                valid_traces=valid_traces,
                invalid_traces=invalid_traces,
                adapter_llm=adapter_llm,
                validator_llm=validator_llm,
                output_dir=output_dir,
                skip_adaptation=skip_adaptation and adapted_model_path is not None,
                adapted_model_path=str(adapted_model_path) if adapted_model_path else None
            )

            audience_results[audience] = {
                "metrics": metrics,
                "misclassified": misclassified,
                "adapted_model_length": len(adapted_model)
            }

            print(f"✓ {audience.upper()} COMPLETED")

        except Exception as e:
            print(f"\n✗ ERROR in {audience}: {e}")
            print(f"Continuing with remaining audiences...\n")
            # Still continue to next audience
            continue

    # Run baseline last if requested
    if not skip_baseline and baseline_last:
        baseline_metrics, baseline_misclassified = run_baseline()

    # Compare metrics (only if baseline was run)
    print(f"\n{'='*80}")
    print("GENERATING COMPARISON REPORT")
    print(f"{'='*80}\n")

    comparison = None
    if not skip_baseline and baseline_metrics:
        comparison = compare_metrics(baseline_metrics, {aud: res['metrics'] for aud, res in audience_results.items()})
    else:
        # No baseline to compare against
        comparison = {
            "baseline_f1": None,
            "audiences": {aud: res['metrics'] for aud, res in audience_results.items()},
            "ranking": list(audience_results.keys())
        }

    # Compile complete results
    results = {
        "baseline": {
            "metrics": baseline_metrics if not skip_baseline else None,
            "misclassified": baseline_misclassified if not skip_baseline else None
        } if not skip_baseline else None,
        "audiences": audience_results,
        "comparison": comparison,
        "config": config
    }

    # Save results JSON
    with open(output_dir / "results.json", 'w') as f:
        # Create serializable version (remove non-serializable trace data)
        results_serializable = {
            "baseline": {
                "metrics": results["baseline"]["metrics"],
                "num_false_negatives": len(results["baseline"]["misclassified"]["false_negatives"]),
                "num_false_positives": len(results["baseline"]["misclassified"]["false_positives"])
            } if not skip_baseline and results["baseline"] else None,
            "audiences": {
                aud: {
                    "metrics": res["metrics"],
                    "adapted_model_length": res["adapted_model_length"],
                    "num_false_negatives": len(res["misclassified"]["false_negatives"]),
                    "num_false_positives": len(res["misclassified"]["false_positives"])
                }
                for aud, res in results["audiences"].items()
            },
            "comparison": results["comparison"],
            "config": results["config"]
        }
        json.dump(results_serializable, f, indent=2)

    print(f"✓ Results saved: {output_dir / 'results.json'}")

    # Generate markdown report
    generate_report(results, output_dir, config)

    print(f"\n{'='*80}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*80}")
    print(f"\nAll results saved to: {output_dir}")
    print(f"View report: {output_dir / 'comparison_report.md'}\n")

    return results


def main():
    """CLI for running adapter benchmarks."""
    parser = argparse.ArgumentParser(
        description='Benchmark adaptive process model explanations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python downstream/benchmark.py \\
    --xes-path benchmark/hiring/hiring_log_high.xes \\
    --textual-model-path benchmark_runs/run_X/hiring_log_high_textual_model.txt \\
    --adapter-llm openai:gpt-5 \\
    --validator-llm gpt-5-mini \\
    --k-valid 100 \\
    --k-invalid 100

This will:
1. Load original model and generate test traces
2. Validate with original model (baseline)
3. Adapt for all 5 audiences using GPT-5
4. Validate each adapted model using GPT-5-mini
5. Compare F1 scores and generate report
6. Complete all benchmarks regardless of performance
        """
    )

    parser.add_argument(
        '--xes-path',
        type=str,
        required=True,
        help='Path to XES event log file'
    )

    parser.add_argument(
        '--textual-model-path',
        type=str,
        required=True,
        help='Path to original textual model file'
    )

    parser.add_argument(
        '--adapter-llm',
        type=str,
        default='openai:gpt-5',
        help='LLM for model adaptation (default: openai:gpt-5)'
    )

    parser.add_argument(
        '--validator-llm',
        type=str,
        default='gpt-5-mini',
        help='LLM for trace validation (default: gpt-5-mini)'
    )

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

    parser.add_argument(
        '--audiences',
        nargs='+',
        choices=['executives', 'process_analysts', 'developers', 'auditors', 'children'],
        help='Specific audiences to test (default: all 5). Order matters.'
    )

    parser.add_argument(
        '--skip-baseline',
        action='store_true',
        help='Skip testing the original model entirely'
    )

    parser.add_argument(
        '--baseline-last',
        action='store_true',
        help='Test baseline after all audiences instead of first'
    )

    parser.add_argument(
        '--adapted-models-dir',
        type=str,
        help='Directory with pre-adapted models (skips adaptation step). Models should be named adapted_{audience}.txt'
    )

    args = parser.parse_args()

    # Verify files exist
    if not Path(args.xes_path).exists():
        print(f"Error: XES file not found: {args.xes_path}")
        return 1

    if not Path(args.textual_model_path).exists():
        print(f"Error: Textual model file not found: {args.textual_model_path}")
        return 1

    # Validate arguments
    if args.skip_baseline and args.baseline_last:
        print("Error: Cannot use both --skip-baseline and --baseline-last")
        return 1

    # Run benchmark
    try:
        results = benchmark_all_audiences(
            xes_path=args.xes_path,
            textual_model_path=args.textual_model_path,
            adapter_llm=args.adapter_llm,
            validator_llm=args.validator_llm,
            k_valid=args.k_valid,
            k_invalid=args.k_invalid,
            audiences=args.audiences,
            skip_adaptation=args.adapted_models_dir is not None,
            adapted_models_dir=args.adapted_models_dir,
            skip_baseline=args.skip_baseline,
            baseline_last=args.baseline_last
        )
        return 0
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
