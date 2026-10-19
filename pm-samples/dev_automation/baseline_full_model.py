#!/usr/bin/env python3
"""
Baseline: Full (non-role-based) textual model generation & benchmark.
Uses the same representations and evaluation as generate_with_roles.py for fair comparison.
Does NOT depend on deepagents or enhanced_agentic_miner.py.
"""

import sys
import json
import argparse
import pickle
from pathlib import Path
from datetime import datetime

import pm4py
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ablation_study import (
    RepresentationDiscoverer,
    AblationMiner,
    AblationMiningConfig,
    load_interpretation_guide,
    sample_valid_traces,
    generate_invalid_traces,
    evaluate_model,
)
from dev_automation.generate_with_roles import ROLE_REPRESENTATIONS


def main():
    parser = argparse.ArgumentParser(description="Baseline full-model benchmark")
    parser.add_argument("--dataset", type=str, default="hiring")
    parser.add_argument("--xes", type=str, default=None)
    parser.add_argument("--model", type=str, default="openai:gpt-5-mini")
    parser.add_argument("--validator-model", type=str, default="gpt-5-mini")
    parser.add_argument("--k-valid", type=int, default=100)
    parser.add_argument("--k-invalid", type=int, default=100)
    args = parser.parse_args()

    xes_path = args.xes or f"benchmark/{args.dataset}/{args.dataset}_log_high.xes"
    if not Path(xes_path).exists():
        print(f"ERROR: {xes_path} not found")
        return 1

    dataset_name = Path(xes_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("dev_automation") / "baseline_runs" / f"run_{dataset_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"BASELINE FULL-MODEL BENCHMARK")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Model: {args.model}")
    print(f"Validator: {args.validator_model}")
    print(f"Output: {output_dir}")

    # Load log & discover representations (same set as role-based)
    print(f"\nLoading event log: {xes_path}")
    log = pm4py.read_xes(xes_path)

    discoverer = RepresentationDiscoverer(log, xes_path)
    for rep_type in ROLE_REPRESENTATIONS:
        try:
            print(f"  Discovering {rep_type.value}...", end=" ")
            discoverer._discover_one(rep_type)
            discoverer.interpretation_guides[rep_type] = load_interpretation_guide(rep_type)
            print("OK")
        except Exception as e:
            print(f"FAILED: {e}")
            discoverer.representations[rep_type] = None
            discoverer.string_representations[rep_type] = f"[Discovery failed: {e}]"

    # Generate full textual model (single model, no roles)
    print(f"\n{'='*60}")
    print("GENERATING FULL TEXTUAL MODEL")
    print(f"{'='*60}")

    mining_config = AblationMiningConfig(
        model_name=args.model,
        prompt_file="GENERATE_AS_IS_MODEL.txt",
    )
    miner = AblationMiner(discoverer, mining_config, output_dir)
    textual_model, model_folder = miner.generate_textual_model(
        ROLE_REPRESENTATIONS, "full_model"
    )
    print(f"  Full model: {len(textual_model)} chars")

    # Generate traces
    print(f"\n{'='*60}")
    print("GENERATING TRACES")
    print(f"{'='*60}")
    valid_traces = sample_valid_traces(xes_path, args.k_valid)
    invalid_traces = generate_invalid_traces(valid_traces, args.k_invalid)
    print(f"  {len(valid_traces)} valid + {len(invalid_traces)} invalid traces")

    traces_dir = output_dir / "traces"
    traces_dir.mkdir(exist_ok=True)
    with open(traces_dir / "valid_traces.pkl", "wb") as f:
        pickle.dump(valid_traces, f)
    with open(traces_dir / "invalid_traces.pkl", "wb") as f:
        pickle.dump(invalid_traces, f)

    # Evaluate
    print(f"\n{'='*60}")
    print("EVALUATING FULL MODEL")
    print(f"{'='*60}")
    metrics = evaluate_model(textual_model, valid_traces, invalid_traces,
                             validator_model=args.validator_model)

    results = {
        "_meta": {
            "xes_path": xes_path,
            "dataset": dataset_name,
            "model_name": args.model,
            "validator_model": args.validator_model,
            "start_time": timestamp,
            "end_time": datetime.now().isoformat(),
            "status": "completed",
        },
        "full_model": {
            "length_chars": len(textual_model),
            "path": str(model_folder / "textual_model.txt"),
        },
        "benchmark": metrics,
    }
    (output_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))

    print(f"\n{'='*60}")
    print("BASELINE RESULTS")
    print(f"{'='*60}")
    print(f"F1={metrics['f1']:.3f} | P={metrics['precision']:.3f} | R={metrics['recall']:.3f}")
    print(f"TP={metrics['TP']} FP={metrics['FP']} TN={metrics['TN']} FN={metrics['FN']}")
    print(f"Results saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
