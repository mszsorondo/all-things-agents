#!/usr/bin/env python3
"""
Test script for the dummy pipeline using the full infrastructure.
Tests the complete pipeline with a small synthetic log.
"""

import os
from pathlib import Path
from enhanced_agentic_miner import (
    MiningConfiguration,
    AgenticProcessMiner,
    RepresentationType
)

def test_dummy_pipeline():
    """Run the full pipeline on dummy data."""

    # Paths
    dummy_log = "benchmark/dummy/dummy_process.xes"
    dummy_prompt = "dummy_prompt.txt"

    # Verify files exist
    if not Path(dummy_log).exists():
        print(f"Error: {dummy_log} not found")
        return 1

    if not Path(dummy_prompt).exists():
        print(f"Error: {dummy_prompt} not found")
        return 1

    # Check for real API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set in environment!")
        print("Please run: export ANTHROPIC_API_KEY='your-key-here'")
        return 1

    print("=" * 70)
    print("DUMMY PIPELINE TEST (Full Run with Haiku 4.5)")
    print("=" * 70)
    print(f"Log: {dummy_log}")
    print(f"Prompt: {dummy_prompt}")
    print()

    # Create configuration using Haiku 4.5 for cost-effective testing
    config = MiningConfiguration(
        representations=[
            RepresentationType.PETRI_NET,
            RepresentationType.BPMN,
            RepresentationType.DFG
        ],
        model_name="anthropic:claude-haiku-4-5",  # Use Haiku for testing
        prompt_file=dummy_prompt,
        output_dir="runs",
        enable_logging=True,
        use_cache=True
    )

    print("Configuration:")
    print(f"  Representations: {[r.value for r in config.representations]}")
    print(f"  Model: {config.model_name}")
    print(f"  Cache enabled: {config.use_cache}")
    print()

    try:
        # Initialize the miner
        miner = AgenticProcessMiner(config)

        # Set log file path and dataset name (needed even when using cache)
        miner.log_file = dummy_log
        miner.xes_path = dummy_log
        miner.dataset_name = Path(dummy_log).stem

        # Discover representations (will use cache if available)
        print("Step 1: Discovering representations...")
        print("-" * 70)
        miner.discover_representations()

        print()
        print("Step 2: Checking discovered representations...")
        print("-" * 70)
        for rep_type, rep in miner.representations.items():
            if rep:
                print(f"  ✓ {rep_type.value}: Discovered successfully")

                # Show basic info
                if rep_type == RepresentationType.PETRI_NET and hasattr(rep, 'net'):
                    print(f"    - Places: {len(rep.net.places)}")
                    print(f"    - Transitions: {len(rep.net.transitions)}")
                elif rep_type == RepresentationType.DFG and rep.representation:
                    if isinstance(rep.representation, dict):
                        dfg = rep.representation.get('dfg', {})
                        print(f"    - Edges: {len(dfg)}")
            else:
                print(f"  ✗ {rep_type.value}: Failed to discover")

        print()
        print("Step 3: Building prompt...")
        print("-" * 70)
        prompt = miner.build_prompt()
        prompt_lines = prompt.split('\n')
        print(f"  Prompt length: {len(prompt)} characters")
        print(f"  Prompt lines: {len(prompt_lines)}")
        print(f"  First 5 lines:")
        for line in prompt_lines[:5]:
            print(f"    {line}")

        # Check if BPMN is truncated
        if "TRUNCATED" in prompt:
            print("  ⚠ WARNING: BPMN appears to be truncated!")
        else:
            print("  ✓ BPMN is complete (no truncation)")

        print()
        print("Step 4: Running agent analysis...")
        print("-" * 70)
        print("  Using Claude Haiku 4.5 (fast and cost-effective)")
        print("  This will make a real API call...")
        print()

        # Actually run the agent!
        miner.run_agent()

        print()
        print("=" * 70)
        print("PIPELINE TEST COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print()
        print("Summary:")
        print(f"  ✓ Log loaded: {miner.log is not None}")
        print(f"  ✓ Representations discovered: {len([r for r in miner.representations.values() if r])}/{len(config.representations)}")
        print(f"  ✓ Prompt built: {len(prompt)} chars")
        print(f"  ✓ Agent analysis completed")
        print(f"  ✓ Results saved to: {miner.run_folder}/")
        print()

        return 0

    except Exception as e:
        print()
        print("=" * 70)
        print(f"ERROR: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()

        return 1

if __name__ == "__main__":
    exit(test_dummy_pipeline())
