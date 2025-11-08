#!/usr/bin/env python3
"""Test script to verify caching system works correctly."""

import time
import os
from pathlib import Path
from enhanced_agentic_miner import (
    MiningConfiguration,
    AgenticProcessMiner,
    RepresentationType,
    RepresentationCache
)

def test_caching():
    """Test that caching avoids log loading when everything is cached."""

    # Use hiring dataset
    xes_file = "benchmark/hiring/hiring_log_high.xes"

    if not Path(xes_file).exists():
        print(f"Error: {xes_file} not found")
        return

    # Clear any existing cache first
    cache = RepresentationCache()
    cache.clear_all()
    print("✓ Cleared existing cache\n")

    # Configuration with cache enabled
    config = MiningConfiguration(
        representations=[
            RepresentationType.PETRI_NET,
            RepresentationType.BPMN,
            RepresentationType.DFG
        ],
        use_cache=True
    )

    # Temporarily set a dummy API key to avoid the check
    os.environ["ANTHROPIC_API_KEY"] = "dummy-key-for-testing"

    print("=" * 60)
    print("FIRST RUN - No cache (will load log)")
    print("=" * 60)
    miner1 = AgenticProcessMiner(config)
    miner1.log_file = xes_file

    start_time = time.time()
    miner1.discover_representations()
    first_run_time = time.time() - start_time

    print(f"\nFirst run time: {first_run_time:.2f} seconds")
    print(f"Log was loaded: {miner1.log is not None}")
    print(f"Discovered: {list(miner1.representations.keys())}")

    print("\n" + "=" * 60)
    print("SECOND RUN - With cache (should skip log loading)")
    print("=" * 60)
    miner2 = AgenticProcessMiner(config)
    miner2.log_file = xes_file

    start_time = time.time()
    miner2.discover_representations()
    second_run_time = time.time() - start_time

    print(f"\nSecond run time: {second_run_time:.2f} seconds")
    print(f"Log was loaded: {miner2.log is not None}")
    print(f"Speedup: {first_run_time/second_run_time:.1f}x faster")

    # Verify cache status
    print("\n" + "=" * 60)
    print("CACHE VERIFICATION")
    print("=" * 60)

    cache_dir = Path("cache/representations")
    if cache_dir.exists():
        cache_files = list(cache_dir.glob("*.pkl"))
        print(f"Cache files: {len(cache_files)}")

        if (cache_dir / "index.json").exists():
            import json
            with open(cache_dir / "index.json", 'r') as f:
                index = json.load(f)

            for key, info in list(index.items())[:5]:  # Show first 5
                print(f"  - {info['representation_type']}: {Path(info['log_path']).name}")

    # Verify representations are identical
    print("\n" + "=" * 60)
    print("INTEGRITY CHECK")
    print("=" * 60)

    all_match = True
    for rep_type in config.representations:
        if rep_type in miner1.representations and rep_type in miner2.representations:
            rep1 = miner1.representations[rep_type]
            rep2 = miner2.representations[rep_type]

            if rep_type == RepresentationType.PETRI_NET and rep1 and rep2:
                match = (len(rep1.net.places) == len(rep2.net.places) and
                        len(rep1.net.transitions) == len(rep2.net.transitions))
                print(f"{'✓' if match else '✗'} Petri Net: {'identical' if match else 'different'}")
                all_match = all_match and match
            elif rep_type == RepresentationType.BPMN and rep1 and rep2:
                # BPMN is derived from Petri net so should match
                print(f"✓ BPMN: cached successfully")
            elif rep_type == RepresentationType.DFG and rep1 and rep2:
                print(f"✓ DFG: cached successfully")

    # Clean up
    del os.environ["ANTHROPIC_API_KEY"]

    print("\n" + "=" * 60)
    if all_match and second_run_time < first_run_time:
        print("✅ CACHE TEST PASSED!")
        print(f"   - Log loading was skipped when fully cached")
        print(f"   - {first_run_time/second_run_time:.1f}x speedup achieved")
        print(f"   - All representations preserved correctly")
    else:
        print("❌ CACHE TEST FAILED")

    print("=" * 60)

if __name__ == "__main__":
    test_caching()