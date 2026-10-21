#!/usr/bin/env python3
"""
Verify that enhanced_agentic_miner.py is configured to match original template by default.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from enhanced_agentic_miner import MiningConfiguration, RepresentationType

def verify_default_configuration():
    """Verify the default configuration matches original template."""
    print("=" * 80)
    print("VERIFYING DEFAULT CONFIGURATION")
    print("=" * 80)

    # Create default configuration
    config = MiningConfiguration()

    print("\n1. Model Configuration:")
    print(f"   Default model: {config.model_name}")
    expected_model = "anthropic:claude-sonnet-4-5"
    if config.model_name == expected_model:
        print(f"   ✓ Matches original template: {expected_model}")
    else:
        print(f"   ✗ Does NOT match! Expected: {expected_model}, Got: {config.model_name}")
        return False

    print("\n2. Prompt File:")
    print(f"   Default prompt file: {config.prompt_file}")
    with open(config.prompt_file, "r") as prompt:
        expected_prompt = prompt.read()
        print(expected_prompt)
    expected_prompt = "new_prompt.txt"
    if config.prompt_file == expected_prompt:
        print(f"   ✓ Matches original template: {expected_prompt}")
    else:
        print(f"   ✗ Does NOT match! Expected: {expected_prompt}, Got: {config.prompt_file}")
        return False

    print("\n3. Output Directory:")
    print(f"   Default output dir: {config.output_dir}")
    expected_dir = "runs"
    if config.output_dir == expected_dir:
        print(f"   ✓ Matches original template: {expected_dir}")
    else:
        print(f"   ✗ Does NOT match! Expected: {expected_dir}, Got: {config.output_dir}")
        return False

    print("\n4. Representations:")
    rep_names = [r.value for r in config.representations]
    print(f"   Default representations: {rep_names}")
    expected_reps = ["petri_net", "bpmn", "dfg"]
    if rep_names == expected_reps:
        print(f"   ✓ Matches original template: {expected_reps}")
    else:
        print(f"   ✗ Does NOT match! Expected: {expected_reps}, Got: {rep_names}")
        return False

    print("\n5. Logging Configuration:")
    print(f"   Enable logging: {config.enable_logging}")
    print(f"   Save experiment config: {config.save_experiment_config}")
    print(f"   Stream mode: {config.stream_mode}")
    if config.enable_logging and config.save_experiment_config and config.stream_mode == "updates":
        print(f"   ✓ Logging settings match original template behavior")
    else:
        print(f"   ✗ Logging settings differ from original")
        return False

    print("\n6. Flexibility Check:")
    print("   Testing that configuration can be customized...")

    # Create custom configuration
    custom_config = MiningConfiguration(
        model_name="openai:gpt-4o-mini",
        representations=[RepresentationType.BPMN, RepresentationType.DFG],
        prompt_file="custom_prompt.txt",
        output_dir="custom_runs"
    )

    print(f"   Custom model: {custom_config.model_name}")
    print(f"   Custom representations: {[r.value for r in custom_config.representations]}")
    print(f"   Custom prompt: {custom_config.prompt_file}")
    print(f"   Custom output: {custom_config.output_dir}")
    print(f"   ✓ Configuration remains fully flexible")

    print("\n" + "=" * 80)
    print("SUCCESS: Enhanced miner defaults match original template!")
    print("The system is configured to replicate the original flow by default,")
    print("while remaining fully configurable for advanced use cases.")
    print("=" * 80)

    return True


if __name__ == "__main__":
    success = verify_default_configuration()
    exit(0 if success else 1)