#!/usr/bin/env python3
"""
Refinement Benchmark Interface (Terminal)

An interactive terminal interface for configuring and running refinement experiments.
Guides users through each configuration option step by step.

Usage:
    python refine_benchmark_interface.py
"""

import subprocess
import json
import sys
from pathlib import Path
from typing import Optional


# Available LLM models
LLM_MODELS = [
    "openai:gpt-5",
    "openai:gpt-5-mini",
    "openai:gpt-4o",
    "openai:o1",
    "openai:o1-mini",
    "anthropic:claude-sonnet-4-20250514",
    "anthropic:claude-opus-4-5-20251101",
    "anthropic:claude-3-5-haiku-20241022",
    "google:gemini-2.0-flash",
    "google:gemini-1.5-pro",
]


def clear_screen():
    """Clear terminal screen."""
    print("\033[2J\033[H", end="")


def print_header(title: str):
    """Print a section header."""
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)
    print()


def print_step(step_num: int, total: int, title: str, description: str):
    """Print step header with description."""
    print()
    print(f"─── Step {step_num}/{total}: {title} ───")
    print()
    print(f"  {description}")
    print()


def select_option(options: list, default: int = 0, prompt: str = "Select option") -> int:
    """
    Display numbered options and get user selection.
    Returns the index of selected option.
    """
    for i, opt in enumerate(options):
        marker = " [default]" if i == default else ""
        print(f"  {i + 1}. {opt}{marker}")
    print()

    while True:
        try:
            user_input = input(f"{prompt} [1-{len(options)}]: ").strip()
            if user_input == "":
                return default
            choice = int(user_input)
            if 1 <= choice <= len(options):
                return choice - 1
            print(f"  Please enter a number between 1 and {len(options)}")
        except ValueError:
            print(f"  Please enter a valid number")


def select_yes_no(prompt: str, default: bool = False) -> bool:
    """Ask yes/no question."""
    default_str = "Y/n" if default else "y/N"
    while True:
        user_input = input(f"{prompt} [{default_str}]: ").strip().lower()
        if user_input == "":
            return default
        if user_input in ("y", "yes"):
            return True
        if user_input in ("n", "no"):
            return False
        print("  Please enter 'y' or 'n'")


def input_text(prompt: str, default: str = "") -> str:
    """Get text input with optional default."""
    if default:
        user_input = input(f"{prompt} [{default}]: ").strip()
        return user_input if user_input else default
    else:
        return input(f"{prompt}: ").strip()


def scan_benchmark_runs() -> list:
    """Scan for available benchmark runs."""
    runs = []
    base_paths = [
        Path("benchmark_runs"),
        Path("runs"),
        Path("multiagent_runs"),
        Path("multiagent_benchmark_runs"),
    ]

    for base_path in base_paths:
        if base_path.exists():
            for run_dir in sorted(base_path.iterdir(), reverse=True):
                if run_dir.is_dir():
                    if (run_dir / "results.json").exists() or (run_dir / "config.json").exists():
                        runs.append(run_dir)

    return runs


def scan_interrupted_runs(parent_run: Path) -> list:
    """Scan for interrupted refinement runs under a parent."""
    interrupted = []
    for refined in sorted(parent_run.glob("refined*")):
        if refined.is_dir():
            config_exists = (refined / "config.json").exists()
            results_exists = (refined / "results.json").exists()
            has_models = any(refined.glob("*_textual_model.txt"))
            if config_exists and not results_exists and has_models:
                interrupted.append(refined)
    return interrupted


def get_run_info(run_path: Path) -> dict:
    """Get info about a benchmark run."""
    info = {"path": run_path}

    config_path = run_path / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
            info["config"] = config
            info["llm_model"] = config.get("llm_model", "unknown")
            info["validator_model"] = config.get("validator_model", config.get("model", "unknown"))
        except Exception:
            pass

    results_path = run_path / "results.json"
    if results_path.exists():
        try:
            with open(results_path) as f:
                results = json.load(f)
            info["results"] = results
            # Get F1 from first problem
            for key, val in results.items():
                if isinstance(val, dict) and "llm" in val:
                    info["f1"] = val["llm"].get("f1", 0)
                    break
        except Exception:
            pass

    return info


def format_run_option(run_info: dict) -> str:
    """Format a run for display in selection list."""
    path = run_info["path"]
    parts = [str(path)]

    if "llm_model" in run_info:
        parts.append(f"LLM: {run_info['llm_model']}")
    if "f1" in run_info and run_info["f1"] is not None:
        parts.append(f"F1: {run_info['f1']:.3f}")

    return " | ".join(parts)


def select_model(prompt: str, default: str, description: str) -> str:
    """Select an LLM model with custom option."""
    print(f"  {description}")
    print()

    # Build options list with default at top if it's in the list
    options = list(LLM_MODELS)
    if default and default not in options:
        options.insert(0, default)

    # Find default index
    default_idx = 0
    if default in options:
        default_idx = options.index(default)

    options.append("(enter custom model)")

    for i, opt in enumerate(options):
        marker = " [default]" if i == default_idx else ""
        print(f"  {i + 1}. {opt}{marker}")
    print()

    while True:
        try:
            user_input = input(f"{prompt} [1-{len(options)}]: ").strip()
            if user_input == "":
                return options[default_idx]
            choice = int(user_input)
            if 1 <= choice <= len(options):
                if choice == len(options):  # Custom option
                    custom = input("  Enter custom model (provider:model): ").strip()
                    return custom if custom else default
                return options[choice - 1]
            print(f"  Please enter a number between 1 and {len(options)}")
        except ValueError:
            print(f"  Please enter a valid number")


def run_refinement(cmd: list):
    """Execute the refinement command."""
    print()
    print("=" * 60)
    print("  EXECUTING REFINEMENT")
    print("=" * 60)
    print()
    print(f"  Command: {' '.join(cmd)}")
    print()
    print("-" * 60)
    print()

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        for line in iter(process.stdout.readline, ""):
            print(line, end="")

        process.wait()
        return process.returncode
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        process.terminate()
        return -1
    except Exception as e:
        print(f"\nError: {e}")
        return -1


def main():
    """Main interactive interface."""
    clear_screen()
    print_header("Refinement Benchmark Interface")
    print("  This wizard will guide you through configuring a refinement experiment.")
    print("  Press Enter to accept default values shown in [brackets].")
    print()

    # =========================================================================
    # Step 1: New or Resume
    # =========================================================================
    print_step(1, 8, "Experiment Type",
               "Choose whether to start a new refinement or resume an interrupted one.")

    options = [
        "Start a new refinement experiment",
        "Resume an interrupted experiment"
    ]
    mode = select_option(options, default=0, prompt="Select mode")
    is_resume = (mode == 1)

    # =========================================================================
    # Step 2: Select Benchmark Run
    # =========================================================================
    print_step(2, 8, "Select Benchmark Run",
               "Choose the benchmark run containing the textual model to refine.")

    runs = scan_benchmark_runs()
    if not runs:
        print("  No benchmark runs found in benchmark_runs/, runs/, etc.")
        print("  Please run a benchmark first using phase5_benchmark.py")
        return

    run_infos = [get_run_info(r) for r in runs]
    run_options = [format_run_option(info) for info in run_infos]

    selected_idx = select_option(run_options, default=0, prompt="Select benchmark run")
    selected_run = run_infos[selected_idx]
    run_folder = selected_run["path"]
    config = selected_run.get("config", {})

    print(f"\n  Selected: {run_folder}")

    # Get default models from parent config
    default_refine_model = config.get("llm_model", "openai:gpt-5-mini")
    default_validator_model = config.get("validator_model", config.get("model", "gpt-5-mini"))

    # =========================================================================
    # If resuming, select the interrupted run
    # =========================================================================
    resume_folder = None
    if is_resume:
        print_step(3, 8, "Select Interrupted Run",
                   "Choose which interrupted refinement to resume.")

        interrupted = scan_interrupted_runs(run_folder)
        if not interrupted:
            print("  No interrupted refinements found for this benchmark run.")
            print("  (Interrupted = has config.json and textual models, but no results.json)")
            print()
            if not select_yes_no("Continue with a new refinement instead?", default=True):
                return
            is_resume = False
        else:
            int_options = [str(p) for p in interrupted]
            int_idx = select_option(int_options, default=0, prompt="Select run to resume")
            resume_folder = interrupted[int_idx]
            print(f"\n  Will resume: {resume_folder}")

    # =========================================================================
    # Step 3/4: Refinement Model (skip if resuming)
    # =========================================================================
    refine_model = default_refine_model
    incremental_mode = False
    if not is_resume:
        print_step(3, 8, "Refinement LLM",
                   "Select which LLM will generate the refined textual model.\n"
                   "  This model analyzes misclassified traces and improves the process description.")

        refine_model = select_model(
            "Select refinement model",
            default=default_refine_model,
            description=f"Original model was: {default_refine_model}"
        )
        print(f"\n  Selected: {refine_model}")

        # =========================================================================
        # Step 4: Refinement Mode
        # =========================================================================
        print_step(4, 8, "Refinement Mode",
                   "Choose how the model should be refined.")

        print("  - Standard: The LLM may rewrite the entire model to fix issues")
        print("  - Incremental: Minimal targeted edits, preserving existing structure")
        print()

        mode_options = [
            "Standard refinement (full rewrite allowed)",
            "Incremental refinement (minimal edits only)"
        ]
        mode_idx = select_option(mode_options, default=0, prompt="Select refinement mode")
        incremental_mode = (mode_idx == 1)
        print(f"\n  Selected: {'Incremental' if incremental_mode else 'Standard'} mode")

    # =========================================================================
    # Step 5/6: Trace Sampling
    # =========================================================================
    step = 5 if not is_resume else 4
    print_step(step, 8, "Trace Sampling",
               "Choose how to sample traces for the full benchmark evaluation.\n"
               "  - New sampling (default): Generate fresh traces to test generalization\n"
               "  - Parent's traces: Reuse exact same traces for controlled comparison")

    use_parent_traces = select_yes_no("Use parent's traces instead of new sampling?", default=False)

    # =========================================================================
    # Step 6/7: Evaluation Options
    # =========================================================================
    step = 6 if not is_resume else 5
    print_step(step, 8, "Evaluation Options",
               "Configure how the refined model will be evaluated.")

    print("  After refinement, evaluation happens in two stages:")
    print("    Stage 1: Quick test on previously misclassified traces")
    print("    Stage 2: Full benchmark on all traces (if Stage 1 shows improvement)")
    print()

    eval_options = [
        "Normal: Run Stage 2 only if Stage 1 shows improvement",
        "Skip Stage 2: Only run Stage 1 (quick validation)",
        "Force Stage 2: Run full benchmark even if no improvement"
    ]
    eval_mode = select_option(eval_options, default=0, prompt="Select evaluation mode")

    skip_full = (eval_mode == 1)
    force_full = (eval_mode == 2)

    # =========================================================================
    # Step 7: Output Folder (skip if resuming)
    # =========================================================================
    output_folder = ""
    if not is_resume:
        print_step(7, 8, "Output Folder",
                   "Optionally specify a custom name for the output folder.\n"
                   "  Leave empty for auto-increment (refined, refined_1, refined_2, ...)")

        output_folder = input_text("Output folder name (or press Enter for auto)", default="")

    # =========================================================================
    # Step 8: Confirmation
    # =========================================================================
    print_step(8, 8, "Confirmation",
               "Review your configuration and confirm to start.")

    print("  Configuration Summary:")
    print("  " + "-" * 40)
    print(f"  Benchmark run:     {run_folder}")
    if is_resume:
        print(f"  Resume folder:     {resume_folder}")
    else:
        print(f"  Refinement LLM:    {refine_model}")
        print(f"  Refinement mode:   {'Incremental' if incremental_mode else 'Standard'}")
    print(f"  Validator LLM:     {default_validator_model}")
    print(f"  Trace sampling:    {'Parent traces' if use_parent_traces else 'New sampling'}")
    print(f"  Evaluation mode:   {eval_options[eval_mode]}")
    if output_folder:
        print(f"  Output folder:     {output_folder}")
    print("  " + "-" * 40)
    print()

    # Build command
    cmd = ["python", "refine_benchmark.py", str(run_folder)]

    if is_resume and resume_folder:
        cmd.extend(["--resume", str(resume_folder)])
    else:
        if refine_model != default_refine_model:
            cmd.extend(["--refine-model", refine_model])
        if incremental_mode:
            cmd.append("--incremental")

    if use_parent_traces:
        cmd.append("--use-parent-traces")

    if skip_full:
        cmd.append("--skip-full-benchmark")
    elif force_full:
        cmd.append("--force-full-benchmark")

    if output_folder:
        cmd.extend(["--output", output_folder])

    print(f"  Command: {' '.join(cmd)}")
    print()

    if not select_yes_no("Run this refinement?", default=True):
        print("\n  Cancelled.")
        return

    # Execute
    exit_code = run_refinement(cmd)

    print()
    print("=" * 60)
    if exit_code == 0:
        print("  REFINEMENT COMPLETED SUCCESSFULLY")
    else:
        print(f"  REFINEMENT EXITED WITH CODE {exit_code}")
    print("=" * 60)
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCancelled by user.")
        sys.exit(0)
