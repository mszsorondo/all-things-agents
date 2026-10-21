#!/usr/bin/env python3
"""
Role-Based Textual Model Generation & Benchmark

Generates one textual AS-IS model per logical role in the process, then evaluates
the concatenated model via trace classification (same benchmark as phase5_benchmark.py).

The agent receives all process representations + organizational context (resource-activity
mapping, handover of work network) and is asked to:
  1. Cluster raw resources into logical roles
  2. Generate one AS-IS textual model per role via tool calls

The hypothesis: if concatenated role models match full-model F1, knowledge is decomposable
and each role model can serve as context for an autonomous agent in production.

Suggested by Paulo (co-director) as a bridge between process mining and multi-agent deployment.
"""

import os
import sys
import json
import time
import random
import pickle
import hashlib
import argparse
from pathlib import Path
from datetime import datetime
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import pm4py
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent

load_dotenv()

# ---------------------------------------------------------------------------
# Import from parent codebase (read-only — we never modify these files)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ablation_study import (
    RepresentationType,
    RepresentationDiscoverer,
    EXCLUDED_FROM_ABLATION,
    load_interpretation_guide,
    sample_valid_traces,
    generate_invalid_traces,
    evaluate_model,
    df_to_key_value_rows,
    create_validator,
)

# ============================================================================
# REPRESENTATIONS TO USE
# ============================================================================
# These are the representations requested for role-based generation.
# Handover of work is discovered separately (it's organizational, not process).

ROLE_REPRESENTATIONS = [
    RepresentationType.ABSTRACT_VARIANTS,
    RepresentationType.DFG,
    RepresentationType.PROCESS_TREE,
    RepresentationType.LOG_ATTRIBUTES,
    RepresentationType.PETRI_NET_INDUCTIVE,
    RepresentationType.DECLARE,
    RepresentationType.LOG_SKELETON,
]


# ============================================================================
# ORGANIZATIONAL CONTEXT DISCOVERY
# ============================================================================

def detect_resource_column(log) -> str:
    """Detect which column holds resource information."""
    candidates = ["org:resource", "resource", "org:role", "org:group"]
    for col in candidates:
        if col in log.columns:
            return col
    raise ValueError(
        f"No resource column found in log. Columns: {list(log.columns)}. "
        "The event log must have resource information for role-based generation."
    )


def build_resource_activity_mapping(log, resource_col: str) -> Dict[str, List[str]]:
    """Map each resource to its unique activities."""
    mapping = {}
    grouped = log.groupby(resource_col)["concept:name"].unique()
    for resource, activities in grouped.items():
        mapping[str(resource)] = sorted(activities.tolist())
    return mapping


def build_handover_of_work_string(log, resource_col: str) -> str:
    """Discover handover of work network and format as text."""
    try:
        hw = pm4py.discover_handover_of_work_network(
            log, resource_key=resource_col
        )
        lines = ["Handover of Work Network (resource → resource: weight):"]
        # Sort by weight descending for readability
        sorted_connections = sorted(
            hw.connections.items(), key=lambda x: x[1], reverse=True
        )
        for (src, tgt), weight in sorted_connections:
            if weight > 0.001:  # Skip near-zero connections
                lines.append(f"  {src} → {tgt}: {weight:.4f}")
        return "\n".join(lines)
    except Exception as e:
        return f"[Handover of work discovery failed: {e}]"


def build_organizational_context(log, resource_col: str) -> str:
    """Build the organizational context block for the LLM prompt."""
    # Resource-activity mapping
    mapping = build_resource_activity_mapping(log, resource_col)
    mapping_lines = ["Resource-Activity Mapping (which resources perform which activities):"]
    for resource, activities in sorted(mapping.items()):
        mapping_lines.append(f"  {resource}: {', '.join(activities)}")

    # Handover of work
    handover = build_handover_of_work_string(log, resource_col)

    # Resource frequency
    resource_counts = log[resource_col].value_counts()
    freq_lines = ["Resource Event Frequency (how many events each resource performs):"]
    for resource, count in resource_counts.items():
        freq_lines.append(f"  {resource}: {count} events")

    return f"""
{'='*60}
ORGANIZATIONAL CONTEXT
{'='*60}

{chr(10).join(mapping_lines)}

{handover}

{chr(10).join(freq_lines)}

IMPORTANT: Some resources above may represent individual instances of the same
logical role (e.g., "Loan Officer 1", "Loan Officer 2" are both "Loan Officer").
You must cluster resources into logical roles based on their activity patterns
and names before generating role models.
"""


# ============================================================================
# SYSTEM PROMPT FOR ROLE-BASED GENERATION
# ============================================================================

ROLE_GENERATION_SYSTEM_PROMPT = """You are a Process Model Translator specialized in converting process mining outputs into role-based textual process descriptions.

Your task has TWO parts:

## PART 1: ROLE IDENTIFICATION

Analyze the organizational context (resource-activity mapping, handover of work network) and cluster the raw resources into LOGICAL ROLES. Multiple resources that perform the same activities or have similar names should be grouped into one role.

For example:
- "Loan Officer 1", "Loan Officer 2", "Loan Officer 3" → one role: "Loan Officer"
- "Strict UW1", "Strict UW2", "Chill UW1" → could be one role: "Underwriter" or two: "Strict Underwriter", "Lenient Underwriter" — use your judgment based on whether their activity patterns differ

## PART 2: PER-ROLE MODEL GENERATION

For EACH logical role you identified, generate a comprehensive AS-IS textual model describing the process FROM THAT ROLE'S PERSPECTIVE. Each model must:

1. Describe what this role does in the process
2. List all activities this role performs
3. Describe when this role gets activated (what triggers their involvement)
4. Describe decision points and criteria this role handles
5. Describe who hands work TO this role and who this role hands work TO (using the handover of work network)
6. Describe all observed variants and paths involving this role
7. Include any rework loops, exceptions, or escalations this role participates in
8. Be self-contained enough that someone assuming this role could understand their responsibilities

CRITICAL BEHAVIORAL FIDELITY RULES:
- Capture ALL observed variants, not just the happy path
- If activities can occur in different orders for this role, explicitly state ALL valid orderings
- If activities are optional or can be skipped, clearly indicate this
- The model describes what ACTUALLY happens (AS-IS), not what should happen ideally
- Do NOT invent activities or behaviors not present in the source data

For EACH role, call the tool:
  write_AS_IS_comprehensive_textual_description_for_role(role_name, content)

Where role_name is the logical role name (e.g., "Loan Officer", "HR Recruiter") and content is the full textual model for that role.

You MUST call this tool once for each logical role you identify. Do not skip any roles.

OUTPUT FORMAT for each role model:
- Title: "AS-IS Process Model — [Role Name]"
- Section 1: Role Overview (purpose, responsibilities)
- Section 2: Activities Performed (complete list with descriptions)
- Section 3: Process Flow (when activated, what happens, decision points)
- Section 4: Interactions (who hands work to this role, who receives work from this role)
- Section 5: Variants and Exceptions (all observed paths)
- Section 6: Key Observations (pain points, inefficiencies observed in the data)
"""

# NOTE: Alternative architecture (preferred for reliability with many roles):
# Instead of a single agent call that generates all role models at once,
# do a two-step approach:
#   Step 1: One LLM call to identify and list logical roles (with justification)
#   Step 2: One LLM call PER role, with the full process context + role-specific
#           focus instructions. This avoids context degradation when the agent
#           has to generate 5-10 detailed models in sequence.
# This alternative is more expensive but more reliable for processes with many roles.
# To implement: split run_role_agent() into identify_roles() + generate_role_model(role).


# ============================================================================
# ROLE-BASED MINER
# ============================================================================

class RoleBasedMiner:
    """Generates per-role textual models using a single LLM agent call."""

    def __init__(self, discoverer: RepresentationDiscoverer, log,
                 resource_col: str, model_name: str = "openai:gpt-5-mini"):
        self.discoverer = discoverer
        self.log = log
        self.resource_col = resource_col
        self.model_name = model_name
        self.role_models: Dict[str, str] = {}  # role_name -> textual model content

    def run(self, output_dir: Path) -> Dict[str, str]:
        """
        Run the agent to generate per-role textual models.

        Returns:
            Dict mapping role_name -> textual model content
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        self.role_models = {}

        # Build the user prompt: process representations + organizational context
        process_prompt = self.discoverer.get_full_prompt_for_representations(ROLE_REPRESENTATIONS)
        org_context = build_organizational_context(self.log, self.resource_col)
        user_prompt = process_prompt + "\n\n" + org_context

        # Save prompts for reproducibility
        (output_dir / "system_prompt.txt").write_text(ROLE_GENERATION_SYSTEM_PROMPT)
        (output_dir / "user_prompt.txt").write_text(user_prompt)

        # Setup tool — captures role models into self.role_models
        def write_AS_IS_comprehensive_textual_description_for_role(role_name: str, content: str):
            """Write the AS-IS textual model for a specific role in the process.

            Args:
                role_name: The logical role name (e.g., "Loan Officer", "HR Recruiter").
                content: The complete textual AS-IS process model from this role's perspective.
            """
            self.role_models[role_name] = content
            # Save individual role model file
            safe_name = role_name.replace(" ", "_").replace("/", "_").lower()
            filepath = output_dir / f"role_{safe_name}.txt"
            filepath.write_text(f"AS-IS Process Model — {role_name}\n{'='*60}\n\n{content}")
            print(f"  [TOOL CALL] Saved role model: {role_name} → {filepath.name} ({len(content)} chars)")

        tools = [write_AS_IS_comprehensive_textual_description_for_role]

        # Create LLM agent
        llm = self._create_llm()

        # For providers that need a model object, pass it; otherwise pass model name
        if self._requires_model_object():
            graph = create_react_agent(
                model=llm,
                tools=tools,
                prompt=ROLE_GENERATION_SYSTEM_PROMPT,
            )
        else:
            graph = create_react_agent(
                model=self.model_name,
                tools=tools,
                prompt=ROLE_GENERATION_SYSTEM_PROMPT,
            )

        inputs = {"messages": [{"role": "user", "content": user_prompt}]}

        print(f"  Running role-based agent ({self.model_name})...")
        print(f"  User prompt: {len(user_prompt)} chars")
        start_time = time.time()

        # Stream agent — log chunks to file
        log_file = output_dir / "agent_logs.txt"
        with open(log_file, "w", encoding="utf-8") as lf:
            lf.write(f"Agent run started: {datetime.now().isoformat()}\n")
            lf.write(f"Model: {self.model_name}\n")
            lf.write(f"{'='*80}\n\n")

            for chunk in graph.stream(inputs, stream_mode="updates"):
                chunk_str = json.dumps(chunk, indent=2, default=str) if isinstance(chunk, dict) else str(chunk)
                lf.write(f"[{datetime.now().isoformat()}]\n{chunk_str}\n{'-'*40}\n")
                lf.flush()

        elapsed = time.time() - start_time
        print(f"  Agent completed in {elapsed:.1f}s — detected {len(self.role_models)} roles")

        if not self.role_models:
            raise RuntimeError(
                "Agent did not generate any role models. "
                "Check agent_logs.txt for details."
            )

        # Save metadata
        metadata = {
            "model_name": self.model_name,
            "elapsed_seconds": elapsed,
            "roles_detected": list(self.role_models.keys()),
            "num_roles": len(self.role_models),
            "resource_column": self.resource_col,
            "raw_resources": sorted(self.log[self.resource_col].unique().tolist()),
            "representations_used": [r.value for r in ROLE_REPRESENTATIONS],
            "timestamp": datetime.now().isoformat(),
        }
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

        return self.role_models

    def _create_llm(self):
        """Create LLM client."""
        model_name = self.model_name
        if ":" in model_name:
            provider, model_id = model_name.split(":", 1)
        else:
            provider, model_id = "openai", model_name

        if "anthropic" in provider.lower():
            return ChatAnthropic(model=model_id)
        elif "grok" in provider.lower() or "xai" in provider.lower():
            return ChatOpenAI(
                model=model_id,
                api_key=os.getenv("XAI_API_KEY"),
                base_url="https://api.x.ai/v1",
            )
        elif "deepseek" in provider.lower():
            return ChatOpenAI(
                model=model_id,
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com",
            )
        else:
            return ChatOpenAI(model=model_id)

    def _requires_model_object(self) -> bool:
        model_lower = self.model_name.lower()
        return "grok" in model_lower or "xai" in model_lower or "deepseek" in model_lower


# ============================================================================
# MODEL CONCATENATION
# ============================================================================

def concatenate_role_models(role_models: Dict[str, str]) -> str:
    """Concatenate per-role models into a single textual model for benchmark evaluation."""
    parts = []
    for i, (role_name, content) in enumerate(sorted(role_models.items()), 1):
        parts.append(
            f"{'='*60}\n"
            f"ROLE {i}: {role_name}\n"
            f"{'='*60}\n\n"
            f"{content}\n"
        )
    header = (
        "This process model is organized by ROLE. Each section describes the process "
        "from the perspective of one participant. Together, all sections describe the "
        "complete end-to-end process.\n\n"
    )
    return header + "\n".join(parts)


# ============================================================================
# BENCHMARK RUNNER
# ============================================================================

class RoleBenchmark:
    """Runs the trace classification benchmark on role-based models."""

    def __init__(self, xes_path: str, output_dir: Path, config: dict):
        self.xes_path = xes_path
        self.output_dir = output_dir
        self.config = config
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {
            "_meta": {
                "xes_path": xes_path,
                "dataset": Path(xes_path).stem,
                "start_time": datetime.now().isoformat(),
                "config": config,
            }
        }

    def run(self):
        """Run the full pipeline: discover → generate roles → benchmark."""

        # Step 1: Load log
        print(f"\nLoading event log: {self.xes_path}")
        log = pm4py.read_xes(self.xes_path)
        resource_col = detect_resource_column(log)
        print(f"Resource column: '{resource_col}' — {log[resource_col].nunique()} unique resources")

        # Step 2: Discover process representations
        print("\nDiscovering process representations...")
        discoverer = RepresentationDiscoverer(log, self.xes_path)
        # Only discover the representations we need (not all 13)
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

        # Step 3: Generate per-role models
        print(f"\n{'='*60}")
        print("GENERATING ROLE-BASED TEXTUAL MODELS")
        print(f"{'='*60}")

        generation_dir = self.output_dir / "role_generation"
        miner = RoleBasedMiner(
            discoverer=discoverer,
            log=log,
            resource_col=resource_col,
            model_name=self.config["model_name"],
        )
        role_models = miner.run(generation_dir)

        # Save role info to results
        self.results["roles"] = {
            "detected_roles": list(role_models.keys()),
            "num_roles": len(role_models),
            "role_model_lengths": {r: len(c) for r, c in role_models.items()},
        }
        self._save_results()

        # Step 4: Concatenate
        concatenated_model = concatenate_role_models(role_models)
        concat_path = self.output_dir / "concatenated_textual_model.txt"
        concat_path.write_text(concatenated_model)
        print(f"\nConcatenated model: {len(concatenated_model)} chars → {concat_path.name}")

        self.results["concatenated_model"] = {
            "path": str(concat_path),
            "length_chars": len(concatenated_model),
        }
        self._save_results()

        # Step 5: Generate traces
        print(f"\n{'='*60}")
        print("GENERATING TRACES")
        print(f"{'='*60}")

        k_valid = self.config["k_valid"]
        k_invalid = self.config["k_invalid"]
        valid_traces = sample_valid_traces(self.xes_path, k_valid)
        invalid_traces = generate_invalid_traces(valid_traces, k_invalid)
        print(f"  {len(valid_traces)} valid + {len(invalid_traces)} invalid traces")

        # Save traces
        traces_dir = self.output_dir / "traces"
        traces_dir.mkdir(exist_ok=True)
        with open(traces_dir / "valid_traces.pkl", "wb") as f:
            pickle.dump(valid_traces, f)
        with open(traces_dir / "invalid_traces.pkl", "wb") as f:
            pickle.dump(invalid_traces, f)

        # Step 6: Evaluate concatenated model
        print(f"\n{'='*60}")
        print("EVALUATING CONCATENATED ROLE MODEL")
        print(f"{'='*60}")

        validator_model = self.config["validator_model"]
        concat_metrics = evaluate_model(
            concatenated_model, valid_traces, invalid_traces,
            validator_model=validator_model,
        )
        self.results["concatenated_benchmark"] = concat_metrics
        self._save_results()
        print(f"\n  Concatenated model F1: {concat_metrics['f1']:.3f}")

        # Step 7: (Optional) Evaluate each role model in isolation
        if self.config.get("evaluate_isolated", False):
            print(f"\n{'='*60}")
            print("EVALUATING ISOLATED ROLE MODELS")
            print(f"{'='*60}")

            self.results["isolated_benchmarks"] = {}
            for role_name, role_content in role_models.items():
                print(f"\n--- Evaluating: {role_name} ---")
                role_metrics = evaluate_model(
                    role_content, valid_traces, invalid_traces,
                    validator_model=validator_model,
                )
                self.results["isolated_benchmarks"][role_name] = role_metrics
                self._save_results()
                print(f"  {role_name} F1: {role_metrics['f1']:.3f}")

        # Finalize
        self.results["_meta"]["end_time"] = datetime.now().isoformat()
        self.results["_meta"]["status"] = "completed"
        self._save_results()

        self._print_summary()
        return self.results

    def _save_results(self):
        (self.output_dir / "results.json").write_text(
            json.dumps(self.results, indent=2, default=str)
        )

    def _print_summary(self):
        print(f"\n{'='*60}")
        print("ROLE-BASED BENCHMARK SUMMARY")
        print(f"{'='*60}")
        print(f"Dataset: {Path(self.xes_path).stem}")
        print(f"Model: {self.config['model_name']}")
        print(f"Validator: {self.config['validator_model']}")
        print(f"Roles detected: {self.results['roles']['num_roles']}")
        for role, length in self.results["roles"]["role_model_lengths"].items():
            print(f"  - {role} ({length} chars)")

        concat = self.results.get("concatenated_benchmark", {})
        if concat:
            print(f"\nConcatenated Model:")
            print(f"  F1={concat['f1']:.3f} | P={concat['precision']:.3f} | R={concat['recall']:.3f}")
            print(f"  TP={concat['TP']} FP={concat['FP']} TN={concat['TN']} FN={concat['FN']}")

        isolated = self.results.get("isolated_benchmarks", {})
        if isolated:
            print(f"\nIsolated Role Models:")
            for role, metrics in isolated.items():
                print(f"  {role}: F1={metrics['f1']:.3f} | P={metrics['precision']:.3f} | R={metrics['recall']:.3f}")

        print(f"\nResults saved to: {self.output_dir}")
        print(f"{'='*60}")


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Role-Based Textual Model Generation & Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on hiring dataset (default)
  python dev_automation/generate_with_roles.py --dataset hiring

  # Run on all datasets
  python dev_automation/generate_with_roles.py --all

  # Specify models
  python dev_automation/generate_with_roles.py --dataset lending --model openai:gpt-4o

  # Also evaluate each role model in isolation
  python dev_automation/generate_with_roles.py --dataset hiring --evaluate-isolated

  # Custom XES file
  python dev_automation/generate_with_roles.py --xes benchmark/hospital/hospital_log_high.xes
""",
    )
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset name (hiring, hospital, lending, renting)")
    parser.add_argument("--xes", type=str, default=None,
                        help="Path to XES file (overrides --dataset)")
    parser.add_argument("--all", action="store_true",
                        help="Run on all four benchmark datasets")
    parser.add_argument("--model", type=str, default="openai:gpt-5-mini",
                        help="LLM for role-based model generation (default: openai:gpt-5-mini)")
    parser.add_argument("--validator-model", type=str, default="gpt-5-mini",
                        help="LLM for trace validation (default: gpt-5-mini)")
    parser.add_argument("--k-valid", type=int, default=100,
                        help="Number of valid traces (default: 100)")
    parser.add_argument("--k-invalid", type=int, default=100,
                        help="Number of invalid traces (default: 100)")
    parser.add_argument("--evaluate-isolated", action="store_true",
                        help="Also evaluate each role model in isolation")
    return parser.parse_args()


def resolve_xes_paths(args) -> List[str]:
    """Resolve which XES files to process."""
    if args.xes:
        return [args.xes]
    if args.all:
        return [
            "benchmark/hiring/hiring_log_high.xes",
            "benchmark/hospital/hospital_log_high.xes",
            "benchmark/lending/lending_log_high.xes",
            "benchmark/renting/renting_log_high.xes",
        ]
    dataset = args.dataset or "hiring"
    return [f"benchmark/{dataset}/{dataset}_log_high.xes"]


def main():
    args = parse_args()
    xes_paths = resolve_xes_paths(args)

    config = {
        "model_name": args.model,
        "validator_model": args.validator_model,
        "k_valid": args.k_valid,
        "k_invalid": args.k_invalid,
        "evaluate_isolated": args.evaluate_isolated,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("ROLE-BASED TEXTUAL MODEL GENERATION & BENCHMARK")
    print("=" * 60)
    print(f"Model: {config['model_name']}")
    print(f"Validator: {config['validator_model']}")
    print(f"Datasets: {[Path(p).stem for p in xes_paths]}")
    print(f"Traces: {config['k_valid']} valid, {config['k_invalid']} invalid")
    print(f"Evaluate isolated: {config['evaluate_isolated']}")

    all_results = {}

    for xes_path in xes_paths:
        if not Path(xes_path).exists():
            print(f"\nERROR: XES file not found: {xes_path}")
            continue

        dataset_name = Path(xes_path).stem
        output_dir = Path("dev_automation") / "role_benchmark_runs" / f"run_{dataset_name}_{timestamp}"

        print(f"\n{'#'*60}")
        print(f"# DATASET: {dataset_name}")
        print(f"# Output: {output_dir}")
        print(f"{'#'*60}")

        benchmark = RoleBenchmark(xes_path, output_dir, config)
        results = benchmark.run()
        all_results[dataset_name] = results

    # Save combined results if multiple datasets
    if len(all_results) > 1:
        combined_dir = Path("dev_automation") / "role_benchmark_runs" / f"combined_{timestamp}"
        combined_dir.mkdir(parents=True, exist_ok=True)
        (combined_dir / "all_results.json").write_text(
            json.dumps(all_results, indent=2, default=str)
        )
        print(f"\nCombined results saved to: {combined_dir / 'all_results.json'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
