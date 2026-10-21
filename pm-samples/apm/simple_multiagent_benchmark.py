#!/usr/bin/env python3
"""Simple Multi-Agent Benchmark - Based on Phase 5 Testing"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from enhanced_agentic_miner import AgenticProcessMiner, MiningConfiguration, RepresentationType

# Import all utility functions from phase5_benchmark
from phase5_benchmark import (
    sample_valid_traces,
    generate_invalid_traces,
    create_validator,
    evaluate_traces,
    calculate_metrics,
    get_trace_cache_key,
    save_traces_to_cache,
    load_traces_from_cache,
    create_run_folder
)


class SimpleHierarchicalMiner(AgenticProcessMiner):
    """
    Simple Hierarchical Multi-Agent Miner
    Overrides run_agent to implement expert analysis and synthesis
    """

    def run_agent(self, system_prompt: Optional[str] = None):
        """Override run_agent to implement hierarchical multi-agent system"""

        # Create run folder first
        self._create_run_folder(self.dataset_name)

        # Ensure representations are discovered (uses cache if available)
        if not self.representations:
            self.discover_representations()

        # Initialize metrics tracking
        total_tokens = 0
        total_time = 0

        # Expert prompts for different aspects
        expert_prompts = {
            "structure": """You are a Process Structure Expert. Analyze the control flow,
            branching patterns, and activity sequences. Focus on the process skeleton.""",

            "performance": """You are a Process Performance Expert. Identify bottlenecks,
            parallel activities, and efficiency patterns in the process.""",

            "compliance": """You are a Process Compliance Expert. Identify mandatory sequences,
            rules, and constraints that must be followed in the process."""
        }

        expert_analyses = {}

        # Phase 1: Expert Analysis
        for expert_name, expert_prompt in expert_prompts.items():
            start_time = time.time()

            # Build prompt with relevant representations
            prompt_parts = [f"Analyze this process from a {expert_name} perspective:\n"]

            # Add relevant representations based on expert type
            if expert_name == "structure":
                relevant_types = [RepresentationType.PETRI_NET, RepresentationType.BPMN, RepresentationType.PROCESS_TREE]
            elif expert_name == "performance":
                relevant_types = [RepresentationType.DFG, RepresentationType.HEURISTICS_NET]
            else:  # compliance
                relevant_types = [RepresentationType.PETRI_NET, RepresentationType.ABSTRACT_VARIANTS]

            for rep_type in relevant_types:
                if rep_type in self.representations:
                    rep = self.representations[rep_type]
                    prompt_parts.append(f"\n{rep_type.value} Representation:\n{rep.description}\n")

            prompt_parts.append("\nProvide a concise analysis of the process from your expert perspective.")
            breakpoint()
            # Call LLM for expert analysis
            llm = ChatOpenAI(model=self.config.model_name, temperature=0)
            messages = [
                SystemMessage(content=expert_prompt),
                HumanMessage(content="\n".join(prompt_parts))
            ]

            response = llm.invoke(messages)
            expert_analyses[expert_name] = response.content

            # Track metrics
            if hasattr(response, 'response_metadata') and 'token_usage' in response.response_metadata:
                tokens = response.response_metadata['token_usage'].get('total_tokens', 0)
                total_tokens += tokens

            total_time += time.time() - start_time

        # Phase 2: Synthesis
        start_time = time.time()

        synthesis_prompt = """You are the Meta-Coordinator synthesizing expert analyses into a
        unified process model description. Create a comprehensive textual model that integrates
        all expert perspectives."""

        synthesis_parts = ["Synthesize these expert analyses into a unified process model:\n"]

        for expert_name, analysis in expert_analyses.items():
            synthesis_parts.append(f"\n{expert_name.upper()} EXPERT ANALYSIS:")
            synthesis_parts.append(analysis)

        synthesis_parts.append("\n\nCreate a comprehensive textual process model description.")

        # Call LLM for synthesis
        llm = ChatOpenAI(model=self.config.model_name, temperature=0)
        messages = [
            SystemMessage(content=synthesis_prompt),
            HumanMessage(content="\n".join(synthesis_parts))
        ]

        response = llm.invoke(messages)
        final_model = response.content

        # Track synthesis metrics
        if hasattr(response, 'response_metadata') and 'token_usage' in response.response_metadata:
            tokens = response.response_metadata['token_usage'].get('total_tokens', 0)
            total_tokens += tokens

        total_time += time.time() - start_time

        # Save outputs
        output_path = self.run_folder / "discovered_model.txt"
        output_path.write_text(final_model)

        # Also save as textual_model.txt for compatibility with benchmark
        textual_path = self.run_folder / "textual_model.txt"
        textual_path.write_text(final_model)

        # Save expert analyses
        experts_folder = self.run_folder / "expert_analyses"
        experts_folder.mkdir(exist_ok=True)
        for expert_name, analysis in expert_analyses.items():
            expert_file = experts_folder / f"{expert_name}_analysis.txt"
            expert_file.write_text(analysis)

        # Save metrics
        metrics = {
            "total_tokens": total_tokens,
            "total_time": total_time,
            "num_experts": len(expert_prompts),
            "architecture": "SimpleHierarchicalMiner"
        }

        metrics_path = self.run_folder / "multiagent_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2))

        # Store metrics for later retrieval
        self.multiagent_metrics = metrics

        print(f"[SimpleHierarchicalMiner] Total tokens: {total_tokens}, Time: {total_time:.2f}s")

        return final_model


def get_as_is_model(xes_path, precomputed_model_path=None, miner_class=None):
    """Generate AS-IS model using specified miner class"""
    if precomputed_model_path:
        return Path(precomputed_model_path).read_text()

    # Default to original AgenticProcessMiner if no class specified
    if miner_class is None:
        miner_class = AgenticProcessMiner

    config = MiningConfiguration(
        representations=[RepresentationType.PETRI_NET, RepresentationType.DFG, RepresentationType.ABSTRACT_VARIANTS],
        model_name="gpt-4o-mini",  # Using same model as original
        prompt_file="ONLY_GENERATE_TEXTUAL_MODEL.txt",
        use_cache=True
    )

    miner = miner_class(config)
    miner.log_file = xes_path
    miner.xes_path = xes_path
    miner.dataset_name = Path(xes_path).stem

    # Discover representations first (will use cache if available)
    miner.discover_representations()
    breakpoint()
    # Run the agent
    miner.run_agent()

    return (Path(miner.run_folder) / "textual_model.txt").read_text()




def main():
    """Main benchmark execution"""
    # Create custom run folder for multiagent
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = Path(f"multiagent_runs/run_{timestamp}")
    run_folder.mkdir(parents=True, exist_ok=True)

    # Configuration
    config = {
        "model": "gpt-4o-mini",
        "k_valid": 100,
        "k_invalid": 100,
        "perturbations": ["random_swap_events"],
        "use_cache": True,
        "architectures": ["single", "hierarchical"]  # Which architectures to test
    }

    problems = [
        "benchmark/hiring/hiring_log_high.xes",
        "benchmark/hospital/hospital_log_high.xes",
        "benchmark/lending/lending_log_high.xes",
        "benchmark/renting/renting_log_high.xes"
    ]

    # Save config
    (run_folder / "config.json").write_text(json.dumps(config, indent=2))

    results = {}

    for xes_path in problems:
        problem_name = Path(xes_path).stem
        print(f"\n{'='*80}")
        print(f"Problem: {problem_name}")
        print(f"{'='*80}")

        problem_results = {}

        # Load or generate traces (shared across architectures)
        cache_key = get_trace_cache_key(xes_path, config["k_valid"], config["k_invalid"], config["perturbations"])
        valid_traces, invalid_traces = load_traces_from_cache(cache_key)

        if valid_traces is None or invalid_traces is None:
            print(f"[{problem_name}] Generating traces...")
            valid_traces = sample_valid_traces(xes_path, k=config["k_valid"])
            invalid_traces = generate_invalid_traces(valid_traces, k=config["k_invalid"])
            if config["use_cache"]:
                save_traces_to_cache(cache_key, valid_traces, invalid_traces)
        else:
            print(f"[{problem_name}] Loaded {len(valid_traces)} valid + {len(invalid_traces)} invalid traces from cache")

        # Test each architecture
        for arch in config["architectures"]:
            print(f"\n--- Testing {arch} architecture ---")

            # Generate model with appropriate miner
            if arch == "single":
                miner_class = AgenticProcessMiner
            elif arch == "hierarchical":
                miner_class = SimpleHierarchicalMiner
            else:
                continue

            # Generate AS-IS model
            print(f"Generating {arch} model...")
            model_start = time.time()
            as_is_model = get_as_is_model(xes_path, miner_class=miner_class)
            model_time = time.time() - model_start

            # Save model
            model_path = run_folder / f"{problem_name}_{arch}_model.txt"
            model_path.write_text(as_is_model)

            # Create validator and evaluate
            validator, system_prompt = create_validator(as_is_model, model_name=config["model"])

            print(f"Evaluating traces with {arch} model...")
            valid_preds, invalid_preds, compute_time, tokens = evaluate_traces(
                valid_traces, invalid_traces, validator
            )

            # Calculate metrics
            metrics = calculate_metrics(valid_preds, invalid_preds, compute_time, tokens)
            metrics["model_generation_time"] = model_time
            metrics["architecture"] = arch

            problem_results[arch] = metrics

            print(f"\n{arch} Results:")
            print(f"  F1={metrics['f1']:.3f} | P={metrics['precision']:.3f} | R={metrics['recall']:.3f}")
            print(f"  Tokens={metrics['total_tokens']} | Time={metrics['total_compute_time_seconds']:.1f}s")

        results[problem_name] = problem_results

        # Save intermediate results
        (run_folder / "results.json").write_text(json.dumps(results, indent=2))

    # Print final summary
    print(f"\n{'='*80}")
    print("BENCHMARK RESULTS SUMMARY")
    print(f"{'='*80}")

    for problem_name, problem_results in results.items():
        print(f"\n{problem_name}:")
        for arch, metrics in problem_results.items():
            print(f"  {arch}:")
            print(f"    F1={metrics['f1']:.3f} | P={metrics['precision']:.3f} | R={metrics['recall']:.3f}")
            print(f"    Tokens={metrics['total_tokens']} | Time={metrics['total_compute_time_seconds']:.1f}s")

    print(f"\n{'='*80}")
    print(f"Results saved to {run_folder}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()