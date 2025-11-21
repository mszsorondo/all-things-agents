"""
Enhanced Phase 5 Benchmark with Multi-Agent Architecture Support
=================================================================

This script extends the Phase 5 benchmark to:
1. Compare multiple architectures (single agent vs multi-agent)
2. Use shared traces and perturbations across architectures
3. Track detailed metrics (tokens, time) for each architecture
4. Provide comparative analysis
"""

import os
import json
import time
import pickle
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import pm4py
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import logging
from dataclasses import dataclass, asdict

# Import benchmark utilities
from benchmark_template import (
    random_swap_events,
    skip_mandatory_activity,
    df_to_key_value_rows
)

# Import miner architectures
from enhanced_agentic_miner import AgenticProcessMiner, MiningConfiguration
from hierarchical_expert_miner import HierarchicalExpertMiner

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Dummy Miner for Testing
class DummyMiner(AgenticProcessMiner):
    """Dummy miner that inherits from AgenticProcessMiner but returns fixed outputs"""

    def run_agent(self, system_prompt: Optional[str] = None):
        """Override run_agent to return dummy output without LLM calls"""
        # Still discover representations to use the cache
        if not self.representations:
            self.discover_representations()

        # Generate dummy model text
        dummy_model = f"""DUMMY MODEL for {self.dataset_name}:
1. Process starts with InitialActivity
2. Then proceeds to either PathA or PathB
3. PathA: Activity1 -> Activity2 -> Activity3
4. PathB: Activity4 -> Activity5
5. Both paths converge at FinalActivity
6. Process ends

Total representations discovered: {len(self.representations)}
Dataset: {self.dataset_name}
"""

        # Write dummy output
        output_path = self.run_folder / "discovered_model.txt"
        output_path.write_text(dummy_model)

        # Set dummy metrics
        self.total_metrics = type('obj', (object,), {
            'tokens_used': 1234,
            'prompt_tokens': 1000,
            'completion_tokens': 234,
            'time_elapsed': 0.5,
            'calls_made': 1
        })()

        return dummy_model

    def get_metrics(self) -> Dict[str, Any]:
        """Return dummy metrics"""
        return {
            "architecture": "DummyMiner",
            "total_tokens": 1234,
            "total_time": 0.5,
            "total_calls": 1,
            "prompt_tokens": 1000,
            "completion_tokens": 234
        }


@dataclass
class ArchitectureMetrics:
    """Metrics for a specific architecture"""
    architecture_name: str
    model_generation_tokens: int = 0
    model_generation_time: float = 0.0
    evaluation_tokens: int = 0
    evaluation_time: float = 0.0
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0


class MultiAgentBenchmark:
    """
    Benchmark system that supports multiple architectures with shared traces
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.architectures = config.get('architectures', [AgenticProcessMiner])
        self.problems = config.get('problems', [])
        self.k_valid = config.get('k_valid', 100)
        self.k_invalid = config.get('k_invalid', 100)
        self.perturbations = config.get('perturbations', ['random_swap'])
        self.model_name = config.get('model_name', 'gpt-4o-mini')
        self.run_folder = self._create_run_folder()
        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)

    def _create_run_folder(self) -> Path:
        """Create a timestamped folder for this benchmark run"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_folder = Path(f"runs/multiagent_{timestamp}")
        run_folder.mkdir(parents=True, exist_ok=True)
        return run_folder

    def get_trace_cache_key(self, xes_path: str) -> str:
        """Generate cache key for traces"""
        xes_stat = os.stat(xes_path)
        key_parts = [
            xes_path,
            str(xes_stat.st_size),
            str(xes_stat.st_mtime),
            str(self.k_valid),
            str(self.k_invalid),
            str(self.perturbations)
        ]
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()

    def get_cached_traces(self, xes_path: str) -> Optional[Tuple[List, List]]:
        """Load cached traces if available"""
        cache_key = self.get_trace_cache_key(xes_path)
        cache_file = self.cache_dir / f"traces_{cache_key}.pkl"

        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                logger.info(f"Loaded cached traces from {cache_file}")
                return cached_data['valid_traces'], cached_data['invalid_traces']
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return None

    def save_cached_traces(self, xes_path: str, valid_traces: List, invalid_traces: List):
        """Save traces to cache"""
        cache_key = self.get_trace_cache_key(xes_path)
        cache_file = self.cache_dir / f"traces_{cache_key}.pkl"

        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'valid_traces': valid_traces,
                    'invalid_traces': invalid_traces
                }, f)
            logger.info(f"Saved traces to cache: {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def generate_traces(self, xes_path: str) -> Tuple[List, List]:
        """Generate or load cached valid and invalid traces"""

        # Try to load from cache first
        cached = self.get_cached_traces(xes_path)
        if cached:
            return cached

        # Load the event log
        log = pm4py.read_xes(xes_path)
        df = pm4py.convert_to_dataframe(log)

        # Sample valid traces
        valid_traces = []
        unique_cases = df['case:case'].unique()

        for _ in range(min(self.k_valid, len(unique_cases))):
            case = df[df['case:case'] == unique_cases[_ % len(unique_cases)]]
            trace_str = df_to_key_value_rows(case)
            valid_traces.append(trace_str)

        # Generate invalid traces through perturbation
        invalid_traces = []
        for i in range(self.k_invalid):
            # Use deepcopy to avoid mutation
            import copy
            case = df[df['case:case'] == unique_cases[i % len(unique_cases)]].copy()

            # Apply perturbation
            if 'random_swap' in self.perturbations:
                case = random_swap_events(case)

            trace_str = df_to_key_value_rows(case)
            invalid_traces.append(trace_str)

        # Save to cache
        self.save_cached_traces(xes_path, valid_traces, invalid_traces)

        return valid_traces, invalid_traces

    def generate_model_with_architecture(self, xes_path: str, architecture_class,
                                        precomputed_path: Optional[str] = None) -> Tuple[str, Dict]:
        """
        Generate process model using specified architecture
        Returns: (model_text, metrics)
        """
        if precomputed_path and os.path.exists(precomputed_path):
            logger.info(f"Loading precomputed model from {precomputed_path}")
            with open(precomputed_path, 'r') as f:
                return f.read(), {"tokens": 0, "time": 0.0}

        logger.info(f"Generating model with {architecture_class.__name__}")

        start_time = time.time()

        # Configure the miner
        config = MiningConfiguration(
            output_dir="temp_output",
            model_name=self.model_name,
            #verbose=False
        )

        # Create miner instance
        miner = architecture_class(config)
        breakpoint()

        # Run the agent
        model_text = miner.run_agent()

        # Get metrics if available
        metrics = {"time": time.time() - start_time}

        if hasattr(miner, 'get_metrics'):
            agent_metrics = miner.get_metrics()
            metrics.update(agent_metrics)
        elif hasattr(miner, 'total_metrics'):
            metrics["tokens"] = miner.total_metrics.tokens_used
            metrics["prompt_tokens"] = miner.total_metrics.prompt_tokens
            metrics["completion_tokens"] = miner.total_metrics.completion_tokens

        return model_text, metrics

    def evaluate_trace(self, trace: str, model_text: str, label: str) -> Dict:
        """
        Evaluate a single trace against the model
        Returns: prediction result with metrics
        """
        # Check if using dummy model for test mode
        if "DUMMY MODEL" in model_text:
            # Dummy evaluation - deterministic based on trace hash
            import random
            random.seed(hash(trace))
            is_correct = random.random() > 0.15  # 85% accuracy
            prediction = label if is_correct else ("INVALID" if label == "VALID" else "VALID")

            return {
                "prediction": prediction,
                "actual": label,
                "correct": prediction == label,
                "compute_time": 0.01,
                "tokens_used": 100
            }

        # Normal evaluation with LLM
        system_prompt = f"""You are evaluating if traces conform to a discovered process model.

AS-IS Process Model:
{model_text}

Your task:
1. Analyze if the given trace follows the AS-IS model
2. Respond with exactly "VALID" if it conforms, "INVALID" if it violates the model
3. Be precise and follow the model structure exactly"""

        user_prompt = f"""Evaluate this trace against the AS-IS model:

{trace}

Does this trace conform to the model? Reply with VALID or INVALID only."""

        start_time = time.time()

        try:
            llm = ChatOpenAI(model=self.model_name, temperature=0)
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]

            response = llm.invoke(messages)
            prediction = "VALID" if "VALID" in response.content.upper() else "INVALID"

            # Extract metrics
            compute_time = time.time() - start_time
            tokens_used = 0

            if hasattr(response, 'response_metadata'):
                metadata = response.response_metadata
                if 'token_usage' in metadata:
                    tokens_used = metadata['token_usage'].get('total_tokens', 0)

            return {
                "prediction": prediction,
                "actual": label,
                "correct": prediction == label,
                "compute_time": compute_time,
                "tokens_used": tokens_used
            }

        except Exception as e:
            logger.error(f"Error evaluating trace: {e}")
            return {
                "prediction": "ERROR",
                "actual": label,
                "correct": False,
                "compute_time": time.time() - start_time,
                "tokens_used": 0
            }

    def evaluate_architecture(self, architecture_class, xes_path: str,
                            valid_traces: List, invalid_traces: List) -> ArchitectureMetrics:
        """
        Evaluate a single architecture on the problem
        """
        metrics = ArchitectureMetrics(architecture_name=architecture_class.__name__)
        #breakpoint()
        # Generate model
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating {architecture_class.__name__}")
        logger.info(f"{'='*60}")

        model_text, model_metrics = self.generate_model_with_architecture(xes_path, architecture_class)

        metrics.model_generation_time = model_metrics.get('time', 0)
        metrics.model_generation_tokens = model_metrics.get('total_tokens', 0)

        # Save model
        model_path = self.run_folder / f"{architecture_class.__name__}_model.txt"
        model_path.write_text(model_text)

        # Evaluate traces
        all_traces = [(t, "VALID") for t in valid_traces] + [(t, "INVALID") for t in invalid_traces]

        logger.info(f"Evaluating {len(all_traces)} traces...")

        for i, (trace, label) in enumerate(all_traces):
            result = self.evaluate_trace(trace, model_text, label)

            # Update metrics
            metrics.evaluation_tokens += result['tokens_used']
            metrics.evaluation_time += result['compute_time']

            if result['correct']:
                if label == "VALID":
                    metrics.true_positives += 1
                else:
                    metrics.true_negatives += 1
            else:
                if label == "VALID":
                    metrics.false_negatives += 1
                else:
                    metrics.false_positives += 1

            # Print progress every 20 traces
            if (i + 1) % 20 == 0:
                current_precision = metrics.true_positives / (metrics.true_positives + metrics.false_positives) \
                    if (metrics.true_positives + metrics.false_positives) > 0 else 0
                current_recall = metrics.true_positives / (metrics.true_positives + metrics.false_negatives) \
                    if (metrics.true_positives + metrics.false_negatives) > 0 else 0

                logger.info(f"  Progress: {i+1}/{len(all_traces)} | "
                          f"Precision: {current_precision:.3f} | "
                          f"Recall: {current_recall:.3f}")

            # Rate limiting (skip for dummy models)
            if "DUMMY MODEL" not in model_text:
                time.sleep(1)

        # Calculate final metrics
        if metrics.true_positives + metrics.false_positives > 0:
            metrics.precision = metrics.true_positives / (metrics.true_positives + metrics.false_positives)

        if metrics.true_positives + metrics.false_negatives > 0:
            metrics.recall = metrics.true_positives / (metrics.true_positives + metrics.false_negatives)

        if metrics.precision + metrics.recall > 0:
            metrics.f1_score = 2 * (metrics.precision * metrics.recall) / (metrics.precision + metrics.recall)

        return metrics

    def run_comparison(self, problem_name: str, xes_path: str) -> Dict[str, ArchitectureMetrics]:
        """
        Run comparison of all architectures on a single problem
        """
        logger.info(f"\n{'#'*70}")
        logger.info(f"# Problem: {problem_name}")
        logger.info(f"{'#'*70}")

        # Generate shared traces
        logger.info("Generating shared traces for all architectures...")
        valid_traces, invalid_traces = self.generate_traces(xes_path)
        logger.info(f"Generated {len(valid_traces)} valid and {len(invalid_traces)} invalid traces")

        # Evaluate each architecture
        results = {}

        for architecture_class in self.architectures:
            metrics = self.evaluate_architecture(
                architecture_class,
                xes_path,
                valid_traces,
                invalid_traces
            )
            results[architecture_class.__name__] = metrics

            # Save intermediate results
            self.save_intermediate_results(problem_name, results)

        return results

    def save_intermediate_results(self, problem_name: str, results: Dict[str, ArchitectureMetrics]):
        """Save intermediate results after each architecture"""
        results_path = self.run_folder / f"{problem_name}_results.json"

        results_data = {
            arch_name: asdict(metrics)
            for arch_name, metrics in results.items()
        }

        results_path.write_text(json.dumps(results_data, indent=2))

    def run_full_benchmark(self) -> Dict:
        """Run full benchmark on all problems"""
        all_results = {}

        for problem in self.problems:
            problem_name = problem['name']
            xes_path = problem['path']

            results = self.run_comparison(problem_name, xes_path)
            all_results[problem_name] = results

            # Save overall results
            self.save_final_results(all_results)

        return all_results

    def save_final_results(self, all_results: Dict):
        """Save final benchmark results with comparison"""
        # Convert to serializable format
        results_data = {}

        for problem_name, problem_results in all_results.items():
            results_data[problem_name] = {
                arch_name: asdict(metrics)
                for arch_name, metrics in problem_results.items()
            }

        # Save detailed results
        final_path = self.run_folder / "benchmark_results.json"
        final_path.write_text(json.dumps(results_data, indent=2))

        # Create comparison summary
        summary = self.create_comparison_summary(all_results)
        summary_path = self.run_folder / "comparison_summary.md"
        summary_path.write_text(summary)

        # Save config
        config_path = self.run_folder / "benchmark_config.json"
        config_path.write_text(json.dumps(self.config, indent=2, default=str))

    def create_comparison_summary(self, all_results: Dict) -> str:
        """Create markdown summary comparing architectures"""
        lines = ["# Multi-Agent Architecture Comparison\n"]
        lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        for problem_name, problem_results in all_results.items():
            lines.append(f"\n## Problem: {problem_name}\n")
            lines.append("\n| Architecture | F1 Score | Precision | Recall | Model Tokens | Eval Tokens | Total Time (s) |")
            lines.append("|--------------|----------|-----------|--------|--------------|-------------|----------------|")

            for arch_name, metrics in problem_results.items():
                total_time = metrics.model_generation_time + metrics.evaluation_time
                lines.append(f"| {arch_name} | {metrics.f1_score:.3f} | {metrics.precision:.3f} | "
                           f"{metrics.recall:.3f} | {metrics.model_generation_tokens:,} | "
                           f"{metrics.evaluation_tokens:,} | {total_time:.1f} |")

        # Overall comparison
        lines.append("\n## Overall Performance Summary\n")
        lines.append("\n| Architecture | Avg F1 | Avg Tokens | Avg Time |")
        lines.append("|--------------|--------|------------|----------|")

        # Calculate averages per architecture
        arch_summaries = {}

        for arch_class in self.architectures:
            arch_name = arch_class.__name__
            f1_scores = []
            total_tokens = []
            total_times = []

            for problem_results in all_results.values():
                if arch_name in problem_results:
                    metrics = problem_results[arch_name]
                    f1_scores.append(metrics.f1_score)
                    total_tokens.append(metrics.model_generation_tokens + metrics.evaluation_tokens)
                    total_times.append(metrics.model_generation_time + metrics.evaluation_time)

            if f1_scores:
                avg_f1 = sum(f1_scores) / len(f1_scores)
                avg_tokens = sum(total_tokens) / len(total_tokens)
                avg_time = sum(total_times) / len(total_times)

                lines.append(f"| {arch_name} | {avg_f1:.3f} | {avg_tokens:,.0f} | {avg_time:.1f} |")

        return "\n".join(lines)


def main():
    """Main execution function"""

    # Configure benchmark
    config = {
        'architectures': [
            #DummyMiner,
            AgenticProcessMiner,  # Baseline single agent
            HierarchicalExpertMiner  # Multi-agent system
        ],
        'problems': [
            {
                'name': 'hiring_log_high',
                'path': 'benchmark/hiring/hiring_log_high.xes'
            },
            {
                'name': 'hospital_log_high',
                'path': 'benchmark/hospital/hospital_log_high.xes'
            },
            {
                'name': 'lending_log_high',
                'path': 'benchmark/lending/lending_log_high.xes'
            },
            {
                'name': 'renting_log_high',
                'path': 'benchmark/renting/renting_log_high.xes'
            }
        ],
        'k_valid': 100,
        'k_invalid': 100,
        'perturbations': ['random_swap'],
        'model_name': 'gpt-4o-mini'
    }

    # Create and run benchmark
    benchmark = MultiAgentBenchmark(config)

    logger.info("Starting Multi-Agent Architecture Benchmark")
    logger.info(f"Comparing architectures: {[a.__name__ for a in config['architectures']]}")
    logger.info(f"Problems: {[p['name'] for p in config['problems']]}")
    logger.info(f"Traces per problem: {config['k_valid']} valid, {config['k_invalid']} invalid")

    # Run benchmark
    results = benchmark.run_full_benchmark()

    logger.info(f"\nBenchmark complete! Results saved to: {benchmark.run_folder}")

    # Print summary
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)

    for problem_name, problem_results in results.items():
        print(f"\n{problem_name}:")
        for arch_name, metrics in problem_results.items():
            print(f"  {arch_name}: F1={metrics.f1_score:.3f}, "
                  f"Tokens={metrics.model_generation_tokens + metrics.evaluation_tokens:,}")


if __name__ == "__main__":
    main()