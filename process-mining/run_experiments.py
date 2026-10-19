"""
Experiment Runner for Multi-Agent System Benchmarking
Executes multiple problems through the pipeline and collects process mining data
"""

import sys
import os
from pathlib import Path
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
import subprocess
import time

# Add parent directory to path to import linear_flow_advanced
sys.path.append(str(Path(__file__).parent.parent))

from benchmark_problems import (
    get_all_problems,
    get_problems_by_difficulty,
    get_problem_by_name,
    BenchmarkProblem
)
from mining_analyzer import MiningAnalyzer
from xes_converter import XESConverter, batch_convert_to_xes

class ExperimentRunner:
    """
    Runs experiments with different problems and architectures
    """

    def __init__(self, base_dir: Path = None):
        """
        Initialize the experiment runner

        Args:
            base_dir: Base directory for experiments (default: current directory)
        """
        self.base_dir = base_dir or Path.cwd()
        self.experiments_dir = self.base_dir / "experiments"
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

        self.results = []
        self.current_run = None

    def run_single_problem(
        self,
        problem: BenchmarkProblem,
        architecture: str = "linear_flow",
        script_name: str = "linear_flow_advanced.py"
    ) -> Dict[str, Any]:
        """
        Run a single problem through the multi-agent pipeline

        Args:
            problem: The benchmark problem to run
            architecture: Architecture type (for logging)
            script_name: Python script to execute

        Returns:
            Dictionary with results and paths
        """
        # Generate run ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"run_{timestamp}_{problem.name}"

        print(f"\n🚀 Running experiment: {problem.name}")
        print(f"   Architecture: {architecture}")
        print(f"   Difficulty: {problem.difficulty}")

        # Prepare the modified script with the problem
        result = self._execute_pipeline(
            problem=problem,
            run_id=run_id,
            architecture=architecture,
            script_name=script_name
        )

        # Collect results
        experiment_result = {
            "run_id": run_id,
            "problem_name": problem.name,
            "problem_difficulty": problem.difficulty,
            "problem_category": problem.category,
            "architecture": architecture,
            "timestamp": timestamp,
            "success": result.get("success", False),
            "error": result.get("error", None),
            "paths": result.get("paths", {}),
            "metrics": result.get("metrics", {})
        }

        self.results.append(experiment_result)
        return experiment_result

    def _execute_pipeline(
        self,
        problem: BenchmarkProblem,
        run_id: str,
        architecture: str,
        script_name: str
    ) -> Dict[str, Any]:
        """
        Execute the multi-agent pipeline with process mining logging

        Args:
            problem: The problem to solve
            run_id: Unique run identifier
            architecture: Architecture type
            script_name: Script to execute

        Returns:
            Execution results
        """
        try:
            # Check if script exists
            script_path = self.base_dir / script_name

            if not script_path.exists():
                return {
                    "success": False,
                    "error": f"Script {script_name} not found"
                }

            # Create a temporary modified script that uses our problem
            temp_script = self._create_temp_script(
                script_path,
                problem,
                run_id,
                architecture
            )

            # Execute the script
            start_time = time.time()

            process = subprocess.run(
                [sys.executable, str(temp_script)],
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout
            )

            execution_time = time.time() - start_time

            # Clean up temp script
            if temp_script.exists():
                temp_script.unlink()

            # Check for experiment directory
            experiment_dir = self.experiments_dir / run_id
            if not experiment_dir.exists():
                # Try to find the most recent run directory
                run_dirs = sorted(self.experiments_dir.glob("run_*"), key=lambda p: p.stat().st_mtime)
                if run_dirs:
                    experiment_dir = run_dirs[-1]
                    run_id = experiment_dir.name

            # Collect paths
            paths = {}
            if experiment_dir.exists():
                paths["experiment_dir"] = str(experiment_dir)
                paths["event_log"] = str(experiment_dir / "event_log.json")
                paths["result"] = str(experiment_dir / "advanced_result.json")

                # Find generated code files
                code_files = list(experiment_dir.glob("*.py"))
                paths["code_files"] = [str(f) for f in code_files if not f.name.startswith("test_")]
                paths["test_files"] = [str(f) for f in code_files if f.name.startswith("test_")]

            # Extract metrics from event log if available
            metrics = {}
            if paths.get("event_log") and Path(paths["event_log"]).exists():
                metrics = self._extract_metrics(Path(paths["event_log"]))

            return {
                "success": process.returncode == 0,
                "error": process.stderr if process.returncode != 0 else None,
                "stdout": process.stdout,
                "execution_time": execution_time,
                "paths": paths,
                "metrics": metrics
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Execution timeout (120 seconds)",
                "paths": {},
                "metrics": {}
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "paths": {},
                "metrics": {}
            }

    def _create_temp_script(
        self,
        original_script: Path,
        problem: BenchmarkProblem,
        run_id: str,
        architecture: str
    ) -> Path:
        """
        Create a temporary modified script with the problem and PM logging

        Args:
            original_script: Path to original script
            problem: Problem to solve
            run_id: Run identifier
            architecture: Architecture type

        Returns:
            Path to temporary script
        """
        # Read original script
        with open(original_script, 'r') as f:
            content = f.read()

        # Modify the initial message to use our problem
        old_message = '''initial_message = HumanMessage(
        content="Create a calculator with add, subtract, multiply, and divide functions. "
                "Include error handling for division by zero and invalid inputs."
    )'''

        new_message = f'''# Injected problem from benchmark
    initial_message = HumanMessage(
        content="""{problem.description}"""
    )

    # Add process mining logging
    from pathlib import Path
    import sys
    sys.path.append(str(Path(__file__).parent / "process-mining"))
    from process_mining_logger import ProcessMiningLogger

    # Initialize PM logger
    pm_logger = ProcessMiningLogger(
        case_id="{run_id}",
        architecture="{architecture}",
        problem="{problem.name}",
        experiment_dir=EXPERIMENT_DIR
    )
    pm_logger.log_user_request(initial_message.content)'''

        content = content.replace(old_message, new_message)

        # Also update the experiment directory name if needed
        content = content.replace(
            'EXPERIMENT_DIR = Path("experiments") / f"run_{timestamp}"',
            f'EXPERIMENT_DIR = Path("experiments") / "{run_id}"'
        )

        # Save temporary script
        temp_path = self.base_dir / f"temp_{run_id}.py"
        with open(temp_path, 'w') as f:
            f.write(content)

        return temp_path

    def _extract_metrics(self, event_log_path: Path) -> Dict[str, Any]:
        """
        Extract key metrics from an event log

        Args:
            event_log_path: Path to event_log.json

        Returns:
            Dictionary of metrics
        """
        try:
            analyzer = MiningAnalyzer(event_log_path)
            stats = analyzer.calculate_basic_statistics()
            perf = analyzer.analyze_performance()

            return {
                "total_events": stats.get("total_events", 0),
                "unique_activities": stats.get("unique_activities", 0),
                "duration_seconds": stats.get("case_duration", {}).get("seconds", 0),
                "tokens_in": perf.get("token_usage", {}).get("total_in", 0),
                "tokens_out": perf.get("token_usage", {}).get("total_out", 0),
                "tools_used": list(perf.get("tool_invocations", {}).keys()),
                "bottlenecks": perf.get("bottlenecks", [])
            }
        except Exception:
            return {}

    def run_difficulty_batch(
        self,
        difficulty: str,
        architecture: str = "linear_flow"
    ) -> List[Dict[str, Any]]:
        """
        Run all problems of a specific difficulty

        Args:
            difficulty: Problem difficulty ('easy', 'medium', 'hard')
            architecture: Architecture to use

        Returns:
            List of results
        """
        problems = get_problems_by_difficulty(difficulty)
        results = []

        print(f"\n📋 Running {len(problems)} {difficulty} problems")

        for problem in problems:
            result = self.run_single_problem(problem, architecture)
            results.append(result)
            time.sleep(2)  # Brief pause between runs

        return results

    def run_all_problems(
        self,
        architecture: str = "linear_flow",
        difficulties: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Run all benchmark problems

        Args:
            architecture: Architecture to use
            difficulties: List of difficulties to run (default: all)

        Returns:
            DataFrame with all results
        """
        if difficulties is None:
            difficulties = ["easy", "medium", "hard"]

        all_results = []

        for difficulty in difficulties:
            results = self.run_difficulty_batch(difficulty, architecture)
            all_results.extend(results)

        # Convert to DataFrame
        df = pd.DataFrame(all_results)

        # Save results
        results_path = self.experiments_dir / f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(results_path, index=False)
        print(f"\n✅ Results saved to {results_path}")

        return df

    def analyze_results(self, results_df: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Analyze experiment results

        Args:
            results_df: DataFrame with results (uses self.results if None)

        Returns:
            Analysis summary
        """
        if results_df is None:
            results_df = pd.DataFrame(self.results)

        if results_df.empty:
            return {"error": "No results to analyze"}

        analysis = {
            "total_experiments": len(results_df),
            "success_rate": (results_df["success"].sum() / len(results_df)) * 100,
            "by_difficulty": {},
            "by_category": {},
            "average_metrics": {},
            "failed_problems": []
        }

        # Success rate by difficulty
        for difficulty in results_df["problem_difficulty"].unique():
            diff_df = results_df[results_df["problem_difficulty"] == difficulty]
            analysis["by_difficulty"][difficulty] = {
                "total": len(diff_df),
                "successful": diff_df["success"].sum(),
                "success_rate": (diff_df["success"].sum() / len(diff_df)) * 100 if len(diff_df) > 0 else 0
            }

        # Average metrics for successful runs
        success_df = results_df[results_df["success"] == True]
        if not success_df.empty:
            metrics_data = []
            for _, row in success_df.iterrows():
                if row["metrics"]:
                    metrics_data.append(row["metrics"])

            if metrics_data:
                metrics_df = pd.DataFrame(metrics_data)
                analysis["average_metrics"] = {
                    "avg_duration": metrics_df["duration_seconds"].mean() if "duration_seconds" in metrics_df else 0,
                    "avg_events": metrics_df["total_events"].mean() if "total_events" in metrics_df else 0,
                    "avg_tokens_in": metrics_df["tokens_in"].mean() if "tokens_in" in metrics_df else 0,
                    "avg_tokens_out": metrics_df["tokens_out"].mean() if "tokens_out" in metrics_df else 0
                }

        # List failed problems
        failed_df = results_df[results_df["success"] == False]
        analysis["failed_problems"] = failed_df[["problem_name", "error"]].to_dict('records')

        return analysis

    def generate_report(self, output_path: Optional[Path] = None) -> str:
        """
        Generate a comprehensive experiment report

        Args:
            output_path: Optional path to save report

        Returns:
            Report text
        """
        df = pd.DataFrame(self.results)
        analysis = self.analyze_results(df)

        report = []
        report.append("=" * 80)
        report.append("MULTI-AGENT SYSTEM BENCHMARK REPORT")
        report.append("=" * 80)
        report.append(f"\nGenerated: {datetime.now().isoformat()}")
        report.append(f"Total Experiments: {analysis['total_experiments']}")
        report.append(f"Overall Success Rate: {analysis['success_rate']:.1f}%")

        report.append("\n📊 RESULTS BY DIFFICULTY")
        for difficulty, stats in analysis['by_difficulty'].items():
            report.append(f"\n{difficulty.upper()}:")
            report.append(f"  Total: {stats['total']}")
            report.append(f"  Successful: {stats['successful']}")
            report.append(f"  Success Rate: {stats['success_rate']:.1f}%")

        report.append("\n⚡ AVERAGE METRICS (Successful Runs)")
        for metric, value in analysis['average_metrics'].items():
            report.append(f"  {metric}: {value:.2f}")

        if analysis['failed_problems']:
            report.append("\n❌ FAILED PROBLEMS")
            for failure in analysis['failed_problems']:
                report.append(f"  - {failure['problem_name']}: {failure['error'][:100]}...")

        report_text = "\n".join(report)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)

        return report_text

def main():
    """Main entry point for running experiments"""
    print("🚀 Multi-Agent System Experiment Runner")
    print("=" * 50)

    runner = ExperimentRunner()

    # Example: Run a single problem
    problem = get_problem_by_name("calculator")
    if problem:
        print(f"\nRunning single problem: {problem.name}")
        result = runner.run_single_problem(problem)
        print(f"Result: {'✅ Success' if result['success'] else '❌ Failed'}")

    # Example: Run all easy problems
    # print("\nRunning all easy problems...")
    # easy_results = runner.run_difficulty_batch("easy")
    # print(f"Completed {len(easy_results)} experiments")

    # Generate report
    report = runner.generate_report()
    print("\n" + report)

if __name__ == "__main__":
    main()