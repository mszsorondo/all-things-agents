#!/usr/bin/env python3
"""
Simple test of the Process Mining system without the full agent flow
"""

import sys
from pathlib import Path
from datetime import datetime

# Add process-mining to path
sys.path.append(str(Path(__file__).parent / "process-mining"))

from process_mining_logger import ProcessMiningLogger, Event
from benchmark_problems import get_problem_by_name
from xes_converter import XESConverter
from mining_analyzer import MiningAnalyzer

def test_pm_logging():
    """Test basic PM logging functionality"""
    print("Testing Process Mining Logger...")

    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = Path("experiments") / f"test_pm_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Initialize PM logger
    pm_logger = ProcessMiningLogger(
        case_id=f"test_{timestamp}",
        architecture="linear_flow",
        problem="calculator",
        experiment_dir=experiment_dir
    )

    # Log some events
    pm_logger.log_user_request("Create a calculator with basic operations")

    pm_logger.log_agent_start("spec_agent", "gpt-4o-mini")
    pm_logger.log_agent_complete("spec_agent", tokens_in=100, tokens_out=200)

    pm_logger.log_transition("spec_agent", "coding_agent", "specification complete")

    pm_logger.log_agent_start("coding_agent", "gpt-4o-mini")
    pm_logger.log_tool_invocation("write_code", "coding_agent",
                                   {"filename": "calculator.py", "lines": 50})
    pm_logger.log_agent_complete("coding_agent", tokens_in=150, tokens_out=300,
                                  tools_used=["write_code"])

    pm_logger.log_transition("coding_agent", "testing_agent", "code complete")

    pm_logger.log_agent_start("testing_agent", "gpt-4o-mini")
    pm_logger.log_tool_invocation("read_code", "testing_agent", {"filename": "calculator.py"})
    pm_logger.log_tool_invocation("write_tests", "testing_agent", {"filename": "test_calculator.py"})
    pm_logger.log_tool_invocation("run_tests", "testing_agent", {"test_file": "test_calculator.py"})
    pm_logger.log_agent_complete("testing_agent", tokens_in=200, tokens_out=400,
                                  tools_used=["read_code", "write_tests", "run_tests"])

    pm_logger.log_transition("testing_agent", "review_agent", "tests complete")

    pm_logger.log_agent_start("review_agent", "gpt-4o-mini")
    pm_logger.log_agent_complete("review_agent", tokens_in=50, tokens_out=100)

    # Finalize
    pm_logger.finalize(success=True)

    print(f"✅ Event log saved to: {experiment_dir / 'event_log.json'}")

    # Test XES conversion
    print("\nTesting XES Converter...")
    event_log = pm_logger.get_event_log()
    converter = XESConverter(event_log.to_dict())

    # Save XES file
    xes_path = experiment_dir / "event_log.xes"
    converter.to_xes_xml(xes_path)
    print(f"✅ XES file saved to: {xes_path}")

    # Save CSV
    csv_path = experiment_dir / "event_log.csv"
    converter.to_csv(csv_path)
    print(f"✅ CSV file saved to: {csv_path}")

    # Test analyzer
    print("\nTesting Mining Analyzer...")
    analyzer = MiningAnalyzer(experiment_dir / "event_log.json")

    stats = analyzer.calculate_basic_statistics()
    print(f"📊 Basic Statistics:")
    print(f"   Total Events: {stats.get('total_events', 0)}")
    print(f"   Unique Activities: {stats.get('unique_activities', 0)}")
    print(f"   Case Duration: {stats.get('case_duration', {}).get('human_readable', 'N/A')}")

    perf = analyzer.analyze_performance()
    print(f"\n⚡ Performance:")
    print(f"   Total Tokens In: {perf.get('token_usage', {}).get('total_in', 0)}")
    print(f"   Total Tokens Out: {perf.get('token_usage', {}).get('total_out', 0)}")
    print(f"   Tools Used: {list(perf.get('tool_invocations', {}).keys())}")

    # Generate report
    report_path = experiment_dir / "analysis_report.txt"
    report = analyzer.generate_report(report_path)
    print(f"\n📝 Analysis report saved to: {report_path}")

    print("\n✅ All tests passed!")
    return experiment_dir

def test_benchmark_problems():
    """Test loading benchmark problems"""
    print("\nTesting Benchmark Problems...")

    from benchmark_problems import get_problem_summary, get_problem_by_name

    summary = get_problem_summary()
    print(f"📚 Total problems available: {summary['total']}")
    print(f"   By difficulty: {summary['by_difficulty']}")

    # Get calculator problem
    calc_problem = get_problem_by_name("calculator")
    if calc_problem:
        print(f"\n✅ Calculator problem loaded:")
        print(f"   Name: {calc_problem.name}")
        print(f"   Difficulty: {calc_problem.difficulty}")
        print(f"   Expected functions: {calc_problem.expected_functions}")

if __name__ == "__main__":
    print("=" * 60)
    print("PROCESS MINING SYSTEM TEST")
    print("=" * 60)

    # Test PM logging
    experiment_dir = test_pm_logging()

    # Test benchmark problems
    test_benchmark_problems()

    print("\n" + "=" * 60)
    print(f"TEST COMPLETE - Check {experiment_dir} for results")
    print("=" * 60)