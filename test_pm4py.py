#!/usr/bin/env python3
"""
Test PM4Py Integration with our event logs
"""

import sys
from pathlib import Path

# Add process-mining to path
sys.path.append(str(Path(__file__).parent / "process-mining"))

from pm4py_integration import analyze_event_log, PM4PyAnalyzer

def test_pm4py():
    """Test PM4Py analysis pipeline"""

    # Find the latest event log
    experiments_dir = Path("experiments")
    event_logs = sorted(experiments_dir.glob("test_pm_*/event_log.json"))

    if not event_logs:
        print("No event logs found. Run test_pm_simple.py first.")
        return

    # Use the latest one
    event_log_path = event_logs[-1]
    output_dir = event_log_path.parent / "pm4py_analysis"

    print(f"Analyzing: {event_log_path}")
    print("=" * 60)

    # Run complete analysis
    results = analyze_event_log(event_log_path, output_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)

    # Models
    print("\n📊 Discovered Models:")
    for model_name, model_data in results['models'].items():
        if 'error' not in model_data:
            if model_name == 'dfg':
                print(f"  {model_name}: {model_data.get('num_edges', 0)} edges")
            else:
                print(f"  {model_name}: {model_data.get('num_places', 0)} places, "
                      f"{model_data.get('num_transitions', 0)} transitions")
        else:
            print(f"  {model_name}: Error - {model_data['error'][:50]}...")

    # Statistics
    if 'statistics' in results and results['statistics']:
        stats = results['statistics']
        print(f"\n📈 Statistics:")
        if 'basic' in stats:
            print(f"  Events: {stats['basic'].get('num_events', 0)}")
            print(f"  Cases: {stats['basic'].get('num_cases', 0)}")
            print(f"  Activities: {stats['basic'].get('num_activities', 0)}")

        if 'variants' in stats:
            print(f"  Process Variants: {stats['variants'].get('num_variants', 0)}")

        if 'performance' in stats:
            print(f"  Avg Case Duration: {stats['performance'].get('avg_case_duration_seconds', 0):.2f} seconds")

    # Conformance
    if 'conformance' in results and results['conformance']:
        print(f"\n✅ Conformance Metrics:")
        for model_type, conf_data in results['conformance'].items():
            if 'fitness' in conf_data and 'error' not in conf_data['fitness']:
                print(f"  {model_type}:")
                print(f"    Fitness: {conf_data['fitness'].get('log_fitness', 0):.3f}")
                if 'precision' in conf_data and not isinstance(conf_data['precision'], dict):
                    print(f"    Precision: {conf_data['precision']:.3f}")
                if 'generalization' in conf_data and not isinstance(conf_data['generalization'], dict):
                    print(f"    Generalization: {conf_data['generalization']:.3f}")
                if 'simplicity' in conf_data and not isinstance(conf_data['simplicity'], dict):
                    print(f"    Simplicity: {conf_data['simplicity']:.3f}")

    # Exported files
    if 'exported_files' in results and results['exported_files']:
        print(f"\n💾 Exported Files:")
        for file_type, file_path in results['exported_files'].items():
            print(f"  {file_type}: {file_path}")

    print("\n✅ PM4Py analysis complete!")

if __name__ == "__main__":
    test_pm4py()