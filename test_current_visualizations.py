#!/usr/bin/env python3
"""
Test Current Visualization Capabilities
Shows what visualizations we currently have
"""

import sys
from pathlib import Path

# Add process-mining to path
sys.path.append(str(Path(__file__).parent / "process-mining"))

from pm4py_integration import PM4PyAnalyzer
from visualizer import ProcessVisualizer

def test_current_setup():
    """Test and demonstrate current visualization capabilities"""

    print("=" * 70)
    print("CURRENT VISUALIZATION CAPABILITIES TEST")
    print("=" * 70)

    # Find the latest event log
    experiments_dir = Path("experiments")
    event_logs = sorted(experiments_dir.glob("test_pm_*/event_log.json"))

    if not event_logs:
        print("\n❌ No event logs found. Run test_pm_simple.py first.")
        return

    event_log_path = event_logs[-1]
    output_dir = event_log_path.parent / "current_visualization_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📁 Using event log: {event_log_path}")
    print(f"📁 Output directory: {output_dir}")

    # ========================================
    # Part 1: PM4Py Process Model Discovery
    # ========================================
    print("\n" + "=" * 70)
    print("PART 1: PM4Py Process Model Discovery")
    print("=" * 70)

    analyzer = PM4PyAnalyzer()
    analyzer.load_event_log(event_log_path)

    print("\n🔍 Discovering process models...")
    models = analyzer.discover_process_models()

    print(f"\n📊 Discovered Models:")
    for model_name, model_data in models.items():
        if 'error' not in model_data:
            if model_name == 'dfg':
                print(f"  ✅ {model_name.upper()}: {model_data.get('num_edges', 0)} edges")
            else:
                print(f"  ✅ {model_name.title()}: "
                      f"{model_data.get('num_places', 0)} places, "
                      f"{model_data.get('num_transitions', 0)} transitions, "
                      f"{model_data.get('num_arcs', 0)} arcs")
        else:
            print(f"  ❌ {model_name}: {model_data['error'][:50]}...")

    # Try to visualize with PM4Py (will fail without Graphviz)
    print(f"\n🎨 Attempting PM4Py visualization (requires Graphviz)...")
    try:
        viz_path = analyzer.visualize_model('heuristics', output_dir / "pm4py_heuristics.png")
        print(f"  ✅ Saved: {viz_path}")
    except Exception as e:
        print(f"  ❌ Failed: {str(e)[:80]}...")
        print(f"     (This is expected without Graphviz installed)")

    # Calculate conformance
    print(f"\n✅ Calculating conformance metrics...")
    conformance = analyzer.calculate_conformance('heuristics')

    if 'fitness' in conformance and 'error' not in conformance['fitness']:
        print(f"\n📈 Conformance Metrics (Heuristics Model):")
        print(f"  Fitness:        {conformance['fitness'].get('log_fitness', 0):.3f}")
        if 'precision' in conformance and not isinstance(conformance['precision'], dict):
            print(f"  Precision:      {conformance['precision']:.3f}")
        if 'generalization' in conformance and not isinstance(conformance['generalization'], dict):
            print(f"  Generalization: {conformance['generalization']:.3f}")
        if 'simplicity' in conformance and not isinstance(conformance['simplicity'], dict):
            print(f"  Simplicity:     {conformance['simplicity']:.3f}")

    # ========================================
    # Part 2: Our Custom Visualizations
    # ========================================
    print("\n" + "=" * 70)
    print("PART 2: Custom Visualizations (NetworkX-based)")
    print("=" * 70)

    visualizer = ProcessVisualizer(event_log_path)

    print("\n🎨 Creating custom visualizations...")
    print("   (These work without Graphviz)")

    results = {}

    # 1. Activity Frequency
    try:
        path = visualizer.plot_activity_frequency(output_dir / "activity_frequency.png")
        results['activity_frequency'] = path
        print(f"  ✅ Activity Frequency: {path.name}")
    except Exception as e:
        print(f"  ❌ Activity Frequency failed: {e}")

    # 2. Resource Utilization
    try:
        path = visualizer.plot_resource_utilization(output_dir / "resource_utilization.png")
        results['resource_utilization'] = path
        print(f"  ✅ Resource Utilization: {path.name}")
    except Exception as e:
        print(f"  ❌ Resource Utilization failed: {e}")

    # 3. Process Flow Graph (from event log)
    try:
        path = visualizer.plot_process_flow_graph(output_dir / "process_flow.png")
        results['process_flow'] = path
        print(f"  ✅ Process Flow Graph: {path.name}")
    except Exception as e:
        print(f"  ❌ Process Flow failed: {e}")

    # 4. Timeline
    try:
        path = visualizer.plot_timeline(output_dir / "timeline.png")
        results['timeline'] = path
        print(f"  ✅ Timeline: {path.name}")
    except Exception as e:
        print(f"  ❌ Timeline failed: {e}")

    # 5. Token Usage
    try:
        path = visualizer.plot_token_usage(output_dir / "token_usage.png")
        results['token_usage'] = path
        print(f"  ✅ Token Usage: {path.name}")
    except Exception as e:
        print(f"  ❌ Token Usage failed: {e}")

    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n✅ Process Models Discovered: {len([m for m in models.values() if 'error' not in m])}/{len(models)}")
    print(f"✅ Custom Visualizations Created: {len(results)}/5")
    print(f"\n📁 All outputs saved to: {output_dir}")

    print("\n" + "=" * 70)
    print("WHAT'S MISSING?")
    print("=" * 70)
    print("""
Currently we have:
  ✅ Process model discovery (4 algorithms)
  ✅ Conformance metrics calculation
  ✅ Custom visualizations (5 types) using NetworkX
  ✅ Activity/resource/timeline analysis

What we DON'T have yet:
  ❌ Direct visualization of discovered Petri nets
  ❌ DFG (Directly-Follows Graph) visualization
  ❌ Process tree visualization
  ❌ Model comparison visualization

These require either:
  1. Graphviz (external dependency)
  2. Custom NetworkX-based renderers (to implement)
    """)

    print("\n💡 TIP: Check the generated PNG files in:")
    print(f"   {output_dir}")
    print("\n🔍 To understand conformance metrics, read:")
    print("   process-mining/CONFORMANCE_METRICS_EXPLAINED.md")

    return output_dir

if __name__ == "__main__":
    output_dir = test_current_setup()

    if output_dir:
        print(f"\n✅ Test complete! Check visualizations in: {output_dir}")