#!/usr/bin/env python3
"""
Test Process Model Visualization
Visualizes the actual discovered Petri nets and DFG models
"""

import sys
from pathlib import Path

# Add process-mining to path
sys.path.append(str(Path(__file__).parent / "process-mining"))

from pm4py_integration import PM4PyAnalyzer
from model_visualizer import ProcessModelVisualizer

def main():
    print("=" * 70)
    print("PROCESS MODEL VISUALIZATION TEST")
    print("=" * 70)

    # Find the latest event log
    experiments_dir = Path("experiments")
    event_logs = sorted(experiments_dir.glob("test_pm_*/event_log.json"))

    if not event_logs:
        print("\n❌ No event logs found. Run test_pm_simple.py first.")
        return

    event_log_path = event_logs[-1]
    output_dir = event_log_path.parent / "process_models"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📁 Using event log: {event_log_path}")
    print(f"📁 Output directory: {output_dir}")

    # Discover process models
    print("\n" + "=" * 70)
    print("STEP 1: Discover Process Models with PM4Py")
    print("=" * 70)

    analyzer = PM4PyAnalyzer()
    analyzer.load_event_log(event_log_path)

    print("\n🔍 Running process discovery algorithms...")
    models = analyzer.discover_process_models()

    print(f"\n✅ Models discovered:")
    for name, data in models.items():
        if 'error' not in data:
            if name == 'dfg':
                print(f"   • {name.upper()}: {data.get('num_edges', 0)} edges")
            else:
                print(f"   • {name.title()}: "
                      f"P={data.get('num_places', 0)}, "
                      f"T={data.get('num_transitions', 0)}, "
                      f"A={data.get('num_arcs', 0)}")
        else:
            print(f"   • {name}: Error")

    # Visualize models
    print("\n" + "=" * 70)
    print("STEP 2: Visualize Discovered Models (NetworkX-based)")
    print("=" * 70)

    visualizer = ProcessModelVisualizer()

    print("\n🎨 Creating model visualizations...")
    model_viz = visualizer.visualize_all_models(models, output_dir)

    print(f"\n✅ Created {len(model_viz)} model visualizations")

    # Create comparison chart
    print("\n🎨 Creating model comparison chart...")
    try:
        comp_path = visualizer.create_model_comparison_chart(
            models,
            output_dir / 'model_comparison.png'
        )
        print(f"✅ Comparison chart: {comp_path.name}")
    except Exception as e:
        print(f"❌ Comparison chart failed: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n📊 Generated files in: {output_dir}")
    for name, path in model_viz.items():
        print(f"   • model_{name}.png - {name.upper()} model visualization")

    print("\n" + "=" * 70)
    print("WHAT YOU'RE SEEING")
    print("=" * 70)
    print("""
**DFG (Directly-Follows Graph):**
  • Shows which activities follow which activities
  • Edge thickness = frequency of that transition
  • Green nodes = start activities
  • Pink nodes = end activities
  • Blue nodes = intermediate activities

**Petri Net Models (Alpha/Heuristics/Inductive):**
  • Circles = Places (states in the process)
  • Squares = Transitions (activities that can occur)
  • Gold circle = Initial marking (where process starts)
  • Pink circle = Final marking (where process ends)
  • Gray squares = Silent transitions (τ) - internal steps
  • Green squares = Visible transitions (actual activities)

**Model Differences:**
  • Alpha: Basic algorithm, good for simple processes
  • Heuristics: Better with noise and complex patterns
  • Inductive: Guarantees sound models (no deadlocks)
    """)

    print("\n💡 TIP: Open the PNG files to see the actual process models!")
    print(f"   Location: {output_dir}")

    return output_dir

if __name__ == "__main__":
    output_dir = main()
    if output_dir:
        print(f"\n✅ Done! Check: {output_dir}")