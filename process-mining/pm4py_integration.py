"""
PM4Py Integration Module for Multi-Agent Process Mining
Provides automated process discovery, conformance checking, and analysis
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import PM4Py components
import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.visualization.dfg import visualizer as dfg_visualizer
from pm4py.visualization.process_tree import visualizer as pt_visualizer
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator
from pm4py.statistics.traces.generic.pandas import case_statistics
from pm4py.statistics.variants.pandas import get as variants_get
from pm4py.algo.filtering.pandas.variants import variants_filter
from pm4py.algo.filtering.pandas.attributes import attributes_filter
from pm4py.algo.filtering.pandas.paths import paths_filter
# Some imports may not be available in all PM4Py versions
try:
    from pm4py.statistics.sojourn_time.pandas import get as soj_time_get
except ImportError:
    soj_time_get = None

try:
    from pm4py.statistics.concurrent_activities.pandas import get as concurrent_activities_get
except ImportError:
    concurrent_activities_get = None

class PM4PyAnalyzer:
    """
    Main class for PM4Py-based process mining analysis
    """

    def __init__(self, event_log_path: Optional[Path] = None):
        """
        Initialize the PM4Py analyzer

        Args:
            event_log_path: Path to event log (JSON, XES, or CSV)
        """
        self.event_log_path = event_log_path
        self.df = None
        self.event_log = None
        self.discovered_models = {}
        self.conformance_results = {}
        self.statistics = {}

    def load_event_log(self, event_log_path: Path, format: str = "auto"):
        """
        Load an event log from various formats

        Args:
            event_log_path: Path to the event log file
            format: File format ('json', 'xes', 'csv', or 'auto')
        """
        self.event_log_path = event_log_path

        if format == "auto":
            format = event_log_path.suffix.lstrip('.')

        if format == "json":
            self._load_from_json(event_log_path)
        elif format == "xes":
            self._load_from_xes(event_log_path)
        elif format == "csv":
            self._load_from_csv(event_log_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

        # Convert dataframe to event log
        if self.df is not None and not self.df.empty:
            self.event_log = pm4py.convert_to_event_log(self.df)

    def _load_from_json(self, json_path: Path):
        """Load event log from custom JSON format"""
        with open(json_path, 'r') as f:
            data = json.load(f)

        events = data.get('events', [])
        rows = []

        for event in events:
            row = {
                'case:concept:name': event.get('case_id', data.get('case_id', 'unknown')),
                'concept:name': event.get('activity', 'unknown'),
                'time:timestamp': pd.to_datetime(event.get('timestamp', datetime.utcnow().isoformat())),
                'org:resource': event.get('resource', 'unknown'),
                'lifecycle:transition': event.get('lifecycle', 'complete')
            }

            # Add custom attributes
            attributes = event.get('attributes', {})
            for key, value in attributes.items():
                if not isinstance(value, (list, dict)):
                    row[f'custom:{key}'] = value

            rows.append(row)

        self.df = pd.DataFrame(rows)
        self.df = self.df.sort_values('time:timestamp')

    def _load_from_xes(self, xes_path: Path):
        """Load event log from XES format"""
        self.event_log = xes_importer.apply(str(xes_path))
        self.df = pm4py.convert_to_dataframe(self.event_log)

    def _load_from_csv(self, csv_path: Path):
        """Load event log from CSV format"""
        self.df = pd.read_csv(csv_path)
        if 'time:timestamp' in self.df.columns:
            self.df['time:timestamp'] = pd.to_datetime(self.df['time:timestamp'])

    def discover_process_models(self) -> Dict[str, Any]:
        """
        Discover process models using multiple algorithms

        Returns:
            Dictionary containing discovered models
        """
        if self.df is None or self.df.empty:
            return {"error": "No event log loaded"}

        results = {}

        # Filter to complete events for cleaner models
        df_complete = self.df[self.df['lifecycle:transition'] == 'complete'].copy()

        # 1. Directly-Follows Graph (DFG)
        print("Discovering DFG...")
        dfg, start_activities, end_activities = pm4py.discover_dfg(df_complete)
        results['dfg'] = {
            'graph': dfg,
            'start_activities': start_activities,
            'end_activities': end_activities,
            'num_edges': len(dfg),
            'num_start_activities': len(start_activities),
            'num_end_activities': len(end_activities)
        }

        # 2. Alpha Miner
        print("Running Alpha Miner...")
        try:
            net_alpha, im_alpha, fm_alpha = pm4py.discover_petri_net_alpha(df_complete)
            results['alpha'] = {
                'net': net_alpha,
                'initial_marking': im_alpha,
                'final_marking': fm_alpha,
                'num_places': len(net_alpha.places),
                'num_transitions': len(net_alpha.transitions),
                'num_arcs': len(net_alpha.arcs)
            }
        except Exception as e:
            results['alpha'] = {'error': str(e)}

        # 3. Heuristics Miner
        print("Running Heuristics Miner...")
        try:
            net_heuristics, im_heuristics, fm_heuristics = pm4py.discover_petri_net_heuristics(
                df_complete,
                dependency_threshold=0.5,
                and_threshold=0.65,
                loop_two_threshold=0.5
            )
            results['heuristics'] = {
                'net': net_heuristics,
                'initial_marking': im_heuristics,
                'final_marking': fm_heuristics,
                'num_places': len(net_heuristics.places),
                'num_transitions': len(net_heuristics.transitions),
                'num_arcs': len(net_heuristics.arcs)
            }
        except Exception as e:
            results['heuristics'] = {'error': str(e)}

        # 4. Inductive Miner
        print("Running Inductive Miner...")
        try:
            tree = pm4py.discover_process_tree_inductive(df_complete)
            net_inductive, im_inductive, fm_inductive = pm4py.convert_to_petri_net(tree)
            results['inductive'] = {
                'tree': tree,
                'net': net_inductive,
                'initial_marking': im_inductive,
                'final_marking': fm_inductive,
                'num_places': len(net_inductive.places) if net_inductive else 0,
                'num_transitions': len(net_inductive.transitions) if net_inductive else 0,
                'num_arcs': len(net_inductive.arcs) if net_inductive else 0
            }
        except Exception as e:
            results['inductive'] = {'error': str(e)}

        self.discovered_models = results
        return results

    def calculate_conformance(self, model_type: str = 'heuristics') -> Dict[str, Any]:
        """
        Calculate conformance checking metrics

        Args:
            model_type: Which discovered model to use ('alpha', 'heuristics', 'inductive')

        Returns:
            Dictionary with conformance metrics
        """
        if model_type not in self.discovered_models:
            return {"error": f"Model {model_type} not discovered yet"}

        model = self.discovered_models[model_type]
        if 'error' in model:
            return {"error": f"Model {model_type} has errors"}

        net = model['net']
        im = model['initial_marking']
        fm = model['final_marking']

        df_complete = self.df[self.df['lifecycle:transition'] == 'complete'].copy()

        results = {}

        # 1. Fitness (Token-based replay)
        print(f"Calculating fitness for {model_type}...")
        try:
            fitness = replay_fitness.apply(
                df_complete,
                net,
                im,
                fm,
                variant=replay_fitness.Variants.TOKEN_BASED
            )
            results['fitness'] = {
                'average_trace_fitness': fitness['average_trace_fitness'],
                'percentage_of_fitting_traces': fitness['percentage_of_fitting_traces'],
                'log_fitness': fitness['log_fitness']
            }
        except Exception as e:
            results['fitness'] = {'error': str(e)}

        # 2. Precision
        print(f"Calculating precision for {model_type}...")
        try:
            precision = precision_evaluator.apply(
                df_complete,
                net,
                im,
                fm,
                variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN
            )
            results['precision'] = precision
        except Exception as e:
            results['precision'] = {'error': str(e)}

        # 3. Generalization
        print(f"Calculating generalization for {model_type}...")
        try:
            generalization = generalization_evaluator.apply(
                df_complete,
                net,
                im,
                fm
            )
            results['generalization'] = generalization
        except Exception as e:
            results['generalization'] = {'error': str(e)}

        # 4. Simplicity
        print(f"Calculating simplicity for {model_type}...")
        try:
            simplicity = simplicity_evaluator.apply(net)
            results['simplicity'] = simplicity
        except Exception as e:
            results['simplicity'] = {'error': str(e)}

        self.conformance_results[model_type] = results
        return results

    def calculate_statistics(self) -> Dict[str, Any]:
        """
        Calculate comprehensive statistics about the event log

        Returns:
            Dictionary with various statistics
        """
        if self.df is None or self.df.empty:
            return {"error": "No event log loaded"}

        stats = {}

        # 1. Basic statistics
        stats['basic'] = {
            'num_events': len(self.df),
            'num_cases': self.df['case:concept:name'].nunique(),
            'num_activities': self.df['concept:name'].nunique(),
            'num_resources': self.df['org:resource'].nunique()
        }

        # 2. Variants (unique paths through the process)
        print("Analyzing process variants...")
        variants = pm4py.get_variants(self.df)
        stats['variants'] = {
            'num_variants': len(variants),
            'top_variants': []
        }

        # Get top 5 variants - handle different return types from pm4py
        if isinstance(variants, dict):
            # Sort by frequency (value could be list of cases or count)
            sorted_items = []
            for variant, value in variants.items():
                if isinstance(value, list):
                    freq = len(value)
                elif isinstance(value, (int, float)):
                    freq = value
                else:
                    freq = 1
                sorted_items.append((variant, freq))

            sorted_items.sort(key=lambda x: x[1], reverse=True)

            for variant, frequency in sorted_items[:5]:
                stats['variants']['top_variants'].append({
                    'variant': str(variant),
                    'frequency': frequency,
                    'percentage': (frequency / stats['basic']['num_cases']) * 100 if stats['basic']['num_cases'] > 0 else 0
                })

        # 3. Performance statistics
        print("Calculating performance metrics...")
        case_durations = []
        for case in self.df['case:concept:name'].unique():
            case_events = self.df[self.df['case:concept:name'] == case]
            if len(case_events) > 0:
                duration = (case_events['time:timestamp'].max() -
                           case_events['time:timestamp'].min()).total_seconds()
                case_durations.append(duration)

        if case_durations:
            stats['performance'] = {
                'avg_case_duration_seconds': np.mean(case_durations),
                'median_case_duration_seconds': np.median(case_durations),
                'min_case_duration_seconds': np.min(case_durations),
                'max_case_duration_seconds': np.max(case_durations)
            }

        # 4. Activity statistics
        activity_freq = self.df['concept:name'].value_counts()
        stats['activities'] = {
            'frequency': activity_freq.to_dict(),
            'most_frequent': activity_freq.index[0] if len(activity_freq) > 0 else None,
            'least_frequent': activity_freq.index[-1] if len(activity_freq) > 0 else None
        }

        # 5. Resource statistics
        resource_freq = self.df['org:resource'].value_counts()
        stats['resources'] = {
            'frequency': resource_freq.to_dict(),
            'most_used': resource_freq.index[0] if len(resource_freq) > 0 else None,
            'least_used': resource_freq.index[-1] if len(resource_freq) > 0 else None
        }

        # 6. Start and end activities
        start_activities = pm4py.get_start_activities(self.df)
        end_activities = pm4py.get_end_activities(self.df)
        stats['start_end'] = {
            'start_activities': start_activities,
            'end_activities': end_activities
        }

        self.statistics = stats
        return stats

    def visualize_model(self, model_type: str = 'heuristics', output_path: Optional[Path] = None) -> Path:
        """
        Visualize a discovered process model

        Args:
            model_type: Which model to visualize
            output_path: Where to save the visualization

        Returns:
            Path to the saved visualization
        """
        if model_type not in self.discovered_models:
            raise ValueError(f"Model {model_type} not discovered yet")

        model = self.discovered_models[model_type]
        if 'error' in model:
            raise ValueError(f"Model {model_type} has errors")

        if output_path is None:
            output_path = Path(f"process_model_{model_type}.png")

        if model_type == 'dfg':
            # Visualize DFG
            parameters = {
                dfg_visualizer.Variants.PERFORMANCE.value.Parameters.FORMAT: "png"
            }
            gviz = dfg_visualizer.apply(
                model['graph'],
                log=self.df,
                parameters=parameters,
                variant=dfg_visualizer.Variants.PERFORMANCE,
                activities_count=model['start_activities']
            )
        elif model_type == 'inductive' and 'tree' in model:
            # Visualize process tree
            parameters = {
                pt_visualizer.Variants.WO_DECORATION.value.Parameters.FORMAT: "png"
            }
            gviz = pt_visualizer.apply(
                model['tree'],
                parameters=parameters
            )
        else:
            # Visualize Petri net
            parameters = {
                pn_visualizer.Variants.WO_DECORATION.value.Parameters.FORMAT: "png"
            }
            gviz = pn_visualizer.apply(
                model['net'],
                model['initial_marking'],
                model['final_marking'],
                parameters=parameters
            )

        # Save the visualization
        pn_visualizer.save(gviz, str(output_path))
        print(f"📊 Model visualization saved to {output_path}")

        return output_path

    def compare_models(self) -> pd.DataFrame:
        """
        Compare all discovered models

        Returns:
            DataFrame with model comparison
        """
        comparison = []

        for model_type, model in self.discovered_models.items():
            if 'error' not in model:
                row = {
                    'model': model_type,
                    'places': model.get('num_places', 0),
                    'transitions': model.get('num_transitions', 0),
                    'arcs': model.get('num_arcs', 0) if model_type != 'dfg' else model.get('num_edges', 0)
                }

                # Add conformance metrics if available
                if model_type in self.conformance_results:
                    conf = self.conformance_results[model_type]
                    if 'fitness' in conf and 'error' not in conf['fitness']:
                        row['fitness'] = conf['fitness'].get('log_fitness', 0)
                    if 'precision' in conf and not isinstance(conf['precision'], dict):
                        row['precision'] = conf['precision']
                    if 'generalization' in conf and not isinstance(conf['generalization'], dict):
                        row['generalization'] = conf['generalization']
                    if 'simplicity' in conf and not isinstance(conf['simplicity'], dict):
                        row['simplicity'] = conf['simplicity']

                comparison.append(row)

        return pd.DataFrame(comparison)

    def export_results(self, output_dir: Path) -> Dict[str, Path]:
        """
        Export all analysis results

        Args:
            output_dir: Directory to save results

        Returns:
            Dictionary with paths to exported files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        exported = {}

        # 1. Save statistics
        stats_path = output_dir / "statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(self.statistics, f, indent=2, default=str)
        exported['statistics'] = stats_path

        # 2. Save model comparison
        if self.discovered_models:
            comparison_df = self.compare_models()
            comparison_path = output_dir / "model_comparison.csv"
            comparison_df.to_csv(comparison_path, index=False)
            exported['comparison'] = comparison_path

        # 3. Save visualizations
        for model_type in self.discovered_models:
            if 'error' not in self.discovered_models[model_type]:
                try:
                    viz_path = output_dir / f"model_{model_type}.png"
                    self.visualize_model(model_type, viz_path)
                    exported[f'viz_{model_type}'] = viz_path
                except Exception as e:
                    print(f"Could not visualize {model_type}: {e}")

        # 4. Save conformance results
        if self.conformance_results:
            conf_path = output_dir / "conformance_results.json"
            with open(conf_path, 'w') as f:
                json.dump(self.conformance_results, f, indent=2, default=str)
            exported['conformance'] = conf_path

        print(f"📁 Results exported to {output_dir}")
        return exported

def analyze_event_log(event_log_path: Path, output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Complete analysis pipeline for an event log

    Args:
        event_log_path: Path to event log file
        output_dir: Optional directory for outputs

    Returns:
        Dictionary with all analysis results
    """
    analyzer = PM4PyAnalyzer()

    # Load event log
    print(f"Loading event log from {event_log_path}...")
    analyzer.load_event_log(event_log_path)

    # Discover process models
    print("\n🔍 Discovering process models...")
    models = analyzer.discover_process_models()

    # Calculate statistics
    print("\n📊 Calculating statistics...")
    stats = analyzer.calculate_statistics()

    # Calculate conformance for best model (usually heuristics)
    print("\n✅ Calculating conformance...")
    conformance = {}
    for model_type in ['heuristics', 'inductive', 'alpha']:
        if model_type in models and 'error' not in models[model_type]:
            conformance[model_type] = analyzer.calculate_conformance(model_type)
            break

    # Export results if output directory specified
    exported_files = {}
    if output_dir:
        print(f"\n💾 Exporting results to {output_dir}...")
        exported_files = analyzer.export_results(output_dir)

    return {
        'models': models,
        'statistics': stats,
        'conformance': conformance,
        'exported_files': exported_files
    }

if __name__ == "__main__":
    # Example usage
    print("PM4Py Integration Module")
    print("=" * 60)

    # Test with a sample event log if it exists
    test_log = Path("experiments") / "test_pm_20251013_205450" / "event_log.json"
    if test_log.exists():
        print(f"\nTesting with {test_log}")
        results = analyze_event_log(test_log, output_dir=Path("pm4py_output"))

        print("\n📊 Analysis Complete!")
        print(f"Models discovered: {list(results['models'].keys())}")
        print(f"Statistics calculated: {list(results['statistics'].keys())}")
        print(f"Files exported: {list(results['exported_files'].keys())}")
    else:
        print("No test event log found. Create one using test_pm_simple.py first.")