"""
Process Mining Analyzer for Multi-Agent Systems
Analyzes event logs to discover patterns, bottlenecks, and performance metrics
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# Import PM4Py if available
try:
    import pm4py
    from pm4py.algo.discovery.alpha import algorithm as alpha_miner
    from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
    from pm4py.algo.discovery.inductive import algorithm as inductive_miner
    from pm4py.visualization.petri_net import visualizer as pn_visualizer
    from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
    from pm4py.statistics.traces.generic.pandas import case_statistics
    from pm4py.algo.filtering.pandas.variants import variants_filter
    PM4PY_AVAILABLE = True
except ImportError:
    PM4PY_AVAILABLE = False
    print("⚠️ PM4Py not installed. Install with: pip install pm4py")

class MiningAnalyzer:
    """
    Analyzes event logs from multi-agent systems to discover process patterns
    """

    def __init__(self, event_log_path: Optional[Path] = None):
        """
        Initialize the analyzer

        Args:
            event_log_path: Path to event_log.json file
        """
        self.event_log = None
        self.df = None
        self.stats = {}

        if event_log_path:
            self.load_event_log(event_log_path)

    def load_event_log(self, event_log_path: Path):
        """Load an event log from JSON file"""
        with open(event_log_path, 'r') as f:
            self.event_log = json.load(f)

        # Convert to dataframe
        self._convert_to_dataframe()

    def _convert_to_dataframe(self):
        """Convert event log to pandas DataFrame"""
        if not self.event_log:
            return

        rows = []
        for event in self.event_log.get('events', []):
            row = {
                'case_id': self.event_log.get('case_id', 'unknown'),
                'activity': event.get('activity', 'unknown'),
                'timestamp': pd.to_datetime(event.get('timestamp', datetime.utcnow().isoformat())),
                'resource': event.get('resource', 'unknown'),
                'lifecycle': event.get('lifecycle', 'complete')
            }

            # Add attributes
            attributes = event.get('attributes', {})
            for key, value in attributes.items():
                if isinstance(value, (list, dict)):
                    row[f'attr_{key}'] = json.dumps(value)
                else:
                    row[f'attr_{key}'] = value

            rows.append(row)

        self.df = pd.DataFrame(rows)
        self.df = self.df.sort_values('timestamp')

    def calculate_basic_statistics(self) -> Dict[str, Any]:
        """Calculate basic process statistics"""
        if self.df is None or self.df.empty:
            return {}

        stats = {
            'total_events': len(self.df),
            'unique_activities': self.df['activity'].nunique(),
            'unique_resources': self.df['resource'].nunique(),
            'case_duration': None,
            'activity_frequencies': {},
            'resource_utilization': {},
            'lifecycle_distribution': {}
        }

        # Calculate case duration
        if not self.df.empty:
            start_time = self.df['timestamp'].min()
            end_time = self.df['timestamp'].max()
            duration = end_time - start_time
            stats['case_duration'] = {
                'seconds': duration.total_seconds(),
                'human_readable': str(duration)
            }

        # Activity frequencies
        activity_counts = self.df['activity'].value_counts()
        stats['activity_frequencies'] = activity_counts.to_dict()

        # Resource utilization
        resource_counts = self.df['resource'].value_counts()
        stats['resource_utilization'] = resource_counts.to_dict()

        # Lifecycle distribution
        lifecycle_counts = self.df['lifecycle'].value_counts()
        stats['lifecycle_distribution'] = lifecycle_counts.to_dict()

        self.stats = stats
        return stats

    def discover_process_variants(self) -> Dict[str, Any]:
        """Discover different execution paths (variants) in the process"""
        if self.df is None or self.df.empty:
            return {}

        # Group by case and create activity sequences
        variants = defaultdict(list)

        # For now, we have single case, so we'll analyze sub-patterns
        activities = self.df[self.df['lifecycle'] == 'complete']['activity'].tolist()

        # Find patterns
        patterns = {
            'full_sequence': activities,
            'agent_sequence': [],
            'tool_usage_sequence': [],
            'transition_sequence': []
        }

        for activity in activities:
            if '_execute' in activity:
                agent = activity.replace('_execute', '')
                patterns['agent_sequence'].append(agent)
            elif activity.startswith('tool_'):
                tool = activity.replace('tool_', '').replace('_invoke', '')
                patterns['tool_usage_sequence'].append(tool)
            elif activity.startswith('transition_'):
                patterns['transition_sequence'].append(activity)

        return patterns

    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze performance metrics"""
        if self.df is None or self.df.empty:
            return {}

        perf = {
            'agent_durations': {},
            'tool_invocations': {},
            'bottlenecks': [],
            'token_usage': {
                'total_in': 0,
                'total_out': 0,
                'by_agent': {}
            }
        }

        # Calculate agent durations
        agents = self.df[self.df['activity'].str.contains('_execute', na=False)]

        for agent in agents['activity'].unique():
            agent_events = agents[agents['activity'] == agent]
            starts = agent_events[agent_events['lifecycle'] == 'start']
            completes = agent_events[agent_events['lifecycle'] == 'complete']

            if not starts.empty and not completes.empty:
                # Match start and complete events
                for _, start in starts.iterrows():
                    complete = completes[completes['timestamp'] > start['timestamp']].head(1)
                    if not complete.empty:
                        duration = (complete.iloc[0]['timestamp'] - start['timestamp']).total_seconds()
                        agent_name = agent.replace('_execute', '')

                        if agent_name not in perf['agent_durations']:
                            perf['agent_durations'][agent_name] = []
                        perf['agent_durations'][agent_name].append(duration)

        # Calculate average durations
        for agent, durations in perf['agent_durations'].items():
            if durations:
                perf['agent_durations'][agent] = {
                    'avg_seconds': np.mean(durations),
                    'min_seconds': np.min(durations),
                    'max_seconds': np.max(durations),
                    'total_seconds': np.sum(durations)
                }

        # Count tool invocations
        tools = self.df[self.df['activity'].str.startswith('tool_', na=False)]
        for tool in tools['activity'].unique():
            tool_name = tool.replace('tool_', '').replace('_invoke', '')
            perf['tool_invocations'][tool_name] = len(tools[tools['activity'] == tool])

        # Identify bottlenecks (agents with longest duration)
        if perf['agent_durations']:
            sorted_agents = sorted(
                perf['agent_durations'].items(),
                key=lambda x: x[1].get('avg_seconds', 0) if isinstance(x[1], dict) else 0,
                reverse=True
            )
            perf['bottlenecks'] = [agent for agent, _ in sorted_agents[:3]]

        # Analyze token usage
        for col in self.df.columns:
            if 'tokens_in' in col:
                perf['token_usage']['total_in'] += self.df[col].sum()
            elif 'tokens_out' in col:
                perf['token_usage']['total_out'] += self.df[col].sum()

        return perf

    def discover_process_model(self, algorithm: str = 'heuristics') -> Any:
        """
        Discover process model using PM4Py

        Args:
            algorithm: Mining algorithm to use ('alpha', 'heuristics', 'inductive')

        Returns:
            Petri net or process tree depending on algorithm
        """
        if not PM4PY_AVAILABLE:
            return {"error": "PM4Py not installed"}

        if self.df is None or self.df.empty:
            return {"error": "No data loaded"}

        # Prepare dataframe for PM4Py
        pm4py_df = self.df.copy()
        pm4py_df = pm4py_df.rename(columns={
            'case_id': 'case:concept:name',
            'activity': 'concept:name',
            'timestamp': 'time:timestamp',
            'resource': 'org:resource'
        })

        # Filter to complete events only for cleaner model
        pm4py_df = pm4py_df[pm4py_df['lifecycle'] == 'complete']

        try:
            if algorithm == 'alpha':
                net, initial_marking, final_marking = alpha_miner.apply(pm4py_df)
            elif algorithm == 'heuristics':
                net, initial_marking, final_marking = heuristics_miner.apply(
                    pm4py_df,
                    parameters={
                        heuristics_miner.Variants.CLASSIC.value.Parameters.DEPENDENCY_THRESH: 0.5,
                        heuristics_miner.Variants.CLASSIC.value.Parameters.AND_MEASURE_THRESH: 0.65
                    }
                )
            elif algorithm == 'inductive':
                net, initial_marking, final_marking = inductive_miner.apply(pm4py_df)
            else:
                return {"error": f"Unknown algorithm: {algorithm}"}

            return {
                'net': net,
                'initial_marking': initial_marking,
                'final_marking': final_marking,
                'places': len(net.places),
                'transitions': len(net.transitions),
                'arcs': len(net.arcs)
            }

        except Exception as e:
            return {"error": str(e)}

    def calculate_conformance(self, net=None, initial_marking=None, final_marking=None) -> Dict[str, float]:
        """
        Calculate conformance metrics using token replay

        Args:
            net: Petri net (if None, will discover using heuristics miner)
            initial_marking: Initial marking
            final_marking: Final marking

        Returns:
            Dictionary with fitness and precision scores
        """
        if not PM4PY_AVAILABLE:
            return {"error": "PM4Py not installed"}

        if net is None:
            # Discover model first
            model = self.discover_process_model('heuristics')
            if 'error' in model:
                return model
            net = model['net']
            initial_marking = model['initial_marking']
            final_marking = model['final_marking']

        # Prepare dataframe
        pm4py_df = self.df.copy()
        pm4py_df = pm4py_df.rename(columns={
            'case_id': 'case:concept:name',
            'activity': 'concept:name',
            'timestamp': 'time:timestamp',
            'resource': 'org:resource'
        })
        pm4py_df = pm4py_df[pm4py_df['lifecycle'] == 'complete']

        try:
            # Token replay for fitness
            replayed_traces = token_replay.apply(
                pm4py_df,
                net,
                initial_marking,
                final_marking
            )

            # Calculate fitness
            fitness = sum(trace['trace_fitness'] for trace in replayed_traces) / len(replayed_traces)

            return {
                'fitness': fitness,
                'total_cases': len(replayed_traces),
                'perfectly_fitting': sum(1 for trace in replayed_traces if trace['trace_is_fit'])
            }

        except Exception as e:
            return {"error": str(e)}

    def compare_architectures(self, event_logs: List[Path]) -> pd.DataFrame:
        """
        Compare multiple event logs from different architectures

        Args:
            event_logs: List of paths to event log files

        Returns:
            DataFrame comparing key metrics
        """
        comparisons = []

        for log_path in event_logs:
            self.load_event_log(log_path)

            # Get basic info
            case_id = self.event_log.get('case_id', 'unknown')
            architecture = self.event_log.get('architecture', 'unknown')
            problem = self.event_log.get('problem', 'unknown')

            # Calculate metrics
            stats = self.calculate_basic_statistics()
            perf = self.analyze_performance()

            comparison = {
                'case_id': case_id,
                'architecture': architecture,
                'problem': problem,
                'total_events': stats.get('total_events', 0),
                'unique_activities': stats.get('unique_activities', 0),
                'duration_seconds': stats.get('case_duration', {}).get('seconds', 0),
                'tokens_in': perf.get('token_usage', {}).get('total_in', 0),
                'tokens_out': perf.get('token_usage', {}).get('total_out', 0),
                'tools_used': len(perf.get('tool_invocations', {})),
                'bottleneck': perf.get('bottlenecks', ['none'])[0] if perf.get('bottlenecks') else 'none'
            }

            comparisons.append(comparison)

        return pd.DataFrame(comparisons)

    def generate_report(self, output_path: Optional[Path] = None) -> str:
        """
        Generate a comprehensive analysis report

        Args:
            output_path: Optional path to save the report

        Returns:
            Report as string
        """
        report = []
        report.append("=" * 60)
        report.append("PROCESS MINING ANALYSIS REPORT")
        report.append("=" * 60)

        # Basic information
        if self.event_log:
            report.append(f"\nCase ID: {self.event_log.get('case_id', 'unknown')}")
            report.append(f"Architecture: {self.event_log.get('architecture', 'unknown')}")
            report.append(f"Problem: {self.event_log.get('problem', 'unknown')}")

        # Statistics
        stats = self.calculate_basic_statistics()
        report.append(f"\n📊 BASIC STATISTICS")
        report.append(f"  Total Events: {stats.get('total_events', 0)}")
        report.append(f"  Unique Activities: {stats.get('unique_activities', 0)}")
        report.append(f"  Case Duration: {stats.get('case_duration', {}).get('human_readable', 'N/A')}")

        report.append(f"\n  Top Activities:")
        for activity, count in list(stats.get('activity_frequencies', {}).items())[:5]:
            report.append(f"    - {activity}: {count}")

        # Performance
        perf = self.analyze_performance()
        report.append(f"\n⚡ PERFORMANCE ANALYSIS")
        report.append(f"  Agent Durations:")
        for agent, duration in perf.get('agent_durations', {}).items():
            if isinstance(duration, dict):
                report.append(f"    - {agent}: {duration.get('avg_seconds', 0):.2f}s avg")

        report.append(f"\n  Tool Invocations:")
        for tool, count in perf.get('tool_invocations', {}).items():
            report.append(f"    - {tool}: {count} times")

        report.append(f"\n  Token Usage:")
        report.append(f"    - Total Input: {perf.get('token_usage', {}).get('total_in', 0)}")
        report.append(f"    - Total Output: {perf.get('token_usage', {}).get('total_out', 0)}")

        report.append(f"\n  Bottlenecks: {', '.join(perf.get('bottlenecks', ['none']))}")

        # Process variants
        variants = self.discover_process_variants()
        report.append(f"\n🔄 PROCESS VARIANTS")
        report.append(f"  Agent Sequence: {' -> '.join(variants.get('agent_sequence', []))}")
        report.append(f"  Tools Used: {', '.join(variants.get('tool_usage_sequence', []))}")

        # Join report
        report_text = "\n".join(report)

        # Save if path provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)

        return report_text

if __name__ == "__main__":
    # Example usage
    print("Process Mining Analyzer - Example Usage")
    print("This module analyzes event logs from multi-agent systems.")
    print("\nFeatures:")
    print("- Basic statistics calculation")
    print("- Performance analysis")
    print("- Process variant discovery")
    print("- Process model discovery (requires PM4Py)")
    print("- Conformance checking")
    print("- Architecture comparison")
    print("- Report generation")