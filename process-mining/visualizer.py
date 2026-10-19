"""
Process Model Visualization Module
Creates visualizations for process mining results without requiring Graphviz
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from datetime import datetime
import networkx as nx

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class ProcessVisualizer:
    """
    Creates various visualizations for process mining results
    """

    def __init__(self, event_log_path: Optional[Path] = None):
        """
        Initialize visualizer

        Args:
            event_log_path: Path to event log file
        """
        self.event_log_path = event_log_path
        self.event_log = None
        self.df = None

        if event_log_path:
            self.load_event_log(event_log_path)

    def load_event_log(self, event_log_path: Path):
        """Load event log from JSON"""
        with open(event_log_path, 'r') as f:
            self.event_log = json.load(f)

        # Convert to dataframe for easier analysis
        events = self.event_log.get('events', [])
        self.df = pd.DataFrame(events)

        if 'timestamp' in self.df.columns:
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])

    def plot_activity_frequency(self, output_path: Optional[Path] = None) -> Path:
        """
        Create bar chart of activity frequencies

        Args:
            output_path: Where to save the plot

        Returns:
            Path to saved plot
        """
        if self.df is None or self.df.empty:
            raise ValueError("No event log loaded")

        # Count activities
        activity_counts = self.df['activity'].value_counts()

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        activity_counts.plot(kind='barh', ax=ax)

        ax.set_xlabel('Frequency')
        ax.set_ylabel('Activity')
        ax.set_title('Activity Frequency in Process')

        # Add value labels
        for i, v in enumerate(activity_counts.values):
            ax.text(v + 0.1, i, str(v), va='center')

        plt.tight_layout()

        if output_path is None:
            output_path = Path("activity_frequency.png")

        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()

        return output_path

    def plot_resource_utilization(self, output_path: Optional[Path] = None) -> Path:
        """
        Create pie chart of resource utilization

        Args:
            output_path: Where to save the plot

        Returns:
            Path to saved plot
        """
        if self.df is None or self.df.empty:
            raise ValueError("No event log loaded")

        # Count resources
        resource_counts = self.df['resource'].value_counts()

        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Pie chart
        colors = sns.color_palette('husl', len(resource_counts))
        wedges, texts, autotexts = ax1.pie(
            resource_counts.values,
            labels=resource_counts.index,
            autopct='%1.1f%%',
            colors=colors,
            startangle=90
        )

        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_weight('bold')

        ax1.set_title('Resource Utilization Distribution')

        # Bar chart
        resource_counts.plot(kind='bar', ax=ax2, color=colors)
        ax2.set_xlabel('Resource')
        ax2.set_ylabel('Number of Activities')
        ax2.set_title('Resource Activity Count')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')

        # Add value labels
        for i, v in enumerate(resource_counts.values):
            ax2.text(i, v + 0.5, str(v), ha='center')

        plt.tight_layout()

        if output_path is None:
            output_path = Path("resource_utilization.png")

        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()

        return output_path

    def plot_process_flow_graph(self, output_path: Optional[Path] = None) -> Path:
        """
        Create a simple process flow graph using NetworkX

        Args:
            output_path: Where to save the plot

        Returns:
            Path to saved plot
        """
        if self.event_log is None:
            raise ValueError("No event log loaded")

        # Build graph from event log
        G = nx.DiGraph()

        # Add nodes from graph data if available
        if 'graph' in self.event_log:
            graph_data = self.event_log['graph']

            # Add nodes
            for node in graph_data.get('nodes', []):
                node_id = node.get('id', '')
                node_type = node.get('type', 'unknown')
                node_name = node.get('name', node_id)

                # Color based on type
                if node_type == 'agent':
                    color = 'lightblue'
                elif node_type == 'tool':
                    color = 'lightgreen'
                elif node_type == 'actor':
                    color = 'lightyellow'
                else:
                    color = 'lightgray'

                G.add_node(node_name, type=node_type, color=color)

            # Add edges
            for edge in graph_data.get('edges', []):
                from_node = edge.get('from', '')
                to_node = edge.get('to', '')
                edge_type = edge.get('type', 'unknown')

                # Map node IDs to names
                from_name = from_node
                to_name = to_node
                for node in graph_data.get('nodes', []):
                    if node.get('id') == from_node:
                        from_name = node.get('name', from_node)
                    if node.get('id') == to_node:
                        to_name = node.get('name', to_node)

                G.add_edge(from_name, to_name, type=edge_type)

        else:
            # Build from event sequence
            events = self.event_log.get('events', [])
            prev_activity = None

            for event in events:
                activity = event.get('activity', 'unknown')
                resource = event.get('resource', 'unknown')

                # Add node
                if activity not in G:
                    G.add_node(activity, resource=resource, color='lightblue')

                # Add edge from previous activity
                if prev_activity and prev_activity != activity:
                    if not G.has_edge(prev_activity, activity):
                        G.add_edge(prev_activity, activity, weight=1)
                    else:
                        G[prev_activity][activity]['weight'] += 1

                prev_activity = activity

        # Create plot
        fig, ax = plt.subplots(figsize=(14, 10))

        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50)

        # Get node colors
        node_colors = [G.nodes[node].get('color', 'lightblue') for node in G.nodes()]

        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=3000,
            alpha=0.8,
            ax=ax
        )

        # Draw edges
        edges = G.edges()
        weights = [G[u][v].get('weight', 1) for u, v in edges]

        nx.draw_networkx_edges(
            G, pos,
            width=[w * 0.5 for w in weights],
            alpha=0.6,
            arrows=True,
            arrowsize=20,
            edge_color='gray',
            ax=ax
        )

        # Draw labels
        nx.draw_networkx_labels(
            G, pos,
            font_size=8,
            font_weight='bold',
            ax=ax
        )

        ax.set_title('Process Flow Graph', fontsize=16, fontweight='bold')
        ax.axis('off')

        plt.tight_layout()

        if output_path is None:
            output_path = Path("process_flow_graph.png")

        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()

        return output_path

    def plot_timeline(self, output_path: Optional[Path] = None) -> Path:
        """
        Create a timeline visualization of events

        Args:
            output_path: Where to save the plot

        Returns:
            Path to saved plot
        """
        if self.df is None or self.df.empty:
            raise ValueError("No event log loaded")

        # Prepare data
        df_viz = self.df.copy()
        df_viz = df_viz.sort_values('timestamp')

        # Assign y-positions for activities
        unique_activities = df_viz['activity'].unique()
        activity_positions = {act: i for i, act in enumerate(unique_activities)}
        df_viz['y_pos'] = df_viz['activity'].map(activity_positions)

        # Create plot
        fig, ax = plt.subplots(figsize=(14, 8))

        # Plot events
        for _, row in df_viz.iterrows():
            # Determine marker and color based on lifecycle
            if row.get('lifecycle', 'complete') == 'start':
                marker = 'o'
                color = 'green'
            else:
                marker = 's'
                color = 'blue'

            ax.scatter(
                row['timestamp'],
                row['y_pos'],
                marker=marker,
                s=100,
                c=color,
                alpha=0.7,
                edgecolors='black',
                linewidth=1
            )

            # Add resource label
            if 'resource' in row:
                ax.text(
                    row['timestamp'],
                    row['y_pos'] + 0.2,
                    row['resource'],
                    fontsize=8,
                    ha='center'
                )

        # Connect events with lines
        for case_id in df_viz.get('case_id', [df_viz.iloc[0].get('case_id', 'default')]).unique():
            case_events = df_viz[df_viz.get('case_id', 'default') == case_id]
            ax.plot(
                case_events['timestamp'],
                case_events['y_pos'],
                'k-',
                alpha=0.3,
                linewidth=1
            )

        # Customize plot
        ax.set_yticks(range(len(unique_activities)))
        ax.set_yticklabels(unique_activities)
        ax.set_xlabel('Timestamp', fontsize=12)
        ax.set_ylabel('Activity', fontsize=12)
        ax.set_title('Process Timeline', fontsize=16, fontweight='bold')

        # Format x-axis
        fig.autofmt_xdate()

        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
                   markersize=10, label='Start', markeredgecolor='black'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='blue',
                   markersize=10, label='Complete', markeredgecolor='black')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if output_path is None:
            output_path = Path("process_timeline.png")

        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()

        return output_path

    def plot_token_usage(self, output_path: Optional[Path] = None) -> Path:
        """
        Plot token usage by agents

        Args:
            output_path: Where to save the plot

        Returns:
            Path to saved plot
        """
        if self.df is None or self.df.empty:
            raise ValueError("No event log loaded")

        # Extract token data from attributes
        token_data = []

        for _, row in self.df.iterrows():
            if 'attributes' in row and isinstance(row['attributes'], dict):
                attrs = row['attributes']
                if 'tokens_in' in attrs or 'tokens_out' in attrs:
                    token_data.append({
                        'activity': row.get('activity', 'unknown'),
                        'agent': attrs.get('agent_type', row.get('resource', 'unknown')),
                        'tokens_in': attrs.get('tokens_in', 0),
                        'tokens_out': attrs.get('tokens_out', 0)
                    })

        if not token_data:
            # Create dummy plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No token usage data available',
                    ha='center', va='center', fontsize=14)
            ax.axis('off')
        else:
            df_tokens = pd.DataFrame(token_data)

            # Aggregate by agent
            agent_tokens = df_tokens.groupby('agent').agg({
                'tokens_in': 'sum',
                'tokens_out': 'sum'
            }).reset_index()

            # Create plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            # Stacked bar chart
            x = range(len(agent_tokens))
            width = 0.35

            ax1.bar(x, agent_tokens['tokens_in'], width, label='Tokens In', color='skyblue')
            ax1.bar(x, agent_tokens['tokens_out'], width, bottom=agent_tokens['tokens_in'],
                    label='Tokens Out', color='lightcoral')

            ax1.set_xlabel('Agent')
            ax1.set_ylabel('Token Count')
            ax1.set_title('Token Usage by Agent')
            ax1.set_xticks(x)
            ax1.set_xticklabels(agent_tokens['agent'], rotation=45, ha='right')
            ax1.legend()

            # Total usage pie chart
            total_in = agent_tokens['tokens_in'].sum()
            total_out = agent_tokens['tokens_out'].sum()

            ax2.pie(
                [total_in, total_out],
                labels=['Tokens In', 'Tokens Out'],
                autopct='%1.1f%%',
                colors=['skyblue', 'lightcoral'],
                startangle=90
            )
            ax2.set_title(f'Total Token Distribution\n({total_in + total_out:,} tokens)')

        plt.tight_layout()

        if output_path is None:
            output_path = Path("token_usage.png")

        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()

        return output_path

    def create_all_visualizations(self, output_dir: Path) -> Dict[str, Path]:
        """
        Create all available visualizations

        Args:
            output_dir: Directory to save visualizations

        Returns:
            Dictionary mapping visualization names to file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {}

        # Activity frequency
        try:
            path = self.plot_activity_frequency(output_dir / "activity_frequency.png")
            results['activity_frequency'] = path
            print(f"✅ Created: {path}")
        except Exception as e:
            print(f"❌ Failed to create activity frequency: {e}")

        # Resource utilization
        try:
            path = self.plot_resource_utilization(output_dir / "resource_utilization.png")
            results['resource_utilization'] = path
            print(f"✅ Created: {path}")
        except Exception as e:
            print(f"❌ Failed to create resource utilization: {e}")

        # Process flow graph
        try:
            path = self.plot_process_flow_graph(output_dir / "process_flow.png")
            results['process_flow'] = path
            print(f"✅ Created: {path}")
        except Exception as e:
            print(f"❌ Failed to create process flow: {e}")

        # Timeline
        try:
            path = self.plot_timeline(output_dir / "timeline.png")
            results['timeline'] = path
            print(f"✅ Created: {path}")
        except Exception as e:
            print(f"❌ Failed to create timeline: {e}")

        # Token usage
        try:
            path = self.plot_token_usage(output_dir / "token_usage.png")
            results['token_usage'] = path
            print(f"✅ Created: {path}")
        except Exception as e:
            print(f"❌ Failed to create token usage: {e}")

        return results

if __name__ == "__main__":
    # Test visualization
    print("Process Visualizer Test")
    print("=" * 60)

    # Find test event log
    from pathlib import Path
    test_logs = sorted(Path("../experiments").glob("test_pm_*/event_log.json"))

    if test_logs:
        event_log_path = test_logs[-1]
        output_dir = event_log_path.parent / "visualizations"

        print(f"Using: {event_log_path}")
        print(f"Output: {output_dir}")

        visualizer = ProcessVisualizer(event_log_path)
        results = visualizer.create_all_visualizations(output_dir)

        print("\n" + "=" * 60)
        print(f"Created {len(results)} visualizations")
    else:
        print("No test event logs found")