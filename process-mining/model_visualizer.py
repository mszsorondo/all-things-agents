"""
Process Model Visualizer
Creates NetworkX-based visualizations of discovered process models (Petri nets, DFG, etc.)
WITHOUT requiring Graphviz
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
import networkx as nx
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np

class ProcessModelVisualizer:
    """
    Visualizes discovered process models using NetworkX and Matplotlib
    """

    def visualize_dfg(self, dfg_data: Dict[str, Any], output_path: Path) -> Path:
        """
        Visualize a Directly-Follows Graph

        Args:
            dfg_data: DFG data with 'graph', 'start_activities', 'end_activities'
            output_path: Where to save the visualization

        Returns:
            Path to saved image
        """
        fig, ax = plt.subplots(figsize=(16, 12))

        # Extract DFG components
        dfg = dfg_data.get('graph', {})
        start_activities = dfg_data.get('start_activities', {})
        end_activities = dfg_data.get('end_activities', {})

        # Create directed graph
        G = nx.DiGraph()

        # Add edges (activities are nodes)
        for (source, target), weight in dfg.items():
            G.add_edge(source, target, weight=weight)

        # Ensure all start and end activities are in the graph
        for activity in start_activities:
            if activity not in G:
                G.add_node(activity)
        for activity in end_activities:
            if activity not in G:
                G.add_node(activity)

        # Use hierarchical layout for better readability
        try:
            pos = nx.spring_layout(G, k=2.5, iterations=50, seed=42)
        except:
            pos = nx.shell_layout(G)

        # Determine node colors
        node_colors = []
        node_sizes = []
        for node in G.nodes():
            if node in start_activities:
                node_colors.append('#90EE90')  # Light green
                node_sizes.append(4000)
            elif node in end_activities:
                node_colors.append('#FFB6C1')  # Light pink
                node_sizes.append(4000)
            else:
                node_colors.append('#87CEEB')  # Sky blue
                node_sizes.append(3500)

        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.9,
            edgecolors='black',
            linewidths=2,
            ax=ax
        )

        # Draw edges with varying thickness based on frequency
        edges = list(G.edges())
        if edges:
            weights = [G[u][v]['weight'] for u, v in edges]
            max_weight = max(weights) if weights else 1

            # Normalize edge widths
            edge_widths = [1 + (w / max_weight) * 8 for w in weights]

            # Color edges by frequency
            edge_colors = plt.cm.YlOrRd([w / max_weight for w in weights])

            nx.draw_networkx_edges(
                G, pos,
                width=edge_widths,
                alpha=0.7,
                arrows=True,
                arrowsize=25,
                arrowstyle='->',
                edge_color=edge_colors,
                connectionstyle='arc3,rad=0.1',
                ax=ax
            )

            # Draw edge labels (frequencies)
            edge_labels = {(u, v): f"{G[u][v]['weight']}" for u, v in edges}
            nx.draw_networkx_edge_labels(
                G, pos,
                edge_labels,
                font_size=10,
                font_weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                ax=ax
            )

        # Draw node labels
        nx.draw_networkx_labels(
            G, pos,
            font_size=9,
            font_weight='bold',
            font_family='sans-serif',
            ax=ax
        )

        # Add legend
        legend_elements = [
            mpatches.Patch(facecolor='#90EE90', edgecolor='black', label='Start Activities'),
            mpatches.Patch(facecolor='#87CEEB', edgecolor='black', label='Intermediate Activities'),
            mpatches.Patch(facecolor='#FFB6C1', edgecolor='black', label='End Activities'),
            mpatches.Patch(facecolor='white', edgecolor='black', label=f'Total Activities: {G.number_of_nodes()}'),
            mpatches.Patch(facecolor='white', edgecolor='black', label=f'Total Transitions: {G.number_of_edges()}')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=11, framealpha=0.9)

        ax.set_title('Directly-Follows Graph (DFG)\nShows the most common activity sequences',
                     fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

        return output_path

    def visualize_petri_net(self, model_data: Dict[str, Any], model_name: str, output_path: Path) -> Path:
        """
        Visualize a Petri net (Alpha, Heuristics, or Inductive miner result)

        Args:
            model_data: Model data with 'net', 'initial_marking', 'final_marking'
            model_name: Name of the model (for title)
            output_path: Where to save

        Returns:
            Path to saved image
        """
        net = model_data.get('net')
        initial_marking = model_data.get('initial_marking')
        final_marking = model_data.get('final_marking')

        if not net:
            raise ValueError(f"No Petri net found in {model_name} model")

        fig, ax = plt.subplots(figsize=(18, 14))

        # Build NetworkX graph
        G = nx.DiGraph()

        # Track node types
        place_nodes = {}
        transition_nodes = {}

        # Add places
        for place in net.places:
            place_id = f"p_{id(place)}"
            place_name = place.name if hasattr(place, 'name') and place.name else f"p{len(place_nodes)}"
            place_nodes[place] = place_id
            G.add_node(place_id, type='place', label=place_name, obj=place)

        # Add transitions
        for transition in net.transitions:
            trans_id = f"t_{id(transition)}"
            trans_label = transition.label if hasattr(transition, 'label') and transition.label else transition.name
            if not trans_label or trans_label == 'None':
                trans_label = f"τ{len(transition_nodes)}"  # Silent transition
            transition_nodes[transition] = trans_id
            G.add_node(trans_id, type='transition', label=trans_label, obj=transition)

        # Add arcs
        for arc in net.arcs:
            source_id = place_nodes.get(arc.source) or transition_nodes.get(arc.source)
            target_id = place_nodes.get(arc.target) or transition_nodes.get(arc.target)
            if source_id and target_id:
                G.add_edge(source_id, target_id)

        # Create layout - try hierarchical if possible
        try:
            # Try to create layers for better readability
            pos = nx.spring_layout(G, k=3, iterations=100, seed=42)
        except:
            pos = nx.circular_layout(G)

        # Separate places and transitions
        place_ids = list(place_nodes.values())
        trans_ids = list(transition_nodes.values())

        # Draw places (circles)
        place_pos = {k: v for k, v in pos.items() if k in place_ids}

        # Determine which places have tokens (from initial marking)
        place_colors = []
        for place_id in place_ids:
            place_obj = G.nodes[place_id]['obj']
            if initial_marking and place_obj in initial_marking:
                place_colors.append('#FFD700')  # Gold for initial
            elif final_marking and place_obj in final_marking:
                place_colors.append('#FF69B4')  # Hot pink for final
            else:
                place_colors.append('#87CEEB')  # Sky blue for others

        nx.draw_networkx_nodes(
            G, pos,
            nodelist=place_ids,
            node_color=place_colors,
            node_shape='o',
            node_size=2500,
            alpha=0.9,
            edgecolors='black',
            linewidths=2.5,
            ax=ax
        )

        # Draw transitions (rectangles by approximation with squares)
        trans_colors = []
        for trans_id in trans_ids:
            label = G.nodes[trans_id]['label']
            if label.startswith('τ'):
                trans_colors.append('#D3D3D3')  # Gray for silent
            else:
                trans_colors.append('#90EE90')  # Light green for visible

        nx.draw_networkx_nodes(
            G, pos,
            nodelist=trans_ids,
            node_color=trans_colors,
            node_shape='s',
            node_size=3500,
            alpha=0.9,
            edgecolors='black',
            linewidths=2.5,
            ax=ax
        )

        # Draw edges
        nx.draw_networkx_edges(
            G, pos,
            width=2.5,
            alpha=0.6,
            arrows=True,
            arrowsize=20,
            arrowstyle='->',
            edge_color='#696969',
            connectionstyle='arc3,rad=0.05',
            ax=ax
        )

        # Draw labels
        labels = {node_id: G.nodes[node_id]['label'] for node_id in G.nodes()}
        nx.draw_networkx_labels(
            G, pos,
            labels,
            font_size=8,
            font_weight='bold',
            font_family='sans-serif',
            ax=ax
        )

        # Add legend
        legend_elements = [
            mpatches.Patch(facecolor='#FFD700', edgecolor='black', label='Initial Place (start)'),
            mpatches.Patch(facecolor='#FF69B4', edgecolor='black', label='Final Place (end)'),
            mpatches.Patch(facecolor='#87CEEB', edgecolor='black', label='Intermediate Place'),
            mpatches.Patch(facecolor='#90EE90', edgecolor='black', label='Visible Transition'),
            mpatches.Patch(facecolor='#D3D3D3', edgecolor='black', label='Silent Transition (τ)'),
            mpatches.Patch(facecolor='white', edgecolor='black',
                          label=f'Places: {len(place_ids)} | Transitions: {len(trans_ids)}')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.95)

        # Title with explanation
        title = f'{model_name.title()} Miner - Petri Net\n'
        if model_name == 'alpha':
            title += 'Basic discovery algorithm (may have issues with loops)'
        elif model_name == 'heuristics':
            title += 'Handles noise and complex patterns well'
        elif model_name == 'inductive':
            title += 'Guarantees sound workflow models'

        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

        return output_path

    def visualize_all_models(self, models: Dict[str, Any], output_dir: Path) -> Dict[str, Path]:
        """
        Visualize all discovered models

        Args:
            models: Dictionary of all discovered models from PM4Py
            output_dir: Directory to save visualizations

        Returns:
            Dictionary mapping model names to output paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {}

        # Visualize DFG
        if 'dfg' in models and 'error' not in models['dfg']:
            try:
                path = self.visualize_dfg(models['dfg'], output_dir / 'model_dfg.png')
                results['dfg'] = path
                print(f"✅ DFG visualization: {path.name}")
            except Exception as e:
                print(f"❌ DFG visualization failed: {e}")

        # Visualize Petri nets
        for model_name in ['alpha', 'heuristics', 'inductive']:
            if model_name in models and 'error' not in models[model_name]:
                try:
                    path = self.visualize_petri_net(
                        models[model_name],
                        model_name,
                        output_dir / f'model_{model_name}.png'
                    )
                    results[model_name] = path
                    print(f"✅ {model_name.title()} Petri net: {path.name}")
                except Exception as e:
                    print(f"❌ {model_name.title()} visualization failed: {e}")

        return results

    def create_model_comparison_chart(self, models: Dict[str, Any], output_path: Path) -> Path:
        """
        Create a comparison chart of model complexities

        Args:
            models: Dictionary of discovered models
            output_path: Where to save the chart

        Returns:
            Path to saved chart
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # Extract statistics
        model_stats = []
        for name, data in models.items():
            if 'error' not in data:
                if name == 'dfg':
                    model_stats.append({
                        'name': 'DFG',
                        'elements': data.get('num_edges', 0),
                        'type': 'dfg'
                    })
                else:
                    model_stats.append({
                        'name': name.title(),
                        'places': data.get('num_places', 0),
                        'transitions': data.get('num_transitions', 0),
                        'arcs': data.get('num_arcs', 0),
                        'type': 'petri'
                    })

        # Chart 1: Petri net complexity
        petri_models = [m for m in model_stats if m['type'] == 'petri']
        if petri_models:
            names = [m['name'] for m in petri_models]
            x = np.arange(len(names))
            width = 0.25

            places = [m['places'] for m in petri_models]
            transitions = [m['transitions'] for m in petri_models]
            arcs = [m['arcs'] for m in petri_models]

            ax1.bar(x - width, places, width, label='Places', color='#87CEEB')
            ax1.bar(x, transitions, width, label='Transitions', color='#90EE90')
            ax1.bar(x + width, arcs, width, label='Arcs', color='#FFB6C1')

            ax1.set_xlabel('Model Type', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
            ax1.set_title('Petri Net Model Complexity Comparison', fontsize=13, fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels(names)
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3, axis='y')

        # Chart 2: Total complexity
        all_names = []
        all_totals = []
        all_colors = []

        for model in model_stats:
            all_names.append(model['name'])
            if model['type'] == 'dfg':
                all_totals.append(model['elements'])
                all_colors.append('#FFA500')  # Orange for DFG
            else:
                all_totals.append(model['places'] + model['transitions'] + model['arcs'])
                all_colors.append('#9370DB')  # Purple for Petri nets

        ax2.barh(all_names, all_totals, color=all_colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax2.set_xlabel('Total Elements', fontsize=12, fontweight='bold')
        ax2.set_title('Overall Model Size Comparison', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')

        # Add value labels
        for i, v in enumerate(all_totals):
            ax2.text(v + 0.5, i, str(v), va='center', fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

        return output_path

if __name__ == "__main__":
    print("Process Model Visualizer")
    print("This module provides NetworkX-based visualization of discovered process models")
    print("Use via pm4py_integration.py or directly with model data")