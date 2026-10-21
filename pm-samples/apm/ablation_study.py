#!/usr/bin/env python3
"""
Ablation Study for Process Mining Representations

This script performs an ablation study to evaluate the contribution of each
process representation to the overall performance of the LLM-based trace validator.

The study:
1. Runs the miner with ALL representations (baseline)
2. For each representation, removes it and runs benchmark
3. Compares results to understand each representation's contribution

All results are stored in ./ablation_results
"""

import os
import sys
import json
import time
import pickle
import hashlib
import argparse
import shutil
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from copy import deepcopy

import pm4py
from pm4py.objects.bpmn.exporter import exporter as bpmn_exporter
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent

load_dotenv()

# =============================================================================
# REPRESENTATION TYPES - Extended from enhanced_agentic_miner.py + agentic_miner_template.py
# =============================================================================

class RepresentationType(Enum):
    """All available process mining representation types for ablation study."""
    # From enhanced_agentic_miner.py
    PETRI_NET_INDUCTIVE = "petri_net_inductive"
    BPMN = "bpmn"
    DFG = "dfg"
    HEURISTICS_NET = "heuristics_net"
    PROCESS_TREE = "process_tree"
    ABSTRACT_VARIANTS = "abstract_variants"

    # New representations from agentic_miner_template.py
    PETRI_NET_ALPHA = "petri_net_alpha"
    PETRI_NET_HEURISTICS = "petri_net_heuristics"
    PETRI_NET_ILP = "petri_net_ilp"
    LOG_SKELETON = "log_skeleton"
    DECLARE = "declare"
    LOG_ATTRIBUTES = "log_attributes"
    PREFIX_TREE = "prefix_tree"  # NOTE: Excluded from ablation until string format is resolved


# Representations to EXCLUDE from ablation study (keep code but don't test yet)
EXCLUDED_FROM_ABLATION = {
    RepresentationType.PREFIX_TREE,  # TODO: Need appropriate string representation format
    RepresentationType.BPMN,  # Excluded: excessive token consumption due to XML syntax
}


# =============================================================================
# INTERPRETATION GUIDES LOADER
# =============================================================================

GUIDES_DIR = Path(__file__).parent / "representation_interpretation_guides"

def load_interpretation_guide(rep_type: RepresentationType) -> str:
    """Load the interpretation guide for a representation type."""
    # Map representation types to guide files
    guide_files = {
        RepresentationType.PETRI_NET_INDUCTIVE: "petri_net_inductive.txt",
        RepresentationType.BPMN: "bpmn.txt",
        RepresentationType.DFG: "dfg.txt",
        RepresentationType.HEURISTICS_NET: "heuristics_net.txt",
        RepresentationType.PROCESS_TREE: "process_tree.txt",
        RepresentationType.ABSTRACT_VARIANTS: "abstract_variants.txt",
        RepresentationType.PETRI_NET_ALPHA: "petri_net_alpha.txt",
        RepresentationType.PETRI_NET_HEURISTICS: "petri_net_heuristics.txt",
        RepresentationType.PETRI_NET_ILP: "petri_net_ilp.txt",
        RepresentationType.LOG_SKELETON: "log_skeleton.txt",
        RepresentationType.DECLARE: "declare.txt",
        RepresentationType.LOG_ATTRIBUTES: "log_attributes.txt",
        RepresentationType.PREFIX_TREE: "prefix_tree.txt",
    }

    guide_file = GUIDES_DIR / guide_files.get(rep_type, f"{rep_type.value}.txt")

    if guide_file.exists():
        return guide_file.read_text()
    else:
        return f"[No interpretation guide available for {rep_type.value}]"


# =============================================================================
# REPRESENTATION DISCOVERY
# =============================================================================

class RepresentationDiscoverer:
    """Discovers all process representations from an event log."""

    def __init__(self, log, log_path: str):
        self.log = log
        self.log_path = log_path
        self.representations: Dict[RepresentationType, Any] = {}
        self.string_representations: Dict[RepresentationType, str] = {}
        self.interpretation_guides: Dict[RepresentationType, str] = {}

    def discover_all(self):
        """Discover all representations."""
        print("\n" + "="*60)
        print("DISCOVERING ALL REPRESENTATIONS")
        print("="*60)

        for rep_type in RepresentationType:
            try:
                print(f"  Discovering {rep_type.value}...", end=" ")
                self._discover_one(rep_type)
                print("OK")
            except Exception as e:
                print(f"FAILED: {e}")
                self.representations[rep_type] = None
                self.string_representations[rep_type] = f"[Discovery failed: {e}]"

            # Load interpretation guide
            self.interpretation_guides[rep_type] = load_interpretation_guide(rep_type)

    def _discover_one(self, rep_type: RepresentationType):
        """Discover a single representation."""

        if rep_type == RepresentationType.PETRI_NET_INDUCTIVE:
            net, im, fm = pm4py.discover_petri_net_inductive(self.log)
            self.representations[rep_type] = (net, im, fm)
            self.string_representations[rep_type] = self._petri_net_to_string(net, im, fm, "Inductive Miner")

        elif rep_type == RepresentationType.PETRI_NET_ALPHA:
            net, im, fm = pm4py.discover_petri_net_alpha(self.log)
            self.representations[rep_type] = (net, im, fm)
            self.string_representations[rep_type] = self._petri_net_to_string(net, im, fm, "Alpha Miner")

        elif rep_type == RepresentationType.PETRI_NET_HEURISTICS:
            net, im, fm = pm4py.discover_petri_net_heuristics(self.log)
            self.representations[rep_type] = (net, im, fm)
            self.string_representations[rep_type] = self._petri_net_to_string(net, im, fm, "Heuristics Miner")

        elif rep_type == RepresentationType.PETRI_NET_ILP:
            net, im, fm = pm4py.discover_petri_net_ilp(self.log)
            self.representations[rep_type] = (net, im, fm)
            self.string_representations[rep_type] = self._petri_net_to_string(net, im, fm, "ILP Miner")

        elif rep_type == RepresentationType.BPMN:
            bpmn = pm4py.discover_bpmn_inductive(self.log)
            self.representations[rep_type] = bpmn
            self.string_representations[rep_type] = self._bpmn_to_string(bpmn)

        elif rep_type == RepresentationType.DFG:
            dfg_abstract = pm4py.llm.abstract_dfg(self.log)
            self.representations[rep_type] = dfg_abstract
            self.string_representations[rep_type] = str(dfg_abstract)

        elif rep_type == RepresentationType.HEURISTICS_NET:
            hnet = pm4py.discover_heuristics_net(self.log)
            self.representations[rep_type] = hnet
            self.string_representations[rep_type] = self._heuristics_net_to_string(hnet)

        elif rep_type == RepresentationType.PROCESS_TREE:
            tree = pm4py.discover_process_tree_inductive(self.log)
            self.representations[rep_type] = tree
            self.string_representations[rep_type] = str(tree)

        elif rep_type == RepresentationType.ABSTRACT_VARIANTS:
            variants = pm4py.llm.abstract_variants(self.log,relative_frequency=True)
            self.representations[rep_type] = variants
            self.string_representations[rep_type] = str(variants)

        elif rep_type == RepresentationType.LOG_SKELETON:
            skeleton = pm4py.discover_log_skeleton(self.log)
            self.representations[rep_type] = skeleton
            self.string_representations[rep_type] = pm4py.llm.abstract_log_skeleton(skeleton)

        elif rep_type == RepresentationType.DECLARE:
            declare = pm4py.discover_declare(self.log)
            self.representations[rep_type] = declare
            self.string_representations[rep_type] = pm4py.llm.abstract_declare(declare)

        elif rep_type == RepresentationType.LOG_ATTRIBUTES:
            attributes = pm4py.llm.abstract_log_attributes(self.log)
            self.representations[rep_type] = attributes
            self.string_representations[rep_type] = str(attributes)

        elif rep_type == RepresentationType.PREFIX_TREE:
            trie = pm4py.discovery.discover_prefix_tree(self.log)
            self.representations[rep_type] = trie
            self.string_representations[rep_type] = self._prefix_tree_to_string(trie)

    def _petri_net_to_string(self, net, im, fm, algorithm: str) -> str:
        """Convert Petri net to string representation."""
        try:
            # Use PM4Py's LLM abstraction if available
            abstract = pm4py.llm.abstract_petri_net(net, im, fm)
            return f"[{algorithm}]\n{abstract}"
        except:
            # Fallback to manual conversion
            places = [f"{p.name}" for p in net.places]
            transitions = [f"{t.name} (label: {t.label})" for t in net.transitions]
            arcs = [f"{a.source.name} -> {a.target.name}" for a in net.arcs]

            return f"""[{algorithm}]
Petri Net Structure:
- Places ({len(places)}): {', '.join(places[:20])}{'...' if len(places) > 20 else ''}
- Transitions ({len(transitions)}): {', '.join(transitions[:20])}{'...' if len(transitions) > 20 else ''}
- Arcs ({len(arcs)}): {', '.join(arcs[:30])}{'...' if len(arcs) > 30 else ''}
- Initial Marking: {str(im)}
- Final Marking: {str(fm)}"""

    def _bpmn_to_string(self, bpmn) -> str:
        """Convert BPMN to string representation."""
        temp_file = "temp_bpmn_export.bpmn"
        try:
            bpmn_exporter.apply(bpmn, temp_file)
            with open(temp_file, 'r') as f:
                bpmn_str = f.read()
            os.remove(temp_file)
            return bpmn_str
        except Exception as e:
            return f"[BPMN export failed: {e}]"

    def _heuristics_net_to_string(self, hnet) -> str:
        """Convert Heuristics Net to string representation."""
        try:
            nodes = list(hnet.nodes)
            activities = hnet.activities
            dfg_matrix = hnet.dfg_matrix
            dependency_matrix = hnet.dependency_matrix

            # Build a readable representation
            lines = ["Heuristics Net:"]
            lines.append(f"\nActivities ({len(activities)}): {', '.join(activities)}")

            # Start and end activities
            if hnet.start_activities:
                start_acts = list(hnet.start_activities[0].keys()) if hnet.start_activities else []
                lines.append(f"\nStart activities: {', '.join(start_acts)}")
            if hnet.end_activities:
                end_acts = list(hnet.end_activities[0].keys()) if hnet.end_activities else []
                lines.append(f"End activities: {', '.join(end_acts)}")

            # Dependencies with measures
            lines.append("\n\nDependency relations (activity -> activity: dependency_value):")
            dep_count = 0
            for source, targets in dependency_matrix.items():
                for target, dep_value in targets.items():
                    if dep_value > 0.5:  # Only show strong dependencies
                        freq = dfg_matrix.get(source, {}).get(target, 0)
                        lines.append(f"  {source} -> {target}: dependency={dep_value:.3f}, frequency={freq}")
                        dep_count += 1
                        if dep_count > 50:  # Limit output
                            lines.append("  ... (truncated)")
                            break
                if dep_count > 50:
                    break

            return "\n".join(lines)
        except Exception as e:
            return f"[Heuristics Net conversion failed: {e}]"

    def _prefix_tree_to_string(self, trie) -> str:
        """Convert prefix tree to string representation."""
        # PM4Py's prefix tree is a complex structure - we'll create a summary
        try:
            def count_nodes(node, depth=0):
                count = 1
                for child in node.children.values():
                    count += count_nodes(child, depth + 1)
                return count

            def get_max_depth(node, depth=0):
                if not node.children:
                    return depth
                return max(get_max_depth(child, depth + 1) for child in node.children.values())

            def get_sample_paths(node, path="", paths=None, max_paths=10):
                if paths is None:
                    paths = []
                if len(paths) >= max_paths:
                    return paths

                current_path = f"{path} -> {node.label}" if path else str(node.label or "START")

                if not node.children:
                    paths.append(current_path)
                else:
                    for child in node.children.values():
                        get_sample_paths(child, current_path, paths, max_paths)
                return paths

            total_nodes = count_nodes(trie)
            max_depth = get_max_depth(trie)
            sample_paths = get_sample_paths(trie)

            return f"""Prefix Tree (Trie) Summary:
- Total nodes: {total_nodes}
- Maximum depth: {max_depth}
- Sample paths (first {len(sample_paths)}):
{chr(10).join('  ' + p for p in sample_paths)}"""
        except Exception as e:
            return f"[Prefix tree summary failed: {e}]"

    def get_full_prompt_for_representations(self, rep_types: List[RepresentationType]) -> str:
        """Build full prompt with interpretation guides and representations."""
        prompt_parts = []

        for i, rep_type in enumerate(rep_types, 1):
            if rep_type in self.string_representations and self.string_representations[rep_type]:
                guide = self.interpretation_guides.get(rep_type, "")
                rep_str = self.string_representations[rep_type]

                prompt_parts.append(f"""
{'='*60}
REPRESENTATION {i}: {rep_type.value.upper()}
{'='*60}

{guide}

--- DISCOVERED MODEL ---
{rep_str}
""")

        return "\n".join(prompt_parts)


# =============================================================================
# ABLATION STUDY MINER
# =============================================================================

@dataclass
class AblationMiningConfig:
    """Configuration for ablation mining."""
    model_name: str = "openai:gpt-5-mini"
    prompt_file: str = "GENERATE_AS_IS_MODEL.txt"
    use_interpretations: bool = True


class AblationMiner:
    """Miner for ablation study that generates textual models."""

    def __init__(self, discoverer: RepresentationDiscoverer, config: AblationMiningConfig,
                 output_dir: Path):
        self.discoverer = discoverer
        self.config = config
        self.output_dir = output_dir

    def generate_textual_model(self, representations: List[RepresentationType],
                                run_name: str) -> Tuple[str, Path]:
        """Generate textual model using specified representations."""
        run_folder = self.output_dir / run_name
        run_folder.mkdir(parents=True, exist_ok=True)

        # Load system prompt
        prompt_path = Path(self.config.prompt_file)
        if prompt_path.exists():
            system_prompt = prompt_path.read_text()
        else:
            system_prompt = "You are a process mining expert. Generate a comprehensive AS-IS textual process model."

        # Build user prompt with representations
        user_prompt = self.discoverer.get_full_prompt_for_representations(representations)

        # Save prompts
        (run_folder / "system_prompt.txt").write_text(system_prompt)
        (run_folder / "user_prompt.txt").write_text(user_prompt)
        (run_folder / "representations_used.json").write_text(
            json.dumps([r.value for r in representations], indent=2)
        )

        # ================================================================
        # BREAKPOINT: Verify representations include interpretation guides
        # ================================================================
        print("\n" + "="*60)
        print("BREAKPOINT: VERIFY REPRESENTATIONS AND GUIDES")
        print("="*60)
        print(f"Run: {run_name}")
        print(f"Representations: {[r.value for r in representations]}")
        print(f"User prompt length: {len(user_prompt)} characters")
        print("\nCheck that each representation has its interpretation guide included.")
        print("User prompt saved to:", run_folder / "user_prompt.txt")
        #breakpoint()  # <-- BREAKPOINT FOR VERIFICATION
        # ================================================================

        # Create LLM
        llm = self._create_llm()

        # Generate model
        messages = [
            ("system", system_prompt),
            ("human", user_prompt)
        ]

        print(f"  Generating textual model for {run_name}...")
        start_time = time.time()
        response = llm.invoke(messages)
        elapsed = time.time() - start_time

        textual_model = response.content.strip()

        # Save textual model
        model_path = run_folder / "textual_model.txt"
        model_path.write_text(textual_model)

        # Save metadata
        metadata = {
            "run_name": run_name,
            "representations": [r.value for r in representations],
            "model_name": self.config.model_name,
            "generation_time_seconds": elapsed,
            "timestamp": datetime.now().isoformat()
        }
        (run_folder / "metadata.json").write_text(json.dumps(metadata, indent=2))

        print(f"  Generated in {elapsed:.1f}s, saved to {model_path}")

        return textual_model, run_folder

    def _create_llm(self):
        """Create LLM client."""
        model_name = self.config.model_name

        if ":" in model_name:
            provider, model_id = model_name.split(":", 1)
        else:
            provider, model_id = "openai", model_name

        if "anthropic" in provider.lower():
            return ChatAnthropic(model=model_id)
        elif "grok" in provider.lower() or "xai" in provider.lower():
            return ChatOpenAI(
                model=model_id,
                api_key=os.getenv("XAI_API_KEY"),
                base_url="https://api.x.ai/v1"
            )
        elif "deepseek" in provider.lower():
            return ChatOpenAI(
                model=model_id,
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com"
            )
        else:
            return ChatOpenAI(model=model_id)


# =============================================================================
# BENCHMARK EVALUATION (adapted from phase5_benchmark.py)
# =============================================================================

def df_to_key_value_rows(df, max_rows=50):
    """Convert DataFrame to key-value text format."""
    df_short = df.head(max_rows).reset_index(drop=True)
    lines = []
    for i, row in df_short.iterrows():
        parts = [f"{col}={row[col]}" for col in df_short.columns]
        lines.append(f"Event number {i}: " + "; ".join(parts))
    return "\n".join(lines)


def random_swap_events(trace, verbose=False):
    """Perturbation: swap two random events in a trace."""
    import random
    from copy import deepcopy
    perturbed_trace = deepcopy(trace)
    num_rows = len(perturbed_trace)
    if num_rows < 2:
        return perturbed_trace
    idx1, idx2 = random.sample(range(num_rows), 2)
    perturbed_trace.iloc[idx1], perturbed_trace.iloc[idx2] = \
        perturbed_trace.iloc[idx2].copy(), perturbed_trace.iloc[idx1].copy()
    return perturbed_trace


def sample_valid_traces(xes_path: str, k: int):
    """Sample k valid traces from event log."""
    import random
    log = pm4py.read_xes(xes_path)
    ignored_columns = ["time", "@@index", "time:timestamp"]
    log = log.drop([col for col in ignored_columns if col in log.columns], axis=1)
    traces = log.groupby('case:case')
    valid = [traces.get_group(tid) for tid in traces.groups.keys()]
    return random.sample(valid, min(k, len(valid)))


def generate_invalid_traces(valid_traces: list, k: int):
    """Generate k invalid traces by perturbing valid traces."""
    import random
    invalid = []
    attempts = 0
    max_attempts = k * 10
    while len(invalid) < k and attempts < max_attempts:
        attempts += 1
        valid_trace = random.choice(valid_traces)
        invalid_trace = random_swap_events(valid_trace)
        is_duplicate = any(invalid_trace.equals(t) for t in invalid)
        if not is_duplicate:
            invalid.append(invalid_trace)
    return invalid


def create_validator(as_is_model: str, model_name: str = "gpt-5-mini"):
    """Create a validator LLM."""
    if ":" in model_name:
        provider, model_id = model_name.split(":", 1)
    else:
        provider, model_id = "openai", model_name

    if "grok" in provider.lower() or "xai" in provider.lower():
        llm = ChatOpenAI(
            model=model_id,
            api_key=os.getenv("XAI_API_KEY"),
            base_url="https://api.x.ai/v1"
        )
    elif "deepseek" in provider.lower():
        llm = ChatOpenAI(
            model=model_id,
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
    else:
        llm = ChatOpenAI(model=model_id, api_key=os.getenv("OPENAI_API_KEY"))

    system_prompt = f"""You are an expert business process analyst. You will be given a business process description and a trace (sequence of activities).

Business Process Description:
{as_is_model}

Your task: Determine if the given trace is VALID (conforms to the observed business process) or INVALID (contains anomalies).

IMPORTANT: The business model describes processes as they are observed, which may include inefficiencies or non-ideal behaviors that still occur in practice.

Respond with ONLY one word: "VALID" or "INVALID"."""

    def validate(trace_df):
        trace_text = df_to_key_value_rows(trace_df)
        messages = [
            ("system", system_prompt),
            ("human", trace_text)
        ]
        start_time = time.time()
        response = llm.invoke(messages)
        compute_time = time.time() - start_time
        tokens_used = response.response_metadata.get('token_usage', {})
        return response.content.strip(), compute_time, tokens_used

    return validate, system_prompt


def evaluate_model(textual_model: str, valid_traces: list, invalid_traces: list,
                   validator_model: str = "gpt-5-mini") -> dict:
    """Evaluate a textual model on trace classification using interleaved evaluation."""
    validator, _ = create_validator(textual_model, model_name=validator_model)

    valid_predictions = []
    invalid_predictions = []
    total_compute_time = 0
    total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    # Helper to compute and print metrics (same as run_with_new_guides.py / phase5_benchmark.py)
    def print_metrics(trace_type: str, idx: int, total: int, pred: str, is_correct: bool, compute_time: float, tokens_used: dict):
        TP = sum(valid_predictions)
        FN = len(valid_predictions) - TP
        TN = sum(invalid_predictions)
        FP = len(invalid_predictions) - TN

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"    {trace_type} trace {idx+1}/{total}: {pred} ({'✓' if is_correct else '✗'}) | P={precision:.3f} R={recall:.3f} F1={f1:.3f} | TP={TP} FP={FP} TN={TN} FN={FN} | Time={compute_time:.2f}s Tokens={tokens_used.get('total_tokens', 0)}")

    # Interleaved evaluation: one valid, one invalid at a time
    max_len = max(len(valid_traces), len(invalid_traces))
    valid_idx = 0
    invalid_idx = 0

    for i in range(max_len):
        # Evaluate one valid trace
        if valid_idx < len(valid_traces):
            trace = valid_traces[valid_idx]
            pred, compute_time, tokens_used = validator(trace)
            is_correct = pred == "VALID"
            valid_predictions.append(is_correct)

            total_compute_time += compute_time
            for key in total_tokens:
                total_tokens[key] += tokens_used.get(key, 0)

            print_metrics("Valid", valid_idx, len(valid_traces), pred, is_correct, compute_time, tokens_used)
            valid_idx += 1
            time.sleep(1)

        # Evaluate one invalid trace
        if invalid_idx < len(invalid_traces):
            trace = invalid_traces[invalid_idx]
            pred, compute_time, tokens_used = validator(trace)
            is_correct = pred == "INVALID"
            invalid_predictions.append(is_correct)

            total_compute_time += compute_time
            for key in total_tokens:
                total_tokens[key] += tokens_used.get(key, 0)

            print_metrics("Invalid", invalid_idx, len(invalid_traces), pred, is_correct, compute_time, tokens_used)
            invalid_idx += 1
            time.sleep(1)

    # Calculate final metrics
    TP = sum(valid_predictions)
    TN = sum(invalid_predictions)
    FN = len(valid_predictions) - TP
    FP = len(invalid_predictions) - TN

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "total_compute_time_seconds": total_compute_time,
        "total_tokens": total_tokens["total_tokens"],
    }


# =============================================================================
# ABLATION STUDY RUNNER
# =============================================================================

class AblationStudy:
    """Main ablation study orchestrator."""

    def __init__(self, xes_path: str, output_dir: Path, config: dict):
        self.xes_path = xes_path
        self.output_dir = output_dir
        self.config = config

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load event log
        print(f"Loading event log: {xes_path}")
        self.log = pm4py.read_xes(xes_path)

        # Initialize discoverer
        self.discoverer = RepresentationDiscoverer(self.log, xes_path)

        # Results storage
        self.results = {
            "_meta": {
                "xes_path": xes_path,
                "start_time": datetime.now().isoformat(),
                "config": config
            }
        }

    def run(self):
        """Run the complete ablation study."""

        # Step 1: Discover all representations
        self.discoverer.discover_all()

        # Save all discovered representations
        self._save_representations()

        # Get list of successfully discovered representations (excluding those not ready for ablation)
        available_reps = [
            rep_type for rep_type in RepresentationType
            if self.discoverer.representations.get(rep_type) is not None
            and rep_type not in EXCLUDED_FROM_ABLATION
        ]

        excluded_count = len([r for r in EXCLUDED_FROM_ABLATION
                              if self.discoverer.representations.get(r) is not None])

        print(f"\nSuccessfully discovered {len(available_reps) + excluded_count}/{len(RepresentationType)} representations")
        if excluded_count > 0:
            print(f"  (Excluding {excluded_count} from ablation: {[r.value for r in EXCLUDED_FROM_ABLATION]})")

        # Initialize miner
        mining_config = AblationMiningConfig(
            model_name=self.config.get("model_name", "openai:gpt-5-mini"),
            prompt_file=self.config.get("prompt_file", "GENERATE_AS_IS_MODEL.txt")
        )
        miner = AblationMiner(self.discoverer, mining_config, self.output_dir)

        # Generate traces for evaluation
        print("\nGenerating traces for evaluation...")
        k_valid = self.config.get("k_valid", 50)
        k_invalid = self.config.get("k_invalid", 50)

        valid_traces = sample_valid_traces(self.xes_path, k_valid)
        invalid_traces = generate_invalid_traces(valid_traces, k_invalid)

        print(f"  Generated {len(valid_traces)} valid + {len(invalid_traces)} invalid traces")

        # Save traces
        traces_dir = self.output_dir / "traces"
        traces_dir.mkdir(exist_ok=True)
        with open(traces_dir / "valid_traces.pkl", "wb") as f:
            pickle.dump(valid_traces, f)
        with open(traces_dir / "invalid_traces.pkl", "wb") as f:
            pickle.dump(invalid_traces, f)

        # Step 2: Run baseline (ALL representations)
        print("\n" + "="*60)
        print("BASELINE: ALL REPRESENTATIONS")
        print("="*60)

        baseline_model, baseline_folder = miner.generate_textual_model(
            available_reps,
            "baseline_all_representations"
        )

        print("\n  Evaluating baseline model...")
        baseline_metrics = evaluate_model(
            baseline_model, valid_traces, invalid_traces,
            self.config.get("validator_model", "gpt-5-mini")
        )

        self.results["baseline"] = {
            "representations": [r.value for r in available_reps],
            "metrics": baseline_metrics,
            "folder": str(baseline_folder)
        }

        print(f"\n  Baseline F1: {baseline_metrics['f1']:.3f}")
        self._save_results()

        # Step 3: Ablation - remove each representation one at a time
        print("\n" + "="*60)
        print("ABLATION: REMOVING ONE REPRESENTATION AT A TIME")
        print("="*60)

        for rep_to_remove in available_reps:
            print(f"\n--- Removing: {rep_to_remove.value} ---")

            ablated_reps = [r for r in available_reps if r != rep_to_remove]
            run_name = f"ablation_without_{rep_to_remove.value}"

            ablated_model, ablated_folder = miner.generate_textual_model(
                ablated_reps,
                run_name
            )

            print(f"\n  Evaluating ablated model (without {rep_to_remove.value})...")
            ablated_metrics = evaluate_model(
                ablated_model, valid_traces, invalid_traces,
                self.config.get("validator_model", "gpt-5-mini")
            )

            f1_delta = ablated_metrics['f1'] - baseline_metrics['f1']

            self.results[f"ablation_{rep_to_remove.value}"] = {
                "removed_representation": rep_to_remove.value,
                "remaining_representations": [r.value for r in ablated_reps],
                "metrics": ablated_metrics,
                "f1_delta": f1_delta,
                "folder": str(ablated_folder)
            }

            impact = "↓" if f1_delta < 0 else ("↑" if f1_delta > 0 else "=")
            print(f"\n  Without {rep_to_remove.value}: F1={ablated_metrics['f1']:.3f} ({impact}{abs(f1_delta):.3f})")

            self._save_results()

        # Step 4: Summary
        self._print_summary(baseline_metrics)

        self.results["_meta"]["end_time"] = datetime.now().isoformat()
        self._save_results()

        return self.results

    def _save_representations(self):
        """Save all discovered representations."""
        reps_dir = self.output_dir / "representations"
        reps_dir.mkdir(exist_ok=True)

        for rep_type in RepresentationType:
            # Save string representation
            rep_str = self.discoverer.string_representations.get(rep_type, "")
            if rep_str:
                (reps_dir / f"{rep_type.value}.txt").write_text(rep_str)

            # Save interpretation guide
            guide = self.discoverer.interpretation_guides.get(rep_type, "")
            if guide:
                (reps_dir / f"{rep_type.value}_guide.txt").write_text(guide)

    def _save_results(self):
        """Save results to JSON."""
        (self.output_dir / "results.json").write_text(
            json.dumps(self.results, indent=2, default=str)
        )

    def _print_summary(self, baseline_metrics):
        """Print ablation study summary."""
        print("\n" + "="*60)
        print("ABLATION STUDY SUMMARY")
        print("="*60)

        print(f"\nBaseline (all representations): F1={baseline_metrics['f1']:.3f}")
        print("\nRepresentation Impact (F1 delta when removed):")

        # Sort by impact (most negative = most important)
        impacts = []
        for key, value in self.results.items():
            if key.startswith("ablation_"):
                impacts.append((
                    value["removed_representation"],
                    value["f1_delta"],
                    value["metrics"]["f1"]
                ))

        impacts.sort(key=lambda x: x[1])  # Sort by delta (ascending = most important first)

        for rep_name, delta, f1 in impacts:
            if delta < -0.01:
                marker = "🔴 CRITICAL"
            elif delta < 0:
                marker = "🟡 Important"
            elif delta > 0.01:
                marker = "🟢 Redundant"
            else:
                marker = "⚪ Neutral"

            print(f"  {marker} {rep_name}: F1={f1:.3f} (Δ={delta:+.3f})")

        print(f"\nResults saved to: {self.output_dir}")


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Ablation Study for Process Mining Representations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run ablation study on hiring dataset
    python ablation_study.py --dataset hiring

    # Run with specific XES file
    python ablation_study.py --xes benchmark/hospital/hospital_log_high.xes

    # Specify models
    python ablation_study.py --dataset hiring --model openai:gpt-4o --validator-model gpt-5-mini
"""
    )

    parser.add_argument("--dataset", type=str, default="hiring",
                        help="Dataset name (hiring, hospital, lending, renting)")
    parser.add_argument("--xes", type=str, default=None,
                        help="Path to XES file (overrides --dataset)")
    parser.add_argument("--model", type=str, default="openai:gpt-5-mini",
                        help="LLM for textual model generation")
    parser.add_argument("--validator-model", type=str, default="gpt-5-mini",
                        help="LLM for trace validation")
    parser.add_argument("--k-valid", type=int, default=30,
                        help="Number of valid traces (default: 30)")
    parser.add_argument("--k-invalid", type=int, default=30,
                        help="Number of invalid traces (default: 30)")
    parser.add_argument("--prompt-file", type=str, default="GENERATE_AS_IS_MODEL.txt",
                        help="System prompt file for model generation")

    return parser.parse_args()


def main():
    args = parse_args()

    # Determine XES path
    if args.xes:
        xes_path = args.xes
    else:
        xes_path = f"benchmark/{args.dataset}/{args.dataset}_log_high.xes"

    if not Path(xes_path).exists():
        print(f"Error: XES file not found: {xes_path}")
        return 1

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = Path(xes_path).stem
    output_dir = Path("ablation_results") / f"ablation_{dataset_name}_{timestamp}"

    # Configuration
    config = {
        "model_name": args.model,
        "validator_model": args.validator_model,
        "k_valid": args.k_valid,
        "k_invalid": args.k_invalid,
        "prompt_file": args.prompt_file
    }

    print("="*60)
    print("ABLATION STUDY FOR PROCESS MINING REPRESENTATIONS")
    print("="*60)
    print(f"Dataset: {xes_path}")
    print(f"Output: {output_dir}")
    print(f"Model: {config['model_name']}")
    print(f"Validator: {config['validator_model']}")
    print(f"Traces: {config['k_valid']} valid, {config['k_invalid']} invalid")

    # Run study
    study = AblationStudy(xes_path, output_dir, config)
    results = study.run()

    print("\n" + "="*60)
    print("ABLATION STUDY COMPLETE")
    print("="*60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
