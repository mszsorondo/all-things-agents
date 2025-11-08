#!/usr/bin/env python3
"""
Enhanced Agentic Process Miner
A modular, configurable system for process mining with LLM-based interpretation.
"""

import os
import json
import shutil
import logging
import argparse
import hashlib
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from abc import ABC, abstractmethod

import pm4py
from pm4py.objects.bpmn.exporter import exporter as bpmn_exporter
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter
from langgraph.prebuilt import create_react_agent


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RepresentationType(Enum):
    """Available process mining representation types."""
    PETRI_NET = "petri_net"
    BPMN = "bpmn"
    DFG = "dfg"
    HEURISTICS_NET = "heuristics_net"
    PROCESS_TREE = "process_tree"


@dataclass
class MiningConfiguration:
    """Configuration for the mining process."""
    representations: List[RepresentationType] = field(default_factory=lambda: [
        RepresentationType.PETRI_NET,
        #RepresentationType.BPMN,
        RepresentationType.DFG,
        RepresentationType.ABSTRACT_VARIANTS,
        RepresentationType.PROCESS_TREE,
        RepresentationType.HEURISTICS_NET
    ])
    model_name: str = "anthropic:claude-sonnet-4-5"  # Match original template
    prompt_file: str = "new_prompt.txt"
    output_dir: str = "runs"
    enable_logging: bool = True
    save_experiment_config: bool = True
    stream_mode: str = "updates"
    use_cache: bool = True  # Enable caching by default


class ProcessRepresentation(ABC):
    """Abstract base class for process representations."""

    def __init__(self, log):
        self.log = log
        self.representation = None
        self.interpretation_guide = ""

    @abstractmethod
    def discover(self):
        """Discover the representation from the event log."""
        pass

    @abstractmethod
    def to_string(self) -> str:
        """Convert the representation to a string format."""
        pass

    @abstractmethod
    def get_interpretation_guide(self) -> str:
        """Return the interpretation guide for this representation."""
        pass

    # TODO: Implement compact string methods in Phase 1.1
    # def to_compact_string(self) -> str:
    #     """Convert to compact format for token optimization."""
    #     pass

    def get_full_description(self) -> str:
        """Get the complete description including interpretation guide."""
        return f"""
{self.get_interpretation_guide()}

DISCOVERED MODEL:
{self.to_string()}
"""


class PetriNetRepresentation(ProcessRepresentation):
    """Petri Net representation with interpretation guide."""

    def discover(self):
        """Discover Petri net using inductive miner."""
        net, im, fm = pm4py.discover_petri_net_inductive(self.log)
        self.net = net
        self.initial_marking = im
        self.final_marking = fm
        self.representation = (net, im, fm)
        return self.representation

    def to_string(self) -> str:
        """Convert Petri net to string representation."""
        if not self.representation:
            return "No Petri net discovered yet"

        places = [f"{p.name}" for p in self.net.places]
        transitions = [f"{t.name} (label: {t.label})" for t in self.net.transitions]
        arcs = [f"{a.source.name} -> {a.target.name}" for a in self.net.arcs]

        return f"""
Petri Net Structure:
- Places ({len(places)}): {', '.join(places[:10])}{'...' if len(places) > 10 else ''}
- Transitions ({len(transitions)}): {', '.join(transitions[:10])}{'...' if len(transitions) > 10 else ''}
- Arcs ({len(arcs)}): {', '.join(arcs[:10])}{'...' if len(arcs) > 10 else ''}
- Initial Marking: {str(self.initial_marking)}
- Final Marking: {str(self.final_marking)}
"""

    def get_interpretation_guide(self) -> str:
        return """
=== PETRI NET INTERPRETATION GUIDE ===
A Petri net is a mathematical model for distributed systems consisting of:

1. PLACES (circles): Represent states or conditions in the process
   - Initial marking: Places with tokens at the start (process entry points)
   - Final marking: Places with tokens at the end (process exit points)
   - Empty places: Intermediate states

2. TRANSITIONS (rectangles): Represent activities or events
   - Labeled transitions: Visible activities (e.g., "Submit Application")
   - Silent/tau transitions: Internal logic without visible activity
   - Parallel transitions: Can fire simultaneously (concurrent activities)

3. ARCS (arrows): Connect places and transitions
   - Place→Transition: Precondition (transition needs token in place to fire)
   - Transition→Place: Postcondition (transition produces token in place)
   - Arc weight: Number of tokens consumed/produced

4. BEHAVIORAL SEMANTICS:
   - A transition is ENABLED when all input places contain tokens
   - Firing consumes tokens from input places, produces in output places
   - Multiple tokens in a place indicate multiple process instances
   - Deadlocks occur when no transition can fire
   - Loops indicate repetitive behavior

5. INDUCTIVE MINER SPECIFICS:
   - Guarantees sound workflow nets (proper completion)
   - May introduce silent transitions for control flow
   - Handles noise and incomplete behavior in logs
"""
class AbstractVariantsRepresentation(ProcessRepresentation):
    """Abstract variants representation with interpretation guide."""

    def discover(self):
        """Discover abstract variants using LLM."""
        self.representation = pm4py.llm.abstract_variants(self.log)
        return self.representation

    def to_string(self) -> str:
        """Convert abstract variants to string representation."""
        if not self.representation:
            return "No abstract variants discovered yet"

        return f"""
Abstract Variants:
{str(self.representation)}
"""


class BPMNRepresentation(ProcessRepresentation):
    """BPMN representation with interpretation guide."""

    def discover(self):
        """Discover BPMN model using inductive miner."""
        self.representation = pm4py.discover_bpmn_inductive(self.log)
        return self.representation

    def to_string(self) -> str:
        """Convert BPMN to XML string."""
        if not self.representation:
            return "No BPMN model discovered yet"

        # Export to temporary file and read back
        temp_file = "temp_bpmn_export.bpmn"
        try:
            bpmn_exporter.apply(self.representation, temp_file)
            with open(temp_file, 'r') as f:
                bpmn_str = f.read()
            os.remove(temp_file)

            # Return full BPMN without truncation to match original template behavior
            return bpmn_str
        except Exception as e:
            logger.error(f"Failed to export BPMN: {e}")
            return f"BPMN model discovered but export failed: {e}"

    def get_interpretation_guide(self) -> str:
        return """
=== BPMN INTERPRETATION GUIDE ===
Business Process Model and Notation (BPMN) is a graphical notation for business processes:

1. EVENTS:
   - Start Event (thin circle): Process initiation point
   - End Event (thick circle): Process termination point
   - Intermediate Events (double circle): Events during execution

2. ACTIVITIES:
   - Task (rounded rectangle): Atomic unit of work
   - Sub-process (rounded rectangle with +): Compound activity
   - Call Activity (thick border): Reference to another process

3. GATEWAYS (diamonds):
   - Exclusive (X or empty): XOR logic - one path only
   - Parallel (+): AND logic - all paths simultaneously
   - Inclusive (O): OR logic - one or more paths
   - Event-based (pentagon inside): Wait for events

4. FLOWS:
   - Sequence Flow (solid arrow): Order of execution
   - Default Flow (slash mark): Taken when no other condition is true
   - Conditional Flow (mini diamond): Has a condition expression

5. INDUCTIVE MINER BPMN SPECIFICS:
   - May contain process trees encoded as nested sub-processes
   - Silent transitions become empty gateways
   - Guarantees proper block structure
   - Loop structures are explicit with backward flows

6. SWIMLANES (if present):
   - Pools: Different organizations/systems
   - Lanes: Different roles within organization

READING THE XML:
- <bpmn:task id="..." name="...">: Activities to perform
- <bpmn:sequenceFlow sourceRef="..." targetRef="...">: Connections
- <bpmn:exclusiveGateway>: Decision points
- <bpmn:parallelGateway>: Concurrent execution points
"""


class DFGRepresentation(ProcessRepresentation):
    """Directly-Follows Graph representation with interpretation guide."""

    def discover(self):
        """Abstract DFG using LLM-based abstraction."""
        try:
            self.representation = pm4py.llm.abstract_dfg(self.log)
        except Exception as e:
            logger.warning(f"LLM abstraction failed, using standard DFG: {e}")
            # Fallback to standard DFG discovery
            dfg, start, end = pm4py.discover_dfg(self.log)
            self.representation = {
                'dfg': dfg,
                'start_activities': start,
                'end_activities': end
            }
        return self.representation

    def to_string(self) -> str:
        """Convert DFG to string representation."""
        if not self.representation:
            return "No DFG discovered yet"

        if isinstance(self.representation, dict) and 'dfg' in self.representation:
            # Standard DFG format
            dfg = self.representation['dfg']
            start = self.representation['start_activities']
            end = self.representation['end_activities']

            edges = [f"{a1} -> {a2} (frequency: {freq})" for (a1, a2), freq in dfg.items()]
            starts = [f"{act} (count: {cnt})" for act, cnt in start.items()]
            ends = [f"{act} (count: {cnt})" for act, cnt in end.items()]

            return f"""
Directly-Follows Graph:
- Start Activities: {', '.join(starts[:5])}{'...' if len(starts) > 5 else ''}
- End Activities: {', '.join(ends[:5])}{'...' if len(ends) > 5 else ''}
- Edges ({len(edges)}):
  {chr(10).join(edges[:20])}{'...' if len(edges) > 20 else ''}
"""
        else:
            # LLM abstracted format
            return str(self.representation)

    def get_interpretation_guide(self) -> str:
        return """
=== DIRECTLY-FOLLOWS GRAPH (DFG) INTERPRETATION GUIDE ===
A DFG shows the frequency of direct succession between activities:

1. NODES: Represent activities/events in the process
   - Size often indicates frequency of occurrence
   - Color may indicate performance metrics

2. EDGES: Show direct succession relationships
   - Edge labels: Frequency of A→B succession
   - Thicker edges: More frequent paths
   - Edge absence: Activities never directly follow each other

3. START/END ACTIVITIES:
   - Start activities: First activities in traces
   - End activities: Last activities in traces
   - Multiple starts/ends indicate process variants

4. INTERPRETATION PATTERNS:
   - High-frequency edges: Main process flow
   - Low-frequency edges: Exceptions or rare variants
   - Self-loops: Activity repetition
   - Disconnected components: Independent sub-processes

5. LLM ABSTRACTION (if applied):
   - Simplifies complex DFGs to essential flows
   - Removes noise and infrequent paths
   - Groups similar activities
   - Highlights main process variants

6. LIMITATIONS:
   - Cannot distinguish XOR from AND gateways
   - May show impossible paths (not all A→B→C means A→C exists)
   - Sensitive to noise in event logs

USE FOR:
- Quick process overview
- Identifying bottlenecks (high-frequency edges)
- Finding process variants
- Detecting loops and rework
"""


class HeuristicsNetRepresentation(ProcessRepresentation):
    """Heuristics Net representation with interpretation guide."""

    def discover(self):
        """Discover Heuristics Net."""
        self.representation = pm4py.discover_heuristics_net(self.log)
        return self.representation

    def to_string(self) -> str:
        """Convert Heuristics Net to string."""
        if not self.representation:
            return "No Heuristics Net discovered yet"

        # The heuristics net has nodes and edges with dependency measures
        nodes = list(self.representation.nodes)

        return f"""
Heuristics Net:
- Nodes ({len(nodes)}): {', '.join([str(n) for n in nodes[:10]])}{'...' if len(nodes) > 10 else ''}
- Uses dependency measures and frequency thresholds
- Handles noise better than simple DFG
"""

    def get_interpretation_guide(self) -> str:
        return """
=== HEURISTICS NET INTERPRETATION GUIDE ===
A Heuristics Net improves on DFG by using dependency measures:

1. DEPENDENCY MEASURE:
   - Calculates |A→B| = (|A>B| - |B>A|) / (|A>B| + |B>A| + 1)
   - Range: -1 to 1
   - Positive: A causes B
   - Negative: B causes A
   - Near 0: No clear dependency

2. ADVANTAGES OVER DFG:
   - Handles noise and incomplete logs
   - Detects true dependencies vs. coincidence
   - Identifies AND/XOR splits and joins
   - Removes infrequent behaviors

3. CONSTRUCTION RULES:
   - Only includes edges above dependency threshold
   - Detects long-distance dependencies
   - Identifies loops based on frequency
   - Constructs split/join semantics

4. READING THE NET:
   - Solid edges: Strong dependencies
   - Dashed edges: Weaker dependencies
   - AND-split: All outgoing edges activated
   - XOR-split: One outgoing edge activated
   - AND-join: Wait for all incoming edges
   - XOR-join: Proceed on any incoming edge
"""


class ProcessTreeRepresentation(ProcessRepresentation):
    """Process Tree representation with interpretation guide."""

    def discover(self):
        """Discover Process Tree using inductive miner."""
        self.representation = pm4py.discover_process_tree_inductive(self.log)
        return self.representation

    def to_string(self) -> str:
        """Convert Process Tree to string."""
        if not self.representation:
            return "No Process Tree discovered yet"

        return str(self.representation)

    def get_interpretation_guide(self) -> str:
        return """
=== PROCESS TREE INTERPRETATION GUIDE ===
A Process Tree is a hierarchical representation of process behavior:

1. OPERATORS (internal nodes):
   - → (sequence): Execute children in order
   - X (exclusive choice): Execute one child
   - + (parallel): Execute all children concurrently
   - ↺ (loop): Repeat body, execute exit condition
   - τ (silent): Hidden/internal activity

2. LEAVES:
   - Activity names: Actual process activities
   - τ: Silent/skip activity

3. SEMANTICS:
   - Read tree top-down, left-to-right
   - Each operator defines control flow for children
   - Guarantees soundness (proper completion)
   - Block-structured (properly nested)

4. EXAMPLE INTERPRETATIONS:
   - →(A, X(B, C), D): Do A, then either B or C, then D
   - +(A, B, C): Do A, B, C in parallel
   - ↺(A, B): Do A repeatedly, exit with B
   - X(→(A, B), →(C, D)): Either do A then B, or C then D

5. INDUCTIVE MINER PROPERTIES:
   - Always produces sound models
   - Handles incomplete logs
   - May introduce τ transitions for fitness
   - Rediscovers original tree if log is complete
"""


class RepresentationCache:
    """
    Caching system for process mining representations.
    Stores discovered representations to avoid recomputation.
    """

    def __init__(self, cache_dir: Path = None, ttl_hours: int = 24):
        """
        Initialize the cache system.

        Args:
            cache_dir: Directory to store cache files
            ttl_hours: Time-to-live for cached items in hours
        """
        self.cache_dir = cache_dir or Path("cache/representations")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours)
        self.cache_index_file = self.cache_dir / "index.json"
        self.cache_index = self._load_index()

    def _load_index(self) -> Dict:
        """Load the cache index from disk."""
        if self.cache_index_file.exists():
            with open(self.cache_index_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_index(self):
        """Save the cache index to disk."""
        with open(self.cache_index_file, 'w') as f:
            json.dump(self.cache_index, f, indent=2)

    def _generate_cache_key(self, log_path: str, representation_type: str,
                           params: Optional[Dict] = None) -> str:
        """
        Generate a unique cache key for a representation.

        Args:
            log_path: Path to the event log
            representation_type: Type of representation
            params: Additional parameters affecting the representation

        Returns:
            SHA256 hash as cache key
        """
        # Get file stats for the log
        log_stats = os.stat(log_path)

        # Create a unique identifier
        cache_data = {
            'log_path': log_path,
            'log_size': log_stats.st_size,
            'log_mtime': log_stats.st_mtime,
            'representation_type': representation_type,
            'params': params or {}
        }

        # Generate hash
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.sha256(cache_str.encode()).hexdigest()

    def _is_valid(self, cache_key: str) -> bool:
        """
        Check if a cached item is still valid.

        Args:
            cache_key: The cache key to check

        Returns:
            True if cache is valid, False otherwise
        """
        if cache_key not in self.cache_index:
            return False

        cache_info = self.cache_index[cache_key]
        cached_time = datetime.fromisoformat(cache_info['timestamp'])

        return datetime.now() - cached_time < self.ttl

    def get(self, log_path: str, representation_type: str,
            params: Optional[Dict] = None) -> Optional[Any]:
        """
        Retrieve a cached representation.

        Args:
            log_path: Path to the event log
            representation_type: Type of representation
            params: Additional parameters

        Returns:
            Cached representation or None if not found/expired
        """
        cache_key = self._generate_cache_key(log_path, representation_type, params)

        if not self._is_valid(cache_key):
            return None

        cache_file = self.cache_dir / f"{cache_key}.pkl"

        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
                logger.info(f"Cache hit for {representation_type} of {Path(log_path).name}")
                return data
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return None

    def set(self, log_path: str, representation_type: str,
            representation: Any, params: Optional[Dict] = None):
        """
        Store a representation in cache.

        Args:
            log_path: Path to the event log
            representation_type: Type of representation
            representation: The representation object to cache
            params: Additional parameters
        """
        cache_key = self._generate_cache_key(log_path, representation_type, params)
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(representation, f)

            # Update index
            self.cache_index[cache_key] = {
                'log_path': log_path,
                'representation_type': representation_type,
                'timestamp': datetime.now().isoformat(),
                'params': params or {}
            }
            self._save_index()

            logger.info(f"Cached {representation_type} for {Path(log_path).name}")
        except Exception as e:
            logger.warning(f"Failed to cache representation: {e}")

    def clear_expired(self):
        """Remove expired cache entries."""
        expired_keys = []

        for cache_key, info in self.cache_index.items():
            if not self._is_valid(cache_key):
                expired_keys.append(cache_key)
                cache_file = self.cache_dir / f"{cache_key}.pkl"
                if cache_file.exists():
                    cache_file.unlink()

        for key in expired_keys:
            del self.cache_index[key]

        if expired_keys:
            self._save_index()
            logger.info(f"Cleared {len(expired_keys)} expired cache entries")

    def clear_all(self):
        """Clear all cached representations."""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()

        self.cache_index = {}
        self._save_index()
        logger.info("Cleared all cache entries")


class AgenticProcessMiner:
    """Main class for agentic process mining."""

    REPRESENTATION_CLASSES = {
        RepresentationType.PETRI_NET: PetriNetRepresentation,
        RepresentationType.BPMN: BPMNRepresentation,
        RepresentationType.DFG: DFGRepresentation,
        RepresentationType.HEURISTICS_NET: HeuristicsNetRepresentation,
        RepresentationType.PROCESS_TREE: ProcessTreeRepresentation
    }

    def __init__(self, config: MiningConfiguration):
        """Initialize the miner with configuration."""
        self.config = config
        self.representations = {}
        self.run_folder = None
        self.log_file = None
        self.log = None  # Event log, loaded only when needed

        # Initialize cache if enabled
        self.cache = RepresentationCache() if self.config.use_cache else None

        # Load API key from environment
        self._setup_api_keys()

    def _setup_api_keys(self):
        """Setup API keys from environment variables."""
        # Check for API keys in environment
        if 'anthropic' in self.config.model_name.lower():
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY not found in environment. "
                    "Please set it using: export ANTHROPIC_API_KEY='your-key'"
                )
        elif 'openai' in self.config.model_name.lower():
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY not found in environment. "
                    "Please set it using: export OPENAI_API_KEY='your-key'"
                )

    def _create_run_folder(self, dataset_name: str) -> Path:
        """Create a folder for this run."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{dataset_name}_{timestamp}"
        run_folder = Path(self.config.output_dir) / run_name
        run_folder.mkdir(parents=True, exist_ok=True)
        self.run_folder = run_folder

        logger.info(f"Created run folder: {run_folder}")
        return run_folder

    def load_event_log(self, xes_path: str):
        """Load an XES event log."""
        logger.info(f"Loading event log from: {xes_path}")
        self.log = pm4py.read_xes(xes_path)
        self.xes_path = xes_path

        # Get dataset name from path
        self.dataset_name = Path(xes_path).stem

        # Log statistics
        num_traces = len(self.log)
        num_events = sum(len(trace) for trace in self.log)

        # Use pm4py's built-in method to get activities
        try:
            activities = pm4py.get_event_attribute_values(self.log, "concept:name")
            logger.info(f"Loaded {num_traces} traces, {num_events} events, {len(activities)} unique activities")
        except Exception as e:
            # Fallback if the above doesn't work
            logger.info(f"Loaded {num_traces} traces, {num_events} events")
            logger.warning(f"Could not extract activities: {e}")
        return self.log

    def check_cache_status(self, log_file: str) -> Tuple[List[RepresentationType], List[RepresentationType]]:
        """
        Check which representations are cached and which need discovery.

        Args:
            log_file: Path to the XES file

        Returns:
            Tuple of (cached_representations, uncached_representations)
        """
        cached = []
        uncached = []

        for rep_type in self.config.representations:
            if self.cache and self.cache.get(log_file, rep_type.value):
                cached.append(rep_type)
            else:
                uncached.append(rep_type)

        return cached, uncached

    def discover_representations(self):
        """Discover all configured representations."""
        logger.info(f"Discovering {len(self.config.representations)} representations")

        # Check cache status BEFORE loading the log
        if self.cache and self.log_file:
            cached_types, uncached_types = self.check_cache_status(self.log_file)

            if cached_types:
                logger.info(f"Found {len(cached_types)} cached representations: {[t.value for t in cached_types]}")

            # Load cached representations
            for rep_type in cached_types:
                cached_rep = self.cache.get(self.log_file, rep_type.value)
                if cached_rep:
                    self.representations[rep_type] = cached_rep
                    logger.info(f"Loaded {rep_type.value} from cache")

            # Only load the log if we need to discover something
            if uncached_types:
                if self.log is None:
                    logger.info(f"Loading event log to discover {len(uncached_types)} representations...")
                    self.load_event_log(self.log_file)

                # Discover uncached representations
                for rep_type in uncached_types:
                    logger.info(f"Discovering {rep_type.value}...")
                    try:
                        rep_class = self.REPRESENTATION_CLASSES[rep_type]
                        representation = rep_class(self.log)
                        representation.discover()
                        self.representations[rep_type] = representation

                        # Store in cache
                        self.cache.set(self.log_file, rep_type.value, representation)
                        logger.info(f"Successfully discovered and cached {rep_type.value}")
                    except Exception as e:
                        logger.error(f"Failed to discover {rep_type.value}: {e}")
                        self.representations[rep_type] = None
            else:
                logger.info("All representations loaded from cache - no log loading needed!")
        else:
            # No cache - traditional flow
            if self.log is None:
                logger.info("Loading event log for discovery...")
                self.load_event_log(self.log_file)

            for rep_type in self.config.representations:
                logger.info(f"Discovering {rep_type.value}...")
                try:
                    rep_class = self.REPRESENTATION_CLASSES[rep_type]
                    representation = rep_class(self.log)
                    representation.discover()
                    self.representations[rep_type] = representation
                    logger.info(f"Successfully discovered {rep_type.value}")
                except Exception as e:
                    logger.error(f"Failed to discover {rep_type.value}: {e}")
                    self.representations[rep_type] = None

    def build_prompt(self) -> str:
        """Build the prompt with all representations and their guides."""
        # Build a simple prompt similar to original template by default
        prompt_parts = ["\nProcess Mining Representations: "]

        # Add each representation in a simple format (matching original template style)
        counter = 1

        for rep_type, representation in self.representations.items():
            if representation:
                if rep_type == RepresentationType.PETRI_NET and hasattr(representation, 'net'):
                    # Special format for Petri net to match original
                    prompt_parts.append(f"    {counter}. Petri Net from inductive miner: {str(representation.net)} ")
                    prompt_parts.append(f"with initial marking {str(representation.initial_marking)}, final marking {str(representation.final_marking)},")
                elif rep_type == RepresentationType.BPMN:
                    prompt_parts.append(f"    {counter}. BPMN extracted from inductive miner: {representation.to_string()}")
                elif rep_type == RepresentationType.DFG:
                    prompt_parts.append(f"    {counter}. DFG (Directly-Follows Graph) abstraction: {representation.to_string()}")
                else:
                    # Generic format for other representations
                    prompt_parts.append(f"    {counter}. {rep_type.value.upper()}: {representation.to_string()}")
                counter += 1

        return "\n".join(prompt_parts)

    def _setup_tools(self):
        """Setup tools for the agent."""
        def write_bpmn_file(fname: str, content: str):
            """Write BPMN content to file."""
            filepath = self.run_folder / fname
            with open(filepath, "w") as file:
                file.write(content)
            logger.info(f"Wrote BPMN file: {filepath}")

        def write_comprehensive_textual_description(fname: str, content: str):
            """Write textual description to file."""
            filepath = self.run_folder / fname
            with open(filepath, "w") as file:
                file.write(content)
            logger.info(f"Wrote description file: {filepath}")

        return [write_bpmn_file, write_comprehensive_textual_description]

    def run_agent(self, system_prompt: Optional[str] = None):
        """Run the LLM agent with discovered representations."""
        # Create run folder
        self._create_run_folder(self.dataset_name)

        # Load system prompt
        if system_prompt is None and os.path.exists(self.config.prompt_file):
            with open(self.config.prompt_file, 'r') as f:
                system_prompt = f.read()
        elif system_prompt is None:
            system_prompt = "You are a process mining expert. Analyze the provided process representations."

        # Copy prompt file to run folder
        if os.path.exists(self.config.prompt_file):
            shutil.copy(self.config.prompt_file, self.run_folder)

        # Build user prompt
        user_prompt = self.build_prompt()

        # Save configuration
        if self.config.save_experiment_config:
            self._save_experiment_config(system_prompt, user_prompt)

        # Setup tools
        tools = self._setup_tools()

        # Create agent
        logger.info(f"Creating agent with model: {self.config.model_name}")

        graph = create_react_agent(
            model=self.config.model_name,
            tools=tools,
            prompt=system_prompt
        )

        # Prepare input
        inputs = {"messages": [{"role": "user", "content": user_prompt}]}
        # Setup logging
        if self.config.enable_logging:
            log_file = self.run_folder / "agent_logs.txt"
            self.log_file = open(log_file, "w", encoding="utf-8")
            self._write_log_header(user_prompt)

        # Stream agent responses
        logger.info("Starting agent stream...")
        try:
            for chunk in graph.stream(inputs, stream_mode=self.config.stream_mode):
                chunk_str = json.dumps(chunk, indent=2, default=str) if isinstance(chunk, dict) else str(chunk)
                print(chunk)

                if self.config.enable_logging and self.log_file:
                    self._write_log_chunk(chunk_str)

            logger.info("Agent processing completed successfully")
        except Exception as e:
            logger.error(f"Agent processing failed: {e}")
            raise
        finally:
            if self.log_file:
                self._write_log_footer()
                self.log_file.close()

    def _save_experiment_config(self, system_prompt: str, user_prompt: str):
        """Save experiment configuration."""
        # Convert config to dict with proper serialization
        config_dict = asdict(self.config)
        # Convert RepresentationType enums to strings
        config_dict["representations"] = [r.value for r in self.config.representations]

        config = {
            "run_timestamp": datetime.now().isoformat(),
            "dataset_name": self.dataset_name,
            "xes_path": self.xes_path,
            "model": self.config.model_name,
            "representations": [r.value for r in self.config.representations],
            "system_prompt_preview": system_prompt[:500] + "..." if len(system_prompt) > 500 else system_prompt,
            "user_prompt_preview": user_prompt[:500] + "..." if len(user_prompt) > 500 else user_prompt,
            "config": config_dict
        }

        config_file = self.run_folder / "experiment_config.json"
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        # Also save full prompts
        with open(self.run_folder / "system_prompt.txt", "w") as f:
            f.write(system_prompt)
        with open(self.run_folder / "user_prompt.txt", "w") as f:
            f.write(user_prompt)

    def _write_log_header(self, user_prompt: str):
        """Write log file header."""
        self.log_file.write(f"{'=' * 80}\n")
        self.log_file.write(f"AGENT RUN LOG\n")
        self.log_file.write(f"Timestamp: {datetime.now().isoformat()}\n")
        self.log_file.write(f"Dataset: {self.dataset_name}\n")
        self.log_file.write(f"Model: {self.config.model_name}\n")
        self.log_file.write(f"{'=' * 80}\n\n")
        self.log_file.write(f"USER PROMPT:\n{'-' * 80}\n{user_prompt}\n\n")
        self.log_file.write(f"{'=' * 80}\nAGENT RESPONSE STREAM:\n{'=' * 80}\n\n")
        self.log_file.flush()

    def _write_log_chunk(self, chunk_str: str):
        """Write a chunk to the log file."""
        self.log_file.write(f"[{datetime.now().isoformat()}] Chunk Update:\n{chunk_str}\n{'-' * 80}\n\n")
        self.log_file.flush()

    def _write_log_footer(self):
        """Write log file footer."""
        self.log_file.write(f"{'=' * 80}\n")
        self.log_file.write(f"END OF RUN\n")
        self.log_file.write(f"Timestamp: {datetime.now().isoformat()}\n")
        self.log_file.write(f"{'=' * 80}\n")


def list_available_datasets(benchmark_dir: str = "benchmark") -> List[Dict[str, Any]]:
    """List all available XES files in the benchmark directory."""
    datasets = []
    benchmark_path = Path(benchmark_dir)

    if not benchmark_path.exists():
        logger.error(f"Benchmark directory not found: {benchmark_dir}")
        return datasets

    for xes_file in benchmark_path.rglob("*.xes"):
        # Skip .gz files
        if xes_file.suffix == ".gz":
            continue

        file_info = {
            "name": xes_file.stem,
            "path": str(xes_file),
            "size_mb": xes_file.stat().st_size / (1024 * 1024),
            "parent_dir": xes_file.parent.name
        }
        datasets.append(file_info)

    return sorted(datasets, key=lambda x: x["size_mb"])


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Enhanced Agentic Process Miner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available datasets
  python enhanced_agentic_miner.py --list

  # Run on hiring dataset with default representations
  python enhanced_agentic_miner.py --dataset hiring

  # Run with specific representations
  python enhanced_agentic_miner.py --dataset hospital --representations petri_net bpmn

  # Use custom model
  python enhanced_agentic_miner.py --dataset lending --model openai:gpt-4o-mini

  # Run on specific XES file
  python enhanced_agentic_miner.py --xes-file /path/to/custom.xes
"""
    )

    parser.add_argument("--list", action="store_true", help="List available datasets")
    parser.add_argument("--dataset", type=str, help="Dataset name (e.g., hiring, hospital)")
    parser.add_argument("--xes-file", type=str, help="Path to specific XES file")
    parser.add_argument(
        "--representations",
        nargs="+",
        choices=[r.value for r in RepresentationType],
        default=["petri_net", "dfg", "abstract_variants"],
        help="Representations to discover"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="anthropic:claude-sonnet-4-5",  # Match original template
        help="LLM model to use"
    )
    parser.add_argument("--prompt-file", type=str, default="new_prompt.txt", help="System prompt file")
    parser.add_argument("--output-dir", type=str, default="runs", help="Output directory")
    parser.add_argument("--no-logging", action="store_true", help="Disable logging")
    parser.add_argument("--benchmark-dir", type=str, default="benchmark", help="Benchmark directory")
    parser.add_argument("--no-cache", action="store_true", help="Disable representation caching")
    parser.add_argument("--clear-cache", action="store_true", help="Clear all cached representations before running")
    parser.add_argument("--clear-expired", action="store_true", help="Clear only expired cache entries")

    args = parser.parse_args()

    # List datasets if requested
    if args.list:
        datasets = list_available_datasets(args.benchmark_dir)
        if not datasets:
            print("No datasets found!")
            return

        print("\nAvailable datasets:")
        print("-" * 60)
        for ds in datasets:
            print(f"{ds['parent_dir']}/{ds['name']}")
            print(f"  Path: {ds['path']}")
            print(f"  Size: {ds['size_mb']:.1f} MB")
            print()
        return

    # Determine XES file path
    xes_path = None
    if args.xes_file:
        xes_path = args.xes_file
    elif args.dataset:
        # Find dataset in benchmark directory
        datasets = list_available_datasets(args.benchmark_dir)
        matching = [ds for ds in datasets if args.dataset in ds['name'].lower() or args.dataset in ds['parent_dir'].lower()]

        if not matching:
            print(f"Dataset '{args.dataset}' not found!")
            print("Use --list to see available datasets")
            return

        if len(matching) > 1:
            print(f"Multiple matches for '{args.dataset}':")
            for ds in matching:
                print(f"  - {ds['parent_dir']}/{ds['name']} ({ds['size_mb']:.1f} MB)")
            print("\nUsing first match. Use --xes-file for specific file.")

        xes_path = matching[0]['path']
    else:
        # Default to hiring dataset to match original template behavior
        print("No dataset specified. Using default: hiring dataset")
        datasets = list_available_datasets(args.benchmark_dir)
        hiring_datasets = [ds for ds in datasets if 'hiring' in ds['name'].lower() or 'hiring' in ds['parent_dir'].lower()]

        if hiring_datasets:
            xes_path = hiring_datasets[0]['path']
            print(f"Using: {xes_path}")
        else:
            print("Default hiring dataset not found. Please specify --dataset or --xes-file")
            parser.print_help()
            return

    # Check if file exists
    if not Path(xes_path).exists():
        print(f"File not found: {xes_path}")
        return

    # Handle cache operations
    if args.clear_cache or args.clear_expired:
        cache = RepresentationCache()
        if args.clear_cache:
            cache.clear_all()
            print("Cleared all cached representations")
        elif args.clear_expired:
            cache.clear_expired()
            print("Cleared expired cache entries")

        if args.clear_cache and not (args.dataset or args.xes_file):
            # If only clearing cache, exit
            return

    # Create configuration
    config = MiningConfiguration(
        representations=[RepresentationType(r) for r in args.representations],
        model_name=args.model,
        prompt_file=args.prompt_file,
        output_dir=args.output_dir,
        enable_logging=not args.no_logging,
        use_cache=not args.no_cache
    )

    print(f"\nConfiguration:")
    print(f"  Dataset: {xes_path}")
    print(f"  Model: {config.model_name}")
    print(f"  Representations: {[r.value for r in config.representations]}")
    print(f"  Output: {config.output_dir}/")
    print(f"  Cache: {'enabled' if config.use_cache else 'disabled'}")
    print()

    # Check API keys
    if 'anthropic' in config.model_name.lower() and not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set in environment!")
        print("Please run: export ANTHROPIC_API_KEY='your-key-here'")
        return
    elif 'openai' in config.model_name.lower() and not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set in environment!")
        print("Please run: export OPENAI_API_KEY='your-key-here'")
        return

    # Run the miner
    try:
        miner = AgenticProcessMiner(config)

        # Set the log file path and dataset name (needed even when using cache)
        miner.log_file = xes_path
        miner.xes_path = xes_path
        miner.dataset_name = Path(xes_path).stem

        print("Discovering representations...")
        # This will now check cache first and only load log if needed
        miner.discover_representations()

        print("Running agent analysis...")
        miner.run_agent()

        print(f"\nResults saved to: {miner.run_folder}/")

    except Exception as e:
        logger.error(f"Process mining failed: {e}")
        print(f"\nERROR: {e}")
        return 1

    print("\nProcess mining completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())