"""
Process Mining Logger for Multi-Agent Systems
Integrates with Langfuse for automatic tracking and creates XES-compatible event logs
"""

import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Literal
from pathlib import Path
from dataclasses import dataclass, field, asdict
# Try to import langfuse decorators, create dummies if not available
try:
    from langfuse.decorators import langfuse_context, observe
except ImportError:
    # Create dummy decorators if not available
    def observe(name=None):
        def decorator(func):
            return func
        return decorator

    class DummyContext:
        def update_current_observation(self, metadata):
            pass
        def get_current_observation_id(self):
            return None
        def get_current_trace_id(self):
            return None

    langfuse_context = DummyContext()
import hashlib

@dataclass
class Event:
    """Represents a single event in the process mining log"""
    case_id: str
    activity: str
    timestamp: str
    resource: str = ""
    lifecycle: Literal["start", "complete"] = "complete"
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary format"""
        return asdict(self)

    def to_xes(self) -> Dict[str, Any]:
        """Convert to XES-compatible format"""
        return {
            "case:concept:name": self.case_id,
            "concept:name": self.activity,
            "time:timestamp": self.timestamp,
            "org:resource": self.resource,
            "lifecycle:transition": self.lifecycle,
            **{f"custom:{k}": v for k, v in self.attributes.items()}
        }

@dataclass
class EventLog:
    """Container for all events in a process execution"""
    case_id: str
    architecture: str
    problem: str
    events: List[Event] = field(default_factory=list)
    graph_nodes: List[Dict[str, Any]] = field(default_factory=list)
    graph_edges: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_event(self, event: Event):
        """Add an event to the log"""
        self.events.append(event)

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire log to dictionary"""
        return {
            "case_id": self.case_id,
            "architecture": self.architecture,
            "problem": self.problem,
            "events": [e.to_dict() for e in self.events],
            "graph": {
                "nodes": self.graph_nodes,
                "edges": self.graph_edges
            },
            "metadata": self.metadata
        }

    def save(self, filepath: Path):
        """Save event log to file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

class ProcessMiningLogger:
    """
    Main logger class that integrates with Langfuse and tracks events
    """

    def __init__(
        self,
        case_id: str,
        architecture: str = "linear_flow",
        problem: str = "unspecified",
        experiment_dir: Optional[Path] = None
    ):
        self.case_id = case_id
        self.architecture = architecture
        self.problem = problem
        self.experiment_dir = experiment_dir

        # Initialize event log
        self.event_log = EventLog(
            case_id=case_id,
            architecture=architecture,
            problem=problem
        )

        # Track current state
        self.current_agent = None
        self.start_times = {}
        self.agent_sequence = []

        # Graph building
        self.node_counter = 0
        self.node_map = {}

    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format"""
        return datetime.utcnow().isoformat() + "Z"

    def _get_langfuse_metadata(self) -> Dict[str, Any]:
        """Extract metadata from current Langfuse context if available"""
        try:
            obs_id = langfuse_context.get_current_observation_id()
            trace_id = langfuse_context.get_current_trace_id()
            return {
                "langfuse_observation_id": obs_id,
                "langfuse_trace_id": trace_id
            }
        except:
            return {}

    def log_user_request(self, request: str):
        """Log the initial user request"""
        event = Event(
            case_id=self.case_id,
            activity="user_request",
            timestamp=self._get_timestamp(),
            resource="user",
            attributes={
                "request": request[:200],  # Truncate long requests
                "request_hash": hashlib.md5(request.encode()).hexdigest(),
                **self._get_langfuse_metadata()
            }
        )
        self.event_log.add_event(event)

        # Add to graph
        node_id = f"node_{self.node_counter}"
        self.node_counter += 1
        self.node_map["user"] = node_id
        self.event_log.graph_nodes.append({
            "id": node_id,
            "type": "actor",
            "name": "user",
            "label": "User Request"
        })

    def log_agent_start(self, agent_name: str, model: str = "gpt-4o-mini"):
        """Log when an agent starts execution"""
        self.current_agent = agent_name
        self.start_times[agent_name] = time.time()

        event = Event(
            case_id=self.case_id,
            activity=f"{agent_name}_execute",
            timestamp=self._get_timestamp(),
            resource=model,
            lifecycle="start",
            attributes={
                "agent_type": agent_name,
                "model": model,
                **self._get_langfuse_metadata()
            }
        )
        self.event_log.add_event(event)

        # Add agent node to graph if not exists
        if agent_name not in self.node_map:
            node_id = f"node_{self.node_counter}"
            self.node_counter += 1
            self.node_map[agent_name] = node_id
            self.event_log.graph_nodes.append({
                "id": node_id,
                "type": "agent",
                "name": agent_name,
                "model": model
            })

    def log_agent_complete(
        self,
        agent_name: str,
        tokens_in: Optional[int] = None,
        tokens_out: Optional[int] = None,
        tools_used: Optional[List[str]] = None,
        success: bool = True
    ):
        """Log when an agent completes execution"""
        duration_ms = None
        if agent_name in self.start_times:
            duration_ms = int((time.time() - self.start_times[agent_name]) * 1000)
            del self.start_times[agent_name]

        event = Event(
            case_id=self.case_id,
            activity=f"{agent_name}_execute",
            timestamp=self._get_timestamp(),
            resource=self.event_log.events[-1].resource if self.event_log.events else "gpt-4o-mini",
            lifecycle="complete",
            attributes={
                "agent_type": agent_name,
                "duration_ms": duration_ms,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "tools_used": tools_used or [],
                "success": success,
                **self._get_langfuse_metadata()
            }
        )
        self.event_log.add_event(event)

        # Track agent sequence for graph edges
        self.agent_sequence.append(agent_name)
        if len(self.agent_sequence) > 1:
            self._add_transition_edge(
                self.agent_sequence[-2],
                self.agent_sequence[-1]
            )

    def log_tool_invocation(
        self,
        tool_name: str,
        agent_name: str,
        parameters: Optional[Dict[str, Any]] = None,
        result_summary: Optional[str] = None
    ):
        """Log when a tool is invoked by an agent"""
        event = Event(
            case_id=self.case_id,
            activity=f"tool_{tool_name}_invoke",
            timestamp=self._get_timestamp(),
            resource=agent_name,
            attributes={
                "tool_name": tool_name,
                "invoking_agent": agent_name,
                "parameters": parameters or {},
                "result_summary": result_summary,
                **self._get_langfuse_metadata()
            }
        )
        self.event_log.add_event(event)

        # Add tool node to graph if not exists
        if tool_name not in self.node_map:
            node_id = f"node_{self.node_counter}"
            self.node_counter += 1
            self.node_map[tool_name] = node_id
            self.event_log.graph_nodes.append({
                "id": node_id,
                "type": "tool",
                "name": tool_name
            })

        # Add edge from agent to tool
        if agent_name in self.node_map and tool_name in self.node_map:
            self.event_log.graph_edges.append({
                "from": self.node_map[agent_name],
                "to": self.node_map[tool_name],
                "type": "uses",
                "timestamp": self._get_timestamp()
            })

    def log_transition(self, from_agent: str, to_agent: str, reason: Optional[str] = None):
        """Log a transition between agents"""
        event = Event(
            case_id=self.case_id,
            activity=f"transition_{from_agent}_to_{to_agent}",
            timestamp=self._get_timestamp(),
            resource="system",
            attributes={
                "from": from_agent,
                "to": to_agent,
                "reason": reason,
                **self._get_langfuse_metadata()
            }
        )
        self.event_log.add_event(event)

    def _add_transition_edge(self, from_agent: str, to_agent: str):
        """Add a transition edge to the graph"""
        if from_agent in self.node_map and to_agent in self.node_map:
            self.event_log.graph_edges.append({
                "from": self.node_map[from_agent],
                "to": self.node_map[to_agent],
                "type": "sequential",
                "timestamp": self._get_timestamp()
            })

    def finalize(self, success: bool = True, error: Optional[str] = None):
        """Finalize the log with summary metadata"""
        # Calculate total duration
        if self.event_log.events:
            start_time = self.event_log.events[0].timestamp
            end_time = self._get_timestamp()

            # Parse timestamps and calculate duration
            from datetime import datetime
            start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
            total_duration_ms = int((end_dt - start_dt).total_seconds() * 1000)
        else:
            total_duration_ms = 0

        self.event_log.metadata = {
            "success": success,
            "error": error,
            "total_duration_ms": total_duration_ms,
            "total_events": len(self.event_log.events),
            "agents_involved": list(set([e.resource for e in self.event_log.events if e.resource != "user" and e.resource != "system"])),
            "tools_used": list(set([e.attributes.get("tool_name") for e in self.event_log.events if "tool_name" in e.attributes])),
            "finalized_at": self._get_timestamp()
        }

        # Save if experiment directory is specified
        if self.experiment_dir:
            log_path = Path(self.experiment_dir) / "event_log.json"
            self.event_log.save(log_path)
            print(f"📊 Process mining log saved to {log_path}")

    def get_event_log(self) -> EventLog:
        """Get the current event log"""
        return self.event_log

    @observe(name="logged_activity")
    def track_with_langfuse(self, activity_name: str, metadata: Dict[str, Any]):
        """
        Decorator method to track activities with Langfuse
        """
        langfuse_context.update_current_observation(
            metadata={
                "activity": activity_name,
                "case_id": self.case_id,
                "architecture": self.architecture,
                **metadata
            }
        )