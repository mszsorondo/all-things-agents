"""
Hierarchical Expert System Architecture for Process Mining
==========================================================

This implementation creates a multi-agent system where specialized agents
analyze different aspects of process models and coordinate through a
hierarchical meta-agent.

Architecture Overview:
- 5 Specialist Agents (Structure, Performance, Compliance, Anomaly, Optimization)
- 1 Meta-Coordinator Agent for synthesis
- Shared knowledge base for inter-agent communication
- Token and time tracking for performance measurement
"""

import os
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
#from langgraph.graph import Graph, StateGraph
from langgraph.prebuilt import create_react_agent
from langchain.tools import tool
import pm4py

from enhanced_agentic_miner import (
    AgenticProcessMiner,
    MiningConfiguration,
    RepresentationType,
    ProcessRepresentation
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AgentMetrics:
    """Track metrics for each agent"""
    tokens_used: int = 0
    time_elapsed: float = 0.0
    calls_made: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class SharedKnowledge:
    """Shared knowledge base for inter-agent communication"""
    structural_insights: List[str] = field(default_factory=list)
    performance_bottlenecks: List[str] = field(default_factory=list)
    compliance_violations: List[str] = field(default_factory=list)
    anomalies_detected: List[str] = field(default_factory=list)
    optimization_suggestions: List[str] = field(default_factory=list)
    consensus_model: Optional[str] = None
    conflicting_views: List[Dict[str, str]] = field(default_factory=list)
    confidence_scores: Dict[str, float] = field(default_factory=dict)


class ExpertType(Enum):
    """Types of expert agents in the hierarchy"""
    STRUCTURE = "structure"
    PERFORMANCE = "performance"
    COMPLIANCE = "compliance"
    ANOMALY = "anomaly"
    OPTIMIZATION = "optimization"
    META_COORDINATOR = "meta_coordinator"


class SpecialistAgent:
    """Base class for specialist agents"""

    def __init__(self, expert_type: ExpertType, model_name: str = "gpt-4o-mini"):
        self.expert_type = expert_type
        self.model = ChatOpenAI(model=model_name, temperature=0)
        self.metrics = AgentMetrics()
        self.tools = self._create_tools()

    def _create_tools(self):
        """Create tools specific to this expert"""

        @tool
        def record_insight(category: str, insight: str) -> str:
            """Record an insight in the shared knowledge base"""
            return f"Recorded {category} insight: {insight}"

        @tool
        def query_knowledge(category: str) -> str:
            """Query the shared knowledge base"""
            return f"Querying {category} knowledge"

        return [record_insight, query_knowledge]

    def analyze(self, representations: Dict[RepresentationType, ProcessRepresentation],
                shared_knowledge: SharedKnowledge) -> Tuple[str, AgentMetrics]:
        """
        Analyze representations from the expert's perspective
        Returns: (analysis_result, metrics)
        """
        start_time = time.time()

        # Build specialized prompt based on expert type
        prompt = self._build_expert_prompt(representations, shared_knowledge)

        # Call LLM with metrics tracking
        messages = [
            SystemMessage(content=self._get_system_prompt()),
            HumanMessage(content=prompt)
        ]

        try:
            response = self.model.invoke(messages)

            # Track metrics if available
            if hasattr(response, 'response_metadata'):
                metadata = response.response_metadata
                if 'token_usage' in metadata:
                    self.metrics.prompt_tokens += metadata['token_usage'].get('prompt_tokens', 0)
                    self.metrics.completion_tokens += metadata['token_usage'].get('completion_tokens', 0)
                    self.metrics.tokens_used = self.metrics.prompt_tokens + self.metrics.completion_tokens

            self.metrics.time_elapsed += time.time() - start_time
            self.metrics.calls_made += 1

            # Update shared knowledge based on analysis
            self._update_shared_knowledge(response.content, shared_knowledge)

            return response.content, self.metrics

        except Exception as e:
            logger.error(f"Error in {self.expert_type.value} agent: {e}")
            return f"Error in analysis: {str(e)}", self.metrics

    def _get_system_prompt(self) -> str:
        """Get specialized system prompt for this expert"""
        prompts = {
            ExpertType.STRUCTURE: """You are a Process Structure Expert. Analyze the control flow,
            branching patterns, loops, and hierarchical decomposition of the process model. Focus on:
            - Sequence flows and gateways (XOR, AND, OR patterns)
            - Process hierarchy and sub-processes
            - Start and end events
            - Activity dependencies and relationships""",

            ExpertType.PERFORMANCE: """You are a Process Performance Expert. Analyze throughput,
            bottlenecks, and efficiency metrics. Focus on:
            - Activity durations and waiting times
            - Resource utilization patterns
            - Throughput and cycle time analysis
            - Performance bottlenecks identification
            - Optimization opportunities for speed""",

            ExpertType.COMPLIANCE: """You are a Process Compliance Expert. Verify adherence to
            rules, standards, and expected behaviors. Focus on:
            - Mandatory activity sequences
            - Business rule violations
            - Process conformance checking
            - Deviation patterns from expected model
            - Regulatory requirement satisfaction""",

            ExpertType.ANOMALY: """You are a Process Anomaly Detection Expert. Identify unusual
            patterns and outliers in the process. Focus on:
            - Infrequent trace patterns
            - Unexpected activity sequences
            - Outlier case durations
            - Unusual resource assignments
            - Potential fraud or error indicators""",

            ExpertType.OPTIMIZATION: """You are a Process Optimization Expert. Suggest improvements
            for efficiency and effectiveness. Focus on:
            - Process simplification opportunities
            - Parallel execution possibilities
            - Resource reallocation suggestions
            - Automation candidates
            - Cost reduction strategies"""
        }

        return prompts.get(self.expert_type, "You are a process mining expert.")

    def _build_expert_prompt(self, representations: Dict[RepresentationType, ProcessRepresentation],
                             shared_knowledge: SharedKnowledge) -> str:
        """Build prompt with relevant representations for this expert"""

        # Select most relevant representations for each expert
        relevant_reps = self._select_relevant_representations(representations)

        prompt_parts = [
            f"Analyze the following process representations from a {self.expert_type.value} perspective:\n"
        ]

        # Add relevant representations
        for rep_type, rep in relevant_reps.items():
            prompt_parts.append(f"\n{rep_type.value.upper()} Representation:\n")
            prompt_parts.append(f"{rep.description}\n")
            prompt_parts.append(f"Interpretation Guide: {rep.interpretation_guide}\n")

        # Add insights from other experts if available
        if shared_knowledge.structural_insights and self.expert_type != ExpertType.STRUCTURE:
            prompt_parts.append(f"\nStructural insights from colleague: {shared_knowledge.structural_insights}\n")

        if shared_knowledge.performance_bottlenecks and self.expert_type != ExpertType.PERFORMANCE:
            prompt_parts.append(f"\nPerformance insights from colleague: {shared_knowledge.performance_bottlenecks}\n")

        prompt_parts.append("\nProvide your expert analysis focusing on your specialization area.")

        return "\n".join(prompt_parts)

    def _select_relevant_representations(self, representations: Dict[RepresentationType, ProcessRepresentation]) -> Dict:
        """Select most relevant representations for this expert type"""

        relevance_map = {
            ExpertType.STRUCTURE: [RepresentationType.PETRI_NET, RepresentationType.BPMN,
                                  RepresentationType.PROCESS_TREE],
            ExpertType.PERFORMANCE: [RepresentationType.DFG, RepresentationType.HEURISTICS_NET,
                                   RepresentationType.ABSTRACT_VARIANTS],
            ExpertType.COMPLIANCE: [RepresentationType.PETRI_NET, RepresentationType.BPMN,
                                  RepresentationType.PROCESS_TREE],
            ExpertType.ANOMALY: [RepresentationType.ABSTRACT_VARIANTS, RepresentationType.DFG,
                               RepresentationType.HEURISTICS_NET],
            ExpertType.OPTIMIZATION: [RepresentationType.HEURISTICS_NET, RepresentationType.DFG,
                                    RepresentationType.BPMN]
        }

        relevant_types = relevance_map.get(self.expert_type, list(RepresentationType))

        return {rep_type: rep for rep_type, rep in representations.items()
                if rep_type in relevant_types}

    def _update_shared_knowledge(self, analysis: str, shared_knowledge: SharedKnowledge):
        """Update shared knowledge based on analysis results"""

        # Parse and categorize insights based on expert type
        if self.expert_type == ExpertType.STRUCTURE:
            shared_knowledge.structural_insights.append(analysis)
        elif self.expert_type == ExpertType.PERFORMANCE:
            shared_knowledge.performance_bottlenecks.append(analysis)
        elif self.expert_type == ExpertType.COMPLIANCE:
            shared_knowledge.compliance_violations.append(analysis)
        elif self.expert_type == ExpertType.ANOMALY:
            shared_knowledge.anomalies_detected.append(analysis)
        elif self.expert_type == ExpertType.OPTIMIZATION:
            shared_knowledge.optimization_suggestions.append(analysis)


class MetaCoordinatorAgent:
    """Meta-coordinator agent that synthesizes insights from all experts"""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model = ChatOpenAI(model=model_name, temperature=0)
        self.metrics = AgentMetrics()

    def synthesize(self, shared_knowledge: SharedKnowledge,
                   expert_analyses: Dict[ExpertType, str]) -> Tuple[str, AgentMetrics]:
        """
        Synthesize all expert insights into a comprehensive process model
        Returns: (synthesis_result, metrics)
        """
        start_time = time.time()

        system_prompt = """You are the Meta-Coordinator for a team of process mining experts.
        Your role is to synthesize insights from multiple specialist agents into a comprehensive,
        coherent process model description.

        You must:
        1. Resolve any conflicting views between experts
        2. Integrate complementary insights
        3. Produce a unified textual process model
        4. Highlight key findings from each expert
        5. Provide confidence scores for different aspects

        The output should be a clear, structured textual description of the discovered process model."""

        # Build synthesis prompt
        prompt_parts = ["Synthesize the following expert analyses:\n"]

        for expert_type, analysis in expert_analyses.items():
            prompt_parts.append(f"\n{expert_type.value.upper()} EXPERT ANALYSIS:")
            prompt_parts.append(f"{analysis}\n")

        prompt_parts.append("\nShared Knowledge Summary:")
        prompt_parts.append(f"- Structural Insights: {len(shared_knowledge.structural_insights)} findings")
        prompt_parts.append(f"- Performance Issues: {len(shared_knowledge.performance_bottlenecks)} bottlenecks")
        prompt_parts.append(f"- Compliance Violations: {len(shared_knowledge.compliance_violations)} issues")
        prompt_parts.append(f"- Anomalies Detected: {len(shared_knowledge.anomalies_detected)} patterns")
        prompt_parts.append(f"- Optimization Opportunities: {len(shared_knowledge.optimization_suggestions)} suggestions")

        prompt_parts.append("\n\nCreate a comprehensive textual process model that incorporates all expert perspectives.")

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content="\n".join(prompt_parts))
        ]

        try:
            response = self.model.invoke(messages)

            # Track metrics
            if hasattr(response, 'response_metadata'):
                metadata = response.response_metadata
                if 'token_usage' in metadata:
                    self.metrics.prompt_tokens = metadata['token_usage'].get('prompt_tokens', 0)
                    self.metrics.completion_tokens = metadata['token_usage'].get('completion_tokens', 0)
                    self.metrics.tokens_used = self.metrics.prompt_tokens + self.metrics.completion_tokens

            self.metrics.time_elapsed = time.time() - start_time
            self.metrics.calls_made = 1

            # Update shared knowledge with consensus
            shared_knowledge.consensus_model = response.content

            return response.content, self.metrics

        except Exception as e:
            logger.error(f"Error in meta-coordinator: {e}")
            return f"Error in synthesis: {str(e)}", self.metrics


class HierarchicalExpertMiner(AgenticProcessMiner):
    """
    Hierarchical Expert System implementation for process mining

    This architecture employs multiple specialist agents coordinated by a meta-agent:
    - Structure Expert: Analyzes control flow patterns
    - Performance Expert: Evaluates efficiency metrics
    - Compliance Expert: Checks rule adherence
    - Anomaly Expert: Detects unusual patterns
    - Optimization Expert: Suggests improvements
    - Meta-Coordinator: Synthesizes all insights
    """

    def __init__(self, config: MiningConfiguration):
        super().__init__(config)
        self.shared_knowledge = SharedKnowledge()
        self.total_metrics = AgentMetrics()

        # Initialize specialist agents
        self.experts = {
            ExpertType.STRUCTURE: SpecialistAgent(ExpertType.STRUCTURE, config.model_name),
            ExpertType.PERFORMANCE: SpecialistAgent(ExpertType.PERFORMANCE, config.model_name),
            ExpertType.COMPLIANCE: SpecialistAgent(ExpertType.COMPLIANCE, config.model_name),
            ExpertType.ANOMALY: SpecialistAgent(ExpertType.ANOMALY, config.model_name),
            ExpertType.OPTIMIZATION: SpecialistAgent(ExpertType.OPTIMIZATION, config.model_name)
        }

        # Initialize meta-coordinator
        self.coordinator = MetaCoordinatorAgent(config.model_name)

    def run_agent(self, system_prompt: Optional[str] = None):
        """
        Override the base run_agent to implement hierarchical multi-agent system
        """
        logger.info("Starting Hierarchical Expert System analysis...")

        # Ensure representations are discovered (uses cache if available)
        if not self.representations:
            logger.info("Discovering process representations...")
            self.discover_representations()

        # Phase 1: Parallel expert analysis
        expert_analyses = {}
        expert_metrics = {}

        logger.info("Phase 1: Expert agents analyzing representations...")

        for expert_type, expert in self.experts.items():
            logger.info(f"  {expert_type.value} expert analyzing...")
            analysis, metrics = expert.analyze(self.representations, self.shared_knowledge)
            expert_analyses[expert_type] = analysis
            expert_metrics[expert_type] = metrics

            # Aggregate metrics
            self.total_metrics.tokens_used += metrics.tokens_used
            self.total_metrics.time_elapsed += metrics.time_elapsed
            self.total_metrics.calls_made += metrics.calls_made
            self.total_metrics.prompt_tokens += metrics.prompt_tokens
            self.total_metrics.completion_tokens += metrics.completion_tokens

            logger.info(f"    Tokens used: {metrics.tokens_used}, Time: {metrics.time_elapsed:.2f}s")

        # Phase 2: Meta-coordination and synthesis
        logger.info("Phase 2: Meta-coordinator synthesizing insights...")

        synthesis, coord_metrics = self.coordinator.synthesize(
            self.shared_knowledge,
            expert_analyses
        )

        # Add coordinator metrics
        self.total_metrics.tokens_used += coord_metrics.tokens_used
        self.total_metrics.time_elapsed += coord_metrics.time_elapsed
        self.total_metrics.calls_made += coord_metrics.calls_made
        self.total_metrics.prompt_tokens += coord_metrics.prompt_tokens
        self.total_metrics.completion_tokens += coord_metrics.completion_tokens

        logger.info(f"  Coordinator tokens: {coord_metrics.tokens_used}, Time: {coord_metrics.time_elapsed:.2f}s")

        # Phase 3: Output generation
        logger.info("Phase 3: Generating outputs...")

        # Write the synthesized model
        output_path = self.run_folder / "discovered_model.txt"
        output_path.write_text(synthesis)

        # Write detailed analysis from each expert
        details_folder = self.run_folder / "expert_analyses"
        details_folder.mkdir(exist_ok=True)

        for expert_type, analysis in expert_analyses.items():
            expert_file = details_folder / f"{expert_type.value}_analysis.txt"
            expert_file.write_text(analysis)

        # Write metrics summary
        metrics_summary = {
            "total_tokens_used": self.total_metrics.tokens_used,
            "total_time_elapsed": self.total_metrics.time_elapsed,
            "total_llm_calls": self.total_metrics.calls_made,
            "prompt_tokens": self.total_metrics.prompt_tokens,
            "completion_tokens": self.total_metrics.completion_tokens,
            "expert_metrics": {
                expert_type.value: {
                    "tokens_used": metrics.tokens_used,
                    "time_elapsed": metrics.time_elapsed,
                    "calls_made": metrics.calls_made
                }
                for expert_type, metrics in expert_metrics.items()
            },
            "coordinator_metrics": {
                "tokens_used": coord_metrics.tokens_used,
                "time_elapsed": coord_metrics.time_elapsed,
                "calls_made": coord_metrics.calls_made
            }
        }

        metrics_path = self.run_folder / "multi_agent_metrics.json"
        metrics_path.write_text(json.dumps(metrics_summary, indent=2))

        # Write shared knowledge
        knowledge_path = self.run_folder / "shared_knowledge.json"
        knowledge_data = {
            "structural_insights": self.shared_knowledge.structural_insights,
            "performance_bottlenecks": self.shared_knowledge.performance_bottlenecks,
            "compliance_violations": self.shared_knowledge.compliance_violations,
            "anomalies_detected": self.shared_knowledge.anomalies_detected,
            "optimization_suggestions": self.shared_knowledge.optimization_suggestions,
            "consensus_model": self.shared_knowledge.consensus_model
        }
        knowledge_path.write_text(json.dumps(knowledge_data, indent=2))

        logger.info(f"\nHierarchical Expert System Analysis Complete!")
        logger.info(f"Total tokens used: {self.total_metrics.tokens_used}")
        logger.info(f"Total time elapsed: {self.total_metrics.time_elapsed:.2f}s")
        logger.info(f"Total LLM calls: {self.total_metrics.calls_made}")
        logger.info(f"Results saved to: {self.run_folder}")

        return synthesis

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get metrics for the multi-agent system
        """
        return {
            "architecture": "HierarchicalExpertSystem",
            "total_tokens": self.total_metrics.tokens_used,
            "total_time": self.total_metrics.time_elapsed,
            "total_calls": self.total_metrics.calls_made,
            "prompt_tokens": self.total_metrics.prompt_tokens,
            "completion_tokens": self.total_metrics.completion_tokens
        }