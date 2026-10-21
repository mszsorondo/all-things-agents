# Generic Agentic Process Mining Algorithm (GAPM)

## Core Focus

Agentic Process Mining is an analytical algorithm that leverages multi-agent systems to discover, interpret, and refine business process models from event logs. The output is enhanced process understanding through refined models and comprehensive descriptions.

## The Generic Algorithm

```
AGENTIC PROCESS MINING ALGORITHM (APM)
=======================================

Input:
  L: Event logs (XES, CSV, or other formats)
  D: Domain context (optional documentation, business rules)
  Φ: Set of process mining algorithms
  Π: Set of prompts for agent reasoning
  C: Configuration (representations to generate, model preferences)

Output:
  M_as_is: AS-IS process model (actual behavior)
  M_ideal: IDEAL process model (optimized behavior)
  A: Comprehensive analysis document
  I: Insights and improvement recommendations

Components:
  MAS: Multi-Agent System for collaborative analysis
  PM_Tools: Process mining algorithm implementations
  LLM: Large Language Model(s) for interpretation

ALGORITHM:

1. INITIALIZATION
   agents ← CONFIGURE_AGENTS(C.agent_types, Π)
   context ← LOAD_CONTEXT(D)
   log ← PREPROCESS(L)

2. MULTI_REPRESENTATION_DISCOVERY
   representations ← {}

   FOR each algorithm φ in Φ:
     IF applicable(φ, log):
       representations[φ] ← DISCOVER(φ, log)

   // Discovered representations might include:
   // - Petri nets (places, transitions, markings)
   // - BPMN fragments (tasks, gateways, flows)
   // - DFG (directly-follows graphs with frequencies)
   // - Heuristics nets (with dependency measures)
   // - Process trees (hierarchical structure)

3. COLLABORATIVE_INTERPRETATION
   interpretations ← []

   FOR each agent in MAS:
     // Each agent analyzes representations from its perspective
     interpretation ← agent.ANALYZE(representations, context, Π)
     interpretations.append(interpretation)

   // Agents identify:
   // - Main process flows
   // - Bottlenecks and inefficiencies
   // - Compliance issues
   // - Rework and loops
   // - Decision points and rules

4. MODEL_SYNTHESIS
   // Consolidate interpretations into unified understanding
   unified_analysis ← MAS.CONSOLIDATE(interpretations)

   // Generate AS-IS model (what actually happens)
   M_as_is ← MAS.GENERATE_AS_IS(unified_analysis, representations)

   // Generate IDEAL model (what should happen)
   M_ideal ← MAS.GENERATE_IDEAL(unified_analysis, context)

   // M_as_is includes: loops, rework, exceptions, violations
   // M_ideal includes: streamlined flow, parallel optimization,
   //                   clear gates, compliance enforcement

5. ANALYSIS_GENERATION
   A ← GENERATE_ANALYSIS_DOCUMENT(
     process_overview: unified_analysis.overview,
     issues_identified: unified_analysis.issues,
     metrics: CALCULATE_METRICS(log, representations),
     variants: EXTRACT_VARIANTS(representations),
     improvements: IDENTIFY_IMPROVEMENTS(M_as_is, M_ideal)
   )

   I ← EXTRACT_INSIGHTS(
     bottlenecks: unified_analysis.bottlenecks,
     optimization_opportunities: M_ideal.improvements,
     compliance_gaps: unified_analysis.violations,
     automation_candidates: unified_analysis.automatable_tasks
   )

6. OUTPUT_GENERATION
   // Format outputs for consumption
   EXPORT_BPMN(M_as_is, "as_is_process.bpmn")
   EXPORT_BPMN(M_ideal, "ideal_process.bpmn")
   EXPORT_DOCUMENT(A, "process_analysis.txt")
   EXPORT_INSIGHTS(I, "insights_report.json")

7. RETURN M_as_is, M_ideal, A, I
```

## Key Components

### 1. Multi-Agent Collaboration

Different agents bring specialized perspectives:

- **Structure Analyst**: Focuses on control flow and process skeleton
- **Performance Analyst**: Identifies bottlenecks and delays
- **Compliance Checker**: Detects violations and governance issues
- **Optimization Expert**: Proposes improvements and simplifications
- **Variant Detective**: Identifies and explains process variations

### 2. Representation Ensemble

Multiple representations capture different aspects:

```python
representations = {
    'petri_net': formal_semantics_and_concurrency,
    'bpmn': business_readable_notation,
    'dfg': frequency_and_main_paths,
    'heuristics': noise_filtered_dependencies,
    'process_tree': hierarchical_structure
}
```

### 3. LLM-Enhanced Interpretation

Agents use LLMs to:
- Understand semantic meaning of activities
- Infer business rules from patterns
- Generate human-readable explanations
- Connect technical models to business objectives

### 4. Dual Model Generation

**AS-IS Model**: Faithful representation of actual behavior
- Preserves all observed paths
- Includes problematic patterns
- Shows real complexity

**IDEAL Model**: Optimized target process
- Removes unnecessary loops
- Streamlines decision logic
- Enforces best practices

## Separation of Concerns

This algorithm focuses ONLY on analysis and model generation. Separate algorithms handle:

### Algorithm 2: Agentic System Proposal (Future Work)
```
Input: M_ideal, I (from APM)
Output: Agent architecture specification
Purpose: Design multi-agent system for process automation
Status: Experimental
```

### Algorithm 3: Incremental Integration (Future Work)
```
Input: Agent specification, Business system interfaces
Output: Deployed agent components
Purpose: Gradually automate process tasks
Status: Conceptual
```

### Algorithm 4: Self-Improvement Loop (Future Work)
```
Input: Execution logs, Performance metrics
Output: Refined models and agents
Purpose: Continuous optimization
Status: Research
```

## Implementation Considerations

### Current Scope (Implemented)
- Event log ingestion
- Multi-algorithm discovery
- LLM-based interpretation
- BPMN generation (AS-IS and IDEAL)
- Analysis documentation

### Near-term Extensions (Planned)
- Richer representation support
- Multi-agent deliberation
- Conformance checking
- Simulation capabilities

### Long-term Vision (Research)
- Automated agent generation
- Live system integration
- Self-improving architectures
- Autonomous process optimization

## Algorithm Properties

### Completeness
The algorithm considers multiple perspectives through different representations and agents, providing comprehensive coverage of process aspects.

### Soundness
Generated models are validated against the original logs to ensure they accurately represent observed behavior (AS-IS) or maintain logical consistency (IDEAL).

### Scalability
Parallel processing of representations and agent analyses enables handling of large event logs.

### Generality
The algorithm is domain-agnostic and works with any event log following standard formats.

## Practical Usage

```python
# Example instantiation
apm = AgenticProcessMiner(
    agents=['structure', 'performance', 'compliance'],
    algorithms=['inductive_miner', 'dfg', 'heuristics'],
    output_format='bpmn'
)

# Run analysis
as_is, ideal, analysis, insights = apm.mine(
    event_log='process.xes',
    domain_context='business_rules.pdf'
)

# Results
print(f"Issues found: {len(analysis.issues)}")
print(f"Improvement potential: {insights.optimization_score}%")
```

## Key Innovation

The core innovation of Agentic Process Mining is using multi-agent collaboration with LLM reasoning to:

1. **Interpret** multiple process representations collectively
2. **Understand** business semantics beyond structural patterns
3. **Generate** both descriptive (AS-IS) and prescriptive (IDEAL) models
4. **Explain** findings in business-friendly language

This goes beyond traditional algorithmic process mining by adding semantic understanding and collaborative reasoning, while staying focused on the analytical task rather than premature automation.

## Conclusion

The Agentic Process Mining Algorithm provides a robust foundation for intelligent process analysis. By focusing on the analytical phase, it delivers immediate value through enhanced understanding and improvement recommendations, while laying groundwork for future automation experiments.

The separation between core process mining and experimental automation extensions ensures the algorithm remains practical and implementable today, while keeping the door open for ambitious future developments.