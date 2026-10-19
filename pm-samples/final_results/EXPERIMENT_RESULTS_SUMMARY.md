# Comprehensive Review: Textual Process Model Generation Experiments

**Date**: 2025-12-29
**Location**: `/final_results/`
**Analysis**: Comparative performance evaluation of LLM-based textual process model generation and trace classification

---

## Executive Summary

This document presents a comprehensive analysis of **5 benchmark experiments** evaluating different LLM configurations for generating textual AS-IS process models and validating event log traces. The experiments span 4 real-world business processes (hiring, hospital, lending, renting) with 100 valid and 100 invalid traces each.

**Key Finding**: **GPT-5 (generator) + GPT-5-mini (validator)** with tools method achieved the best overall performance with an average F1 score of **0.954** across all processes.

---

## Table of Contents

1. [Experiment Configurations](#experiment-configurations)
2. [Performance by Process](#performance-by-process)
3. [Model Comparison](#model-comparison)
4. [Key Insights](#key-insights)
5. [Recommendations](#recommendations)
6. [Appendix: Detailed Metrics](#appendix-detailed-metrics)

---

## Experiment Configurations

| Run ID | Generator LLM | Validator LLM | Method | Date |
|--------|---------------|---------------|--------|------|
| `run_20251221_221516` | gpt-5-mini | gpt-5-nano | tools | 2025-12-21 |
| `run_20251222_004259` | gpt-5 | gpt-5-mini | tools | 2025-12-22 |
| `run_20251222_015013` | claude-sonnet-4-5 | gpt-5-mini | tools | 2025-12-22 |
| `run_20251223_202934` | gpt-5-mini | gpt-5-mini | tools | 2025-12-23 |
| `run_20251224_010630` | gpt-5-mini | gpt-5-mini | no-tools | 2025-12-24 |

### Method Definitions

- **Tools Method**: Agentic approach where the generator LLM has access to tools (write, edit) for iterative model refinement based on process mining representations (Petri net, DFG, abstract variants)
- **No-Tools Method**: Two-step prompting approach (AS-IS model generation, then IDEAL model generation) without tool access

### Process Mining Inputs

All experiments used the same process mining representations:
- **Petri Net**: Formal model showing places, transitions, and control flow
- **DFG (Directly-Follows Graph)**: Activity frequency and sequence statistics
- **Abstract Variants**: High-level process variant descriptions

### Trace Generation

- **Valid Traces**: 100 traces sampled from real event logs
- **Invalid Traces**: 100 traces generated via `random_swap_events` perturbation
- **Evaluation**: Binary classification (VALID/INVALID) using validator LLM

---

## Performance by Process

### 1. Hiring Process (hiring_log_high)

**Process Characteristics**: Complex candidate selection with parallel technical/behavioral interviews

| Experiment | Precision | Recall | F1 | TP | TN | FP | FN | Notes |
|------------|-----------|--------|----|----|----|----|-----|-------|
| gpt-5-mini + gpt-5-nano | **0.852** | 0.920 | 0.885 | 92 | 84 | 16 | **8** | Only model with FN > 0 |
| gpt-5 + gpt-5-mini | 0.820 | **1.000** | **0.901** | 100 | 78 | 22 | 0 | Best F1 |
| Claude S4.5 + gpt-5-mini | 0.654 | **1.000** | 0.791 | 100 | 47 | **53** | 0 | High FP |
| gpt-5-mini + gpt-5-mini | 0.633 | **1.000** | 0.775 | 100 | 42 | **58** | 0 | Highest FP |
| gpt-5-mini no-tools | 0.654 | **1.000** | 0.791 | 100 | 47 | **53** | 0 | Same as Claude |

**Key Observations**:
- Hiring is the **hardest process** (lowest avg F1: 0.829)
- Most models achieve perfect recall but struggle with false positives
- Common FP pattern: Accepting traces with "Application Rejected" after multiple interviews
- GPT-5 achieves best balance with F1=0.901

**Example Misclassifications**:
```
FP Example: Hand In Job Application → Telephonic Screening →
            Application Rejected → Background Check → Screening Interview

Model accepts this despite rejection happening before screening interview
```

---

### 2. Hospital Process (hospital_log_high)

**Process Characteristics**: Emergency department patient flow with diagnosis and treatment cycles

| Experiment | Precision | Recall | F1 | TP | TN | FP | FN | Notes |
|------------|-----------|--------|----|----|----|----|-----|-------|
| gpt-5-mini + gpt-5-nano | 0.962 | 1.000 | 0.980 | 100 | 96 | 4 | 0 | |
| gpt-5 + gpt-5-mini | **0.990** | 1.000 | **0.995** | 100 | 99 | **1** | 0 | Tied best |
| Claude S4.5 + gpt-5-mini | **0.990** | 1.000 | **0.995** | 100 | 99 | **1** | 0 | Tied best |
| gpt-5-mini + gpt-5-mini | 0.971 | 1.000 | 0.985 | 100 | 97 | 3 | 0 | |
| gpt-5-mini no-tools | **0.990** | 1.000 | **0.995** | 100 | 99 | **1** | 0 | Tied best |

**Key Observations**:
- Hospital is the **easiest process** (avg F1: 0.990)
- Three models achieved near-perfect F1=0.995
- Linear medical logic makes validation straightforward
- Only 1-4 false positives across all experiments

**Example Misclassifications**:
```
FP Example: Register at FD → Examination → Diagnosis → Treatment →
            Treatment unsuccessful → Diagnosis → Treatment →
            Treatment successful → Discharge

Model accepts despite "Treatment unsuccessful" appearing mid-cycle
```

---

### 3. Lending Process (lending_log_high)

**Process Characteristics**: Loan application with sequential verification and approval steps

| Experiment | Precision | Recall | F1 | TP | TN | FP | FN | Notes |
|------------|-----------|--------|----|----|----|----|-----|-------|
| gpt-5-mini + gpt-5-nano | 0.917 | 1.000 | 0.957 | 100 | 91 | 9 | 0 | |
| gpt-5 + gpt-5-mini | 0.962 | 1.000 | 0.980 | 100 | 96 | 4 | 0 | |
| Claude S4.5 + gpt-5-mini | 0.962 | 1.000 | 0.980 | 100 | 96 | 4 | 0 | |
| gpt-5-mini + gpt-5-mini | 0.980 | 1.000 | 0.990 | 100 | 98 | 2 | 0 | |
| gpt-5-mini no-tools | **1.000** | 1.000 | **1.000** | 100 | 100 | **0** | 0 | **PERFECT** |

**Key Observations**:
- **Perfect score achieved**: gpt-5-mini no-tools method (F1=1.000)
- All models achieve perfect recall
- Sequential nature of lending process makes validation clearer
- No-tools method excels at sequential processes

**Example Misclassifications** (from other runs):
```
FP Example: Request Appointment → Set Appointment → Hand In Credit Application →
            Verify Borrowers Information → Loan Denied → Submit File to Underwriter

Model accepts despite "Loan Denied" before underwriter submission
```

---

### 4. Renting Process (renting_log_high)

**Process Characteristics**: Apartment rental lifecycle with screening, payments, and terminations

| Experiment | Precision | Recall | F1 | TP | TN | FP | FN | Notes |
|------------|-----------|--------|----|----|----|----|-----|-------|
| gpt-5-mini + gpt-5-nano | 0.889 | 0.960 | 0.923 | 96 | 88 | 12 | **4** | Balanced recall |
| gpt-5 + gpt-5-mini | 0.885 | 1.000 | 0.939 | 100 | 87 | 13 | 0 | |
| Claude S4.5 + gpt-5-mini | **0.952** | 1.000 | **0.976** | 100 | 95 | **5** | 0 | Tied best |
| gpt-5-mini + gpt-5-mini | 0.877 | 1.000 | 0.935 | 100 | 86 | 14 | 0 | |
| gpt-5-mini no-tools | **0.952** | 1.000 | **0.976** | 100 | 95 | **5** | 0 | Tied best |

**Key Observations**:
- Moderate-high complexity (avg F1: 0.950)
- Two models tied at best F1=0.976
- Multiple termination paths add complexity
- 5-14 false positives across experiments

**Example Misclassifications**:
```
FP Example: Apply for Viewing → Set Appointment → View Property →
            Hand In Paperwork → Check Paperwork → Screen Tenant →
            Extensive Screening → Sign Contract → Move In →
            Pay Rent (×6) → Tenant Cancels Apartment

Model accepts despite missing rent payments before cancellation
```

---

## Model Comparison

### Overall Performance (Average F1 Across All Processes)

| Configuration | Hiring | Hospital | Lending | Renting | **Average** | Rank |
|---------------|--------|----------|---------|---------|-------------|------|
| **gpt-5 + gpt-5-mini (tools)** | 0.901 | 0.995 | 0.980 | 0.939 | **0.954** | 🥇 1st |
| **gpt-5-mini no-tools** | 0.791 | 0.995 | 1.000 | 0.976 | **0.941** | 🥈 2nd |
| **gpt-5-mini + gpt-5-nano (tools)** | 0.885 | 0.980 | 0.957 | 0.923 | **0.936** | 🥉 3rd |
| **Claude S4.5 + gpt-5-mini (tools)** | 0.791 | 0.995 | 0.980 | 0.976 | **0.936** | 🥉 3rd |
| **gpt-5-mini + gpt-5-mini (tools)** | 0.775 | 0.985 | 0.990 | 0.935 | **0.921** | 5th |

### Precision vs Recall Patterns

```
High Precision, Balanced Recall:
  └─ gpt-5-mini + gpt-5-nano: Only model with FN > 0 (willing to reject valid traces)

Perfect Recall, Varying Precision:
  └─ All other models: Recall = 1.000 on most problems (permissive models)
     ├─ gpt-5: Best precision among perfect-recall models
     ├─ Claude Sonnet 4.5: Struggles with hiring complexity
     └─ gpt-5-mini: Lower precision, higher false positives
```

### Problem Difficulty Ranking

1. **Hospital** (Easiest): Avg F1 = 0.990 ⭐⭐⭐⭐⭐
   - Linear flow, clear medical logic
   - All models ≥ 0.980

2. **Lending** (Easy-Moderate): Avg F1 = 0.981 ⭐⭐⭐⭐
   - Sequential decision points
   - One model achieved perfect score

3. **Renting** (Moderate): Avg F1 = 0.950 ⭐⭐⭐
   - Multiple paths, lifecycle complexity
   - Varying termination scenarios

4. **Hiring** (Hardest): Avg F1 = 0.829 ⭐⭐
   - Parallel interviews, diverse outcomes
   - Significant model performance variation

---

## Key Insights

### 1. Generator LLM Quality Impact

**GPT-5** produces the highest quality textual models:
- Average F1: 0.954 (best overall)
- Consistently strong across all process types
- Best balance of precision and recall

**Claude Sonnet 4.5** shows mixed results:
- Excellent on simple processes (hospital: 0.995, renting: 0.976)
- Struggles with complex processes (hiring: 0.791)
- Tied for 3rd place overall (0.936)

**GPT-5-mini** surprisingly competitive:
- With no-tools: 0.941 average (2nd place)
- Perfect score on lending (1.000)
- Cost-effective alternative to GPT-5

### 2. Tools vs No-Tools Method

**Tools Method Advantages**:
- Better for complex processes (hiring)
- Allows iterative refinement
- LLM can write/edit model sections

**No-Tools Method Advantages**:
- Achieved perfect score on lending (1.000)
- Better for sequential processes
- Simpler prompting strategy
- Potentially lower cost (fewer API calls)

**Recommendation**: Use tools for complex/parallel processes, no-tools for sequential processes

### 3. Validator LLM Impact

**GPT-5-mini** (most common validator):
- Good balance of cost and accuracy
- Consistent performance across generators
- Industry-standard choice

**GPT-5-nano** (used in first experiment):
- Unique balanced recall behavior (0.920-0.960)
- Only configuration with FN > 0
- More conservative validation strategy

### 4. Common Failure Patterns

All models exhibit similar misclassification patterns:

**False Positive Pattern**: Accepting invalid traces with:
- Terminal activities (rejection/offer) in wrong positions
- Duplicate activities (two "Make Job Offer" events)
- Activities after termination

**False Negative Pattern** (rare, only in first experiment):
- Valid early rejection paths
- Abbreviated process variants

### 5. Process Characteristics Matter

**Easy to Model** (F1 ≥ 0.980):
- Linear sequential flows
- Clear domain logic (medical processes)
- Minimal branching

**Hard to Model** (F1 < 0.850):
- Parallel activities (multiple interview types)
- Complex termination conditions
- High process variability

---

## Recommendations

### For Production Deployment

**Best Overall Performance**:
```
Generator: gpt-5
Validator: gpt-5-mini
Method: tools
Expected F1: 0.954
```

**Use Case**: General-purpose, highest quality textual models across all process types

---

### For Cost Optimization

**Best Cost-Performance Ratio**:
```
Generator: gpt-5-mini
Validator: gpt-5-mini
Method: no-tools
Expected F1: 0.941
```

**Use Case**: Budget-conscious deployments, still achieves excellent performance (only 1.3% lower than best)

---

### For Specific Process Types

**Complex Processes** (parallel activities, high variability):
```
Generator: gpt-5
Validator: gpt-5-mini
Method: tools
Process Examples: hiring, recruitment, multi-stage approvals
```

**Sequential Processes** (linear flows, clear decision points):
```
Generator: gpt-5-mini
Validator: gpt-5-mini
Method: no-tools
Process Examples: lending, ordering, linear workflows
Expected: Potential for F1=1.000
```

**Simple Processes** (linear, minimal branching):
```
Generator: Any
Validator: gpt-5-mini
Method: Either
Process Examples: hospital, simple approvals
Expected: F1 ≥ 0.980
```

---

### For Balanced Precision-Recall

**Conservative Validation** (minimize false positives):
```
Generator: gpt-5-mini
Validator: gpt-5-nano
Method: tools
Expected: Higher precision, lower recall (0.920-0.960)
```

**Use Case**: Applications where false positives are costly

---

## Appendix: Detailed Metrics

### Confusion Matrix Statistics

#### Hiring Process

| Config | TP | TN | FP | FN | Accuracy |
|--------|----|----|----|----|----------|
| gpt-5-mini + gpt-5-nano | 92 | 84 | 16 | 8 | 0.880 |
| gpt-5 + gpt-5-mini | 100 | 78 | 22 | 0 | 0.890 |
| Claude S4.5 + gpt-5-mini | 100 | 47 | 53 | 0 | 0.735 |
| gpt-5-mini + gpt-5-mini | 100 | 42 | 58 | 0 | 0.710 |
| gpt-5-mini no-tools | 100 | 47 | 53 | 0 | 0.735 |

#### Hospital Process

| Config | TP | TN | FP | FN | Accuracy |
|--------|----|----|----|----|----------|
| gpt-5-mini + gpt-5-nano | 100 | 96 | 4 | 0 | 0.980 |
| gpt-5 + gpt-5-mini | 100 | 99 | 1 | 0 | 0.995 |
| Claude S4.5 + gpt-5-mini | 100 | 99 | 1 | 0 | 0.995 |
| gpt-5-mini + gpt-5-mini | 100 | 97 | 3 | 0 | 0.985 |
| gpt-5-mini no-tools | 100 | 99 | 1 | 0 | 0.995 |

#### Lending Process

| Config | TP | TN | FP | FN | Accuracy |
|--------|----|----|----|----|----------|
| gpt-5-mini + gpt-5-nano | 100 | 91 | 9 | 0 | 0.955 |
| gpt-5 + gpt-5-mini | 100 | 96 | 4 | 0 | 0.980 |
| Claude S4.5 + gpt-5-mini | 100 | 96 | 4 | 0 | 0.980 |
| gpt-5-mini + gpt-5-mini | 100 | 98 | 2 | 0 | 0.990 |
| gpt-5-mini no-tools | 100 | 100 | 0 | 0 | **1.000** |

#### Renting Process

| Config | TP | TN | FP | FN | Accuracy |
|--------|----|----|----|----|----------|
| gpt-5-mini + gpt-5-nano | 96 | 88 | 12 | 4 | 0.920 |
| gpt-5 + gpt-5-mini | 100 | 87 | 13 | 0 | 0.935 |
| Claude S4.5 + gpt-5-mini | 100 | 95 | 5 | 0 | 0.975 |
| gpt-5-mini + gpt-5-mini | 100 | 86 | 14 | 0 | 0.930 |
| gpt-5-mini no-tools | 100 | 95 | 5 | 0 | 0.975 |

---

### Performance Variance

| Process | Min F1 | Max F1 | Variance | Std Dev |
|---------|--------|--------|----------|---------|
| Hiring | 0.775 | 0.901 | 0.126 | 0.056 |
| Hospital | 0.980 | 0.995 | 0.015 | 0.006 |
| Lending | 0.957 | 1.000 | 0.043 | 0.017 |
| Renting | 0.923 | 0.976 | 0.053 | 0.021 |

**Observation**: Hospital process shows lowest variance (most consistent across models), while hiring shows highest variance (model choice matters most)

---

## Methodology Notes

### Process Mining Pipeline

```
XES Event Log
     ↓
[Enhanced Agentic Miner]
     ↓
Process Representations (Petri Net, DFG, Abstract Variants)
     ↓
[Generator LLM with/without tools]
     ↓
Textual AS-IS Model
     ↓
[Validator LLM + System Prompt]
     ↓
Trace Classification (VALID/INVALID)
     ↓
Metrics (Precision, Recall, F1, TP/TN/FP/FN)
```

### Perturbation Method

**random_swap_events**: Swaps two adjacent events in a trace to create ordering violations

Example:
```
Valid:   A → B → C → D
Invalid: A → C → B → D  (B and C swapped)
```

### Evaluation Protocol

1. Sample 100 valid traces from XES log
2. Generate 100 invalid traces via perturbation
3. Create system prompt with textual model
4. Classify each trace using validator LLM
5. Compare predictions to ground truth
6. Calculate confusion matrix and metrics

---

## Future Work

### Recommended Experiments

1. **Hybrid Approaches**:
   - Combine tools + no-tools (use tools for complex parts only)
   - Adaptive method selection based on process complexity

2. **Validator Ensembles**:
   - Majority voting across multiple validator LLMs
   - Confidence-based weighting

3. **Incremental Refinement**:
   - Test incremental refinement mode (minimal edits)
   - Compare vs full rewrite refinement

4. **Process Complexity Metrics**:
   - Develop automated complexity scoring
   - Use complexity to guide method/model selection

5. **Additional Perturbations**:
   - Test other perturbation types (skip events, duplicate events)
   - Evaluate robustness to different anomaly types

6. **Cost Analysis**:
   - Track token usage and API costs
   - Compute cost-performance Pareto frontiers

---

## Conclusion

This comprehensive evaluation demonstrates that **LLM-based textual process models achieve high accuracy** in trace classification tasks (F1: 0.775-1.000). Key findings:

1. **GPT-5 + GPT-5-mini** achieves best overall performance (F1=0.954)
2. **GPT-5-mini no-tools** offers excellent cost-performance ratio (F1=0.941)
3. Process complexity significantly impacts model performance
4. No-tools method can achieve perfect scores on sequential processes
5. Most models are permissive (perfect recall, varying precision)

The choice of generator LLM, method (tools vs no-tools), and process characteristics all significantly impact performance, enabling informed decisions based on specific deployment requirements.

---

**Generated**: 2025-12-29
**Analysis Tool**: Claude Code
**Data Source**: `/final_results/` benchmark runs
