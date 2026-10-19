# dev_automation/ — Role-Based Textual Model Generation

## Overview

This module implements **role-decomposed process model generation**: instead of producing one monolithic textual model describing an entire process, the system generates one AS-IS textual model per logical role (participant) in the process. The concatenation of these role models is then evaluated with the same trace classification benchmark used for full models.

The goal is to demonstrate that process knowledge is **decomposable by role** without losing semantic fidelity — and that each isolated model can serve as a "job description" for an autonomous agent in a multi-agent system.

Suggested by Paulo (co-director) as a bridge between process mining discovery and multi-agent deployment.

## How It Works

### Pipeline

```
XES Event Log
  |
  ├─ pm4py: Discover process representations
  |    (Abstract Variants, DFG, Process Tree, Log Attributes,
  |     Petri Net, DECLARE, Log Skeleton)
  |
  ├─ pm4py: Discover organizational context
  |    - Resource → Activity mapping
  |    - Handover of Work network (resource-to-resource DFG)
  |    - Resource event frequencies
  |
  ├─ LLM Agent (single call, tools mode)
  |    1. Receives all representations + organizational context
  |    2. Clusters raw resources into logical roles
  |       (e.g., "Loan Officer 1-5" → "Loan Officer")
  |    3. Calls write_AS_IS_comprehensive_textual_description_for_role(role_name, content)
  |       once per detected role
  |
  ├─ Concatenate role models into single document
  |
  └─ Trace Classification Benchmark
       - Sample valid traces from log
       - Generate invalid traces via perturbation (random event swaps)
       - GPT-5-mini classifies each trace as VALID/INVALID using
         the concatenated model as context
       - Compute Precision, Recall, F1
```

### Key Design Decision: Role Clustering

Raw event logs often have many individual resources (e.g., "Loan Officer 1" through "Loan Officer 5") that represent the same logical role. The LLM agent clusters these into logical roles based on:
- Activity patterns (which activities each resource performs)
- Naming conventions
- Handover of work relationships

This avoids generating redundant models for individual resource instances.

> **Alternative architecture (recommended for production):** Instead of a single agent call that generates all role models at once, use a two-step approach: (1) one LLM call to identify roles, (2) one LLM call per role with focused instructions. This avoids context degradation with many roles. See the comment at line ~170 in `generate_with_roles.py`.

## Scripts

| File | Purpose |
|---|---|
| `generate_with_roles.py` | Main script: role detection, per-role model generation, concatenated benchmark |
| `baseline_full_model.py` | Baseline: generates a single full textual model with the same representations, for fair comparison |

Both scripts import from the parent codebase (`ablation_study.py`) without modifying it.

## Usage

```bash
# Role-based generation + benchmark (single dataset)
python dev_automation/generate_with_roles.py --dataset hiring

# With specific models
python dev_automation/generate_with_roles.py --dataset hiring \
  --model openai:gpt-5-mini --validator-model gpt-5-mini

# Also evaluate each role model in isolation
python dev_automation/generate_with_roles.py --dataset hiring --evaluate-isolated

# All four benchmark datasets
python dev_automation/generate_with_roles.py --all

# Baseline for comparison (same representations, single full model)
python dev_automation/baseline_full_model.py --dataset hiring

# Smaller test run
python dev_automation/generate_with_roles.py --dataset hiring --k-valid 30 --k-invalid 30
```

## Output Structure

```
dev_automation/
├── role_benchmark_runs/
│   └── run_{dataset}_{timestamp}/
│       ├── results.json                    # Full results with metrics
│       ├── concatenated_textual_model.txt  # All role models joined
│       ├── role_generation/
│       │   ├── role_{name}.txt             # Individual role model
│       │   ├── ...                         # One file per detected role
│       │   ├── metadata.json               # Roles detected, timing, resources
│       │   ├── system_prompt.txt           # Agent system prompt
│       │   ├── user_prompt.txt             # Full prompt sent to agent
│       │   └── agent_logs.txt              # Agent execution log
│       └── traces/
│           ├── valid_traces.pkl
│           └── invalid_traces.pkl
│
├── baseline_runs/
│   └── run_{dataset}_{timestamp}/
│       ├── results.json
│       ├── full_model/
│       │   └── textual_model.txt           # Single full model
│       └── traces/
│           ├── valid_traces.pkl
│           └── invalid_traces.pkl
```

## Experiment Results

All experiments run with **gpt-5-mini** for both textual model generation and trace validation, using **100 valid + 100 invalid traces** per dataset.

### Summary — All 4 Datasets

| Dataset | Full Model F1 | Role-Based F1 | Delta | Roles Detected | Resources → Roles |
|---|---|---|---|---|---|
| **Hiring** | 0.769 | 0.766 | -0.003 | 6 | 8 → 6 |
| **Hospital** | 0.990 | 0.947 | -0.043 | 2 | 7 → 2 |
| **Lending** | 0.990 | 0.990 | 0.000 | 4 | 19 → 4 |
| **Renting** | 0.976 | 0.935 | -0.041 | 5 | 15 → 5 |
| **Average** | **0.931** | **0.910** | **-0.021** | | |

All runs maintain **perfect recall** (1.000) for both full and role-based models — the only variation is in precision (false positives).

### Per-Dataset Details

#### Hiring (8 resources → 6 roles)

| Role | Resources Clustered |
|---|---|
| Application Intake | Front Desk, Online System |
| HR Screener | HR-dummy, HR-pro, Senior Partner |
| Background Check Specialist | HR-pro, HR-dummy |
| Technical Interviewer | Coding Interviewer |
| Hiring Decision Maker | HR-pro and Senior Partner |
| External Background Vendor | deepResearch Inc. |

| | Full Model | Role-Based |
|---|---|---|
| F1 | 0.769 | 0.766 |
| Precision | 0.625 | 0.621 |
| TP=100 FP=60 TN=40 FN=0 | TP=100 FP=61 TN=39 FN=0 |

Near-identical performance. The 1 FP difference is within statistical noise. Hiring is the hardest dataset (high variability, parallel interviews), yet role decomposition loses almost nothing.

#### Hospital (7 resources → 2 roles)

| Role | Resources Clustered |
|---|---|
| Doctor | Doc. Pro, Doc. Low |
| Nurse | Nurse 1-5 |

| | Full Model | Role-Based |
|---|---|---|
| F1 | 0.990 | 0.947 |
| Precision | 0.980 | 0.908 |
| TP=100 FP=2 TN=98 FN=0 | TP=99 FP=10 TN=90 FN=1 |

Largest gap (-4.3% F1). Hospital has a very linear process where activities interleave tightly between doctors and nurses. Splitting by role may lose some of the strict sequential constraints that span across roles (e.g., "nurse does X, then doctor does Y, then nurse does Z"). This is the only dataset where the role-based model produced a false negative.

#### Lending (19 resources → 4 roles)

| Role | Resources Clustered |
|---|---|
| Appointment Agent | Hotline, Resource 1-5 |
| Online System | Online System |
| Loan Officer | Loan Officer 1-5 |
| Underwriter | Chill UW1-3, Strict UW1-3, MostChillUW |

| | Full Model | Role-Based |
|---|---|---|
| F1 | 0.990 | 0.990 |
| Precision | 0.980 | 0.980 |
| TP=100 FP=2 TN=98 FN=0 | TP=100 FP=2 TN=98 FN=0 |

Identical results. Lending has clear sequential decision points and well-separated role responsibilities. The clustering is excellent — 19 individual resources compressed to 4 logical roles with zero information loss. This is the strongest evidence for decomposability.

#### Renting (15 resources → 5 roles)

| Role | Resources Clustered |
|---|---|
| Real Estate Agent | Real Estate Agent 1-5 |
| Hotline | Hotline |
| Tenant | Tenant |
| Lawyer | Lawyer 1-3 |
| Landlord | Landlord 1-5 |

| | Full Model | Role-Based |
|---|---|---|
| F1 | 0.976 | 0.935 |
| Precision | 0.952 | 0.877 |
| TP=100 FP=5 TN=95 FN=0 | TP=100 FP=14 TN=86 FN=0 |

Moderate gap (-4.1% F1). Renting has multiple termination paths and interactions between several roles. The role-based model is more permissive (14 FP vs 5), suggesting that some cross-role ordering constraints are weakened when described from isolated perspectives.

### Observations

1. **Knowledge is decomposable.** Across all 4 datasets, the average F1 drop is only 2.1%. The role-based decomposition preserves the vast majority of process semantics.

2. **Perfect recall is preserved.** All role-based models maintain 1.000 recall (except hospital with one FN). The decomposition does not cause the model to reject valid traces — it only occasionally accepts invalid ones.

3. **Degradation correlates with cross-role coupling.** The two datasets with the largest gaps (hospital, renting) have tight interleaving between roles. When activities strictly alternate between participants, splitting by role makes it harder to express those sequential constraints. Conversely, lending (clear handoff boundaries) and hiring (parallel independent interviews) decompose cleanly.

4. **Role clustering works well.** The LLM successfully compressed 8-19 raw resources into 2-6 logical roles across all datasets, grouping instances of the same role type (e.g., "Loan Officer 1-5" → "Loan Officer").

5. **Each role model is independently meaningful.** Each file describes a coherent "job description" — what the role does, when it activates, who it interacts with, what decisions it makes. These are directly usable as system prompts for autonomous agents.

6. **Direct path to multi-agent deployment.** Each role model can serve as the system prompt for an autonomous agent that handles that role's responsibilities. The handover of work network provides the interaction protocol between agents.

## Dependencies

Same as the parent codebase:
- pm4py >= 2.7.0
- langchain, langchain_openai, langgraph
- API keys in `.env`: `OPENAI_API_KEY` (and optionally `XAI_API_KEY`, `DEEPSEEK_API_KEY`, `ANTHROPIC_API_KEY`)

Does **not** depend on `deepagents` or `enhanced_agentic_miner.py`.
