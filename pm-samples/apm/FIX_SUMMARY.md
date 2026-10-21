# Fix Summary: Ablation Study Mode

## Problem
The initial implementation tried to use 11 representations from `ablation_study.py`, but these don't exist in `enhanced_agentic_miner.py`:

```
ValueError: 'petri_net_inductive' is not a valid RepresentationType
```

## Root Cause
Two different representation systems exist in the codebase:

### 1. `enhanced_agentic_miner.py` (lines 37-44)
```python
class RepresentationType(Enum):
    PETRI_NET = "petri_net"
    BPMN = "bpmn"
    DFG = "dfg"
    HEURISTICS_NET = "heuristics_net"
    PROCESS_TREE = "process_tree"
    ABSTRACT_VARIANTS = "abstract_variants"
```
**Total: 6 representations** (BPMN excluded from ablation)

### 2. `ablation_study.py` (lines 45-62)
```python
class RepresentationType(Enum):
    PETRI_NET_INDUCTIVE = "petri_net_inductive"
    BPMN = "bpmn"
    DFG = "dfg"
    HEURISTICS_NET = "heuristics_net"
    PROCESS_TREE = "process_tree"
    ABSTRACT_VARIANTS = "abstract_variants"
    PETRI_NET_ALPHA = "petri_net_alpha"
    PETRI_NET_HEURISTICS = "petri_net_heuristics"
    PETRI_NET_ILP = "petri_net_ilp"
    LOG_SKELETON = "log_skeleton"
    DECLARE = "declare"
    LOG_ATTRIBUTES = "log_attributes"
    PREFIX_TREE = "prefix_tree"
```
**Total: 13 representations** (with its own discovery logic)

## Solution
Since `run_with_new_guides.py` uses `enhanced_agentic_miner.py`, the ablation study is limited to the **5 representations** supported there:

1. ✅ `petri_net`
2. ✅ `dfg`
3. ✅ `heuristics_net`
4. ✅ `process_tree`
5. ✅ `abstract_variants`

BPMN is excluded from ablation due to excessive token usage.

## Changes Made

### 1. Updated `run_with_new_guides.py`
- Changed `valid_reps` list from 11 representations to 5 (lines 155-162)
- Updated docstring to list correct representations (lines 22-28)
- Updated usage examples (lines 11-19)

### 2. Updated `ABLATION_MODE_USAGE.md`
- Corrected all references from 11 to 5 representations
- Added clarification about relationship with `ablation_study.py`
- Updated performance estimates (20 runs instead of 44)
- Fixed example outputs

## Current Usage

### ✅ Correct: Ablation study with 5 representations
```bash
python run_with_new_guides.py --one-rep openai:gpt-5-mini 5 5
```

**Output structure**:
```
ablation_runs/ablation_study_<timestamp>/
├── petri_net_only/
├── dfg_only/
├── heuristics_net_only/
├── process_tree_only/
├── abstract_variants_only/
└── ablation_summary.json
```

### ✅ Correct: Single representation
```bash
python run_with_new_guides.py --one-rep petri_net openai:gpt-5-mini 100 100
```

### ✅ Correct: Default mode (3 representations)
```bash
python run_with_new_guides.py openai:gpt-5-mini 100 100
```

Uses: `petri_net`, `dfg`, `abstract_variants` (default from `enhanced_agentic_miner.py`)

## For More Representations

If you need all 13 representations including multiple Petri Net variants:
```bash
python ablation_study.py --dataset hiring --k-valid 50 --k-invalid 50
```

This uses its own discovery logic and supports:
- Multiple Petri Net algorithms (Inductive, Alpha, Heuristics, ILP)
- Log Skeleton
- DECLARE constraints
- Log Attributes
- And more...

## Recommendation

**For most users**: Use `run_with_new_guides.py --one-rep` (5 representations, simpler, faster)

**For comprehensive research**: Use `ablation_study.py` (13 representations, more detailed)
