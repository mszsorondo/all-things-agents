# Ablation Study Mode - Usage Guide

## Overview

The `run_with_new_guides.py` script now supports three distinct modes for representation testing.

## Three Modes of Operation

### 1. **Default Mode** (No `--one-rep` flag)
Uses the default 3 representations together in a single benchmark run.

```bash
# Uses: petri_net, dfg, abstract_variants (default)
python run_with_new_guides.py openai:gpt-5-mini 100 100
```

**Output**: Single benchmark run with combined representations
**Folder**: `benchmark_runs/new_guides_run_<timestamp>/`

---

### 2. **Single Representation Mode** (`--one-rep <name>`)
Runs benchmark with only ONE specific representation.

```bash
# Test only DFG
python run_with_new_guides.py --one-rep dfg openai:gpt-5-mini 100 100

# Test only Petri Net (Inductive)
python run_with_new_guides.py --one-rep petri_net_inductive
```

**Output**: Single benchmark run with one representation
**Folder**: `benchmark_runs/new_guides_run_<timestamp>/`

---

### 3. **Ablation Study Mode** (`--one-rep` without name) ⭐ **NEW**
Automatically runs 5 separate benchmarks, one for EACH representation.

```bash
# Run ablation study with all 5 representations
python run_with_new_guides.py --one-rep

# With custom parameters
python run_with_new_guides.py --one-rep openai:gpt-5-mini 50 50
```

**Output**: 5 separate benchmark runs
**Folder Structure**:
```
ablation_runs/ablation_study_<timestamp>/
├── petri_net_only/
│   └── results.json
├── dfg_only/
│   └── results.json
├── heuristics_net_only/
│   └── results.json
├── process_tree_only/
│   └── results.json
├── abstract_variants_only/
│   └── results.json
└── ablation_summary.json  ⭐ Comparison table
```

---

## Available Representations (for modes 2 & 3)

The following 5 representations are used in ablation study mode (supported by `enhanced_agentic_miner.py`):

1. **petri_net** - Petri Net (Inductive Miner)
2. **dfg** - Directly-Follows Graph
3. **heuristics_net** - Heuristics Net
4. **process_tree** - Process Tree
5. **abstract_variants** - Abstract Process Variants

**Note**: `bpmn` is excluded from ablation study due to excessive token usage.

**For more representations**: Use `ablation_study.py` which supports 13 different representations including multiple Petri Net variants (alpha, heuristics, ILP), log skeleton, DECLARE constraints, and log attributes.

---

## Ablation Study Output

### ablation_summary.json
Contains aggregated results for comparison:

```json
{
  "petri_net_inductive": {
    "results": { ... },
    "run_folder": "ablation_runs/.../petri_net_inductive_only"
  },
  "dfg": {
    "results": { ... },
    "run_folder": "ablation_runs/.../dfg_only"
  },
  ...
}
```

### Console Output
After all runs complete, a comparison table is printed:

```
ABLATION STUDY SUMMARY - ALL REPRESENTATIONS
================================================================================

hiring_log_high:
Representation            Precision    Recall       F1
-------------------------------------------------------------
petri_net                0.850        0.920        0.883
dfg                      0.780        0.850        0.813
heuristics_net           0.860        0.910        0.884
process_tree             0.820        0.900        0.858
abstract_variants        0.900        0.880        0.890

hospital_log_high:
...
```

---

## Use Cases

### Research: Which representation is most valuable?
```bash
# Run full ablation study to identify best single representation
python run_with_new_guides.py --one-rep openai:gpt-5-mini 100 100
```

### Development: Test specific representation quickly
```bash
# Test only DFG (fastest)
python run_with_new_guides.py --one-rep dfg openai:gpt-5-mini 20 20
```

### Production: Use default optimized set
```bash
# Use the 3 best representations (fastest + good quality)
python run_with_new_guides.py openai:gpt-5-mini 100 100
```

---

## Relationship with ablation_study.py

**Key Difference**:
- `run_with_new_guides.py`: Uses `enhanced_agentic_miner.py` → **5 representations**
- `ablation_study.py`: Has its own discovery logic → **13 representations**

**When to use which**:
- Use `run_with_new_guides.py --one-rep` for quick ablation studies with the standard miner
- Use `ablation_study.py` for comprehensive ablation with all PM4Py representation variants

Both use the same interpretation guides directory: `representation_interpretation_guides/`

---

## Performance Notes

**Ablation Study Runtime**:
- 5 representations × 4 datasets = 20 benchmark runs
- Estimated time: ~1-2 hours (depends on LLM and trace counts)
- Can be interrupted and resumed (each representation saves results)

**Recommendations**:
- For quick tests: Use smaller trace counts (20-50)
- For research: Use standard counts (100-100)
- For production decisions: Use larger counts (200-200)

---

## Examples

```bash
# Quick ablation study (20 traces each)
python run_with_new_guides.py --one-rep openai:gpt-5-mini 20 20

# Standard ablation study
python run_with_new_guides.py --one-rep openai:gpt-5-mini 100 100

# Comprehensive ablation study
python run_with_new_guides.py --one-rep anthropic:claude-sonnet-4-5 200 200

# Test single representation
python run_with_new_guides.py --one-rep process_tree openai:gpt-5-mini 50 50

# Use default mode (backward compatible)
python run_with_new_guides.py openai:gpt-5-mini 100 100
```
