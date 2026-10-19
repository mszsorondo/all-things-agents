# Final Ablation Study Implementation

## What Changed

✅ **Now supports ALL 11 representations from `ablation_study.py`** (as you requested!)

Instead of modifying `enhanced_agentic_miner.py`, I **reused the existing code** from `ablation_study.py`:
- Imports `RepresentationDiscoverer` and `AblationMiner`
- Uses ablation_study's discovery logic when `--one-rep` is active
- Keeps enhanced_agentic_miner for default mode (backward compatible)

## Available Representations

When you run `--one-rep` mode, you now get **11 representations**:

1. ✅ `petri_net_inductive` - Petri Net (Inductive Miner)
2. ✅ `petri_net_alpha` - Petri Net (Alpha Miner)
3. ✅ `petri_net_heuristics` - Petri Net (Heuristics Miner)
4. ✅ `petri_net_ilp` - Petri Net (ILP Miner)
5. ✅ `dfg` - Directly-Follows Graph
6. ✅ `heuristics_net` - Heuristics Net
7. ✅ `process_tree` - Process Tree
8. ✅ `abstract_variants` - Abstract Process Variants
9. ✅ `log_skeleton` - Log Skeleton
10. ✅ `declare` - DECLARE Constraints
11. ✅ `log_attributes` - Log Attributes

**Excluded**: `bpmn` (too many tokens), `prefix_tree` (no string format)

## Usage

### Full Ablation Study (11 representations)
```bash
python run_with_new_guides.py --one-rep openai:gpt-5-mini 5 5
```

**Output**: 11 separate benchmark runs in:
```
ablation_runs/ablation_study_<timestamp>/
├── petri_net_inductive_only/
├── petri_net_alpha_only/
├── petri_net_heuristics_only/
├── petri_net_ilp_only/
├── dfg_only/
├── heuristics_net_only/
├── process_tree_only/
├── abstract_variants_only/
├── log_skeleton_only/
├── declare_only/
├── log_attributes_only/
└── ablation_summary.json
```

### Single Representation
```bash
python run_with_new_guides.py --one-rep log_skeleton openai:gpt-5-mini 100 100
python run_with_new_guides.py --one-rep petri_net_alpha openai:gpt-5-mini 50 50
```

### Default Mode (Unchanged)
```bash
python run_with_new_guides.py openai:gpt-5-mini 100 100
```
Uses enhanced_agentic_miner with default 3 representations: petri_net, dfg, abstract_variants

## Backward Compatibility

If you use the old name `petri_net`, it automatically maps to `petri_net_inductive`:
```bash
python run_with_new_guides.py --one-rep petri_net openai:gpt-5-mini 50 50
# Automatically uses petri_net_inductive
```

## Important: Breakpoint Handling

`ablation_study.py` has a breakpoint at line 397 for verification. You'll hit it **44 times** (11 reps × 4 datasets).

**Solution**: Run with breakpoints disabled:
```bash
PYTHONBREAKPOINT=0 python run_with_new_guides.py --one-rep openai:gpt-5-mini 5 5
```

Or just press `c` (continue) each time it pauses.

See `ABLATION_BREAKPOINT_NOTE.md` for details.

## Code Architecture

```
run_with_new_guides.py
│
├── run_problem()                      # Uses enhanced_agentic_miner.py
│   └── For default mode (no --one-rep)
│   └── Uses: petri_net, dfg, abstract_variants
│
└── run_problem_ablation_style()       # Uses ablation_study.py
    └── For --one-rep mode (ablation study)
    └── Uses: RepresentationDiscoverer + AblationMiner
    └── Supports: All 11 representations
```

## Files Modified

1. ✅ `run_with_new_guides.py` - Added ablation_study integration
2. ✅ `ABLATION_MODE_USAGE.md` - Updated documentation
3. ✅ `ABLATION_BREAKPOINT_NOTE.md` - New guide for breakpoint handling
4. ✅ `FINAL_ABLATION_IMPLEMENTATION.md` - This file
5. ❌ `ablation_study.py` - **NOT MODIFIED** (as requested)

## Performance Estimate

Full ablation study:
- **11 representations** × **4 datasets** = **44 benchmark runs**
- With 5 traces each: ~30-45 minutes
- With 100 traces each: ~2-3 hours

## Comparison Table Example

After running the ablation study, you'll see:

```
ABLATION STUDY SUMMARY - ALL REPRESENTATIONS
================================================================================

hiring_log_high:
Representation            Precision    Recall       F1
-------------------------------------------------------------
petri_net_inductive      0.850        0.920        0.883
petri_net_alpha          0.790        0.870        0.828
petri_net_heuristics     0.845        0.905        0.874
petri_net_ilp            0.810        0.885        0.846
dfg                      0.780        0.850        0.813
heuristics_net           0.860        0.910        0.884
process_tree             0.820        0.900        0.858
abstract_variants        0.900        0.880        0.890
log_skeleton             0.875        0.895        0.885
declare                  0.830        0.870        0.849
log_attributes           0.795        0.860        0.826
```

## Test It Now!

```bash
# Quick test with 5 traces (disable breakpoint)
PYTHONBREAKPOINT=0 python run_with_new_guides.py --one-rep openai:gpt-5-mini 5 5

# Or test single representation
python run_with_new_guides.py --one-rep log_skeleton openai:gpt-5-mini 20 20
```

✅ **You now have full ablation study support with all 11 representations!**
