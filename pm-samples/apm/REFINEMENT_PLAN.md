# Enhanced Refinement System for Process Mining Benchmarks

## Overview

This document describes enhancements to the benchmark refinement pipeline for LLM-based process trace validation. The goal is to create an iterative improvement loop where textual process models can be refined based on classification errors, tested efficiently, and improved recursively until satisfactory performance is achieved.

---

## Motivation & Problem Statement

### Current Workflow
1. Run `phase5_benchmark.py` to generate a textual AS-IS model from event logs
2. The model is used by an LLM validator to classify traces as VALID or INVALID
3. Some traces are misclassified (False Negatives and False Positives)
4. Run `refine_benchmark.py` to ask an LLM to improve the model based on errors

### Limitations of Current System

**1. No control over refinement LLM**
- Currently uses the same LLM that generated the original model
- Sometimes a more capable model (e.g., Claude Opus, GPT-4o) would produce better refinements
- Need ability to specify: `--refine-model anthropic:claude-opus-4-5-20251101`

**2. Wasteful full benchmark runs**
- After refinement, the system immediately runs a full benchmark (200 traces)
- This is expensive in time and tokens
- If the refinement didn't even fix the specific errors it was shown, the full benchmark is pointless
- Need a quick "sanity check" on just the misclassified traces first

**3. Traces are ephemeral**
- Valid and invalid traces are cached globally but not stored with the benchmark run
- If the cache is cleared, the exact traces used cannot be reproduced
- Refinements should be comparable on the SAME traces as the original
- Need traces stored in each run folder

**4. One-shot refinement**
- Can only refine the original model once
- Sometimes multiple rounds of refinement are needed
- Each refinement might fix some errors but introduce others
- Need ability to refine the refined model (recursively)

---

## What We Want to Achieve

### 1. Configurable Refinement LLM (`--refine-model`)
Allow specifying which LLM generates the refined textual model, independent of what generated the original.

**Use cases:**
- Use a more capable model for refinement: `--refine-model anthropic:claude-opus-4-5-20251101`
- Test if a cheaper model can refine effectively: `--refine-model openai:gpt-4o-mini`
- Compare refinement quality across models

### 2. Two-Stage Evaluation
Split the evaluation into two stages:

**Stage 1: Misclassified Traces Test (Quick Validation)**
- Test the refined model ONLY on the traces that were previously misclassified
- This is a focused sanity check: "Did the refinement fix what it was supposed to fix?"
- Fast and cheap (typically 10-50 traces instead of 200)
- Results: "X/Y previously misclassified traces are now correct"

**Stage 2: Full Benchmark (Conditional)**
- Only run if Stage 1 shows improvement (at least 1 trace fixed)
- Uses the same traces as the parent benchmark for fair comparison
- OR can use `--new-sampling` for fresh traces (generalization test)
- Can be skipped with `--skip-full-benchmark` to only do Stage 1
- Can be forced with `--force-full-benchmark` even if Stage 1 shows no improvement

**Why this matters:**
- Saves time and cost when refinement clearly didn't work
- Provides quick feedback loop for iterative refinement
- Separates "did it fix the specific errors?" from "did it generalize?"

### 3. Persistent Trace Storage
Store the actual traces used in each benchmark run, not just in a global cache.

**Structure:**
```
benchmark_runs/run_TIMESTAMP/
├── traces/
│   └── {problem}/
│       ├── valid_traces.pkl      # The 100 valid traces used
│       └── invalid_traces.pkl    # The 100 invalid traces used
```

**Benefits:**
- Reproducibility: Can re-run evaluation on exact same traces
- Comparison: Refinements are evaluated on same traces as original
- Portability: Run folder is self-contained, can be shared/archived
- Inheritance: Refinements can inherit parent's traces or generate new ones

### 4. Recursive Refinements
Allow refining a refined model, creating a chain of improvements.

**Folder structure:**
```
benchmark_runs/run_ORIGINAL/
├── textual_model.txt           # Original model
├── results.json                # F1=0.85
└── refined/                    # First refinement
    ├── textual_model.txt       # Refined model v1
    ├── results.json            # F1=0.90
    └── refined/                # Second refinement
        ├── textual_model.txt   # Refined model v2
        ├── results.json        # F1=0.93
        └── refined/            # Can continue...
```

**Use cases:**
- Iterative improvement until target F1 is reached
- Fix errors introduced by previous refinement
- Try different refinement strategies at different depths
- Track the evolution of model quality

**Command:**
```bash
# Refine the original
python refine_benchmark.py benchmark_runs/run_X

# Refine the first refinement
python refine_benchmark.py benchmark_runs/run_X/refined

# Refine the second refinement
python refine_benchmark.py benchmark_runs/run_X/refined/refined
```

---

## Implementation Plan

### 1. Add `--refine-model` CLI Argument

**File:** `refine_benchmark.py`

```python
parser.add_argument(
    "--refine-model",
    type=str,
    default=None,  # Falls back to config["llm_model"]
    help="LLM for refinement (default: same as parent benchmark)"
)
```

In `main()`:
```python
refine_model_spec = args.refine_model or config.get("llm_model", "openai:gpt-4o-mini")
llm = create_llm(refine_model_spec)
```

---

### 2. Two-Stage Evaluation

**New CLI arguments:**
```python
parser.add_argument(
    "--use-parent-traces",
    action="store_true",
    help="Reuse parent's traces instead of generating new ones (default: new sampling)"
)
parser.add_argument(
    "--skip-full-benchmark",
    action="store_true",
    help="Only run misclassified-traces test, skip full benchmark"
)
parser.add_argument(
    "--force-full-benchmark",
    action="store_true",
    help="Run full benchmark even if misclassified test doesn't improve"
)
```

**Logic:**
```python
# Stage 1: Test on misclassified traces only
misclassified_results = evaluate_on_misclassified(refined_model, misclassified_traces)

# Check improvement (STRICT: at least 1 trace must be fixed to continue)
improved = misclassified_results["fixed"] > 0

if args.skip_full_benchmark:
    return  # Done

if not improved and not args.force_full_benchmark:
    print("No improvement on misclassified traces - skipping full benchmark")
    print("Use --force-full-benchmark to run anyway")
    return

# Stage 2: Full benchmark (default: new sampling for generalization test)
# k_valid/k_invalid inherited from parent config
k_valid = config.get("k_valid", 100)
k_invalid = config.get("k_invalid", 100)

if args.use_parent_traces:
    valid_traces, invalid_traces = load_traces_from_run_folder(run_folder, problem_name)
else:
    # Default: generate new traces to test generalization
    valid_traces, invalid_traces = sample_new_traces(xes_path, k_valid, k_invalid)
```

---

### 3. Store Traces in Run Folder

**Files:** `phase5_benchmark.py` and `refine_benchmark.py`

**New folder structure:**
```
benchmark_runs/run_TIMESTAMP/
├── config.json
├── results.json
├── traces/                          # NEW: Stored traces
│   └── {problem}/
│       ├── valid_traces.pkl         # List of valid trace DataFrames
│       └── invalid_traces.pkl       # List of invalid trace DataFrames
├── {problem}_textual_model.txt
├── {problem}_system_prompt.txt
└── misclassified/{problem}/
```

**In `phase5_benchmark.py`:**
```python
def save_traces_to_run_folder(run_folder, problem_name, valid_traces, invalid_traces):
    traces_folder = run_folder / "traces" / problem_name
    traces_folder.mkdir(parents=True, exist_ok=True)

    with open(traces_folder / "valid_traces.pkl", "wb") as f:
        pickle.dump(valid_traces, f)
    with open(traces_folder / "invalid_traces.pkl", "wb") as f:
        pickle.dump(invalid_traces, f)

def load_traces_from_run_folder(run_folder, problem_name):
    traces_folder = run_folder / "traces" / problem_name
    valid_path = traces_folder / "valid_traces.pkl"
    invalid_path = traces_folder / "invalid_traces.pkl"

    if valid_path.exists() and invalid_path.exists():
        with open(valid_path, "rb") as f:
            valid_traces = pickle.load(f)
        with open(invalid_path, "rb") as f:
            invalid_traces = pickle.load(f)
        return valid_traces, invalid_traces
    return None, None
```

**Update results.json:**
```json
{
  "problem_name": {
    "traces": {
      "valid_traces_path": "traces/problem_name/valid_traces.pkl",
      "invalid_traces_path": "traces/problem_name/invalid_traces.pkl",
      "k_valid": 100,
      "k_invalid": 100,
      "source": "generated" | "inherited_from_parent"
    }
  }
}
```

---

### 4. Recursive Refinements

**Key changes:**

1. **Save config.json in refined folder:**
```python
refined_config = {
    "parent_run": str(run_folder),
    "refine_model": refine_model_spec,
    "validator_model": config.get("validator_model"),
    "k_valid": config.get("k_valid"),
    "k_invalid": config.get("k_invalid"),
    "trace_source": "inherited" | "new_sampling",
    "refinement_timestamp": datetime.now().isoformat()
}
(refined_folder / "config.json").write_text(json.dumps(refined_config, indent=2))
```

2. **Results.json structure matches parent:**
```python
refined_results = {
    "_meta": {
        "parent_run": str(run_folder),
        "refinement_depth": parent_depth + 1,
        ...
    },
    "problem_name": {
        "status": "completed",
        "traces": {...},           # Same structure as parent
        "llm": {...},              # Metrics
        "misclassified_traces": {...}
    }
}
```

3. **Copy/inherit traces:**
```python
if args.new_sampling:
    # Generate new traces
    valid_traces, invalid_traces = sample_and_generate(xes_path, k_valid, k_invalid)
    save_traces_to_run_folder(refined_folder, problem_name, valid_traces, invalid_traces)
else:
    # Inherit from parent
    valid_traces, invalid_traces = load_traces_from_run_folder(run_folder, problem_name)
    # Copy to refined folder for self-containment
    save_traces_to_run_folder(refined_folder, problem_name, valid_traces, invalid_traces)
```

---

## Files to Modify

| File | Changes |
|------|---------|
| `phase5_benchmark.py` | Add `save_traces_to_run_folder()`, call it after trace generation |
| `refine_benchmark.py` | Add `--refine-model`, `--use-parent-traces`, two-stage evaluation, recursive support |

---

## New CLI Interface

```bash
# Basic refinement (generates new traces by default for generalization test)
python refine_benchmark.py benchmark_runs/run_20251222_004259

# Specify refinement LLM
python refine_benchmark.py benchmark_runs/run_20251222_004259 --refine-model anthropic:claude-sonnet-4-20250514

# Reuse parent's traces (for controlled comparison)
python refine_benchmark.py benchmark_runs/run_20251222_004259 --use-parent-traces

# Only test on misclassified traces (quick validation)
python refine_benchmark.py benchmark_runs/run_20251222_004259 --skip-full-benchmark

# Force full benchmark even if no improvement
python refine_benchmark.py benchmark_runs/run_20251222_004259 --force-full-benchmark

# Refine a refined model (recursive)
python refine_benchmark.py benchmark_runs/run_20251222_004259/refined

# Multiple refinements from same parent (auto-increments: refined_1, refined_2, etc.)
python refine_benchmark.py benchmark_runs/run_X  # Creates refined/
python refine_benchmark.py benchmark_runs/run_X  # Creates refined_1/
python refine_benchmark.py benchmark_runs/run_X  # Creates refined_2/

# Custom output folder name
python refine_benchmark.py benchmark_runs/run_X --output refined_gpt4o
python refine_benchmark.py benchmark_runs/run_X -o refined_claude

# Combine options
python refine_benchmark.py benchmark_runs/run_X/refined --refine-model openai:gpt-4o --use-parent-traces
```

---

## Output Example

```
Benchmark run: benchmark_runs/run_20251222_004259
Refinement model: anthropic:claude-sonnet-4-20250514
Validator model: gpt-4o-mini

============================================================
Processing: hiring_log_high
============================================================
  Misclassified: 22 FN, 0 FP
  Loaded 22 misclassified traces

  --- Refinement ---
  Calling LLM for refinement...
  Refinement completed in 45.2s

  --- Stage 1: Misclassified Traces Test ---
  Testing refined model on 22 previously misclassified traces...
  Result: 18/22 now correct (82% improvement)

  --- Stage 2: Full Benchmark ---
  Generating new traces (100 valid + 100 invalid)
  ...

  hiring_log_high:
    Stage 1 (misclassified): 18/22 fixed (82%)
    Stage 2 (full): F1=0.95 (was 0.90) | FN=4 (was 22) | FP=2 (was 0)

Results saved to benchmark_runs/run_20251222_004259/refined/
```

---

## Implementation Status

**✓ IMPLEMENTED** (2025-12-27)

All features have been implemented:

- [x] `save_traces_to_run_folder()` and `load_traces_from_run_folder()` added to `phase5_benchmark.py`
- [x] `--refine-model` CLI argument added to `refine_benchmark.py`
- [x] `--new-sampling`, `--skip-full-benchmark`, `--force-full-benchmark` flags added
- [x] `load_misclassified_traces_for_eval()` and `evaluate_on_misclassified()` functions added
- [x] Two-stage evaluation logic implemented in `main()`
- [x] `config.json` saved in refined folder for recursive refinement
- [x] Trace loading updated to support run folder storage with cache fallback
- [x] Results structure compatible for recursive refinement (refined folders can be refined again)
- [x] Auto-incrementing folder names (`refined`, `refined_1`, `refined_2`, etc.)
- [x] Custom output folder via `--output` / `-o` flag
