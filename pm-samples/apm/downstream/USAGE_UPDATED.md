# Updated Benchmark Usage Guide

The benchmark script now supports selecting specific audiences, controlling test order, and choosing when to test the baseline.

## New Features

1. **Select Audiences**: Test only specific audiences (in the order you specify)
2. **Control Baseline Timing**: Test baseline first (default), last, or skip entirely
3. **Pre-adapted Models**: Optionally load already-adapted models to skip adaptation

---

## Usage Examples

### 1. Test Only Executives (Baseline Last)

```bash
python downstream/benchmark.py \
  --xes-path benchmark/hiring/hiring_log_high.xes \
  --textual-model-path benchmark_runs/run_20251222_004259/hiring_log_high_textual_model.txt \
  --audiences executives \
  --baseline-last \
  --k-valid 50 \
  --k-invalid 50
```

**What happens:**
1. Adapts model for executives using GPT-5
2. Tests adapted executives model
3. Tests original model (baseline) last

---

### 2. Test Multiple Audiences (Custom Order, Skip Baseline)

```bash
python downstream/benchmark.py \
  --xes-path benchmark/hiring/hiring_log_high.xes \
  --textual-model-path benchmark_runs/run_20251222_004259/hiring_log_high_textual_model.txt \
  --audiences executives developers auditors \
  --skip-baseline \
  --k-valid 100 \
  --k-invalid 100
```

**What happens:**
1. Adapts for executives, developers, auditors (in that order)
2. Tests each adapted model
3. Skips testing original model entirely
4. Report shows F1 scores without baseline comparison

---

### 3. Test All Audiences (Baseline First - Default Behavior)

```bash
python downstream/benchmark.py \
  --xes-path benchmark/hiring/hiring_log_high.xes \
  --textual-model-path benchmark_runs/run_20251222_004259/hiring_log_high_textual_model.txt \
  --k-valid 100 \
  --k-invalid 100
```

**What happens:**
1. Tests original model first (baseline)
2. Adapts all 5 audiences
3. Tests each adapted model
4. Compares all against baseline

---

### 4. Use Pre-Adapted Models (Side Feature)

If you already adapted models separately:

```bash
# First, adapt models manually
python downstream/adapters.py \
  --model-path benchmark_runs/run_20251222_004259/hiring_log_high_textual_model.txt \
  --audience executives \
  --output-path /tmp/my_adaptations/adapted_executives.txt

# Then benchmark using the pre-adapted model
python downstream/benchmark.py \
  --xes-path benchmark/hiring/hiring_log_high.xes \
  --textual-model-path benchmark_runs/run_20251222_004259/hiring_log_high_textual_model.txt \
  --adapted-models-dir /tmp/my_adaptations \
  --audiences executives \
  --k-valid 50 \
  --k-invalid 50
```

**What happens:**
1. Loads pre-adapted model from `/tmp/my_adaptations/adapted_executives.txt`
2. Skips adaptation step (saves time and API costs)
3. Tests the pre-adapted model
4. Tests baseline (unless `--skip-baseline` or `--baseline-last`)

---

## All Command-Line Arguments

```
Required:
  --xes-path PATH               Path to XES event log file
  --textual-model-path PATH     Path to original textual model

Optional:
  --adapter-llm MODEL          LLM for adaptation (default: openai:gpt-5)
  --validator-llm MODEL        LLM for validation (default: gpt-5-mini)
  --k-valid N                  Number of valid traces (default: 100)
  --k-invalid N                Number of invalid traces (default: 100)

  --audiences A1 A2 A3         Specific audiences to test (default: all 5)
                               Choices: executives, process_analysts,
                                       developers, auditors, children
                               Order matters!

  --skip-baseline              Skip testing original model entirely
  --baseline-last              Test original model after all audiences

  --adapted-models-dir DIR     Load pre-adapted models from directory
                               (skips adaptation step)
```

---

## Common Workflows

### Workflow 1: Quick Executive Test

Test only the executives adaptation quickly:

```bash
python downstream/benchmark.py \
  --xes-path benchmark/hiring/hiring_log_high.xes \
  --textual-model-path benchmark_runs/run_20251222_004259/hiring_log_high_textual_model.txt \
  --audiences executives \
  --k-valid 10 \
  --k-invalid 10
```

### Workflow 2: Compare Three Audiences

```bash
python downstream/benchmark.py \
  --xes-path benchmark/hiring/hiring_log_high.xes \
  --textual-model-path benchmark_runs/run_20251222_004259/hiring_log_high_textual_model.txt \
  --audiences executives process_analysts developers \
  --baseline-last \
  --k-valid 50 \
  --k-invalid 50
```

###Workflow 3: Test Adapted Model Quality (No Baseline Needed)

If you just want to see F1 scores for adaptations without comparing to baseline:

```bash
python downstream/benchmark.py \
  --xes-path benchmark/hiring/hiring_log_high.xes \
  --textual-model-path benchmark_runs/run_20251222_004259/hiring_log_high_textual_model.txt \
  --audiences executives \
  --skip-baseline \
  --k-valid 100 \
  --k-invalid 100
```

### Workflow 4: Iterate on Prompts

When refining adapter prompts:

```bash
# Edit adapter_prompts/executives.txt

# Test the refined prompt
python downstream/benchmark.py \
  --xes-path benchmark/hiring/hiring_log_high.xes \
  --textual-model-path benchmark_runs/run_20251222_004259/hiring_log_high_textual_model.txt \
  --audiences executives \
  --baseline-last \
  --k-valid 20 \
  --k-invalid 20
```

---

## Output Differences

### With Baseline (Default)

Report shows:
- Baseline F1 score
- Δ F1 for each audience (vs baseline)
- Ranking by F1 preservation

### Without Baseline (`--skip-baseline`)

Report shows:
- No baseline row in table
- "N/A" in Δ F1 column
- Ranking by absolute F1 score (highest first)

### Baseline Last (`--baseline-last`)

Same output as default, but baseline tested after all adaptations (useful for seeing adapted results first in the console output).

---

## Tips

1. **Use `--baseline-last`** when you want to see adapted model results immediately without waiting for baseline
2. **Use `--skip-baseline`** when iterating on prompts and you don't need comparison
3. **Use `--audiences`** to focus on specific audiences and save time/money
4. **Order matters** for `--audiences` - they're tested in the order you specify
5. **Start with small k** (10-20) for quick tests, then scale to 100+ for production

---

## Answer to Your Question

To test the refined executives model AFTER the original model:

```bash
python downstream/benchmark.py \
  --xes-path benchmark/hiring/hiring_log_high.xes \
  --textual-model-path benchmark_runs/run_20251222_004259/hiring_log_high_textual_model.txt \
  --audiences executives \
  --baseline-last \
  --k-valid 100 \
  --k-invalid 100
```

This will:
1. Adapt the model for executives using GPT-5
2. Test the executives adaptation
3. Test the original model last

The report will show both results with Δ F1 comparison.
