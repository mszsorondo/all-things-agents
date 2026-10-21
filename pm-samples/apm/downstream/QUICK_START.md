# Quick Start Guide

## Setup (30 seconds)

```bash
# 1. Set your OpenAI API key
export OPENAI_API_KEY="your-key-here"

# 2. Navigate to project directory
cd /home/marco/Escritorio/austral/all-things-agents/pm-samples
```

## Test Single Adaptation (2 minutes)

Adapt a model for executives:

```bash
python downstream/adapters.py \
  --model-path benchmark_runs/run_20251231_005033/hiring_log_high_textual_model.txt \
  --audience executives \
  --output-path /tmp/adapted_executives.txt

# View the result
less /tmp/adapted_executives.txt
```

Try other audiences:
- `executives` - C-level decision makers
- `process_analysts` - Business process analysts
- `developers` - Software engineers
- `auditors` - Compliance officers
- `children` - Educational/simple language

## Run Small Benchmark (5 minutes)

Test with 10 traces per audience:

```bash
python downstream/benchmark.py \
  --xes-path benchmark/hiring/hiring_log_high.xes \
  --textual-model-path benchmark_runs/run_20251231_005033/hiring_log_high_textual_model.txt \
  --k-valid 10 \
  --k-invalid 10
```

Check results:
```bash
cd downstream/adapters_benchmark_results/run_*/
cat comparison_report.md
```

## Run Production Benchmark (20-30 minutes)

Full benchmark with 100 traces per audience:

```bash
python downstream/benchmark.py \
  --xes-path benchmark/hiring/hiring_log_high.xes \
  --textual-model-path benchmark_runs/run_20251231_005033/hiring_log_high_textual_model.txt \
  --k-valid 100 \
  --k-invalid 100
```

## Understanding Results

The benchmark creates a directory with:

- `comparison_report.md` - Read this first! Shows F1 scores and rankings
- `adapted_*.txt` - 5 adapted models (one per audience)
- `results.json` - All metrics in JSON format
- `misclassified/` - Traces that were misclassified (for debugging)

**Key Metric:** Δ F1 (delta from baseline)
- Close to 0 = Good semantic preservation
- Negative = Some information loss
- Positive = Better than original (rare but possible)

## Troubleshooting

**"OPENAI_API_KEY not set"**
```bash
export OPENAI_API_KEY="sk-..."
```

**"Model file not found"**
```bash
# Find available models
ls benchmark_runs/*/hiring_log_high_textual_model.txt
```

**Want to use different LLM?**
```bash
# Use Claude for adaptation
export ANTHROPIC_API_KEY="sk-ant-..."
python downstream/adapters.py \
  --llm-model anthropic:claude-3-5-sonnet-20241022 \
  ...
```

## Next Steps

1. ✅ Test single adaptation (confirm API works)
2. ✅ Review adapted outputs (check quality)
3. ✅ Run small benchmark (verify end-to-end)
4. ✅ Review comparison report (assess preservation)
5. ✅ Run production benchmark (get real results)
6. 🔧 Refine prompts if needed (edit `adapter_prompts/*.txt`)

## Full Documentation

See `downstream/README.md` for complete documentation.
