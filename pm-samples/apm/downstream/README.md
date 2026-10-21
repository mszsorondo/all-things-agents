# Adaptive Process Model Explanation System

Transform technical process mining models into audience-specific explanations while preserving complete semantic fidelity for trace validation.

## Overview

This system addresses a key challenge in process mining: technical models (Petri nets, DFGs) are difficult for non-experts to understand. The solution adapts these models to different audiences (executives, analysts, developers, auditors, children) using LLMs while maintaining **zero information loss** - all behavioral information needed for trace validation is preserved.

## Architecture

### Components

1. **`adapters.py`** - Core adaptation engine
   - Transforms textual models using audience-specific prompts
   - Supports multiple LLM providers (OpenAI, Anthropic, xAI, DeepSeek)
   - Uses GPT-5 by default for high-quality transformations

2. **`adapter_prompts/`** - Audience-specific instructions
   - 5 carefully crafted prompts emphasizing zero information loss
   - Each prompt provides transformation rules without allowing deletions
   - Maintains complete process behavior in adapted language

3. **`benchmark.py`** - Validation system
   - Tests semantic fidelity via trace classification
   - Compares adapted models against original baseline
   - Uses GPT-5-mini for cost-effective validation
   - Always completes all benchmarks regardless of performance

### Supported Audiences

- **executives** - C-level decision makers (strategic insights, business impact)
- **process_analysts** - Business process analysts (BPMN terminology, variants)
- **developers** - Software engineers (state machines, API specs, pseudocode)
- **auditors** - Compliance officers (controls, risks, authorization chains)
- **children** - Educational/simple language (storytelling, concrete examples)

## Installation

Ensure dependencies are installed:

```bash
pip install langchain-openai langchain-anthropic pm4py
```

Set up API keys:

```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"  # Optional
export XAI_API_KEY="your-key"        # Optional
export DEEPSEEK_API_KEY="your-key"   # Optional
```

## Usage

### Adapt a Single Model

Transform a textual model for a specific audience:

```bash
python downstream/adapters.py \
  --model-path benchmark_runs/run_X/hiring_log_high_textual_model.txt \
  --audience executives \
  --llm-model openai:gpt-5 \
  --output-path adapted_executives.txt
```

**Arguments:**
- `--model-path` - Path to original textual model
- `--audience` - Target audience (executives|process_analysts|developers|auditors|children)
- `--llm-model` - LLM to use (format: `provider:model`)
- `--output-path` - Where to save adapted model

**Example with different LLMs:**

```bash
# Use Claude for adaptation
python downstream/adapters.py \
  --model-path mymodel.txt \
  --audience developers \
  --llm-model anthropic:claude-3-5-sonnet-20241022 \
  --output-path adapted_dev.txt

# Use xAI Grok
python downstream/adapters.py \
  --model-path mymodel.txt \
  --audience process_analysts \
  --llm-model xai:grok-2-latest \
  --output-path adapted_analysts.txt
```

### Adapt for All Audiences

Batch adaptation using shell loop:

```bash
MODEL="benchmark_runs/run_20251231_005033/hiring_log_high_textual_model.txt"

for audience in executives process_analysts developers auditors children; do
  python downstream/adapters.py \
    --model-path "$MODEL" \
    --audience "$audience" \
    --llm-model openai:gpt-5 \
    --output-path "adapted_${audience}.txt"
done
```

### Run Complete Benchmark

Validate that adaptations preserve semantic fidelity:

```bash
python downstream/benchmark.py \
  --xes-path benchmark/hiring/hiring_log_high.xes \
  --textual-model-path benchmark_runs/run_X/hiring_log_high_textual_model.txt \
  --adapter-llm openai:gpt-5 \
  --validator-llm gpt-5-mini \
  --k-valid 100 \
  --k-invalid 100
```

**Arguments:**
- `--xes-path` - Path to XES event log (for generating test traces)
- `--textual-model-path` - Path to original textual model
- `--adapter-llm` - LLM for model adaptation (default: `openai:gpt-5`)
- `--validator-llm` - LLM for trace classification (default: `gpt-5-mini`)
- `--k-valid` - Number of valid traces to test (default: 100)
- `--k-invalid` - Number of invalid traces to generate (default: 100)

**What the benchmark does:**

1. **Baseline**: Validates original model against test traces → F1_baseline
2. **Adaptation**: Adapts model for all 5 audiences using GPT-5
3. **Validation**: Tests each adapted model using GPT-5-mini
4. **Comparison**: Calculates F1 scores and deltas vs baseline
5. **Reporting**: Generates comprehensive markdown report
6. **Completion**: Runs all benchmarks regardless of performance

**Note:** The benchmark ALWAYS completes all 5 audiences. Poor performance on one audience does not interrupt the process.

## Output Structure

Benchmark creates timestamped results directory:

```
downstream/adapters_benchmark_results/run_YYYYMMDD_HHMMSS/
├── config.json                      # Benchmark configuration
├── original_textual_model.txt       # Baseline model
├── adapted_executives.txt           # 5 adapted versions
├── adapted_process_analysts.txt
├── adapted_developers.txt
├── adapted_auditors.txt
├── adapted_children.txt
├── results.json                     # Complete metrics data
├── comparison_report.md             # Human-readable summary
├── traces/
│   ├── valid_traces.pkl             # Test traces
│   └── invalid_traces.pkl
└── misclassified/                   # Misclassified traces by audience
    ├── original/
    ├── executives/
    ├── process_analysts/
    ├── developers/
    ├── auditors/
    └── children/
```

## Interpreting Results

### Comparison Report

The `comparison_report.md` contains:

1. **Configuration** - Benchmark settings
2. **Executive Summary** - Baseline performance overview
3. **Performance Table** - F1 scores and deltas for all audiences
4. **Ranking** - Audiences sorted by F1 preservation
5. **Detailed Results** - Per-audience metrics and misclassifications
6. **Observations** - Notable findings and recommendations

### Key Metrics

- **F1 Score** - Harmonic mean of precision and recall (0-1, higher better)
- **Δ F1** - Change from baseline (close to 0 means good preservation)
- **Precision** - Correct validations / total validated (avoid false positives)
- **Recall** - Correct validations / total valid (avoid false negatives)

### Success Criteria

- ✅ **Perfect Preservation**: Δ F1 ≈ 0 (within ±0.05)
- ⚠️ **Minor Degradation**: Δ F1 between -0.05 and -0.10
- ❌ **Significant Loss**: Δ F1 < -0.10 (consider refining prompt)

## Example Workflow

Complete workflow from model to validated adaptations:

```bash
# 1. Set up environment
export OPENAI_API_KEY="your-key"
cd downstream/

# 2. Test single adaptation
python adapters.py \
  --model-path ../benchmark_runs/run_20251231_005033/hiring_log_high_textual_model.txt \
  --audience executives \
  --output-path /tmp/test_adapted.txt

# 3. Review adapted model
less /tmp/test_adapted.txt

# 4. Run full benchmark (small test)
python benchmark.py \
  --xes-path ../benchmark/hiring/hiring_log_high.xes \
  --textual-model-path ../benchmark_runs/run_20251231_005033/hiring_log_high_textual_model.txt \
  --k-valid 20 \
  --k-invalid 20

# 5. Review results
cd adapters_benchmark_results/run_*/
cat comparison_report.md

# 6. Run production benchmark
cd ../..
python benchmark.py \
  --xes-path ../benchmark/hiring/hiring_log_high.xes \
  --textual-model-path ../benchmark_runs/run_20251231_005033/hiring_log_high_textual_model.txt \
  --k-valid 100 \
  --k-invalid 100
```

## Design Principles

### Zero Information Loss

The core requirement: adapted models must preserve ALL behavioral information.

**What is preserved:**
- Every activity and its sequence
- Every decision point and branching logic
- All loops, reworks, and exceptional paths
- All frequencies and performance data
- All start/end conditions
- All stakeholder roles

**What is adapted:**
- Language and terminology (technical → audience-appropriate)
- Presentation structure (technical flow → audience format)
- Emphasis and focus (highlight what matters to audience)

### LLM Selection Strategy

**Adaptation uses GPT-5** (high quality):
- Complex transformation task requiring nuance
- Must balance simplification with completeness
- Quality matters more than cost

**Validation uses GPT-5-mini** (cost effective):
- Simple binary classification task
- Repeated many times (100+ traces)
- Speed and cost matter

### Prompt Engineering

Each audience prompt follows a strict template:

1. **Critical Requirement** - Zero information loss mandate
2. **Preserve List** - What must NOT be removed
3. **Adapt Rules** - How to transform (translate, don't delete)
4. **Maintain Completeness** - Verification checklist
5. **Output Specification** - Expected format

## Troubleshooting

### API Key Issues

```bash
# Verify keys are set
env | grep API_KEY

# Test with minimal example
python -c "from langchain_openai import ChatOpenAI; print(ChatOpenAI(model='gpt-5-mini').invoke('test'))"
```

### Adaptation Failures

If adaptation fails:
1. Check API key for selected provider
2. Verify model name is correct (`provider:model` format)
3. Try with default GPT-5: `--llm-model openai:gpt-5`
4. Check original model is valid text file

### Poor Benchmark Performance

If F1 scores are low:
1. Review misclassified traces in `misclassified/` folders
2. Check if original model itself has low baseline
3. Consider refining adapter prompt for problematic audience
4. Verify validator LLM is working correctly

## Extending the System

### Adding New Audiences

1. Create `adapter_prompts/new_audience.txt` following template
2. Update `valid_audiences` list in `adapters.py` (line ~90)
3. Update `audiences` list in `benchmark.py` (line ~460)
4. Test with single adaptation first

### Adding New Perturbations

To improve invalid trace generation, edit `benchmark.py`:

```python
from benchmark_template import skip_mandatory_activity, duplicate_activity

# In generate_invalid_traces call, add more perturbations
perturbations = [random_swap_events, skip_mandatory_activity]
```

### Custom Validation Prompts

Modify the system prompt in `create_validator()` (imported from `phase5_benchmark.py`) to adjust validation behavior.

## Related Files

- `../phase5_benchmark.py` - Core validation infrastructure
- `../benchmark_template.py` - Trace manipulation utilities
- `../enhanced_agentic_miner.py` - Original model generation

## Citation

If you use this system in research, please cite:

```
Adaptive Process Model Explanation System
Part of: All Things Agents - Process Mining Samples
Repository: pm-samples/downstream/
```

## Support

For issues or questions:
1. Check this README
2. Review example outputs in `adapters_benchmark_results/`
3. Read adapter prompts to understand transformation logic
4. Examine `comparison_report.md` for insights

## License

Part of the All Things Agents project.
