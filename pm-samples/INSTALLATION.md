# Installation & Setup Guide

## Quick Setup

```bash
# 1. Create environment (choose one)
conda create -n pm-samples python=3.10 && conda activate pm-samples
# OR: python -m venv venv && source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API keys
cp .env.example .env
# Edit .env and add your keys (get from links below)

# 4. Verify
python -c "import pm4py, langchain; from dotenv import load_dotenv; import os; load_dotenv(); print('✓ Setup OK')"
```

## Requirements

- **Python**: 3.9+ (tested with 3.10-3.13)
- **API Keys**: At least one of:
  - OpenAI: https://platform.openai.com/api-keys
  - Anthropic: https://console.anthropic.com/

## Dependencies Installed

```
pm4py>=2.7.0                  # Process mining
langgraph, langchain          # Agent framework
langchain-openai              # GPT models
langchain-anthropic           # Claude models
python-dotenv                 # .env loading
pandas, numpy                 # Data processing
```

Optional: `pip install deepagents` (if using deep agent mode)

## API Keys (.env file)

```bash
# Required: At least ONE
OPENAI_API_KEY=sk-proj-YOUR-KEY-HERE
ANTHROPIC_API_KEY=sk-ant-api03-YOUR-KEY-HERE

# Optional
XAI_API_KEY=xai-YOUR-KEY-HERE
DEEPSEEK_API_KEY=sk-YOUR-KEY-HERE
```

## Running Scripts

```bash
# Enhanced Agentic Miner
python enhanced_agentic_miner.py --xes-path benchmark/hiring/hiring_log_high.xes --model-name openai:gpt-4o-mini

# Phase 5 Benchmark
python phase5_benchmark.py --xes-path benchmark/hiring/hiring_log_high.xes --precomputed-model-path PATH --validator-model gpt-4o-mini

# Refine Benchmark
python refine_benchmark.py --xes-path benchmark/hiring/hiring_log_high.xes --model-name openai:gpt-4o

# Run With New Guides
python run_with_new_guides.py --xes-path benchmark/hiring/hiring_log_high.xes --model-name openai:gpt-4o-mini

# Ablation Study
python ablation_study.py --xes-path benchmark/hiring/hiring_log_high.xes --model-name openai:gpt-4o-mini

# Downstream: Adapt Model
python downstream/adapters.py --model-path PATH --audience executives --output-path output.txt

# Downstream: Benchmark
python downstream/benchmark.py --xes-path PATH --textual-model-path PATH --audiences executives --k-valid 100 --k-invalid 100
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: pm4py` | `pip install -r requirements.txt` |
| `ModuleNotFoundError: dotenv` | `pip install python-dotenv` |
| API key not found | Check `.env` exists in project root, verify with `python -c "from dotenv import load_dotenv; import os; load_dotenv(); print(os.getenv('OPENAI_API_KEY'))"` |
| `ImportError: deepagents` | `pip install deepagents` (or skip if not needed) |
| Wrong Python version | Use `python3.10` explicitly or create venv with specific version |

## Quick Test

```bash
# Small test (costs ~$0.10)
python downstream/benchmark.py \
  --xes-path benchmark/hiring/hiring_log_high.xes \
  --textual-model-path benchmark_runs/run_20251231_005033/hiring_log_high_textual_model.txt \
  --audiences executives \
  --k-valid 5 --k-invalid 5
```

---

**Complete setup in 4 commands:**
```bash
conda create -n pm-samples python=3.10 && conda activate pm-samples
pip install -r requirements.txt
cp .env.example .env && nano .env  # Add your API keys
python enhanced_agentic_miner.py --help  # Verify
```
