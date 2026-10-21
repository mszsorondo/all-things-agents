# CLAUDE.md

## Project Overview

This is the **Agentic Process Mining (APM)** framework — a system that uses LLMs to generate textual process models from event logs, then evaluates them via trace classification benchmarks. It implements the methodology from a master's thesis at Universidad Austral.

The core idea: discover multiple process representations (Petri nets, DFGs, variants, etc.) from XES event logs, feed them to an LLM agent that produces a textual AS-IS model, then validate that model's quality by checking if a small/cheap LLM (GPT-5-mini) can use it to correctly classify valid vs. invalid traces.

## Architecture

### Core Pipeline
```
XES Event Log
  → discover_representations() [Petri Net, DFG, Variants, Process Tree, BPMN, Heuristics Net]
  → build_prompt() [combine all textual abstractions]
  → LLM Agent (run_agent or run_without_tools)
  → Textual AS-IS Model + Textual IDEAL Model
  → Model Fidelity Benchmark (trace classification with small LLM)
  → Metrics (Precision, Recall, F1)
  → Refinement cycle (optional iterative improvement)
```

### Key Files
- `enhanced_agentic_miner.py` — Core mining engine: `AgenticProcessMiner`, `MiningConfiguration`, `RepresentationType`, representation classes, caching system
- `phase5_benchmark.py` — Model Fidelity Benchmark: trace sampling, perturbation, LLM validation, metrics
- `benchmark_template.py` — Trace perturbation functions (`random_swap_events`, `df_to_key_value_rows`) and formatting utilities
- `refine_benchmark.py` — Optimize-Execute-Evaluate refinement cycle (iterative model improvement guided by misclassified traces)
- `refine_benchmark_interface.py` — Interactive CLI wizard for configuring refinement experiments
- `phase5_benchmark_multiagent.py` — Multi-agent architecture comparison (experimental, largely abandoned due to cost)

### Prompt Files
- `ONLY_GENERATE_TEXTUAL_MODEL.txt` — Agent mode (tool calling): generate AS-IS only
- `GENERATE_AS_IS_MODEL.txt` — No-tools mode step 1: generate AS-IS with emphasis on capturing ALL variants
- `GENERATE_IDEAL_MODEL.txt` — No-tools mode step 2: generate optimized IDEAL model

### Benchmark Datasets
Located in `benchmark/`:
- `hiring/hiring_log_high.xes` — Parallel interviews, high variability (hardest)
- `hospital/hospital_log_high.xes` — Linear medical flow (easiest)
- `lending/lending_log_high.xes` — Sequential with clear decision points
- `renting/renting_log_high.xes` — Multiple termination paths

### Output Structure
```
benchmark_runs/run_YYYYMMDD_HHMMSS/
  results.json, config.json
  {problem}_textual_model.txt
  {problem}_system_prompt.txt
  traces/{problem}/valid_traces.pkl, invalid_traces.pkl
  misclassified/{problem}/fn_XXXX.pkl, fp_XXXX.pkl
  refined/  (refinement iterations nest here)
```

## Commands

```bash
# Run benchmark (default: tools method, gpt-5-mini)
python phase5_benchmark.py

# Run with specific configuration
python phase5_benchmark.py --method no-tools --llm openai:gpt-4o --validator-llm gpt-5-mini

# Resume interrupted run
python phase5_benchmark.py --resume benchmark_runs/run_YYYYMMDD_HHMMSS

# Run refinement (interactive)
python refine_benchmark_interface.py

# Run refinement (direct)
python refine_benchmark.py --parent-run benchmark_runs/run_YYYYMMDD_HHMMSS --refinement-llm openai:gpt-5 --mode incremental
```

## Conventions

- LLM model format: `provider:model` (e.g., `openai:gpt-5-mini`, `anthropic:claude-sonnet-4-20250514`, `xai:grok-4-1-fast-reasoning`, `deepseek:DeepSeek-V3.2-Thinking`)
- Validator model format: just model name (e.g., `gpt-5-mini`) — always OpenAI-compatible
- Results are always saved incrementally after each problem completes (crash-resilient)
- Traces are cached by content hash of XES file + sampling params
- Representations are cached by SHA-256 of XES file stats
- Language: codebase is in English, thesis is in Spanish

## Key Design Decisions

- **Textual model as universal intermediate representation**: The AS-IS model is a natural language description, not a formal notation. It serves as a semantic bridge between algorithmic PM outputs and human/LLM understanding.
- **Small model validation**: Using GPT-5-mini as validator is intentional — it's a "clarity test" for the generated model. If a small model can classify traces correctly, the textual model is clear enough.
- **Two generation modes**: "tools" (ReAct agent with tool calling) vs. "no-tools" (direct two-step conversation). Different LLMs perform differently with each.
- **Refinement = prompt optimization**: The textual model IS the prompt. Misclassified traces act as "textual gradients" guiding iterative improvement.
- **Semantic overfitting risk**: More refinement iterations can degrade performance. The system tracks all iterations and selects the best model per process, not necessarily the latest.

## Dependencies

- Python 3.8+
- pm4py >= 2.7.0 (process mining algorithms)
- langchain, langchain_openai, langgraph (LLM orchestration)
- pandas (trace DataFrames)
- API keys needed in `.env`: OPENAI_API_KEY, optionally XAI_API_KEY, DEEPSEEK_API_KEY, ANTHROPIC_API_KEY
