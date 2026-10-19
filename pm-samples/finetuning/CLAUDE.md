# CLAUDE.md — finetuning/

## What This Is

Fine-tuning pipeline for the APM (Agentic Process Mining) textual model generator. Instead of using API-based LLMs (GPT-5, Claude, etc.) to generate textual process models, we train a local model via REINFORCE using trace classification performance as the training signal.

**The key insight**: the textual model IS the prompt for the downstream validator. By fine-tuning the generator to produce textual models that yield better classification metrics, we are doing *prompt optimization via parameter updates*.

## Architecture

```
              REINFORCE Training Loop
              ========================

  XES Event Log
       |
       v
  [Cached Representations]  ──>  Prompt (system + representations)
       (Petri Net, DFG,              |
        Variants, Process Tree)       v
                              [Local Model (QLoRA)]  ──>  Textual AS-IS Model
                                      ^                         |
                                      |                         v
                              [Update LoRA weights]    [Parallel Trace Classification]
                                      ^                  (k queries via ThreadPoolExecutor)
                                      |                         |
                                      |                         v
                              [REINFORCE Loss]  <───  Reward = f(TP, FP, TN, FN)
                              -advantage * log_prob
```

### Two-pass generation for REINFORCE:
1. **Pass 1 (no grad)**: `model.generate()` — fast autoregressive sampling
2. **Pass 2 (with grad)**: Single forward pass over [prompt + generated] to compute log-probs of the generated tokens. This is what gets differentiated.

### Variance reduction:
- EMA baseline: `reward_baseline = ema * old + (1-ema) * reward` (initialized at 0, not first reward)
- Running reward normalization via Welford's algorithm: `advantage = (reward - baseline) / running_std`
- Per-problem training eliminates cross-problem noise in the baseline
- Without these, REINFORCE has extremely high variance with sparse scalar rewards

## Files

| File | Purpose |
|------|---------|
| `config.py` | Dataclasses: ModelConfig, TrainingConfig, EvalConfig, FinetuningConfig |
| `losses.py` | 8 reward functions + registry. Each takes (TP, FP, TN, FN) → scalar |
| `model.py` | QLoRA model loading, generation, log-prob computation, checkpoint save/load |
| `evaluator.py` | Trace classification via OpenAI API (gpt-5-mini) with parallel calls |
| `trainer.py` | Per-problem REINFORCE training loop with logging and checkpointing |
| `run.py` | CLI entry point with argparse |

## How to Run

```bash
# From the pm-samples/ directory:
python -m finetuning.run --model meta-llama/Llama-3.1-8B-Instruct --loss-fn f1

# With custom settings:
python -m finetuning.run \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --loss-fn anti_permissiveness \
  --epochs 20 \
  --k-valid 30 --k-invalid 30 \
  --k-parallel 32 \
  --lr 5e-5 \
  --early-stopping-patience 5

# See all options:
python -m finetuning.run --help
```

## Target Environment

- **GPU**: NVIDIA T4 (16GB VRAM) on Lightning AI — only the generator runs locally
- **Validator**: OpenAI API (gpt-5-mini) — no local VRAM cost for the validator
- **Quantization**: 4-bit QLoRA (NF4) — fits 7-8B models comfortably
- **Models that fit**: Llama 3.1 8B, Mistral 7B, Phi-3, Qwen2.5 7B
- **Models that are tight**: 13B models (possible with aggressive settings)

## Dependencies (beyond parent project)

```
torch
transformers
peft
bitsandbytes
accelerate
```

The parent project dependencies (pm4py, langchain, langchain_openai) are also required since we import from the existing codebase.

## Available Loss Functions

| Name | What it does | When to use |
|------|-------------|-------------|
| `f1` | Standard F1 score | Baseline, balanced FP/FN treatment |
| `fbeta_precision` | F-beta with beta=0.5 | When FP is the dominant error (most cases) |
| `mcc` | Matthews Correlation Coefficient | Balanced metric for binary classification |
| `balanced_accuracy` | Avg of sensitivity + specificity | Less sensitive to class imbalance |
| `asymmetric_fp` | Weighted FP/FN penalty (FP 2x) | Directly targets the permissive model problem |
| `precision_at_high_recall` | Maximize precision with recall floor | Keep perfect recall while improving precision |
| `anti_permissiveness` | Specificity * recall_gate | Specifically rewards rejecting invalid traces |
| `adaptive_fp_fn` | Auto-adjusts based on error distribution | Handles both permissive and over-specified states |

See `losses.py` for 6 additional suggestions for future implementation (Tversky, log-barrier, curriculum, etc.).

## Imports from Parent Codebase

This module imports heavily from the parent `pm-samples/` directory:

- **`enhanced_agentic_miner`**: `AgenticProcessMiner`, `MiningConfiguration`, `RepresentationType` — for discovering and caching process representations
- **`phase5_benchmark`**: `create_validator`, `sample_valid_traces`, `generate_invalid_traces`, `calculate_metrics`, trace cache functions — for evaluation
- **`benchmark_template`**: `random_swap_events`, `df_to_key_value_rows` — used transitively via phase5_benchmark

Path resolution is handled via `PROJECT_ROOT = Path(__file__).resolve().parent.parent` + `sys.path.insert`.

## Output Structure

```
finetuning_runs/run_YYYYMMDD_HHMMSS/
  config.json                                  # Full FinetuningConfig
  summary.json                                 # Per-problem best reward, epoch, F1
  {problem_name}/                              # One directory per problem
    history.json                               # Per-epoch metrics and rewards
    epoch_{N}_textual_model.txt                # Generated model at each epoch
    checkpoint_epoch_{N}/                      # LoRA adapter weights
    final_adapter/                             # Final LoRA adapter for this problem
```

## Key Design Decisions

1. **REINFORCE over SFT**: We don't have "gold" textual models. The training signal is the downstream task performance (trace classification), making this a reinforcement learning problem. The textual model is the "action", F1/reward is the "return".

2. **API validator (gpt-5-mini)**: Uses `create_validator` from `phase5_benchmark.py` so the validation prompt is identical to the main benchmark. The validator is frozen — only the generator's LoRA weights are updated. API calls are parallelized via ThreadPoolExecutor.

3. **One model per problem**: Each problem (hiring, hospital, lending, renting) gets its own LoRA adapter trained from scratch. This avoids cross-problem interference and lets each model specialize for its process structure. GPU memory is freed between problems.

4. **Reward normalization**: Running Welford statistics normalize the advantage by the reward's standard deviation, making gradient magnitude consistent across different loss functions and reward scales.

5. **Representations include Process Tree**: The default benchmark uses Petri Net + DFG + Abstract Variants. We add Process Tree for richer structural information since the local model may need more explicit guidance.

## Known Limitations

- Single T4 assumed, no multi-GPU
- Single-sample REINFORCE (1 generation per step) — variance is mitigated by EMA baseline + normalization but could benefit from multi-sample
- The two-pass log-prob computation loads full [prompt + generated] — may OOM on T4 with very long prompts
- API costs scale with epochs × problems × traces_per_step
