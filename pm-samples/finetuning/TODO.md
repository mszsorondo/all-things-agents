# TODO — finetuning/

## High Priority
- [ ] Verify `miner.build_prompt()` method name matches `enhanced_agentic_miner.py` — `build_generation_prompt` in model.py depends on it
- [ ] Cache representations once per training run instead of calling `discover_representations()` every step
- [ ] Add LR warmup scheduler (linear warmup + cosine decay)
- [ ] Add early stopping when reward plateaus across N epochs
- [ ] Test VRAM budget: generator (8B 4-bit ~5GB) + validator (3B 4-bit ~2GB) on T4 (16GB) — check if both fit simultaneously

## Medium Priority
- [ ] Add SFT pre-training mode: fine-tune on the best existing textual models from `benchmark_runs/` before REINFORCE
- [ ] Add DPO mode as alternative to REINFORCE, using (better model, worse model) pairs from different benchmark runs
- [ ] Implement the additional loss functions suggested in `losses.py` comments (Tversky, log-barrier, curriculum, contrastive)
- [ ] Chunk the two-pass log-prob computation for very long prompts to avoid OOM

## Low Priority
- [ ] Multi-GPU support (model parallelism or separate GPUs for generator/validator)
- [ ] Wandb/tensorboard logging integration
- [ ] Evaluation against the full benchmark pipeline (phase5_benchmark.py) at end of training
- [ ] Support resuming interrupted training runs from checkpoint
