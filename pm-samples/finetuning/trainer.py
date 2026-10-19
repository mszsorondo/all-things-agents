"""REINFORCE training loop — one model per problem.

For each problem (XES log), trains a separate LoRA adapter from scratch.
Training signal: trace classification metrics from a downstream API validator.

Each training step (= 1 epoch for 1 problem):
  1. Build prompt from cached representations
  2. Generate textual model (sampling, no grad)
  3. Classify traces via API using the generated model
  4. Compute reward from classification metrics
  5. Compute log-probs of the generated text (with grad)
  6. REINFORCE loss = -normalized_advantage * mean_log_prob
  7. Backpropagate and update LoRA weights
"""

import json
import time
import math
import random
from pathlib import Path
from datetime import datetime
from dataclasses import asdict

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from finetuning.config import FinetuningConfig, PROJECT_ROOT
from finetuning.model import (
    load_model,
    RepresentationStore,
    generate_textual_model,
    compute_sequence_log_probs,
    save_adapter,
)
from finetuning.evaluator import evaluate_textual_model
from finetuning.losses import get_loss_fn


class RunningStats:
    """Welford's online algorithm for running mean/variance.

    Used to normalize rewards so that gradient magnitude is consistent
    regardless of the reward function's range.
    """

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, x: float):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

    def std(self) -> float:
        if self.n < 2:
            return 1.0
        return max(math.sqrt(self.m2 / self.n), 1e-8)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int):
    """Linear warmup then cosine decay to 0."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return current_step / max(1, num_warmup_steps)
        progress = (current_step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


def create_run_folder(config: FinetuningConfig) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = PROJECT_ROOT / config.training.output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(config), indent=2, default=str))
    return run_dir


def check_early_stopping(history: list, patience: int) -> bool:
    """Stop if mean_reward hasn't improved in the last `patience` epochs."""
    if len(history) < patience + 1:
        return False
    recent = history[-patience:]
    best_before = max(e["reward"] for e in history[:-patience])
    best_recent = max(e["reward"] for e in recent)
    if best_recent <= best_before:
        print(f"  Early stopping: no reward improvement in {patience} epochs "
              f"(best before: {best_before:.4f}, best recent: {best_recent:.4f})")
        return True
    return False


def train_single_problem(
    config: FinetuningConfig,
    xes_path: str,
    problem_dir: Path,
    rep_store: RepresentationStore,
):
    """Train a single LoRA adapter for one problem."""
    problem_name = Path(xes_path).stem
    print(f"\n{'#'*60}")
    print(f"# Training: {problem_name}")
    print(f"# Output:   {problem_dir}")
    print(f"{'#'*60}")

    model, tokenizer = load_model(config.model)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    # 1 optimizer step per epoch, so total_steps = num_epochs
    total_training_steps = config.training.num_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.training.warmup_steps,
        num_training_steps=total_training_steps,
    )

    loss_fn = get_loss_fn(config.loss_fn)
    reward_baseline = 0.0
    reward_stats = RunningStats()

    history = []

    for epoch in range(config.training.num_epochs):
        epoch_start = time.time()

        # --- Step 1: Build prompt ---
        prompt = rep_store.build_generation_prompt(xes_path, tokenizer)

        # --- Step 2: Generate textual model (no grad) ---
        print(f"\n[Epoch {epoch+1}/{config.training.num_epochs}] [{problem_name}] Generating...")
        text, input_ids, generated_ids = generate_textual_model(
            model, tokenizer, prompt, config.model
        )
        print(f"  {len(generated_ids)} tokens generated")

        # Save generated model
        model_path = problem_dir / f"epoch_{epoch+1}_textual_model.txt"
        model_path.write_text(text)

        # --- Step 3: Evaluate via trace classification (API) ---
        print(f"  Classifying {config.eval.k_valid}+{config.eval.k_invalid} traces...")
        metrics = evaluate_textual_model(text, xes_path, config.eval)
        print(f"  F1={metrics['f1']:.3f} P={metrics['precision']:.3f} "
              f"R={metrics['recall']:.3f} TP={metrics['TP']} FP={metrics['FP']} "
              f"TN={metrics['TN']} FN={metrics['FN']}")

        # --- Step 4: Compute reward with normalization ---
        raw_reward = loss_fn(metrics["TP"], metrics["FP"], metrics["TN"], metrics["FN"])
        reward_stats.update(raw_reward)

        # Normalize advantage by running std for consistent gradient scale
        advantage = (raw_reward - reward_baseline) / reward_stats.std()

        # Update EMA baseline after computing advantage
        ema = config.training.reward_baseline_ema
        reward_baseline = ema * reward_baseline + (1 - ema) * raw_reward

        print(f"  Reward={raw_reward:.4f} Baseline={reward_baseline:.4f} Adv={advantage:.4f}")

        # --- Step 5: Compute log-probs (with grad) ---
        mean_log_prob = compute_sequence_log_probs(model, input_ids, generated_ids)

        # --- Step 6: REINFORCE loss ---
        loss = -advantage * mean_log_prob

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, config.training.max_grad_norm)
        optimizer.step()
        scheduler.step()

        epoch_time = time.time() - epoch_start
        current_lr = scheduler.get_last_lr()[0]

        epoch_entry = {
            "epoch": epoch + 1,
            "reward": raw_reward,
            "loss": loss.item(),
            "reward_baseline": reward_baseline,
            "advantage": advantage,
            "learning_rate": current_lr,
            "metrics": metrics,
            "epoch_time_seconds": epoch_time,
        }
        history.append(epoch_entry)

        print(f"  Loss={loss.item():.4f} LR={current_lr:.2e} Time={epoch_time:.1f}s")

        # Save checkpoint
        if (epoch + 1) % config.training.save_every_epoch == 0:
            ckpt_path = problem_dir / f"checkpoint_epoch_{epoch+1}"
            save_adapter(model, str(ckpt_path))
            print(f"  Saved checkpoint: {ckpt_path}")

        # Save history incrementally
        (problem_dir / "history.json").write_text(json.dumps(history, indent=2, default=str))

        # Early stopping
        if config.training.early_stopping_patience and check_early_stopping(
            history, config.training.early_stopping_patience
        ):
            break

    # Final save
    save_adapter(model, str(problem_dir / "final_adapter"))
    print(f"  Final adapter saved: {problem_dir / 'final_adapter'}")

    # Free GPU memory before next problem
    del model, tokenizer, optimizer, scheduler, trainable_params
    torch.cuda.empty_cache()

    return history


def train(config: FinetuningConfig):
    """Main training entry point. Trains one model per problem."""
    random.seed(config.training.seed)
    torch.manual_seed(config.training.seed)

    run_dir = create_run_folder(config)
    print(f"Run directory: {run_dir}")

    problems = [str(PROJECT_ROOT / p) for p in config.problems]

    # Discover representations once for all problems
    print("Discovering representations (one-time)...")
    rep_store = RepresentationStore(config)
    for xes_path in problems:
        rep_store.discover(xes_path)
    print(f"  Cached representations for {len(problems)} problems")

    all_histories = {}

    for xes_path in problems:
        problem_name = Path(xes_path).stem
        problem_dir = run_dir / problem_name
        problem_dir.mkdir(parents=True, exist_ok=True)

        history = train_single_problem(config, xes_path, problem_dir, rep_store)
        all_histories[problem_name] = history

    # Save combined summary
    summary = {
        problem: {
            "best_reward": max(e["reward"] for e in hist),
            "best_epoch": max(range(len(hist)), key=lambda i: hist[i]["reward"]) + 1,
            "final_reward": hist[-1]["reward"],
            "final_f1": hist[-1]["metrics"]["f1"],
            "epochs_trained": len(hist),
        }
        for problem, hist in all_histories.items()
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*60}")
    print("Training complete — Summary")
    print(f"{'='*60}")
    for problem, s in summary.items():
        print(f"  {problem}: best_reward={s['best_reward']:.4f} (epoch {s['best_epoch']}), "
              f"final_f1={s['final_f1']:.3f}")
    print(f"{'='*60}")

    return run_dir, all_histories
