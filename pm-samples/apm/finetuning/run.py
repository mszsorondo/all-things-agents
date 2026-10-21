#!/usr/bin/env python3
"""CLI entry point for fine-tuning.

Trains one LoRA adapter per problem. Each problem gets an independent
model, optimizer, and training loop.

Usage:
    python -m finetuning.run --model meta-llama/Llama-3.1-8B-Instruct --loss-fn anti_permissiveness
    python -m finetuning.run --help
"""

import argparse

from finetuning.config import FinetuningConfig, ModelConfig, TrainingConfig, EvalConfig
from finetuning.losses import list_loss_functions
from finetuning.trainer import train


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune a local LLM for textual process model generation (one model per problem)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available loss functions:
{chr(10).join(f'  {name:30s} {doc}' for name, doc in list_loss_functions().items())}

Examples:
  # Default: Llama 3.1 8B generator, gpt-5-mini validator, F1 reward
  python -m finetuning.run

  # Anti-permissiveness loss on Mistral
  python -m finetuning.run --model mistralai/Mistral-7B-Instruct-v0.3 --loss-fn anti_permissiveness

  # Fast iteration: fewer traces, more concurrent API calls
  python -m finetuning.run --k-valid 30 --k-invalid 30 --k-parallel 32

  # Precision-focused with lower learning rate and early stopping
  python -m finetuning.run --loss-fn precision_at_high_recall --lr 5e-5 --epochs 20 --early-stopping-patience 5
""",
    )

    # Generator model
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                        help="HuggingFace model ID or local path for generator")
    parser.add_argument("--quant-bits", type=int, default=4, choices=[4, 8],
                        help="Generator quantization bits (4 or 8)")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--max-new-tokens", type=int, default=2048,
                        help="Max tokens for textual model generation")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Generation temperature")

    # Training
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs per problem")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--early-stopping-patience", type=int, default=None,
                        help="Stop if no reward improvement for N epochs (default: disabled)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="finetuning_runs",
                        help="Output directory for runs")

    # Evaluation / validator
    parser.add_argument("--k-valid", type=int, default=50, help="Valid traces per problem")
    parser.add_argument("--k-invalid", type=int, default=50, help="Invalid traces per problem")
    parser.add_argument("--k-parallel", type=int, default=16,
                        help="Concurrent API calls for trace classification")
    parser.add_argument("--validator-model", type=str, default="gpt-5-mini",
                        help="OpenAI-compatible model name for trace validation")

    # Loss
    parser.add_argument("--loss-fn", type=str, default="f1",
                        choices=list(list_loss_functions().keys()),
                        help="Reward/loss function for REINFORCE")

    return parser.parse_args()


def main():
    args = parse_args()

    config = FinetuningConfig(
        model=ModelConfig(
            model_name_or_path=args.model,
            quantization_bits=args.quant_bits,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            max_new_tokens=args.max_new_tokens,
            generation_temperature=args.temperature,
        ),
        training=TrainingConfig(
            num_epochs=args.epochs,
            learning_rate=args.lr,
            early_stopping_patience=args.early_stopping_patience,
            seed=args.seed,
            output_dir=args.output_dir,
        ),
        eval=EvalConfig(
            k_valid=args.k_valid,
            k_invalid=args.k_invalid,
            k_parallel=args.k_parallel,
            validator_model=args.validator_model,
        ),
        loss_fn=args.loss_fn,
    )

    print("=" * 60)
    print("FINE-TUNING: Agentic Process Mining (per-problem)")
    print("=" * 60)
    print(f"Generator:  {config.model.model_name_or_path} ({config.model.quantization_bits}-bit)")
    print(f"LoRA:       r={config.model.lora_r} alpha={config.model.lora_alpha}")
    print(f"Validator:  {config.eval.validator_model} (API)")
    print(f"Loss:       {config.loss_fn}")
    print(f"Epochs:     {config.training.num_epochs} per problem")
    print(f"LR:         {config.training.learning_rate}")
    patience = config.training.early_stopping_patience
    print(f"Early stop: {'disabled' if patience is None else f'patience={patience}'}")
    print(f"Traces:     {config.eval.k_valid} valid + {config.eval.k_invalid} invalid (parallel={config.eval.k_parallel})")
    print(f"Problems:   {[p.split('/')[-2] for p in config.problems]}")
    print("=" * 60)

    train(config)


if __name__ == "__main__":
    main()
