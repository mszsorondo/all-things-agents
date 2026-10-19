"""Configuration dataclasses for the fine-tuning pipeline."""

import sys
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from enhanced_agentic_miner import RepresentationType


@dataclass
class ModelConfig:
    """Local model configuration for T4 GPU (16GB VRAM)."""
    model_name_or_path: str = "meta-llama/Llama-3.1-8B-Instruct"
    quantization_bits: int = 4  # 4-bit QLoRA for T4 memory
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    max_new_tokens: int = 2048
    generation_temperature: float = 0.7
    generation_top_p: float = 0.9
    torch_dtype: str = "float16"


@dataclass
class TrainingConfig:
    """Training hyperparameters for REINFORCE."""
    num_epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    reward_baseline_ema: float = 0.9  # exponential moving average for variance reduction
    max_grad_norm: float = 1.0
    warmup_steps: int = 2
    save_every_epoch: int = 1
    early_stopping_patience: Optional[int] = None  # None = disabled
    seed: int = 42
    output_dir: str = "finetuning_runs"


@dataclass
class EvalConfig:
    """Trace classification evaluation configuration."""
    k_valid: int = 50   # valid traces per problem (smaller than benchmark's 100 for speed)
    k_invalid: int = 50
    k_parallel: int = 16  # concurrent API calls for trace classification
    validator_model: str = "gpt-5-mini"  # OpenAI-compatible model name for create_validator
    resample_traces_each_step: bool = True  # True: fresh random sample each step. False: cache and reuse.
    use_trace_cache: bool = False  # only relevant when resample_traces_each_step=False


@dataclass
class FinetuningConfig:
    """Top-level configuration combining all sub-configs."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    loss_fn: str = "f1"
    problems: List[str] = field(default_factory=lambda: [
        "benchmark/hiring/hiring_log_high.xes",
        "benchmark/hospital/hospital_log_high.xes",
        "benchmark/lending/lending_log_high.xes",
        "benchmark/renting/renting_log_high.xes",
    ])
    representations: List[RepresentationType] = field(default_factory=lambda: [
        RepresentationType.PETRI_NET,
        RepresentationType.DFG,
        RepresentationType.ABSTRACT_VARIANTS,
        RepresentationType.PROCESS_TREE,
    ])
    prompt_file: str = "GENERATE_AS_IS_MODEL.txt"
