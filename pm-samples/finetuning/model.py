"""Model loading (QLoRA on T4) and generation with log-probability computation.

Two-pass approach for REINFORCE:
  Pass 1 (no grad): Fast autoregressive generation via model.generate()
  Pass 2 (with grad): Single forward pass over [prompt + generated] to get log-probs
"""

import sys
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

from finetuning.config import ModelConfig, FinetuningConfig
from enhanced_agentic_miner import AgenticProcessMiner, MiningConfiguration, RepresentationType


def load_model(config: ModelConfig):
    """Load a quantized model with LoRA adapters for T4 GPU.

    Returns:
        model: PeftModel with LoRA adapters (trainable)
        tokenizer: Corresponding tokenizer
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=(config.quantization_bits == 4),
        load_in_8bit=(config.quantization_bits == 8),
        bnb_4bit_compute_dtype=getattr(torch, config.torch_dtype),
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=getattr(torch, config.torch_dtype),
    )

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


def load_checkpoint(config: ModelConfig, adapter_path: str):
    """Load a model with a previously saved LoRA adapter."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=(config.quantization_bits == 4),
        load_in_8bit=(config.quantization_bits == 8),
        bnb_4bit_compute_dtype=getattr(torch, config.torch_dtype),
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=getattr(torch, config.torch_dtype),
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Representation caching (discover once per run, not per step)
# ---------------------------------------------------------------------------

class RepresentationStore:
    """Discovers and caches representations once per XES file for the entire run.

    Avoids re-instantiating AgenticProcessMiner and re-checking disk cache
    every training step. Representations are immutable across epochs.
    """

    def __init__(self, config: FinetuningConfig):
        self.config = config
        self._cache: Dict[str, str] = {}       # xes_path -> representations_text
        self._system_prompt: str = self._load_system_prompt()

    def _load_system_prompt(self) -> str:
        prompt_file_path = PROJECT_ROOT / self.config.prompt_file
        if prompt_file_path.exists():
            return prompt_file_path.read_text()
        return ""

    def discover(self, xes_path: str) -> str:
        """Discover representations for a given XES file (cached after first call)."""
        if xes_path in self._cache:
            return self._cache[xes_path]

        mining_config = MiningConfiguration(
            representations=self.config.representations,
            model_name="local",  # placeholder, not used for discovery
            prompt_file=self.config.prompt_file,
            use_cache=True,
        )
        miner = AgenticProcessMiner(mining_config)
        miner.log_file = xes_path  # used as cache key for disk cache
        miner.discover_representations()

        representations_text = miner.build_prompt()
        self._cache[xes_path] = representations_text
        return representations_text

    def build_generation_prompt(self, xes_path: str, tokenizer) -> str:
        """Build the full chat prompt for a given XES file."""
        representations_text = self.discover(xes_path)
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": representations_text},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_textual_model(model, tokenizer, prompt: str, config: ModelConfig):
    """Pass 1: Generate textual model via sampling (no gradients, fast).

    Returns:
        text: Generated textual model string
        input_ids: Prompt token IDs (1, prompt_len)
        generated_ids: Generated token IDs (gen_len,) without prompt
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=config.max_new_tokens,
            temperature=config.generation_temperature,
            top_p=config.generation_top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_ids = output[0, input_ids.shape[1]:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text, input_ids, generated_ids


def compute_sequence_log_probs(model, input_ids: torch.Tensor, generated_ids: torch.Tensor) -> torch.Tensor:
    """Pass 2: Compute log-probabilities of the generated sequence (with gradients).

    Single forward pass over the full sequence [prompt + generated].
    Extracts log-probs only for generated tokens (prompt is masked).

    Returns:
        mean_log_prob: Scalar tensor with gradient attached
    """
    full_ids = torch.cat([input_ids, generated_ids.unsqueeze(0)], dim=1)

    outputs = model(full_ids)
    logits = outputs.logits  # (1, seq_len, vocab_size)

    # Shift: position t predicts token t+1
    shift_logits = logits[:, :-1, :]
    shift_labels = full_ids[:, 1:]

    # Per-token log probs (negative cross-entropy)
    log_probs = -F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="none",
    )  # (seq_len - 1,)

    # Mask: keep only generated tokens
    prompt_len = input_ids.shape[1]
    gen_start = prompt_len - 1  # shift offset
    gen_log_probs = log_probs[gen_start:]

    return gen_log_probs.mean()


def save_adapter(model, path: str):
    """Save LoRA adapter weights."""
    Path(path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
