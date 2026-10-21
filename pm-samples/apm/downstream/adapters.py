#!/usr/bin/env python3
"""
Adaptive Process Model Explanation System - Core Adaptation Engine

Transforms technical textual process models to audience-specific explanations
while preserving semantic fidelity for trace validation.
"""

import os
import argparse
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


def create_llm_client(model_name: str):
    """
    Create LLM client based on provider:model format.

    Args:
        model_name: Format "provider:model" (e.g., "openai:gpt-4o", "anthropic:claude-3-5-sonnet-20241022")

    Returns:
        LLM client object (ChatOpenAI, ChatAnthropic, etc.)

    Raises:
        ValueError: If model_name format is invalid or provider unsupported
    """
    if ':' not in model_name:
        raise ValueError(f"Invalid model format: {model_name}. Expected 'provider:model'")

    provider, model = model_name.split(':', 1)
    provider = provider.lower()

    if provider == 'openai':
        from langchain_openai import ChatOpenAI
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        return ChatOpenAI(model=model, api_key=api_key, temperature=0)

    elif provider == 'anthropic':
        from langchain_anthropic import ChatAnthropic
        api_key = os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        return ChatAnthropic(model=model, api_key=api_key, temperature=0)

    elif provider == 'xai':
        from langchain_openai import ChatOpenAI
        api_key = os.environ.get('XAI_API_KEY')
        if not api_key:
            raise ValueError("XAI_API_KEY environment variable not set")
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url="https://api.x.ai/v1",
            temperature=0
        )

    elif provider == 'deepseek':
        from langchain_openai import ChatOpenAI
        api_key = os.environ.get('DEEPSEEK_API_KEY')
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable not set")
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url="https://api.deepseek.com",
            temperature=0
        )

    else:
        raise ValueError(f"Unsupported provider: {provider}. Supported: openai, anthropic, xai, deepseek")


def load_adapter_prompt(audience: str) -> str:
    """
    Load adapter prompt for the specified audience.

    Args:
        audience: Target audience (executives|process_analysts|developers|auditors|children)

    Returns:
        Adapter prompt text

    Raises:
        FileNotFoundError: If prompt file doesn't exist
        ValueError: If audience is invalid
    """
    valid_audiences = ['executives', 'process_analysts', 'developers', 'auditors', 'children']
    if audience not in valid_audiences:
        raise ValueError(f"Invalid audience: {audience}. Valid options: {', '.join(valid_audiences)}")

    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    prompt_path = script_dir / 'adapter_prompts' / f'{audience}.txt'

    if not prompt_path.exists():
        raise FileNotFoundError(f"Adapter prompt not found: {prompt_path}")

    return prompt_path.read_text()


def adapt_textual_model(model_text: str, audience: str, llm_model: str = 'openai:gpt-5') -> str:
    """
    Adapt textual model for target audience using LLM.

    Args:
        model_text: Original textual model content
        audience: Target audience (executives|process_analysts|developers|auditors|children)
        llm_model: LLM to use (format: provider:model)

    Returns:
        Adapted model text
    """
    # Load adapter instructions
    adapter_prompt = load_adapter_prompt(audience)

    # Create LLM client
    llm = create_llm_client(llm_model)

    # Construct full prompt
    full_prompt = f"""{adapter_prompt}

---

ORIGINAL TECHNICAL PROCESS MODEL:

{model_text}

---

Now adapt this process model for the target audience, following all guidelines above.
"""

    # Invoke LLM
    messages = [("human", full_prompt)]
    response = llm.invoke(messages)

    return response.content


def main():
    """CLI for adapting textual models to different audiences."""
    parser = argparse.ArgumentParser(
        description='Adapt technical process models for different audiences',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Adapt for executives using GPT-5
  python downstream/adapters.py \\
    --model-path benchmark_runs/run_X/hiring_log_high_textual_model.txt \\
    --audience executives \\
    --llm-model openai:gpt-5 \\
    --output-path adapted_executives.txt

  # Adapt for developers using Claude
  python downstream/adapters.py \\
    --model-path mymodel.txt \\
    --audience developers \\
    --llm-model anthropic:claude-3-5-sonnet-20241022 \\
    --output-path adapted_developers.txt

Valid audiences: executives, process_analysts, developers, auditors, children
        """
    )

    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to original textual model file'
    )

    parser.add_argument(
        '--audience',
        type=str,
        required=True,
        choices=['executives', 'process_analysts', 'developers', 'auditors', 'children'],
        help='Target audience for adaptation'
    )

    parser.add_argument(
        '--llm-model',
        type=str,
        default='openai:gpt-5',
        help='LLM model to use (format: provider:model, default: openai:gpt-5)'
    )

    parser.add_argument(
        '--output-path',
        type=str,
        required=True,
        help='Path where adapted model will be saved'
    )

    args = parser.parse_args()

    # Load original model
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        return 1

    print(f"Loading original model from: {model_path}")
    model_text = model_path.read_text()

    # Adapt model
    print(f"Adapting for audience: {args.audience}")
    print(f"Using LLM: {args.llm_model}")

    try:
        adapted_text = adapt_textual_model(model_text, args.audience, args.llm_model)
    except Exception as e:
        print(f"Error during adaptation: {e}")
        return 1

    # Save adapted model
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(adapted_text)

    print(f"Adapted model saved to: {output_path}")
    print(f"Original length: {len(model_text)} characters")
    print(f"Adapted length: {len(adapted_text)} characters")

    return 0


if __name__ == '__main__':
    exit(main())
