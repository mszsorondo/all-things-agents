"""Fine-tuning pipeline for Agentic Process Mining textual model generation.

Trains a local LLM (via QLoRA on T4 GPU) to generate better textual process models
by using trace classification results as a training signal (REINFORCE).
"""
