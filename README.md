this branch contains code (and readme!) that was either generated from chatgpt/deepseek, didn't really test this

# Communication Skills Coach

This project is a wrapper application around an open‑source LLM designed to help learners improve their verbal communication skills. It supports chat and voice interactions, provides skill-training activities (impromptu speaking, storytelling, conflict resolution), and assesses presentations with actionable feedback.

## Features
- **LLM Integration:** Uses an open-source model (e.g., Mistral‑7B) via Hugging Face Transformers with quantization (8‑bit using bitsandbytes).
- **Chat Interface:** Text‑based conversation with a communication coach.
- **Voice Interface:** Integrates Whisper for speech‑to‑text and (optionally) TTS.
- **Training Modules:** Includes three training exercises with pre‑defined prompts.
- **Presentation Assessment:** Evaluates presentation scripts or audio with structured feedback.
- **Interface:** Built with Gradio for a web‑based UI.
- **Extensibility:** Easily swap models via configuration and robust error handling.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt

