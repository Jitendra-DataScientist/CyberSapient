import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from functools import lru_cache

# Change this to your model of choice from Hugging Face.
MODEL_NAME = "mistral-7b"  # Example model; ensure it supports inference with quantization

@lru_cache(maxsize=1)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",   # Automatically use GPU if available
        load_in_8bit=True    # Apply 8-bit quantization to reduce memory usage
    )
    return model, tokenizer

def generate_response(prompt: str, max_new_tokens: int = 150) -> str:
    model, tokenizer = load_model()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

