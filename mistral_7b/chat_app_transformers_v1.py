"""
    uses mistralai/Mistral-7B-v0.1 running locally
"""
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv

# Get the Hugging Face token from the environment variable
hf_token = os.getenv("hf_token")

if not hf_token:
    raise ValueError("Hugging Face token not found in environment variables.")

# Load the tokenizer and model with the Hugging Face token
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", use_auth_token=hf_token)
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", use_auth_token=hf_token)


# Set pad_token_id if not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def generate_response(prompt, max_length=100):
    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate a response using the model
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],  # Pass the attention mask
        max_length=max_length,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,  # Explicitly set pad_token_id
    )

    # Decode the generated tokens back into text
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response

# Example usage in a chat loop
print("Chat with Mistral-7B! Type 'exit' to end the chat.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    # Generate a response
    response = generate_response(user_input)
    print(f"Mistral-7B: {response}")

