"""
    uses mistralai/Mistral-7B-v0.1 running locally
"""
import os
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get the Hugging Face token from the environment variable
hf_token = os.getenv("hf_token")

if not hf_token:
    raise ValueError("Hugging Face token not found in environment variables.")

# Load the tokenizer and model with the Hugging Face token
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", token=hf_token)
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", token=hf_token).to("cuda")

# Set pad_token_id if not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def generate_response(prompt, context="", max_new_tokens=500):
    # Combine the context with the new prompt
    full_prompt = context + "\nYou: " + prompt + "\nMistral-7B:" if context else prompt

    # Tokenize the input prompt
    inputs = tokenizer(full_prompt, return_tensors="pt")

    # Generate a response using the model
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],  # Pass the attention mask
        max_new_tokens=max_new_tokens,  # Control the number of new tokens generated
        pad_token_id=tokenizer.pad_token_id,  # Explicitly set pad_token_id
    )

    # Decode the generated tokens back into text
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the new response (remove the input and context)
    response = response[len(full_prompt):].strip()

    return response

# Example usage in a chat loop
print("Chat with Mistral-7B! Type 'exit' to end the chat.")
context = ""
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Goodbye!")
        # exit()
        sys.exit()

    # Generate a response
    response = generate_response(user_input, context)
    print(f"Mistral-7B: {response}")

    # Update the context with the new conversation
    context += f"\nYou: {user_input}\nMistral-7B: {response}"
