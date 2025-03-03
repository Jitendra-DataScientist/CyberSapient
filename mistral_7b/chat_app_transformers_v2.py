"""
    uses mistralai/Mistral-7B-v0.2 running locally
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
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", token=hf_token)
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", token=hf_token, device_map="auto")

# Set pad_token_id if not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def generate_response(messages, max_new_tokens=100):

    # Tokenize the input prompt
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")

    # Generate a response using the model
    # outputs = model.generate(
    #     inputs["input_ids"],
    #     attention_mask=inputs["attention_mask"],  # Pass the attention mask
    #     max_new_tokens=max_new_tokens,  # Control the number of new tokens generated
    #     pad_token_id=tokenizer.pad_token_id,  # Explicitly set pad_token_id
    # )
    outputs = model.generate(
        inputs,
        attention_mask=inputs["attention_mask"],  # Pass the attention mask
        max_new_tokens=max_new_tokens,  # Control the number of new tokens generated
        pad_token_id=tokenizer.pad_token_id,  # Explicitly set pad_token_id
    )

    # Decode the generated tokens back into text
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the new response (remove the input and context)
    # response = response[len(full_prompt):].strip()

    return response

# Example usage in a chat loop
print("Chat with Mistral-7B! Type 'exit' to end the chat.")
messages = []
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Goodbye!")
        exit()
    messages.append({"role": "user", "content": user_input})
    # Generate a response
    response = generate_response(messages)
    print(f"Mistral-7B: {response}")
    messages.append({"role": "assistant", "content": response})
