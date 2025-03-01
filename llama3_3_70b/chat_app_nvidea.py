import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env
load_dotenv()

# Get the NVIDIA API key from the environment variable
nvidia_key = os.getenv("nvidia_key")
if not nvidia_key:
    raise ValueError("NVIDIA API key not found in environment variables.")

# Create an OpenAI instance configured to hit the NVIDIA API endpoint
client_ai = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=nvidia_key)

def generate_response(prompt, context="", max_new_tokens=100):
    """
    Generates a response by sending the prompt (and any conversation context)
    to the NVIDIA API for Llama-3.3.
    """
    # Build the full prompt with context, appending an indicator for the model's response
    if context:
        full_prompt = context + "\nYou: " + prompt + "\nLlama-3.3:"
    else:
        full_prompt = prompt

    # Prepare the messages payload for the API (using a single user message)
    messages = [{"role": "user", "content": full_prompt}]

    # Call the NVIDIA API (via the OpenAI interface) for the Llama-3.3 model
    response_obj = client_ai.chat.completions.create(
        model="meta/llama-3.3-70b-instruct",
        messages=messages,
        temperature=0.5,
        top_p=1,
        max_tokens=max_new_tokens,
        stream=False
    )

    # Extract and return the generated text response
    response = response_obj.choices[0].message.content.strip()
    return response

# Start a simple chat loop with Llama-3.3
print("Chat with Llama-3.3! Type 'exit' to end the chat.")
context = ""
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    # Generate and display the response
    response = generate_response(user_input, context)
    print(f"Llama-3.3: {response}")

    # Append the latest exchange to the conversation context
    context += f"\nYou: {user_input}\nLlama-3.3: {response}"
