"""
    > uses vllm for running mistralai/Mistral-7B-v0.1 locally
    > didn't work as the runtime wasn't able to detect the NVIDEA GPU
"""
import os
from vllm import LLM, SamplingParams
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get the Hugging Face token from the environment variable
hf_token = os.getenv("hf_token")

if not hf_token:
    raise ValueError("Hugging Face token not found in environment variables.")

# Set the Hugging Face token as an environment variable for authentication
os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

# Initialize the vLLM model
model = LLM(model="mistralai/Mistral-7B-v0.1")

# Define sampling parameters for generation
sampling_params = SamplingParams(
    max_tokens=100,  # Maximum number of tokens to generate
    n=1,  # Number of output sequences
    stop=None,  # Stop generation at these tokens
    temperature=0.7,  # Sampling temperature
    top_p=0.9,  # Nucleus sampling probability
)

def generate_response(prompt):
    # Generate a response using the vLLM model
    outputs = model.generate(prompt, sampling_params=sampling_params)

    # Extract the generated text from the output
    response = outputs[0].outputs[0].text

    return response

# Example usage in a chat loop
print("Chat with Mistral-7B using vLLM! Type 'exit' to end the chat.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    # Generate a response
    response = generate_response(user_input)
    print(f"Mistral-7B: {response}")
