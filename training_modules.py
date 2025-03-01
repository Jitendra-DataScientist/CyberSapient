from model_wrapper import generate_response
import yaml

def load_training_prompts(config_file="config.yaml"):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config.get("training_prompts", {})

def impromptu_speaking():
    prompts = load_training_prompts()
    topic = prompts.get("impromptu", "Discuss the importance of empathy.")
    prompt = (
        f"Please deliver a brief speech on the following topic: '{topic}'. "
        "After that, provide detailed feedback on your speech focusing on structure, fluency, and clarity."
    )
    return generate_response(prompt)

def storytelling():
    prompts = load_training_prompts()
    story_prompt = prompts.get("storytelling", "Tell a story about a significant life event.")
    prompt = (
        f"Please share your story: '{story_prompt}'. "
        "Then provide feedback on the narrative quality, engagement, and overall flow of the story."
    )
    return generate_response(prompt)

def conflict_resolution():
    prompts = load_training_prompts()
    scenario = prompts.get("conflict_resolution", "Your teammate is frustrated. How do you address this?")
    prompt = (
        f"Imagine the following scenario: '{scenario}'. "
        "Provide your response and then give feedback on the diplomatic nature of your answer, suggesting improvements where needed."
    )
    return generate_response(prompt)

