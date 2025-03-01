from model_wrapper import generate_response

def assess_presentation(text_input: str) -> dict:
    prompt = (
        "Analyze the following presentation script for structure (intro, body, conclusion), delivery (pacing, use of filler words), "
        "and content (persuasiveness, vocabulary). Provide scores and actionable feedback:\n\n"
        f"{text_input}"
    )
    response = generate_response(prompt)
    # In a production scenario, you might parse the response into structured data.
    return {"feedback": response}

