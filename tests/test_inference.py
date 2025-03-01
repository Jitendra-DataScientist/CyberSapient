import pytest
from model_wrapper import generate_response

def test_generate_response():
    prompt = "Hello, how are you?"
    response = generate_response(prompt)
    assert isinstance(response, str)
    assert len(response) > 0

