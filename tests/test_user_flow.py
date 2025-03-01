import pytest
from app import chat_interface

def test_chat_flow():
    user_input = "I need help with my presentation."
    feedback = chat_interface(user_input)
    assert isinstance(feedback, str)
    assert len(feedback) > 0

