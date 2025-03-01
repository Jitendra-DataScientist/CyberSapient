import whisper

# Load the Whisper model once; adjust the model size (e.g., "base", "small", "medium") as needed.
whisper_model = whisper.load_model("base")

def transcribe_audio(audio_path: str) -> str:
    result = whisper_model.transcribe(audio_path)
    return result['text']

