import gradio as gr
from model_wrapper import generate_response
from voice import transcribe_audio
from training_modules import impromptu_speaking, storytelling, conflict_resolution
from presentation import assess_presentation

def chat_interface(user_input):
    # Provide a conversational prompt with expert feedback instructions.
    prompt = (
        "You are a communication expert. Respond to my input with detailed feedback on clarity, tone, and suggestions for improvement.\n\n"
        f"User: {user_input}\nFeedback:"
    )
    return generate_response(prompt)

def voice_interface(audio_file):
    text = transcribe_audio(audio_file)
    prompt = (
        "You are a communication expert. Provide detailed feedback on the following speech input regarding clarity, tone, and improvement suggestions.\n\n"
        f"User: {text}\nFeedback:"
    )
    return generate_response(prompt)

def presentation_interface(text_input):
    result = assess_presentation(text_input)
    return result["feedback"]

# Training module interfaces
def impromptu_interface():
    return impromptu_speaking()

def storytelling_interface():
    return storytelling()

def conflict_interface():
    return conflict_resolution()

with gr.Blocks() as demo:
    gr.Markdown("# Communication Skills Coach")
    
    with gr.Tabs():
        with gr.Tab("Chat"):
            chat_input = gr.Textbox(label="Enter your message", placeholder="Type your communication challenge here...")
            chat_output = gr.Textbox(label="Coach's Feedback")
            gr.Button("Send").click(fn=chat_interface, inputs=chat_input, outputs=chat_output)
        
        with gr.Tab("Voice"):
            voice_input = gr.Audio(source="upload", type="filepath", label="Upload your audio")
            voice_output = gr.Textbox(label="Coach's Feedback")
            gr.Button("Send Voice").click(fn=voice_interface, inputs=voice_input, outputs=voice_output)
        
        with gr.Tab("Training"):
            gr.Markdown("### Training Modules")
            with gr.Accordion("Impromptu Speaking", open=False):
                impromptu_button = gr.Button("Start Impromptu Speaking")
                impromptu_output = gr.Textbox(label="Feedback")
                impromptu_button.click(fn=impromptu_interface, inputs=[], outputs=impromptu_output)
            with gr.Accordion("Storytelling", open=False):
                storytelling_button = gr.Button("Start Storytelling")
                storytelling_output = gr.Textbox(label="Feedback")
                storytelling_button.click(fn=storytelling_interface, inputs=[], outputs=storytelling_output)
            with gr.Accordion("Conflict Resolution", open=False):
                conflict_button = gr.Button("Start Conflict Resolution")
                conflict_output = gr.Textbox(label="Feedback")
                conflict_button.click(fn=conflict_interface, inputs=[], outputs=conflict_output)
        
        with gr.Tab("Presentation"):
            presentation_input = gr.Textbox(
                label="Enter your presentation script",
                placeholder="Paste your presentation script here..."
            )
            presentation_output = gr.Textbox(label="Feedback Report")
            gr.Button("Assess Presentation").click(fn=presentation_interface, inputs=presentation_input, outputs=presentation_output)

demo.launch()

