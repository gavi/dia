import gradio as gr
import tempfile
import os
from dia.model import Dia

model = None  # Lazy loading to save memory

def load_model():
    global model
    if model is None:
        model = Dia.from_pretrained("nari-labs/Dia-1.6B", compute_dtype="float16")
    return model

def clone_voice(script_to_read, recorded_audio, text_to_generate):
    model = load_model()
    
    # Generate audio with cloned voice - recorded_audio is already a file path
    output = model.generate(
        script_to_read + text_to_generate,
        audio_prompt=recorded_audio,
        use_torch_compile=True,
        verbose=True
    )
    
    # Save and return output audio
    output_path = "output.mp3"
    model.save_audio(output_path, output)
    return output_path

# Create Gradio interface
with gr.Blocks() as app:
    gr.Markdown("# Voice Cloning Demo")
    
    with gr.Row():
        with gr.Column():
            script_input = gr.Textbox(
                label="Script to Read",
                placeholder="Enter the script you will read...",
                lines=3,
                value="[S1] The situation at the border is escalating. Are you prepared for the consequences if this leads to war?"
            )
            gr.Markdown("""
                **Instructions:**
                1. Record yourself reading the script above
                2. Enter new text you want to generate with your voice
                3. Click Generate
            """)
            audio_input = gr.Audio(
                label="Record Your Voice",
                sources=["microphone"],
                type="filepath"
            )
            target_text = gr.Textbox(
                label="Text to Generate",
                placeholder="Enter the text you want to generate with your voice...",
                lines=3
            )
            submit_btn = gr.Button("Generate")
        
        with gr.Column():
            output_audio = gr.Audio(label="Generated Audio")
    
    submit_btn.click(
        fn=clone_voice,
        inputs=[script_input, audio_input, target_text],
        outputs=output_audio
    )

if __name__ == "__main__":
    app.launch() 