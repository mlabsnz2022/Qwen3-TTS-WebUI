import torch
import soundfile as sf
import gradio as gr
from qwen_tts import Qwen3TTSModel
import os
import gc

# Global variables to manage models
CUSTOM_VOICE_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
BASE_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

current_model = None
current_model_id = None

def load_model(model_id):
    global current_model, current_model_id
    
    if current_model_id == model_id:
        return current_model, "Model already loaded."
    
    print(f"Loading model {model_id}...")
    
    # Unload previous model
    if current_model is not None:
        del current_model
        current_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("Previous model unloaded.")

    # Load new model
    current_model = Qwen3TTSModel.from_pretrained(
        model_id,
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )
    current_model_id = model_id
    return current_model, f"Successfully loaded {model_id}"

# Initial load (CustomVoice)
load_model(CUSTOM_VOICE_MODEL_ID)

def tts_preset(text, language, speaker, instruct):
    global current_model
    if not text:
        return None, "Please enter some text."
    
    # Ensure correct model is loaded
    load_model(CUSTOM_VOICE_MODEL_ID)
    
    lang = language if language else "Auto"
    
    try:
        with torch.no_grad():
            wavs, sr = current_model.generate_custom_voice(
                text=text,
                language=lang,
                speaker=speaker,
                instruct=instruct if instruct else "",
            )
        
        output_path = "temp_preset.wav"
        sf.write(output_path, wavs[0], sr)
        return output_path, "Generation successful!"
    except Exception as e:
        return None, f"Error: {str(e)}"
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def tts_clone(text, language, ref_audio, ref_text):
    global current_model
    if not text or ref_audio is None:
        return None, "Please provide text and reference audio."
    
    # Ensure Base model is loaded for cloning
    load_model(BASE_MODEL_ID)
    
    lang = language if language else "Auto"
    
    try:
        # ref_audio in Gradio is typically a filepath
        with torch.no_grad():
            wavs, sr = current_model.generate_voice_clone(
                text=text,
                language=lang,
                ref_audio=ref_audio,
                ref_text=ref_text if ref_text else "",
            )
        
        output_path = "temp_clone.wav"
        sf.write(output_path, wavs[0], sr)
        return output_path, "Cloning successful!"
    except Exception as e:
        return None, f"Error: {str(e)}"
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def shutdown():
    print("Shutting down Gradio server...")
    os._exit(0)

# Get supported speakers and languages for Preset UI
preset_speakers = [s.lower() for s in current_model.get_supported_speakers()]
preset_languages = current_model.get_supported_languages()

with gr.Blocks(title="Qwen3-TTS WebUI (Multi-Mode)") as demo:
    gr.Markdown("# Qwen3-TTS WebUI")
    status_msg = gr.Markdown("Current Model: " + CUSTOM_VOICE_MODEL_ID)
    
    with gr.Tabs():
        # TAB 1: PRESET VOICES
        with gr.Tab("Preset Voices"):
            with gr.Row():
                with gr.Column():
                    p_text = gr.Textbox(label="Text to Speak", placeholder="Enter text here...", lines=3)
                    p_lang = gr.Dropdown(choices=["Auto"] + preset_languages, label="Language", value="Auto")
                    p_speaker = gr.Dropdown(choices=preset_speakers, label="Speaker", value="vivian")
                    p_instruct = gr.Textbox(label="Instruction (Optional)", placeholder="e.g., Very happy, whispered...")
                    p_gen_btn = gr.Button("Generate Audio", variant="primary")
                with gr.Column():
                    p_audio = gr.Audio(label="Generated Audio", type="filepath")
                    p_info = gr.Textbox(label="Status Info", interactive=False)

        # TAB 2: VOICE CLONE
        with gr.Tab("Voice Clone (Zero-Shot)"):
            gr.Markdown("Upload a short clip of the voice you want to clone.")
            with gr.Row():
                with gr.Column():
                    c_text = gr.Textbox(label="Text to Speak", placeholder="Enter text here...", lines=3)
                    c_lang = gr.Dropdown(choices=["Auto", "English", "Chinese", "Japanese", "Korean"], label="Language", value="Auto")
                    c_ref_audio = gr.Audio(label="Reference Audio (WAV/MP3)", type="filepath", sources=["upload", "microphone"], interactive=True)
                    c_ref_text = gr.Textbox(label="Reference Transcript (Optional but recommended)", placeholder="What is being said in the reference audio?")
                    c_gen_btn = gr.Button("Clone & Generate", variant="primary")
                with gr.Column():
                    c_audio = gr.Audio(label="Generated Audio", type="filepath")
                    c_info = gr.Textbox(label="Status Info", interactive=False)

    with gr.Row():
        exit_btn = gr.Button("Exit / Stop Server", variant="stop")

    # Callbacks
    p_gen_btn.click(
        fn=tts_preset,
        inputs=[p_text, p_lang, p_speaker, p_instruct],
        outputs=[p_audio, p_info]
    ).then(
        fn=lambda: "Current Model: " + current_model_id,
        outputs=status_msg
    )

    c_gen_btn.click(
        fn=tts_clone,
        inputs=[c_text, c_lang, c_ref_audio, c_ref_text],
        outputs=[c_audio, c_info]
    ).then(
        fn=lambda: "Current Model: " + current_model_id,
        outputs=status_msg
    )

    exit_btn.click(fn=shutdown)

if __name__ == "__main__":
    # share=True creates a public HTTPS link, which often fixes microphone permission issues
    demo.launch(share=True, inbrowser=True)
