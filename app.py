import torch
import soundfile as sf
import gradio as gr
from qwen_tts import Qwen3TTSModel
import os
import gc
import shutil

# Global variables to manage models
CUSTOM_VOICE_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
BASE_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

current_model = None
current_model_id = None

CUSTOM_VOICES_DIR = "/home/mlabs/Documents/dev/tts/custom_voices"
os.makedirs(CUSTOM_VOICES_DIR, exist_ok=True)

def get_custom_voices():
    if not os.path.exists(CUSTOM_VOICES_DIR):
        return []
    return [os.path.splitext(f)[0] for f in os.listdir(CUSTOM_VOICES_DIR) if f.endswith(".wav")]

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
    
    custom_voices = get_custom_voices()
    
    if speaker in custom_voices:
        # Custom voice selected - use Base model for cloning logic
        load_model(BASE_MODEL_ID)
        ref_audio = os.path.join(CUSTOM_VOICES_DIR, f"{speaker}.wav")
        ref_text_path = os.path.join(CUSTOM_VOICES_DIR, f"{speaker}.txt")
        ref_text = ""
        if os.path.exists(ref_text_path):
            try:
                with open(ref_text_path, "r", encoding="utf-8") as f:
                    ref_text = f.read().strip()
            except Exception as e:
                print(f"Warning: Could not read ref_text for {speaker}: {e}")

        try:
            with torch.no_grad():
                # If ref_text is empty, we must use x_vector_only_mode=True
                # Otherwise, we use x_vector_only_mode=False (ICL mode)
                x_vector_only = not bool(ref_text)
                wavs, sr = current_model.generate_voice_clone(
                    text=text,
                    language=language if language != "Auto" else "Auto",
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                    x_vector_only_mode=x_vector_only
                )
            output_path = "temp_preset.wav"
            sf.write(output_path, wavs[0], sr)
            return output_path, f"Generation with custom voice '{speaker}' successful! (Mode: {'X-Vector' if x_vector_only else 'ICL'})"
        except Exception as e:
            return None, f"Error with custom voice: {str(e)}"
    else:
        # Standard preset speaker - use CustomVoice model
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

def save_voice(name, ref_audio, ref_text):
    if not name or ref_audio is None:
        return "Please provide a name and reference audio to save."
    
    name = name.strip().replace(" ", "_")
    dest_path = os.path.join(CUSTOM_VOICES_DIR, f"{name}.wav")
    text_path = os.path.join(CUSTOM_VOICES_DIR, f"{name}.txt")
    
    try:
        shutil.copy(ref_audio, dest_path)
        if ref_text:
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(ref_text)
        return f"Voice '{name}' saved successfully!"
    except Exception as e:
        return f"Error saving voice: {str(e)}"

def delete_voice(name):
    if not name:
        return "No voice selected."
    
    custom_voices = get_custom_voices()
    if name not in custom_voices:
        return f"'{name}' is a preset voice and cannot be deleted."
    
    try:
        wav_path = os.path.join(CUSTOM_VOICES_DIR, f"{name}.wav")
        txt_path = os.path.join(CUSTOM_VOICES_DIR, f"{name}.txt")
        if os.path.exists(wav_path):
            os.remove(wav_path)
        if os.path.exists(txt_path):
            os.remove(txt_path)
        return f"Voice '{name}' deleted successfully!"
    except Exception as e:
        return f"Error deleting voice: {str(e)}"

def update_speaker_dropdown():
    base_speakers = [s.lower() for s in current_model.get_supported_speakers()]
    custom_voices = get_custom_voices()
    all_speakers = base_speakers + custom_voices
    return gr.Dropdown(choices=all_speakers)

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
                    p_speaker = gr.Dropdown(choices=preset_speakers + get_custom_voices(), label="Speaker", value="vivian")
                    p_instruct = gr.Textbox(label="Instruction (Optional)", placeholder="e.g., Very happy, whispered...")
                    p_gen_btn = gr.Button("Generate Audio", variant="primary")
                    p_delete_btn = gr.Button("Delete Voice", variant="secondary")
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
                    c_save_name = gr.Textbox(label="Voice Name (to save)", placeholder="e.g., MyAwesomeVoice")
                    c_save_btn = gr.Button("Save Voice Model", variant="secondary")
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

    p_delete_btn.click(
        fn=delete_voice,
        inputs=[p_speaker],
        outputs=[p_info]
    ).then(
        fn=update_speaker_dropdown,
        outputs=[p_speaker]
    )

    c_gen_btn.click(
        fn=tts_clone,
        inputs=[c_text, c_lang, c_ref_audio, c_ref_text],
        outputs=[c_audio, c_info]
    ).then(
        fn=lambda: "Current Model: " + current_model_id,
        outputs=status_msg
    )

    c_save_btn.click(
        fn=save_voice,
        inputs=[c_save_name, c_ref_audio, c_ref_text],
        outputs=[c_info]
    ).then(
        fn=update_speaker_dropdown,
        outputs=[p_speaker]
    )

    exit_btn.click(fn=shutdown)

if __name__ == "__main__":
    # share=True creates a public HTTPS link, which often fixes microphone permission issues
    demo.launch(share=True, inbrowser=True)
