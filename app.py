import torch
import soundfile as sf
import gradio as gr
from qwen_tts import Qwen3TTSModel
import os
import gc
import shutil
import time
from pydub import AudioSegment

# Global variables to manage models
CUSTOM_VOICE_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
BASE_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
VOICE_DESIGN_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"

current_model = None
current_model_id = None

CUSTOM_VOICES_DIR = "/home/mlabs/Documents/dev/tts/custom_voices"
os.makedirs(CUSTOM_VOICES_DIR, exist_ok=True)

SESSION_DIR = "/home/mlabs/Documents/dev/tts/session_outputs"
os.makedirs(SESSION_DIR, exist_ok=True)

STYLE_TAGS = {
    "Whispered": "(whispering)",
    "Shouting": "(shouting)",
    "Excited": "(excitedly)",
    "Happy": "(happily)",
    "Serious": "(seriously)",
    "Slow": "(slowly)",
    "Fast": "(quickly)",
}

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

def convert_to_mp3(wav_path, mp3_path):
    try:
        audio = AudioSegment.from_wav(wav_path)
        audio.export(mp3_path, format="mp3")
        return mp3_path
    except Exception as e:
        print(f"MP3 Conversion Error: {e}")
        return wav_path

def tts_preset(text, language, speaker, instruct, output_format):
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
            
            if output_format == "mp3":
                mp3_path = "temp_preset.mp3"
                output_path = convert_to_mp3(output_path, mp3_path)
                
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
            
            if output_format == "mp3":
                mp3_path = "temp_preset.mp3"
                output_path = convert_to_mp3(output_path, mp3_path)
                
            return output_path, "Generation successful!"
        except Exception as e:
            return None, f"Error: {str(e)}"
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

def tts_voice_design(text, instruct, language, output_format):
    global current_model
    if not text or not instruct:
        return None, "Please provide both text and voice design instructions."
    
    # Ensure VoiceDesign model is loaded
    load_model(VOICE_DESIGN_MODEL_ID)
    
    lang = language if language else "Auto"
    
    try:
        with torch.no_grad():
            wavs, sr = current_model.generate_voice_design(
                text=text.strip(),
                language=lang,
                instruct=instruct.strip()
            )
        
        output_path = "temp_design.wav"
        sf.write(output_path, wavs[0], sr)
        
        if output_format == "mp3":
            mp3_path = "temp_design.mp3"
            output_path = convert_to_mp3(output_path, mp3_path)
            
        return output_path, "Dialogue generation successful!"
    except Exception as e:
        return None, f"Error: {str(e)}"
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def get_instruct_ids(instruct, model):
    if not instruct:
        return None
    # Follow the format used in Qwen3TTSModel._build_instruct_text
    prompt = f"<|im_start|>user\n{instruct}<|im_end|>\n"
    inputs = model.processor(text=prompt, return_tensors="pt", padding=True)
    target_device = getattr(model, "device", "cuda:0")
    return [inputs["input_ids"].to(target_device)]

def tts_clone(text, language, ref_audio, ref_text, output_format, instruct):
    global current_model
    if not text or ref_audio is None:
        return None, "Please provide text and reference audio."
    
    # Ensure Base model is loaded for cloning
    load_model(BASE_MODEL_ID)
    
    lang = language if language else "Auto"
    
    try:
        # Prepare instruction IDs if provided
        instruct_ids = get_instruct_ids(instruct, current_model)
        
        # ref_audio in Gradio is typically a filepath
        with torch.no_grad():
            wavs, sr = current_model.generate_voice_clone(
                text=text,
                language=lang,
                ref_audio=ref_audio,
                ref_text=ref_text if ref_text else "",
                instruct_ids=instruct_ids
            )
        
        output_path = "temp_clone.wav"
        sf.write(output_path, wavs[0], sr)
        
        if output_format == "mp3":
            mp3_path = "temp_clone.mp3"
            output_path = convert_to_mp3(output_path, mp3_path)
            
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

def add_to_history(new_audio, history):
    if not new_audio:
        return history
    
    # Create a unique copy in session_outputs
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    ext = os.path.splitext(new_audio)[1]
    filename = f"gen_{timestamp}{ext}"
    dest_path = os.path.join(SESSION_DIR, filename)
    shutil.copy(new_audio, dest_path)
    
    # Store the path in state
    history.append(dest_path)
    return history

# Custom CSS for the scrollable grid
custom_css = """
.history-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 10px;
    max-height: 400px;
    overflow-y: auto;
    padding: 10px;
    border: 1px solid #ddd;
    border-radius: 8px;
}
.history-card {
    padding: 8px;
    background: #2a2a2a;
    border-radius: 6px;
    display: flex;
    flex-direction: column;
    gap: 5px;
}
.history-card p {
    margin: 0;
    font-size: 0.8em;
    color: #ccc;
    word-break: break-all;
}
"""



# Get supported speakers and languages for Preset UI
preset_speakers = [s.lower() for s in current_model.get_supported_speakers()]
preset_languages = current_model.get_supported_languages()

with gr.Blocks(title="Qwen3-TTS WebUI (Multi-Mode)") as demo:
    gr.Markdown("# Qwen3-TTS WebUI")
    status_msg = gr.Markdown("Current Model: " + CUSTOM_VOICE_MODEL_ID)
    history_state = gr.State([])
    
    def get_actual_speaker(preset, custom):
        if custom and custom != "- None -":
            return custom
        return preset

    def update_custom_dropdown():
        custom_voices = ["- None -"] + get_custom_voices()
        return gr.Dropdown(choices=custom_voices, value="- None -")

    def insert_tag(text, tag):
        if not text:
            return tag + " "
        if text.endswith(" "):
            return text + tag + " "
        return text + " " + tag + " "

    def render_history_grid(history_list):
        if not history_list:
            return
        gr.Markdown("### Session History")
        with gr.Group(elem_classes="history-grid"):
            for path in reversed(history_list):
                with gr.Group(elem_classes="history-card"):
                    gr.Markdown(f"**{os.path.basename(path)}**")
                    # Using a minimal audio player. Gradio's gr.Audio is somewhat chunky, 
                    # but setting labels and interactive=False makes it cleaner.
                    gr.Audio(path, show_label=False, container=False)

    with gr.Tabs():
        # TAB 1: PRESET VOICES
        with gr.Tab("Preset Voices"):
            with gr.Row():
                with gr.Column():
                    p_text = gr.Textbox(label="Text to Speak", placeholder="Enter text here...", lines=3)
                    with gr.Row():
                        for label, tag in STYLE_TAGS.items():
                            btn = gr.Button(label, size="sm", variant="secondary")
                            btn.click(fn=insert_tag, inputs=[p_text, gr.State(tag)], outputs=[p_text])
                    
                    p_lang = gr.Dropdown(choices=["Auto"] + preset_languages, label="Language", value="Auto")
                    with gr.Row():
                        p_speaker = gr.Dropdown(choices=preset_speakers, label="Preset Voices", value="vivian")
                        p_custom_speaker = gr.Dropdown(choices=["- None -"] + get_custom_voices(), label="Custom Voices", value="- None -")
                    p_instruct = gr.Textbox(label="Instruction (Optional)", placeholder="e.g., Very happy, whispered...")
                    p_format = gr.Radio(choices=["wav", "mp3"], label="Output Format", value="wav")
                    p_gen_btn = gr.Button("Generate Audio", variant="primary")
                    p_delete_btn = gr.Button("Delete Custom Voice", variant="secondary")
                with gr.Column():
                    p_audio = gr.Audio(label="Generated Audio", type="filepath")
                    p_info = gr.Textbox(label="Status Info", interactive=False)
                    
                    @gr.render(inputs=history_state)
                    def render_p_history(history_list):
                        render_history_grid(history_list)

        # TAB 2: VOICE CLONE
        with gr.Tab("Voice Clone (Zero-Shot)"):
            gr.Markdown("Upload a short clip of the voice you want to clone.")
            with gr.Row():
                with gr.Column():
                    c_text = gr.Textbox(label="Text to Speak", placeholder="Enter text here...", lines=3)
                    with gr.Row():
                        for label, tag in STYLE_TAGS.items():
                            btn = gr.Button(label, size="sm", variant="secondary")
                            btn.click(fn=insert_tag, inputs=[c_text, gr.State(tag)], outputs=[c_text])
                            
                    c_lang = gr.Dropdown(choices=["Auto", "English", "Chinese", "Japanese", "Korean"], label="Language", value="Auto")
                    c_ref_audio = gr.Audio(label="Reference Audio (WAV/MP3)", type="filepath", sources=["upload", "microphone"], interactive=True)
                    c_ref_text = gr.Textbox(label="Reference Transcript (Optional but recommended)", placeholder="What is being said in the reference audio?")
                    c_save_name = gr.Textbox(label="Voice Name (to save)", placeholder="e.g., MyAwesomeVoice")
                    c_save_btn = gr.Button("Save Voice Model", variant="secondary")
                    c_instruct = gr.Textbox(label="Instruction (Optional)", placeholder="e.g., Speak with a Cantonese accent, or aggressive style...")
                    c_format = gr.Radio(choices=["wav", "mp3"], label="Output Format", value="wav")
                    c_gen_btn = gr.Button("Clone & Generate", variant="primary")
                with gr.Column():
                    c_audio = gr.Audio(label="Generated Audio", type="filepath")
                    c_info = gr.Textbox(label="Status Info", interactive=False)
                    
                    @gr.render(inputs=history_state)
                    def render_c_history(history_list):
                        render_history_grid(history_list)

        # TAB 3: DIALOGUE (VOICE DESIGN)
        with gr.Tab("Dialogue (Timbre Reuse)"):
            gr.Markdown("### Create Multi-Character Dialogues with Persistent Timbres")
            with gr.Row():
                with gr.Column():
                    d_text = gr.Textbox(
                        label="Dialogue Script", 
                        placeholder="CharacterName: Dialogue text...", 
                        lines=10,
                        value='Lucas:H-hey! You dropped your... uh... calculus notebook? I mean, I think it\'s yours? Maybe?\nMia:Oh wow, my mortal enemy - Mr. Thompson\'s problem sets. Thanks for rescuing me from that F.\nLucas:No problem! I actually... kinda finished those already? If you want to compare answers or something...\nMia:Is this your sneaky way of saying you want to study together, Lucas? Because I saw you staring during lab partners sign-up.'
                    )
                    d_instruct = gr.Textbox(
                        label="Character Timbre Definitions", 
                        placeholder='"CharacterName": "Description..."', 
                        lines=5,
                        value='"Lucas": "Male, 17 years old, tenor range, gaining confidence - deeper breath support now, though vowels still tighten when nervous"\n"Mia": "Female, 16 years old, mezzo-soprano range, softening - lowering register to intimate speaking voice, consonants softening"'
                    )
                    d_lang = gr.Dropdown(choices=["Auto"] + preset_languages, label="Language", value="Auto")
                    d_format = gr.Radio(choices=["wav", "mp3"], label="Output Format", value="wav")
                    d_gen_btn = gr.Button("Generate Dialogue", variant="primary")
                with gr.Column():
                    d_audio = gr.Audio(label="Generated Audio", type="filepath")
                    d_info = gr.Textbox(label="Status Info", interactive=False)
                    
                    @gr.render(inputs=history_state)
                    def render_d_history(history_list):
                        render_history_grid(history_list)

    with gr.Row():
        exit_btn = gr.Button("Exit / Stop Server", variant="stop")

    # Callbacks
    p_gen_btn.click(
        fn=lambda t, l, ps, cs, i, f: tts_preset(t, l, get_actual_speaker(ps, cs), i, f),
        inputs=[p_text, p_lang, p_speaker, p_custom_speaker, p_instruct, p_format],
        outputs=[p_audio, p_info]
    ).then(
        fn=add_to_history,
        inputs=[p_audio, history_state],
        outputs=history_state
    ).then(
        fn=lambda: "Current Model: " + current_model_id,
        outputs=status_msg
    )

    p_delete_btn.click(
        fn=delete_voice,
        inputs=[p_custom_speaker],
        outputs=[p_info]
    ).then(
        fn=update_custom_dropdown,
        outputs=[p_custom_speaker]
    )

    c_gen_btn.click(
        fn=tts_clone,
        inputs=[c_text, c_lang, c_ref_audio, c_ref_text, c_format, c_instruct],
        outputs=[c_audio, c_info]
    ).then(
        fn=add_to_history,
        inputs=[c_audio, history_state],
        outputs=history_state
    ).then(
        fn=lambda: "Current Model: " + current_model_id,
        outputs=status_msg
    )

    c_save_btn.click(
        fn=save_voice,
        inputs=[c_save_name, c_ref_audio, c_ref_text],
        outputs=[c_info]
    ).then(
        fn=update_custom_dropdown,
        outputs=[p_custom_speaker]
    )

    d_gen_btn.click(
        fn=tts_voice_design,
        inputs=[d_text, d_instruct, d_lang, d_format],
        outputs=[d_audio, d_info]
    ).then(
        fn=add_to_history,
        inputs=[d_audio, history_state],
        outputs=history_state
    ).then(
        fn=lambda: "Current Model: " + current_model_id,
        outputs=status_msg
    )

    exit_btn.click(fn=shutdown)

if __name__ == "__main__":
    # share=True creates a public HTTPS link, which often fixes microphone permission issues
    demo.launch(share=True, inbrowser=True, css=custom_css)
