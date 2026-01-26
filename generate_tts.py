import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel
import argparse

def generate_tts(text, language="Auto", speaker="Vivian", instruct="", output_file="output_custom_voice.wav"):
    print(f"Loading model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice...")
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        device_map="cuda:0",
        dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2", # Skipped due to missing CUDA toolkit for compilation
    )

    print(f"Generating audio for text: {text}")
    wavs, sr = model.generate_custom_voice(
        text=text,
        language=language,
        speaker=speaker,
        instruct=instruct,
    )

    sf.write(output_file, wavs[0], sr)
    print(f"Audio saved to {output_file}")
    return output_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3-TTS Generation Script")
    parser.add_argument("--text", type=str, required=True, help="Text to convert to speech")
    parser.add_argument("--language", type=str, default="Auto", help="Language (e.g., Chinese, English, Auto)")
    parser.add_argument("--speaker", type=str, default="Vivian", help="Speaker name")
    parser.add_argument("--instruct", type=str, default="", help="Instruction for the voice (e.g., 'Very happy')")
    parser.add_argument("--output", type=str, default="output.wav", help="Output filename")

    args = parser.parse_args()
    generate_tts(args.text, args.language, args.speaker, args.instruct, args.output)
