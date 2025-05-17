#!/usr/bin/env python3
"""
XTTS2 Batch Text-to-Speech Processor with Enhanced Chunk Handling
"""

import os
import re
import torch
import torchaudio
from pydub import AudioSegment
from nltk.tokenize import sent_tokenize
import nltk
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# Configuration constants
CONFIG_PATH = "/home/jari/PycharmProjects/TTS/models/xtts2/config.json"
CHECKPOINT_DIR = "/home/jari/PycharmProjects/TTS/models/xtts2/"
SPEAKER_WAV = "small.wav"
TEXT_FILE = "text_to_be_spoken.txt"
OUTPUT_DIR = "/home/jari/PycharmProjects/TTS/Jari/tts_output"
OUTPUT_FILE = "combined_audio.wav"
MAX_CHUNK_LENGTH = 200  # Reduced for safety margin
SILENCE_THRESHOLD = -35
SILENCE_LEN = 500
MAX_SENTENCES_PER_CHUNK = 4

nltk.download('punkt', quiet=True)


def clean_text(text):
    """Enhanced text cleaning with punctuation spacing"""
    text = re.sub(r'(?<=[.,!?])(?=[^\s])', r' ', text)  # Fix missing spaces
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^A-Za-z0-9 .,!?\'"\-:;()…]', '', text)
    return text.strip()


def split_text(text, max_length=MAX_CHUNK_LENGTH):
    """Safer splitting with sentence limits"""
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    sentence_count = 0

    for sentence in sentences:
        projected_length = len(current_chunk) + len(sentence) + 1
        sentence_count += 1

        if (projected_length > max_length) or (sentence_count > MAX_SENTENCES_PER_CHUNK):
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
                sentence_count = 0

        current_chunk += sentence + " "

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    print(f"\n🔍 Split into {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}: {len(chunk)} chars")
    return chunks


def initialize_model():
    print("🔄 Initializing XTTS2 model...")
    config = XttsConfig()
    config.load_json(CONFIG_PATH)
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=CHECKPOINT_DIR, use_deepspeed=False)
    model.cuda()
    return model


def generate_voice_latents(model):
    print("🔊 Computing voice conditioning latents...")
    return model.get_conditioning_latents(audio_path=[SPEAKER_WAV])


def validate_audio(output_path, chunk):
    """Validate audio duration meets expectations"""
    audio = AudioSegment.from_wav(output_path)
    duration = len(audio) / 1000  # Convert to seconds
    min_expected = len(chunk) * 0.055  # 55ms per character minimum

    if duration < min_expected:
        print(f"⚠️ Short audio: {duration:.1f}s (expected >{min_expected:.1f}s)")
        return False
    return True


def process_text_chunks(model, latents, chunks):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for i, chunk in enumerate(chunks, 1):
        print(f"\n📄 Processing chunk {i}/{len(chunks)}:")
        print(chunk)
        print(f"Character count: {len(chunk)}/{MAX_CHUNK_LENGTH}")

        if len(chunk) > MAX_CHUNK_LENGTH:
            print(f"⚠️ Truncating {len(chunk) - MAX_CHUNK_LENGTH} characters")
            chunk = chunk[:MAX_CHUNK_LENGTH]

        # Initial generation attempt
        out = model.inference(
            text=chunk,
            language="en",
            gpt_cond_latent=latents[0],
            speaker_embedding=latents[1],
            temperature=0.65,
            repetition_penalty=1.5,
            speed=1.0,
            length_penalty=1.0,
            enable_text_splitting=False
        )

        output_path = os.path.join(OUTPUT_DIR, f"chunk_{i:03d}.wav")
        torchaudio.save(output_path, torch.tensor(out["wav"]).unsqueeze(0), 24000)

        # Validation and retry if needed
        if not validate_audio(output_path, chunk):
            print("🔄 Retrying with adjusted parameters...")
            out = model.inference(
                text=chunk,
                language="en",
                gpt_cond_latent=latents[0],
                speaker_embedding=latents[1],
                temperature=0.75,
                repetition_penalty=1.3,
                speed=0.95,
                length_penalty=1.2,
                enable_text_splitting=False
            )
            torchaudio.save(output_path, torch.tensor(out["wav"]).unsqueeze(0), 24000)

        print(f"✅ Saved: {output_path}")


def combine_and_clean_audio():
    print("\n🔊 Combining audio segments...")
    audio_files = sorted(
        [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".wav")],
        key=lambda f: int(re.search(r'(\d+)', f).group())
    )

    combined = AudioSegment.from_wav(os.path.join(OUTPUT_DIR, audio_files[0]))
    for audio_file in audio_files[1:]:
        combined += AudioSegment.from_wav(os.path.join(OUTPUT_DIR, audio_file))

    combined.export(OUTPUT_FILE, format="wav")

    print("\n🔇 Removing silence...")
    audio = AudioSegment.from_wav(OUTPUT_FILE)
    audio = audio.strip_silence(
        silence_len=SILENCE_LEN,
        silence_thresh=SILENCE_THRESHOLD,
        padding=100
    )
    audio.export(OUTPUT_FILE, format="wav")
    print(f"🎉 Final output: {OUTPUT_FILE}")

    return audio_files


def cleanup_temp_files(temp_files):
    print("\n🧹 Cleaning temporary files...")
    deleted_count = 0
    for temp_file in temp_files:
        try:
            os.remove(os.path.join(OUTPUT_DIR, temp_file))
            deleted_count += 1
        except Exception as e:
            print(f"⚠️ Error deleting {temp_file}: {str(e)}")
    print(f"🗑️ Removed {deleted_count}/{len(temp_files)} files")


def main():
    model = initialize_model()
    voice_latents = generate_voice_latents(model)

    with open(TEXT_FILE, "r", encoding="utf-8") as f:
        raw_text = f.read()

    cleaned_text = clean_text(raw_text)
    chunks = split_text(cleaned_text)

    process_text_chunks(model, voice_latents, chunks)
    temp_files = combine_and_clean_audio()
    cleanup_temp_files(temp_files)


if __name__ == "__main__":
    main()
