#!/usr/bin/env python3
"""
XTTS2 Batch Text-to-Speech Processor
- Splits long text into model-friendly chunks
- Generates speech using voice cloning
- Combines outputs and removes silence
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
SPEAKER_WAV = "small.wav"  # Reference audio for voice cloning
TEXT_FILE = "text_to_be_spoken.txt"
OUTPUT_DIR = "/home/jari/PycharmProjects/TTS/Jari/tts_output"
OUTPUT_FILE = "combined_audio.wav"
MAX_CHUNK_LENGTH = 240  # Optimal chunk size for XTTS2 (characters)
SILENCE_THRESHOLD = -40  # dBFS for silence removal
SILENCE_LEN = 100  # ms of silence to detect

# Initialize NLTK resources
nltk.download('punkt', quiet=True)


def clean_text(text):
    """
    Normalize and sanitize input text
    Args:
        text: Raw input text
    Returns:
        str: Cleaned text with standardized formatting
    """
    text = re.sub(r'\s+', ' ', text)  # Collapse multiple whitespaces
    text = re.sub(r'[^A-Za-z0-9 .,!?\'"\-:;()…]', '', text)  # Remove special chars
    return text.strip()


def split_text(text, max_length=MAX_CHUNK_LENGTH):
    """
    Split text into chunks respecting sentence boundaries and max length
    Args:
        text: Input text to split
        max_length: Maximum character length per chunk
    Returns:
        list: Text chunks optimized for TTS processing
    """
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # Account for space that will be added
        projected_length = len(current_chunk) + len(sentence) + 1

        if projected_length > max_length:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""

            # Handle sentences exceeding max_length
            while len(sentence) > max_length:
                # Find optimal split points (prioritizing punctuation)
                split_pos = max(
                    sentence[:max_length].rfind('. '),
                    sentence[:max_length].rfind(', '),
                    sentence[:max_length].rfind('! '),
                    sentence[:max_length].rfind('? '),
                    sentence[:max_length].rfind(' ')
                )

                if split_pos == -1:
                    split_pos = max_length - 1

                chunks.append(sentence[:split_pos].strip() + "...")
                sentence = "..." + sentence[split_pos + 1:].lstrip()

        current_chunk += sentence + " "

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def initialize_model():
    """Initialize and configure XTTS2 model"""
    print("🔄 Initializing XTTS2 model...")
    config = XttsConfig()
    config.load_json(CONFIG_PATH)
    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config,
        checkpoint_dir=CHECKPOINT_DIR,
        use_deepspeed=False
    )
    model.cuda()
    return model


def generate_voice_latents(model):
    """Generate voice conditioning latents from reference audio"""
    print("🔊 Computing voice conditioning latents...")
    return model.get_conditioning_latents(audio_path=[SPEAKER_WAV])


def process_text_chunks(model, latents, chunks):
    """Process text chunks through TTS pipeline"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for i, chunk in enumerate(chunks, 1):
        if len(chunk) > MAX_CHUNK_LENGTH:
            print(f"⚠️ Chunk {i} exceeds limit ({len(chunk)} chars). Truncating...")
            chunk = chunk[:MAX_CHUNK_LENGTH]

        print(f"🎙 Processing chunk {i}/{len(chunks)}...")

        # Generate audio with optimized parameters
        out = model.inference(
            text=chunk,
            language="en",
            gpt_cond_latent=latents[0],
            speaker_embedding=latents[1],
            temperature=0.45,  # Balances creativity and stability
            repetition_penalty=2.0,  # Reduces word repetition
            speed=1.1,  # Slightly faster than default
            enable_text_splitting=False  # Respect our chunk boundaries
        )

        output_path = os.path.join(OUTPUT_DIR, f"chunk_{i:03d}.wav")
        torchaudio.save(
            output_path,
            torch.tensor(out["wav"]).unsqueeze(0),
            24000  # Sample rate
        )
        print(f"✅ Saved: {output_path}")


def combine_and_clean_audio():
    """Combine audio chunks and remove silence"""
    print("\n🔊 Combining audio segments...")

    # Get and sort chunk files numerically
    audio_files = sorted(
        [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".wav")],
        key=lambda f: int(re.search(r'(\d+)', f).group())
    )

    if not audio_files:
        raise FileNotFoundError("No audio chunks found for combining")

    # Initialize with first segment
    combined = AudioSegment.from_wav(os.path.join(OUTPUT_DIR, audio_files[0]))

    # Concatenate remaining segments
    for audio_file in audio_files[1:]:
        combined += AudioSegment.from_wav(os.path.join(OUTPUT_DIR, audio_file))

    # Save raw combined file
    combined.export(OUTPUT_FILE, format="wav")

    # Remove silence and re-export
    print("\n🔇 Removing silence from final output...")
    audio = AudioSegment.from_wav(OUTPUT_FILE)
    audio = audio.strip_silence(
        silence_len=SILENCE_LEN,
        silence_thresh=SILENCE_THRESHOLD
    )
    audio.export(OUTPUT_FILE, format="wav")
    print(f"🎉 Final output saved to: {OUTPUT_FILE}")


def main():
    """Main processing pipeline"""
    # Initialize TTS system
    model = initialize_model()
    voice_latents = generate_voice_latents(model)

    # Process input text
    print("📖 Processing input text...")
    with open(TEXT_FILE, "r", encoding="utf-8") as f:
        raw_text = f.read()

    cleaned_text = clean_text(raw_text)
    chunks = split_text(cleaned_text)

    # Generate speech
    process_text_chunks(model, voice_latents, chunks)

    # Post-process audio
    combine_and_clean_audio()


if __name__ == "__main__":
    main()
