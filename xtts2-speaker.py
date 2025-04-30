import os
import re
import torch
import torchaudio
from pydub import AudioSegment

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt')


CONFIG_PATH = "./models/xtts2/config.json"
CHECKPOINT_DIR = "./models/xtts2/"
SPEAKER_WAV = "small.wav"
TEXT_FILE = "text_to_be_spoken.txt"
OUTPUT_DIR = "./tts_output"
OUTPUT_FILE = "combined_audio.wav"
MAX_CHUNK_LENGTH = 240


def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with one
    text = re.sub(r'[^A-Za-z0-9 .,!?\'"\-:;()…]', '', text)  # Keep essential punctuation
    return text.strip()


def split_text(text, max_length=240):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # Calculate total length if we add this sentence + space
        new_length = len(current_chunk) + len(sentence) + 1

        if new_length > max_length:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""

            # Process oversized sentence
            while len(sentence) > max_length:
                # Prefer splitting at punctuation
                split_pos = max(
                    sentence[:max_length].rfind('. '),
                    sentence[:max_length].rfind(', '),
                    sentence[:max_length].rfind('! '),
                    sentence[:max_length].rfind('? ')
                )

                if split_pos == -1:
                    split_pos = sentence[:max_length].rfind(' ')

                if split_pos == -1:
                    split_pos = max_length - 1

                # Add ellipsis to imply continuation
                chunk_part = sentence[:split_pos].strip() + "..."
                chunks.append(chunk_part)
                sentence = "..." + sentence[split_pos + 1:].lstrip()  # Avoid double spaces

        # Add sentence to current chunk (with space)
        current_chunk += sentence + " "

    # Add the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks

print("🔄 Ladataan XTTS2-malli...")
config = XttsConfig()
config.load_json(CONFIG_PATH)
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir=CHECKPOINT_DIR, use_deepspeed=False)
model.cuda()


print("🔊 Lasketaan voice clone -latentit...")
gpt_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[SPEAKER_WAV])


print("📖 Luetaan ja pilkotaan teksti...")
with open(TEXT_FILE, "r", encoding="utf-8") as f:
    raw_text = f.read()

cleaned_text = clean_text(raw_text)
chunks = split_text(cleaned_text, MAX_CHUNK_LENGTH)


os.makedirs(OUTPUT_DIR, exist_ok=True)


for i, chunk in enumerate(chunks, 1):
    if len(chunk) > MAX_CHUNK_LENGTH:
        print(f"⚠️ Chunk {i} is too long ({len(chunk)} chars). Truncating...")
        chunk = chunk[:MAX_CHUNK_LENGTH]
    print(f"🎙 Generoidaan osa {i}/{len(chunks)}...")

    out = model.inference(
        text=chunk,
        language="en",
        gpt_cond_latent=gpt_latent,
        speaker_embedding=speaker_embedding,
        temperature=0.45,
        repetition_penalty=2.0,
        speed=1.1,  # Slightly faster speech
        enable_text_splitting=False  # Force the model to process your splits as-is
    )

    output_path = os.path.join(OUTPUT_DIR, f"osa_{i}.wav")
    torchaudio.save(output_path, torch.tensor(out["wav"]).unsqueeze(0), 24000)

    print(f"✅ Tallennettu: {output_path}")

print("\n🎉 Valmista! Kaikki osat on generoitu kansioon:", OUTPUT_DIR)


# Haetaan kaikki WAV-tiedostot kansiosta
audio_files = [f for f in os.listdir(OUTPUT_DIR ) if f.endswith(".wav")]

# Järjestetään tiedostot numeroinnilla (osa_1.wav, osa_2.wav -> osa_1, osa_2 ...)
audio_files.sort(key=lambda f: int(re.search(r'(\d+)', f).group()))

# Ladataan ensimmäinen tiedosto
combined = AudioSegment.from_wav(os.path.join(OUTPUT_DIR , audio_files[0]))

# Yhdistetään kaikki tiedostot
for audio_file in audio_files[1:]:
    sound = AudioSegment.from_wav(os.path.join(OUTPUT_DIR , audio_file))
    combined += sound  # Yhdistetään tiedostot

# Tallennetaan yhdistetty tiedosto
combined.export(OUTPUT_FILE, format="wav")

# Poistetaan tyhjät
audio = AudioSegment.from_wav(OUTPUT_FILE)
audio = audio.strip_silence(silence_len=100, silence_thresh=-40)
audio.export(OUTPUT_FILE, format="wav")
