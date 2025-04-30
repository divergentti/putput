import os
import re
import torch
import torchaudio

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
MAX_CHUNK_LENGTH = 250


def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with one
    text = re.sub(r'[^A-Za-z0-9 .,!?\'"\-:;()…]', '', text)  # Keep essential punctuation
    return text.strip()


def split_text(text, max_length=250):
    sentences = nltk.sent_tokenize(text)
    chunks, current_chunk, current_len = [], [], 0
    for sentence in sentences:
        sentence_len = len(sentence)
        if current_len + sentence_len > max_length:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk, current_len = [], 0
            # Handle sentences longer than max_length
            while sentence_len > max_length:
                split_index = sentence[:max_length].rfind(' ')
                chunks.append(sentence[:split_index])
                sentence = sentence[split_index+1:]
                sentence_len = len(sentence)
        current_chunk.append(sentence)
        current_len += sentence_len
    if current_chunk:
        chunks.append(" ".join(current_chunk))
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
    print(f"🎙 Generoidaan osa {i}/{len(chunks)}...")

    out = model.inference(
        text=chunk,
        language="en",
        gpt_cond_latent=gpt_latent,
        speaker_embedding=speaker_embedding,
        temperature=0.3,
        repetition_penalty=2.0
    )

    output_path = os.path.join(OUTPUT_DIR, f"osa_{i}.wav")
    torchaudio.save(output_path, torch.tensor(out["wav"]).unsqueeze(0), 24000)

    print(f"✅ Tallennettu: {output_path}")

print("\n🎉 Valmista! Kaikki osat on generoitu kansioon:", OUTPUT_DIR)
