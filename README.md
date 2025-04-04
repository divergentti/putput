**Tiny Scripts Collection**

A set of small but powerful Python utilities for steganography and LLM interactions.

**1. Steganography Tool**

Hide messages or watermarks in images using adaptive LSB, DCT-based embedding, or error-corrected methods. 
Passes png to jpeg conversions and size adjustments.

**Features**

✅ Adaptive LSB
    Dynamically adjusts LSBs (1-3 bits per channel) based on local pixel complexity.
    Maximizes data hiding in textured areas, minimizes artifacts in smooth regions.
    
✅ DCT-based Steganography
    Embeds data in mid-frequency DCT coefficients (YCbCr color space).
    More robust against JPEG compression/resizing than simple LSB methods.
    
✅ Error Correction
    Uses (7,4) Hamming codes to detect and correct single-bit errors.

**Usage**

Embed a message (DCT mode): python3 steganography.py embed input.png "secret" dct  
Extract from a DCT-processed image: python3 steganography.py extract output.jpg dct  

Notes

🔹 Not encryption: For true secrecy, pre-encrypt data (e.g., AES).

🔹 Best for PNGs: Lossless format preserves hidden data best.

🔹 Capacity: Works best with small messages (<1% of image size).

**Use Cases**

    📌 Invisible watermarking (copyright protection).
    🎓 Learning steganography techniques.
    🏗 Lightweight Python (no heavy dependencies).

**2. 2xLLM-blaa-blaa.py**

Make two LLMs chat with each other—one local (e.g., via LM Studio), the other via Mistral.ai API.

**Features**

✅ Multi-LLM Discussions
    Run a conversation between a local LLM and a cloud-based LLM (Mistral.ai).
    Flexible setup (swap models easily).
✅ API + Local Combo
    Useful for comparison testing or automated debates.

python3 2xLLM-blaa-blaa.py

**Possible Use Cases**

    🤖 Testing model responses side-by-side.
    🧠 Automated debate simulations.
    🔍 Comparing local vs. cloud-based LLMs.

**Contributing**

Feel free to fork, improve, or suggest changes!

🔗 Repository: github.com/divergentti/putput
✍️ Author: Divergentti / Jari Hiltunen
