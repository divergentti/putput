This repository contains some tiny scripts.

1. Steganography Tool

Hide messages or watermarks in images using adaptive LSB, DCT, or error-corrected methods.
Key Features

    Adaptive LSB

        Dynamically adjusts LSBs (1-3 bits/channel) based on local pixel complexity.

        Hides more data in textured areas, fewer in smooth regions.

    DCT-based Steganography

        Embeds data in mid-frequency coefficients (YCbCr color space).

        Better survives JPEG compression/resizing than spatial-domain methods.

    Error Correction

        Uses (7,4) Hamming codes to detect/correct single-bit errors.

Usage

# Embed a message (DCT mode):
python3 steganography.py embed input.png "secret" dct  

# Extract from a DCT-processed image:
python3 steganography.py extract output.jpg dct  

Important Notes

    Not encryption: For confidentiality, pre-encrypt data (e.g., AES).

    Best for PNGs: Lossless format preserves hidden data; JPEGs may degrade it.

    Capacity: Small messages (<1% of image size) work best.

Why Use This?

    Watermarking: Embed invisible copyright tags.

    Education: Learn steganography fundamentals.

    Lightweight: Pure Python, no heavy dependencies.
