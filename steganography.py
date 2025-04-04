"""
Version 0.0.2 by Divergentti / Jari Hiltunen

Steganography is not encryption. It hides the message, but it doesn't make it unreadable without knowing how to
extract it. Anyone with basic image analysis tools could potentially detect the hidden data. For true confidentiality,
use proper encryption methods (e.g., AES) before embedding.

PNG uses lossless compression, which means that altering individual bits is less likely to introduce noticeable
artifacts compared to lossy formats like JPEG.

Capacity: The amount of data you can hide depends on the image size and color depth. Each bit requires one pixel
(or part of a pixel). A small message will work fine; larger messages might become visible as distortions in the image.

Usage: python3 steganography.py [mode] <image_path> [method]

Modes:
- extract = This mode specifically focuses on extracting hidden messages
- embed = Embeds message using method

Methods:
- either adaptive_lsb (default) or dct

Examples:

python3 steganography.py embed thecat.png "Hidden message" dct
python3 steganography.py extract dct_encrypted_thecat.png

Adaptive LSB Modification

Analyzes image complexity in local 3x3 pixel regions
Uses 1-3 LSBs per channel based on region complexity
Hides more data in textured/complex areas (less noticeable)
Preserves image quality in smooth areas

2. Hamming Code Error Correction

Adds redundancy to detect and correct bit errors during extraction
Implements (7,4) Hamming code for single-bit error correction
Includes matrix-based encoding/decoding functions
Makes your steganography more robust against image modifications

3. DCT-based Steganography

Operates in frequency domain using Discrete Cosine Transform
Embeds data in mid-frequency DCT coefficients (less perceptible)
Converts image to YCbCr color space and modifies Y channel
More resistant to statistical analysis than spatial domain methods

"""

from PIL import Image
import sys
import numpy as np
import random
from scipy.fftpack import dct, idct
import math

START_CHAR = '#'  # Choose unique characters not likely to appear in the image or message
STOP_CHAR = '$'


# ------------------ Hamming Code Implementation ------------------

def generate_hamming_matrix(r):
    """Generate Hamming code matrices for encoding and decoding."""
    n = 2 ** r - 1
    k = 2 ** r - r - 1

    # Generate parity-check matrix H
    H = np.zeros((r, n), dtype=int)
    for i in range(n):
        # Convert column index to binary representation
        binary = format(i + 1, f'0{r}b')
        for j in range(r):
            H[j, i] = int(binary[j])

    # Generate generator matrix G
    # First find positions of parity bits
    parity_positions = [2 ** i - 1 for i in range(r)]
    data_positions = [i for i in range(n) if i not in parity_positions]

    G = np.zeros((k, n), dtype=int)
    for i, pos in enumerate(data_positions):
        G[i, pos] = 1

        # XOR with corresponding parity bits
        col = format(pos + 1, f'0{r}b')
        for j in range(r):
            if col[j] == '1':
                G[i, parity_positions[j]] = 1

    return H, G


def hamming_encode(data, r=3):
    """Encode data using Hamming code with parameter r."""
    _, G = generate_hamming_matrix(r)
    k = G.shape[0]  # Number of data bits
    n = G.shape[1]  # Total codeword length

    # Ensure data length is multiple of k
    if len(data) % k != 0:
        padding = k - (len(data) % k)
        data = data + '0' * padding

    encoded_data = ""
    for i in range(0, len(data), k):
        block = data[i:i + k]
        data_vector = np.array([int(bit) for bit in block])

        # Encode block using generator matrix
        codeword = np.remainder(np.dot(data_vector, G), 2)
        encoded_data += ''.join(map(str, codeword))

    return encoded_data


def hamming_decode(encoded_data, r=3):
    """Decode Hamming-encoded data with error correction."""
    H, _ = generate_hamming_matrix(r)
    n = H.shape[1]  # Codeword length
    k = n - r  # Data length

    # Process each block
    decoded_data = ""
    for i in range(0, len(encoded_data), n):
        block = encoded_data[i:i + n]
        if len(block) < n:  # Handle potential padding at the end
            break

        received = np.array([int(bit) for bit in block])

        # Calculate syndrome
        syndrome = np.remainder(np.dot(H, received), 2)

        # Convert syndrome to decimal for error position
        error_pos = 0
        for j in range(r):
            error_pos += syndrome[j] * (2 ** (r - j - 1))

        # Correct error if detected
        if error_pos > 0:
            received[error_pos - 1] = 1 - received[error_pos - 1]

        # Extract data bits (remove parity bits)
        parity_positions = [2 ** i - 1 for i in range(r)]
        data_positions = [j for j in range(n) if j not in parity_positions]

        for pos in data_positions:
            if pos < len(received):
                decoded_data += str(received[pos])

    return decoded_data


# ------------------ Adaptive LSB Implementation ------------------

def pixel_complexity(pixel_region):
    """Calculate local complexity for adaptive LSB."""
    # Calculate standard deviation as complexity measure
    return np.std(pixel_region)


def get_embedding_capacity(complexity, threshold_low=5, threshold_high=15):
    """Determine number of LSBs to use based on complexity."""
    if complexity < threshold_low:
        return 1  # Low complexity - use only 1 LSB
    elif complexity < threshold_high:
        return 2  # Medium complexity - use 2 LSBs
    else:
        return 3  # High complexity - use 3 LSBs


def adaptive_lsb_embed(img, binary_message):
    """Embeds a message using adaptive LSB steganography."""
    width, height = img.size
    pixels = np.array(img)

    # Prepare message index
    message_length = len(binary_message)
    data_index = 0

    # Embed message
    for x in range(0, width - 2, 3):
        for y in range(0, height - 2, 3):
            # Check if we've embedded the entire message
            if data_index >= message_length:
                break

            # Get 3x3 pixel region for complexity analysis
            region = pixels[y:y + 3, x:x + 3]
            complexity = pixel_complexity(region)

            # Determine embedding capacity
            capacity = get_embedding_capacity(complexity)

            # Embed in center pixel with determined capacity
            pixel = list(img.getpixel((x + 1, y + 1)))

            # For each color channel
            for i in range(3):
                # Create bit mask based on capacity
                mask = (1 << capacity) - 1
                # Clear the LSBs
                pixel[i] = pixel[i] & ~mask

                # Embed bits
                bits_to_embed = 0
                for j in range(capacity):
                    if data_index < message_length:
                        bits_to_embed |= int(binary_message[data_index], 2) << j
                        data_index += 1

                pixel[i] |= bits_to_embed

            img.putpixel((x + 1, y + 1), tuple(pixel))

    return img, data_index


def adaptive_lsb_extract(img, message_length):
    """Extracts a message using adaptive LSB steganography."""
    width, height = img.size
    pixels = np.array(img)
    binary_message = ""
    data_index = 0

    for x in range(0, width - 2, 3):
        for y in range(0, height - 2, 3):
            # Check if we've extracted enough bits
            if data_index >= message_length:
                break

            # Get 3x3 pixel region for complexity analysis
            region = pixels[y:y + 3, x:x + 3]
            complexity = pixel_complexity(region)

            # Determine embedding capacity
            capacity = get_embedding_capacity(complexity)

            # Extract from center pixel
            pixel = list(img.getpixel((x + 1, y + 1)))

            # For each color channel
            for i in range(3):
                # Extract bits
                for j in range(capacity):
                    if data_index < message_length:
                        bit = (pixel[i] >> j) & 1
                        binary_message += str(bit)
                        data_index += 1

    return binary_message


# ------------------ DCT Implementation ------------------

def rgb_to_ycbcr(img):
    """Convert RGB image to YCbCr color space."""
    pixels = np.array(img)
    height, width, _ = pixels.shape

    ycbcr = np.zeros_like(pixels, dtype=float)

    for y in range(height):
        for x in range(width):
            r, g, b = pixels[y, x]

            # RGB to YCbCr conversion
            ycbcr[y, x, 0] = 0.299 * r + 0.587 * g + 0.114 * b
            ycbcr[y, x, 1] = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
            ycbcr[y, x, 2] = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b

    return ycbcr


def ycbcr_to_rgb(ycbcr):
    """Convert YCbCr image back to RGB color space."""
    height, width, _ = ycbcr.shape
    rgb = np.zeros_like(ycbcr, dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            y_val, cb, cr = ycbcr[y, x]

            # YCbCr to RGB conversion
            r = y_val + 1.402 * (cr - 128)
            g = y_val - 0.344136 * (cb - 128) - 0.714136 * (cr - 128)
            b = y_val + 1.772 * (cb - 128)

            # Clip values to valid range
            rgb[y, x, 0] = min(max(0, round(r)), 255)
            rgb[y, x, 1] = min(max(0, round(g)), 255)
            rgb[y, x, 2] = min(max(0, round(b)), 255)

    return rgb


def embed_in_dct_block(block, bits, alpha=5):
    """Embed bits in the mid-frequency coefficients of an 8x8 DCT block."""
    # Apply DCT
    dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')

    # Mid-frequency coefficients to use (zigzag order)
    positions = [(1, 2), (2, 1), (2, 2), (1, 3), (3, 1)]

    for i, pos in enumerate(positions):
        if i < len(bits):
            # Modify coefficient to embed bit
            if bits[i] == '1':
                # Ensure coefficient is positive and at least alpha
                if dct_block[pos] > 0:
                    dct_block[pos] = max(dct_block[pos], alpha)
                else:
                    dct_block[pos] = alpha
            else:
                # Ensure coefficient is negative and at most -alpha
                if dct_block[pos] < 0:
                    dct_block[pos] = min(dct_block[pos], -alpha)
                else:
                    dct_block[pos] = -alpha

    # Apply inverse DCT
    idct_block = idct(idct(dct_block.T, norm='ortho').T, norm='ortho')
    return idct_block


def extract_from_dct_block(block):
    """Extract bits from the mid-frequency coefficients of an 8x8 DCT block."""
    # Apply DCT
    dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')

    # Same positions used for embedding
    positions = [(1, 2), (2, 1), (2, 2), (1, 3), (3, 1)]
    bits = ""

    for pos in positions:
        # Extract bit based on coefficient sign
        if dct_block[pos] > 0:
            bits += '1'
        else:
            bits += '0'

    return bits


def dct_embed(img, binary_message):
    """Embed message using DCT-based steganography."""
    # Convert to YCbCr
    ycbcr = rgb_to_ycbcr(img)

    # Get image dimensions
    height, width, _ = ycbcr.shape

    # Ensure dimensions are multiples of 8
    height_pad = height - (height % 8)
    width_pad = width - (width % 8)

    # Prepare message index
    message_length = len(binary_message)
    data_index = 0

    # Process 8x8 blocks
    for y in range(0, height_pad, 8):
        for x in range(0, width_pad, 8):
            if data_index >= message_length:
                break

            # Get the Y channel block
            block = ycbcr[y:y + 8, x:x + 8, 0]

            # Determine bits to embed in this block
            bits_to_embed = binary_message[data_index:min(data_index + 5, message_length)]
            data_index += len(bits_to_embed)

            # Embed bits
            modified_block = embed_in_dct_block(block, bits_to_embed)

            # Update block
            ycbcr[y:y + 8, x:x + 8, 0] = modified_block

    # Convert back to RGB
    rgb = ycbcr_to_rgb(ycbcr)

    # Create new image
    dct_img = Image.fromarray(rgb)
    return dct_img, data_index


def dct_extract(img, message_length):
    """Extract message using DCT-based steganography."""
    # Convert to YCbCr
    pixels = np.array(img)
    ycbcr = rgb_to_ycbcr(img)

    # Get image dimensions
    height, width, _ = ycbcr.shape

    # Ensure dimensions are multiples of 8
    height_pad = height - (height % 8)
    width_pad = width - (width % 8)

    # Prepare for extraction
    binary_message = ""
    bits_needed = message_length

    # Process 8x8 blocks
    for y in range(0, height_pad, 8):
        for x in range(0, width_pad, 8):
            if len(binary_message) >= bits_needed:
                break

            # Get the Y channel block
            block = ycbcr[y:y + 8, x:x + 8, 0]

            # Extract bits
            bits = extract_from_dct_block(block)

            # Add bits to message
            remaining = bits_needed - len(binary_message)
            binary_message += bits[:min(5, remaining)]

    return binary_message


# ------------------ Enhanced Steganography Functions ------------------

def embed_message(image_path, message, method="adaptive_lsb"):
    """Embeds a message into an image using selected steganography method."""
    try:
        img = Image.open(image_path)
        img = img.convert("RGB")
        width, height = img.size

        # Add start and stop markers to message
        full_message = START_CHAR + message + STOP_CHAR

        # Convert message to binary
        binary_message = ''.join(format(ord(char), '08b') for char in full_message)
        message_length = len(binary_message)

        # Store message length at the beginning for extraction
        length_binary = format(message_length, '032b')

        # Apply Hamming encoding for error correction
        if method != "dct":  # DCT has its own error resilience
            binary_message = hamming_encode(binary_message)
            message_length = len(binary_message)

        # Choose steganography method
        if method == "adaptive_lsb":
            # Check capacity
            if message_length > (width // 3) * (height // 3) * 9:  # Rough estimate
                raise ValueError("Message is too large for adaptive LSB.")

            # Embed message length first
            img_copy = img.copy()
            length_embedded_img, _ = adaptive_lsb_embed(img_copy, length_binary)

            # Then embed actual message
            modified_img, embedded_bits = adaptive_lsb_embed(length_embedded_img, binary_message)

            if embedded_bits < message_length:
                raise ValueError("Could not embed entire message.")

        elif method == "dct":
            # Check capacity
            max_bits = (width // 8) * (height // 8) * 5
            if message_length > max_bits:
                raise ValueError("Message is too large for DCT embedding.")

            # Embed message length and data
            img_copy = img.copy()
            img_with_length, _ = adaptive_lsb_embed(img_copy, length_binary)
            modified_img, embedded_bits = dct_embed(img_with_length, binary_message)

            if embedded_bits < message_length:
                raise ValueError("Could not embed entire message.")
        else:
            raise ValueError("Unknown steganography method.")

        # Save modified image
        modified_image_path = f"{method}_encrypted_{image_path}"
        modified_img.save(modified_image_path)

        return modified_image_path, method

    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None, None
    except ValueError as e:
        print(f"ValueError: {e}")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None, None


def extract_message(image_path, method="adaptive_lsb"):
    """Extracts a message from an image using the specified steganography method."""
    try:
        img = Image.open(image_path)
        img = img.convert("RGB")

        # First extract the message length (always embedded with adaptive LSB)
        length_binary = adaptive_lsb_extract(img, 32)
        message_length = int(length_binary, 2)

        # Extract the message using appropriate method
        if method == "adaptive_lsb":
            binary_message = adaptive_lsb_extract(img, message_length)
        elif method == "dct":
            binary_message = dct_extract(img, message_length)
        else:
            raise ValueError("Unknown steganography method.")

        # Apply Hamming decoding if needed
        if method != "dct":
            binary_message = hamming_decode(binary_message)

        # Convert binary to characters
        chars = []
        for i in range(0, len(binary_message), 8):
            byte = binary_message[i:i + 8]
            if len(byte) == 8:  # Ensure we have a full byte
                chars.append(chr(int(byte, 2)))

        extracted = ''.join(chars)

        # Find start and stop markers
        start_index = extracted.find(START_CHAR)
        if start_index == -1:
            print("Start marker not found.")
            return None

        stop_index = extracted.find(STOP_CHAR, start_index + 1)
        if stop_index == -1:
            print("Stop marker not found.")
            return None

        # Extract message between markers
        return extracted[start_index + 1:stop_index]

    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


def detect_method(image_path):
    """Attempt to detect which steganography method was used."""
    # Simple detection based on filename
    if "dct_encrypted_" in image_path:
        return "dct"
    elif "adaptive_lsb_encrypted_" in image_path:
        return "adaptive_lsb"
    else:
        # Default to adaptive_lsb (which is used for length encoding in all methods)
        return "adaptive_lsb"


if __name__ == "__main__":
    # Define command modes
    EMBED_MODE = "embed"
    EXTRACT_MODE = "extract"

    # Parse command-line arguments
    if len(sys.argv) < 3:
        print("Usage:")
        print("  For embedding: python steganography.py embed <image_path> <message> [method]")
        print("  For extraction: python steganography.py extract <image_path> [method]")
        print()
        print("  <image_path>: Path to the image file.")
        print("  <message>: The message to be embedded (only for embed mode).")
        print("  [method]: Optional - Steganography method to use:")
        print("            'adaptive_lsb' (default) - Adaptive LSB with Hamming code")
        print("            'dct' - DCT-based steganography")
        print()
        print("  If method is not specified during extraction, the program will")
        print("  attempt to detect it based on the filename.")
        sys.exit(1)

    mode = sys.argv[1].lower()

    if mode == EMBED_MODE:
        if len(sys.argv) < 4:
            print("Error: Embedding mode requires both image path and message.")
            sys.exit(1)

        image_file = sys.argv[2]
        message_to_embed = sys.argv[3]

        # Default to adaptive_lsb if no method specified
        method = "adaptive_lsb"
        if len(sys.argv) >= 5:
            method = sys.argv[4]

        # Embed message
        modified_image, used_method = embed_message(image_file, message_to_embed, method)

        if modified_image:
            print(f"Message embedded successfully using {used_method}.")
            print(f"Modified image saved as: {modified_image}")

            # Extract message to verify
            extracted_message = extract_message(modified_image, used_method)
            if extracted_message:
                print(f"Verification - Extracted Message: {extracted_message}")

                if extracted_message != message_to_embed:
                    print("Warning: Extracted message differs from original message.")
            else:
                print("Warning: Failed to verify the embedded message.")

    elif mode == EXTRACT_MODE:
        image_file = sys.argv[2]

        # Determine method
        if len(sys.argv) >= 4:
            method = sys.argv[3]
        else:
            method = detect_method(image_file)
            print(f"Auto-detected method: {method}")

        # Extract message
        extracted_message = extract_message(image_file, method)
        if extracted_message:
            print(f"Extracted Message: {extracted_message}")
        else:
            print("Failed to extract any message.")

    else:
        print(f"Error: Unknown mode '{mode}'. Use 'embed' or 'extract'.")
        sys.exit(1)
