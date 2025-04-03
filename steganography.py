"""
Version 0.0.1 by Divergentti / Jari Hiltunen

Simple steganography script for png files.

Important:

Steganography is not encryption. It hides the message, but it doesn't make it unreadable without knowing how to
extract it. Anyone with basic image analysis tools could potentially detect the hidden data. For true confidentiality,
use proper encryption methods (e.g., AES) before embedding.

PNG uses lossless compression, which means that altering individual bits is less likely to introduce noticeable
artifacts compared to lossy formats like JPEG.

Capacity: The amount of data you can hide depends on the image size and color depth. Each bit requires one pixel
(or part of a pixel). A small message will work fine; larger messages might become visible as distortions in the image.

Installation:

Install Pillow: pip install Pillow

Running:
Save as steganography.py.
Run with arguments: python3 steganography.py your_image.png "Greetings from Finland"

Further Improvements and Considerations:

Error Correction/Detection: Implement error correction codes (e.g., Hamming code) to make the embedded message more
robust against minor data corruption.
Key-Based Steganography: Use a key to determine which bits to modify, making it harder for someone without the key to
extract the message. This adds a layer of security but doesn't provide true encryption.
Adaptive LSB Modification: Adjust the number of LSBs modified based on the image content (e.g., avoid modifying LSBs
in areas with high contrast).
More Complex Steganographic Techniques: Explore other techniques like Discrete Cosine Transform (DCT) steganography,
which can hide data within JPEG images more effectively but are significantly more complex to implement.
Encryption: always encrypt your message before embedding it using a strong encryption algorithm
(e.g., AES). This is crucial for protecting the confidentiality of the data.
"""

from PIL import Image
import sys

START_CHAR = '#'  # Choose unique characters not likely to appear in the image or message
STOP_CHAR = '$'


def embed_message(image_path, message):
    """Embeds a message into a PNG image using LSB steganography with start/stop markers."""
    try:
        img = Image.open(image_path)
        img = img.convert("RGB")
        width, height = img.size

        # Convert the message to binary representation including start and stop chars
        binary_message = ''.join(format(ord(char), '08b') for char in START_CHAR + message + STOP_CHAR)
        message_length = len(binary_message)

        if message_length > width * height * 3:
            raise ValueError("Message is too large to fit into the image.")

        data_index = 0
        for x in range(width):
            for y in range(height):
                pixel = list(img.getpixel((x, y)))

                for i in range(3):
                    if data_index < message_length:
                        pixel[i] = (pixel[i] & ~1) | int(binary_message[data_index % message_length], 2)
                        data_index += 1
                img.putpixel((x, y), tuple(pixel))

        modified_image_path = "encrypted_" + image_path
        img.save(modified_image_path)
        return modified_image_path

    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except ValueError as e:
        print(f"ValueError: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


def extract_message(image_path):
    """Extracts a message from a PNG image using LSB steganography with start/stop markers."""
    try:
        img = Image.open(image_path)
        width, height = img.size
        binary_message = ""

        for x in range(width):
            for y in range(height):
                pixel = list(img.getpixel((x, y)))
                for i in range(3):
                    binary_message += str(pixel[i] & 1)

        # Find start and stop markers
        start_index = binary_message.find(format(ord(START_CHAR), '08b'))
        if start_index == -1:
            print("Start marker not found.")
            return None

        stop_index = binary_message.find(format(ord(STOP_CHAR), '08b'), start_index + len(format(ord(START_CHAR), '08b')))
        if stop_index == -1:
            print("Stop marker not found.")
            return None

        # Extract the message between the markers
        extracted_binary = binary_message[start_index + len(format(ord(START_CHAR), '08b')):stop_index]
        extracted_message = ''.join([chr(int(extracted_binary[i:i+8], 2)) for i in range(0, len(extracted_binary), 8)])

        return extracted_message

    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python steganography.py <image_path> <message>")
        print("  <image_path>: Path to the PNG image file.")
        print("  <message>: The message to be embedded (ASCII characters).")
        sys.exit(1)

    image_file = sys.argv[1]
    message_to_embed = sys.argv[2]

    modified_image = embed_message(image_file, message_to_embed)

    if modified_image:
        print(f"Message embedded successfully. Modified image saved as: {modified_image}")

        extracted_message = extract_message(modified_image)
        if extracted_message:
            print(f"Extracted Message: {extracted_message}")
        else:
            print("Failed to extract the message.")
