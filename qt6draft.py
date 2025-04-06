"""
Not functional yet!

sudo apt-get install -y libxcb-cursor-dev

"""

import sys
import os
import io
from PIL import Image, ExifTags  # Need to install: pip install pillow
from PIL.ExifTags import TAGS
import numpy as np  # Need to install: pip install numpy
from scipy.fftpack import dct, idct
import reedsolo  # Need to install: pip install reedsolo
import struct
import zlib  # For CRC32 checksum
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QRadioButton, QGroupBox, QLineEdit, QPushButton, QComboBox,
    QLabel, QTextEdit, QFileDialog, QStatusBar, QMessageBox
)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt  # Need to install: pip install pyqt6

debug = True

# Constants
MAX_MESSAGE_SIZE = 100000  # Safety limit for message size in bits
MAX_DCT_ALPHA = 25  # Increased strength for DCT embedding

# Reed-Solomon error correction parameters
RS_REDUNDANCY = 20  # Increased redundancy for better error correction


class CoordinateTracker:
    """Track used DCT block coordinates"""
    def __init__(self, max_coords):
        self.used = set()
        self.max_coords = max_coords

    def add(self, x, y):
        if (x, y) in self.used or len(self.used) >= self.max_coords:
            return False
        self.used.add((x, y))
        return True

# Add to constants
DCT_SIZE = 8
ZIGZAG_ORDER = [
    (0,0), (0,1), (1,0), (2,0), (1,1), (0,2), (0,3), (1,2),
    (2,1), (3,0), (4,0), (3,1), (2,2), (1,3), (0,4), (0,5),
    (1,4), (2,3), (3,2), (4,1), (5,0), (6,0), (5,1), (4,2),
    (3,3), (2,4), (1,5), (0,6), (0,7), (1,6), (2,5), (3,4),
    (4,3), (5,2), (6,1), (7,0), (7,1), (6,2), (5,3), (4,4),
    (3,5), (2,6), (1,7), (2,7), (3,6), (4,5), (5,4), (6,3),
    (7,2), (7,3), (6,4), (5,5), (4,6), (3,7), (4,7), (5,6),
    (6,5), (7,4), (7,5), (6,6), (5,7), (6,7), (7,6), (7,7)
]

def is_mid_frequency(coeff_index):
    """Check if coefficient is in mid-frequency range"""
    return 5 < coeff_index < 58

def quantize_coefficients(dct_block, quant_table):
    """Quantize DCT coefficients using JPEG quantization table"""
    return np.round(dct_block / quant_table)





def crc32_checksum(data):
    """Calculate CRC32 checksum for data verification."""
    return zlib.crc32(data) & 0xffffffff


def text_to_rs_encoded(text, redundancy=RS_REDUNDANCY):
    """Convert text to Reed-Solomon encoded bytes with checksum."""
    # Initialize RS encoder
    rs = reedsolo.RSCodec(redundancy)

    # Convert text to bytes
    text_bytes = text.encode('utf-8')

    # Add checksum
    checksum = struct.pack('>I', crc32_checksum(text_bytes))
    data_with_checksum = checksum + text_bytes

    # Encode with Reed-Solomon
    encoded = rs.encode(data_with_checksum)

    return encoded


def rs_encoded_to_text(encoded_data, redundancy=RS_REDUNDANCY):
    """Convert Reed-Solomon encoded bytes back to text, with error correction and checksum verification."""
    # Initialize RS decoder
    rs = reedsolo.RSCodec(redundancy)

    try:
        # Decode data (this will correct errors up to the redundancy limit)
        decoded, _, _ = rs.decode(encoded_data)

        # Extract and verify checksum
        if len(decoded) < 4:
            if debug:
                print("Decoded data too short to contain checksum")
            return None

        stored_checksum = struct.unpack('>I', decoded[:4])[0]
        text_bytes = decoded[4:]
        calculated_checksum = crc32_checksum(text_bytes)

        if stored_checksum != calculated_checksum:
            if debug:
                print(f"Checksum verification failed: stored={stored_checksum}, calculated={calculated_checksum}")
            return None

        return text_bytes.decode('utf-8')
    except reedsolo.ReedSolomonError as e:
        if debug:
            print(f"Reed-Solomon decoding error: {e}")
        return None
    except UnicodeDecodeError as e:
        if debug:
            print(f"Unicode decode error (possibly corrupted data): {e}")
        return None


def bytes_to_binary_string(data):
    """Convert bytes to a binary string with debug output."""
    binary = ''.join(format(byte, '08b') for byte in data)
    if debug:
        print(f"Original Bytes: {data.hex()}")
        print(f"Generated Binary: {binary[:64]}... (truncated)")
    return ''.join([format(byte, '08b') for byte in data])

def binary_string_to_bytes(binary_string):
    """Convert binary string back to bytes with debug output."""
    bytes_data = bytes()
    for i in range(0, len(binary_string), 8):
        byte_str = binary_string[i:i+8]
        if len(byte_str) < 8:
            break  # Discard incomplete byte
        byte = int(byte_str, 2)
        bytes_data += struct.pack('B', byte)
    if debug:
        print(f"Binary String (first 64 bits): {binary_string[:64]}...")
        print(f"Converted Bytes: {bytes_data.hex()}")
    binary_string = binary_string[:len(binary_string) // 8 * 8]  # Trim to whole bytes
    return bytes([int(binary_string[i:i + 8], 2) for i in range(0, len(binary_string), 8)])


# Better length encoding with redundancy and checksum
def encode_length_with_magic_marker(length):
    """Encode length with magic marker and CRC32 checksum."""
    magic = 0xA55A
    length_data = struct.pack('>HI', magic, length)
    checksum = struct.pack('>I', crc32_checksum(length_data))
    return bytes_to_binary_string(length_data + checksum) * 3  # Triple redundancy


def decode_length_with_magic_marker(binary_data):
    """Decode length from redundant binary data."""
    bytes_data = binary_string_to_bytes(binary_data)

    # Check all 3 redundant copies
    for i in [0, 10, 20]:  # Positions of redundant copies
        if i + 10 > len(bytes_data):
            continue

        chunk = bytes_data[i:i + 10]
        try:
            magic, length = struct.unpack('>HI', chunk[:6])
            stored_checksum = struct.unpack('>I', chunk[6:10])[0]

            if magic == 0xA55A and crc32_checksum(chunk[:6]) == stored_checksum:
                return min(length, MAX_MESSAGE_SIZE)
        except:
            continue

    return 0  # No valid copy found

def adaptive_lsb_embed(img, binary_message):
    """Enhanced adaptive LSB algorithm focusing on stable areas."""
    width, height = img.size
    pixels = np.array(img)
    img_copy = img.copy()
    message_length = len(binary_message)
    data_index = 0

    for y in range(1, height - 1, 2):
        for x in range(1, width - 1, 2):
            if data_index >= message_length:
                break

            region = pixels[max(0, y - 1):y + 2, max(0, x - 1):x + 2]
            if np.var(region) < 100:
                pixel = list(img.getpixel((x, y)))
                for i in range(3):
                    if data_index < message_length:
                        # Clear 2 LSBs and embed the same bit twice
                        bit = int(binary_message[data_index])
                        pixel[i] = (pixel[i] & ~3) | (bit << 1) | bit
                        data_index += 1
                img_copy.putpixel((x, y), tuple(pixel))

    return img_copy, data_index


def adaptive_lsb_extract(img, message_length):
    """Extract bits from smooth regions using first LSB."""
    binary_message = []
    width, height = img.size

    for y in range(1, height - 1, 2):
        for x in range(1, width - 1, 2):
            if len(binary_message) >= message_length:
                break

            pixel = img.getpixel((x, y))
            # Extract only the first LSB (ignore redundancy)
            for channel in pixel[:3]:  # RGB channels
                if len(binary_message) < message_length:
                    binary_message.append(str((channel >> 1) & 1))  # Use first redundant bit

    return ''.join(binary_message)

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


def embed_in_dct_block(block, bits, alpha=MAX_DCT_ALPHA):
    """Embed bits in carefully selected medium-frequency coefficients of an 8x8 DCT block."""
    # Apply DCT
    dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')

    # Use coefficients that are less affected by JPEG compression
    # These are generally in the medium frequency range (not too low, not too high)
    positions = [(3, 3), (4, 2), (2, 4), (3, 4), (4, 3), (5, 2), (2, 5)]  # Added more positions

    for i, pos in enumerate(positions):
        if i < len(bits):
            # Use strong embedding with controlled quantization
            coef_value = abs(dct_block[pos])
            sign = 1 if dct_block[pos] >= 0 else -1

            # Quantize coefficient based on bit
            if bits[i] == '1':
                # Make it positive and large enough
                new_value = max(coef_value, alpha) if sign >= 0 else alpha
                dct_block[pos] = new_value
            else:
                # Make it negative and large enough
                new_value = max(coef_value, alpha) if sign < 0 else alpha
                dct_block[pos] = -new_value

    # Apply inverse DCT with rounding to avoid floating point errors
    idct_block = idct(idct(dct_block.T, norm='ortho').T, norm='ortho')
    return np.round(idct_block).astype(np.float64)


def extract_from_dct_block(block):
    """Extract bits from the medium-frequency coefficients of an 8x8 DCT block."""
    # Apply DCT
    dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')

    # Same positions used for embedding
    positions = [(3, 3), (4, 2), (2, 4), (3, 4), (4, 3), (5, 2), (2, 5)]  # Added more positions
    bits = ""

    for pos in positions:
        # Extract bit based on coefficient sign
        if dct_block[pos] >= 0:
            bits += '1'
        else:
            bits += '0'

    return bits

def dct_embed(img, binary_message):
    """Embed message using improved DCT-based steganography with a focus on dark regions."""
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

    # Define a threshold for dark regions
    dark_threshold = 100  # Adjust this value based on your needs

    # Process 8x8 blocks with spacing for resilience
    for y in range(0, height_pad, 8):
        for x in range(0, width_pad, 8):
            if data_index >= message_length:
                break

            block = ycbcr[y:y+8, x:x+8, 0]  # Use Y channel

            # Embed only in dark regions to minimize visual artifacts
            if np.mean(block) < dark_threshold:
                bits = binary_message[data_index:data_index+7]  # 7 bits per block
                if len(bits) < 7:
                    bits = bits.ljust(7, '0')  # Pad if necessary

                modified_block = embed_in_dct_block(block, bits)
                ycbcr[y:y+8, x:x+8, 0] = modified_block
                data_index += 7

    # Convert back to RGB
    rgb_array = ycbcr_to_rgb(ycbcr)
    result_img = Image.fromarray(rgb_array.astype(np.uint8))
    return result_img, data_index


def dct_extract(img, message_length, password=None):
    """Improved DCT extraction with Java-inspired features"""
    ycbcr = rgb_to_ycbcr(img)
    height, width, _ = ycbcr.shape
    quant_table = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])

    # Initialize password-based random generator
    seed = int(hashlib.sha256(password.encode()).hexdigest(), 16) & 0xffffffff if password else 0
    rand = random.Random(seed)

    coord_tracker = CoordinateTracker((width * height) // (DCT_SIZE ** 2))
    bits = []

    while len(bits) < message_length:
        # Find unused coordinates
        while True:
            xb = rand.randint(0, (width // DCT_SIZE) - 1)
            yb = rand.randint(0, (height // DCT_SIZE) - 1)
            if coord_tracker.add(xb, yb):
                break

        # Extract block and convert to Y channel
        block = ycbcr[yb * DCT_SIZE:(yb + 1) * DCT_SIZE, xb * DCT_SIZE:(xb + 1) * DCT_SIZE, 0]
        block = block.astype(np.float64) - 128  # Center around zero

        # Perform DCT
        dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')

        # Quantize coefficients
        quantized = quantize_coefficients(dct_block, quant_table)

        # Select random mid-frequency coefficient
        while True:
            coeff_index = rand.randint(1, len(ZIGZAG_ORDER) - 2)
            if is_mid_frequency(coeff_index):
                i, j = ZIGZAG_ORDER[coeff_index]
                break

        # Extract LSB
        bits.append(str(int(quantized[i, j]) & 1))

        if len(bits) >= message_length:
            break

    return ''.join(bits[:message_length])



def secure_hybrid_embed(image_path, message, start_char='#', stop_char='$'):
    """Embeds a message into an image using improved hybrid steganography."""
    try:
        # Convert to absolute path
        abs_path = os.path.abspath(image_path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Image file not found at {abs_path}")

        # Open image and get original format
        img = Image.open(abs_path)
        original_format_to_save = img.format  # Save original format for later
        original_format = img.format


        # Convert to RGB if needed
        if img.mode != "RGB":
           img = img.convert("RGB")

        # Convert to PNG in-memory if original format is not PNG
        if original_format.upper() != 'PNG':
            png_buffer = io.BytesIO()
            img.save(png_buffer, format='PNG')
            png_buffer.seek(0)
            img = Image.open(png_buffer)
            original_format = 'PNG'  # Embedding is done on PNG version

        # Add start and stop markers
        full_message = start_char + message + stop_char

        # Apply Reed-Solomon encoding
        rs_encoded = text_to_rs_encoded(full_message)

        # Convert to binary
        binary_message = bytes_to_binary_string(rs_encoded)
        message_length = len(binary_message)

        # Check if message is too large
        if message_length > MAX_MESSAGE_SIZE:
            raise ValueError(f"Message too large: {message_length} bits (max {MAX_MESSAGE_SIZE})")

        # Create length information with redundancy and checksum
        length_binary = encode_length_with_magic_marker(message_length)

        # Embed length information using enhanced LSB
        img_copy = img.copy()
        img_with_length, length_bits_embedded = adaptive_lsb_embed(img_copy, length_binary)

        if length_bits_embedded < len(length_binary):
            print(f"Warning: Could only embed {length_bits_embedded} of {len(length_binary)} length bits")

        # Embed main message using improved DCT
        modified_img, embedded_bits = dct_embed(img_with_length, binary_message)

        # Verify embedding success
        if embedded_bits < message_length:
            print(f"Warning: Could only embed {embedded_bits} of {message_length} bits")

        # Save in original format
        original_dir = os.path.dirname(abs_path)
        original_name = os.path.basename(abs_path)
        base_name, ext = os.path.splitext(original_name)
        modified_image_path = os.path.join(original_dir, f"encrypted_{base_name}{ext}")

        # Use high quality settings for lossy formats
        if original_format_to_save == "JPEG":
            modified_img.save(modified_image_path, original_format_to_save, quality=100, subsampling=0)
        else:
            modified_img.save(modified_image_path, format='PNG')

        return modified_image_path

    except Exception as e:
        print(f"Error in secure_hybrid_embed: {e}")
        import traceback
        traceback.print_exc()
        return None


def secure_hybrid_extract(image_path, password=None, start_char='#', stop_char='$'):
    """Updated extraction with password support"""
    try:
        img = Image.open(image_path).convert("RGB")

        # Length extraction remains the same
        length_info_size = len(encode_length_with_magic_marker(0))
        length_binary = adaptive_lsb_extract(img, length_info_size * 2)
        message_length = decode_length_with_magic_marker(length_binary)


        if debug:
            print(f"Length_info: {length_info_size}")
            print(f"Length_binary: {length_binary}")

        # Decode the length with validation
        message_length = decode_length_with_magic_marker(length_binary)

        if message_length <= 0 or message_length > MAX_MESSAGE_SIZE:
            print(f"Invalid message length detected: {message_length}")
            return None

        if debug:
            print(f"Detected message length: {message_length} bits")

        # Use new DCT extraction with password
        binary_message = dct_extract(img, message_length, password)

        #binary_message = dct_extract(img, message_length)
        if len(binary_message) < message_length:
            return f"ERROR: Only {len(binary_message)}/{message_length} bits extracted"

        extracted_length = len(binary_message)
        if extracted_length < message_length:
            print(f"Warning: Could only extract {extracted_length} of {message_length} bits")
            # Pad with zeros if needed
            binary_message = binary_message.ljust(message_length, '0')

        # Convert binary to bytes
        encoded_bytes = binary_string_to_bytes(binary_message)

        # Apply Reed-Solomon decoding with error correction
        decoded_text = rs_encoded_to_text(encoded_bytes)
        if not decoded_text:
            print("Reed-Solomon decoding failed")
            return None

        # Find start and stop markers
        start_index = decoded_text.find(start_char)
        if start_index == -1:
            print("Start marker not found")
            return None

        stop_index = decoded_text.find(stop_char, start_index + 1)
        if stop_index == -1:
            print("Stop marker not found")
            return None

        if not decoded_text:
            print("Reed-Solomon decoding failed")
            return "DECODING FAILED: Reed-Solomon error"

        # Find start and stop markers
        start_index = decoded_text.find(start_char)
        if start_index == -1:
            return "MARKER ERROR: Start character not found"

        return decoded_text[start_index + 1:stop_index]

    except Exception as e:
        print(f"Error in secure_hybrid_extract: {e}")
        import traceback
        traceback.print_exc()
        return None


# ------------------------------------- The GUI part

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Steganography Tool")
        self.setGeometry(100, 100, 800, 600)
        self.setup_ui()
        self.create_menu()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Mode selection
        self.mode_group = QGroupBox("Operation Mode")
        mode_layout = QHBoxLayout()
        self.encrypt_radio = QRadioButton("Encrypt")
        self.decrypt_radio = QRadioButton("Decrypt")
        self.encrypt_radio.setChecked(True)
        mode_layout.addWidget(self.encrypt_radio)
        mode_layout.addWidget(self.decrypt_radio)
        self.mode_group.setLayout(mode_layout)
        layout.addWidget(self.mode_group)

        # Input type selection
        self.input_type_group = QGroupBox("Input Type")
        input_type_layout = QHBoxLayout()
        self.file_radio = QRadioButton("File")
        self.folder_radio = QRadioButton("Folder")
        self.file_radio.setChecked(True)
        input_type_layout.addWidget(self.file_radio)
        input_type_layout.addWidget(self.folder_radio)
        self.input_type_group.setLayout(input_type_layout)
        layout.addWidget(self.input_type_group)

        # Path selection
        path_layout = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.browse_btn = QPushButton("Browse")
        path_layout.addWidget(self.path_edit)
        path_layout.addWidget(self.browse_btn)
        layout.addLayout(path_layout)

        self.password_label = QLabel("Password:")
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        layout.addWidget(self.password_label)
        layout.addWidget(self.password_input)

        # Message input
        self.message_label = QLabel("Secret Message:")
        self.message_input = QLineEdit()
        layout.addWidget(self.message_label)
        layout.addWidget(self.message_input)

        # Action button
        self.action_btn = QPushButton("Encrypt")
        layout.addWidget(self.action_btn)

        # Preview area
        preview_layout = QHBoxLayout()
        self.preview_label = QLabel()
        self.preview_label.setFixedSize(400, 400)
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.exif_text = QTextEdit()
        self.exif_text.setReadOnly(True)
        preview_layout.addWidget(self.preview_label)
        preview_layout.addWidget(self.exif_text)
        layout.addLayout(preview_layout)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Connect signals
        self.encrypt_radio.toggled.connect(self.update_ui_mode)
        self.browse_btn.clicked.connect(self.handle_browse)
        self.action_btn.clicked.connect(self.handle_action)
        self.file_radio.toggled.connect(self.update_path_field)

    def create_menu(self):
        menu_bar = self.menuBar()
        help_menu = menu_bar.addMenu("Help")
        about_action = help_menu.addAction("About")
        about_action.triggered.connect(self.show_help)

    def update_ui_mode(self, checked):
        if checked:
            self.action_btn.setText("Encrypt")
            self.message_label.show()
            self.message_input.show()
            self.input_type_group.show()
        else:
            self.action_btn.setText("Decrypt")
            self.message_label.hide()
            self.message_input.hide()
            self.input_type_group.hide()
            self.file_radio.setChecked(True)

    def update_path_field(self):
        if self.file_radio.isChecked():
            self.path_edit.setPlaceholderText("Select file...")
        else:
            self.path_edit.setPlaceholderText("Select folder...")

    def handle_browse(self):
        if self.encrypt_radio.isChecked():
            if self.file_radio.isChecked():
                path, _ = QFileDialog.getOpenFileName(
                    self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
            else:
                path = QFileDialog.getExistingDirectory(self, "Select Folder")
        else:
            path, _ = QFileDialog.getOpenFileName(
                self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")

        if path:
            self.path_edit.setText(path)
            if self.encrypt_radio.isChecked() and self.file_radio.isChecked():
                self.update_preview(path)
            elif self.decrypt_radio.isChecked():
                self.update_preview(path)

    def update_preview(self, path):
        # Update image preview
        pixmap = QPixmap(path)
        if not pixmap.isNull():
            scaled = pixmap.scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio)
            self.preview_label.setPixmap(scaled)
        else:
            self.preview_label.clear()

        # Update EXIF data
        try:
            with Image.open(path) as img:
                exif_info = self.get_exif_data(img)
                self.exif_text.setPlainText(exif_info)
        except Exception as e:
            self.exif_text.setPlainText(f"Error loading EXIF: {str(e)}")

    def get_exif_data(self, image):
        try:
            exif_data = image.getexif()
            if not exif_data:
                return "No EXIF data found"

            exif_info = []
            for tag_id, value in exif_data.items():
                tag_name = TAGS.get(tag_id, tag_id)
                exif_info.append(f"{tag_name}: {value}")
            return "\n".join(exif_info)
        except Exception as e:
            return f"EXIF Error: {str(e)}"

    def handle_action(self):
        path = self.path_edit.text()
        if not path:
            self.status_bar.showMessage("Please select input path")
            return

        if self.encrypt_radio.isChecked():
            self.handle_encrypt()
        else:
            self.handle_decrypt()

    def handle_encrypt(self):
        path = self.path_edit.text()
        message = self.message_input.text().strip()  # Clean whitespace
        password = self.password_input.text()

        if not path:
            self.status_bar.showMessage("Please select an input file or folder!")
            return
        if not message:
            self.status_bar.showMessage("Message is required!")
            return

        try:
            if self.file_radio.isChecked():
                if not os.path.exists(path):
                    raise FileNotFoundError(f"File not found: {path}")

                output_path = secure_hybrid_embed(path, message)

                if not os.path.exists(output_path):
                    raise RuntimeError(f"Failed to create output file at {output_path}")

                # Verification phase
                success_msg = (f"Encryption successful using\n"
                               f"Original: {os.path.basename(path)}\n"
                               f"Saved to: {os.path.basename(output_path)}")

                try:
                    decrypted = secure_hybrid_extract(output_path)
                    if decrypted is None:
                        success_msg += "\n\nDecryption verification FAILED - No message found"
                    elif decrypted != message:
                        success_msg += (f"\n\nDECRYPTION MISMATCH!\n"
                                        f"Original: {message}\n"
                                        f"Decrypted: {decrypted}")
                    else:
                        success_msg += "\n\nDecryption verified - Message matches perfectly"
                except Exception as e:
                    success_msg += f"\n\nDecryption verification ERROR: {str(e)}"

                # Show comprehensive result
                QMessageBox.information(self, "Encryption Result", success_msg)
                self.status_bar.showMessage(f"Saved encrypted file: {output_path}")

                # Update preview
                self.update_preview(output_path)
            else:
                self.process_folder(path, message)

        except FileNotFoundError as e:
            error_msg = f"File not found: {str(e)}"
            self.status_bar.showMessage(error_msg)
            QMessageBox.critical(self, "Error", error_msg)
        except Exception as e:
            error_msg = f"Encryption Error: {str(e)}"
            self.status_bar.showMessage(error_msg)
            QMessageBox.critical(self, "Error", error_msg)

    def handle_decrypt(self):
        path = self.path_edit.text()
        password = self.password_input.text()

        try:
            message = secure_hybrid_extract(path, password)

            if message:
                QMessageBox.information(self, "Decrypted Message", message)
                self.status_bar.showMessage("Message extracted successfully")
            else:
                self.status_bar.showMessage("No message found or extraction failed")
                QMessageBox.warning(self, "Warning", "No message found or extraction failed")
        except Exception as e:
            error_msg = f"Decryption Error: {str(e)}"
            self.status_bar.showMessage(error_msg)
            QMessageBox.critical(self, "Error", error_msg)

    def process_folder(self, folder_path, message):
        supported_ext = ('.png', '.jpg', '.jpeg', '.bmp')
        count = 0

        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(supported_ext):
                    file_path = os.path.join(root, file)
                    try:
                        secure_hybrid_embed(file_path, message)
                        count += 1
                    except Exception as e:
                        if debug:
                            print(f"Error processing {file}: {str(e)}")

        self.status_bar.showMessage(f"Processed {count} files successfully")

    def show_help(self):
        help_text = """Steganography Tool Help

Encryption Method:
- DCT: Better for JPEGs, survives format conversions and resizing.
- All types will be first converted to PNG and then repacked to original type.

Usage Tips:
1. For encryption: Select file/folder, enter message, choose method
2. For decryption: Select encrypted file (method detection is automatic)
3. DCT method survives LinkedIn-style JPEG conversion and resizing
4. EXIF data is preserved during encryption

Note: Always keep original files as some platforms may alter images"""
        QMessageBox.information(self, "Help", help_text)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
