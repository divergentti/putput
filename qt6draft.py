"""
Version 0.0.1 by Divergentti / Jari Hiltunen (07.04.2025)

Steganography ≠ Encryption: It hides messages but doesn’t make them unreadable without extraction.
Basic image analysis tools can detect hidden data. For confidentiality, encrypt data first
(e.g., AES) before embedding. This script also supports non-secret use cases like copyright watermarking
(e.g., embedding invisible ownership markers).

PNG vs. JPEG Workflows:
    PNG’s lossless compression allows bit-level edits with minimal artifacts.
    JPEG’s lossy DCT-based compression distorts hidden data during conversions.


Capacity Limits: Data size depends on image resolution/color depth.
Small messages work well; larger ones risk visible distortions (1 bit ≈ 1 pixel/subpixel).

1. Adaptive LSB Modification

Analyzes image complexity in local 3x3 pixel regions
Uses 1-3 LSBs per channel based on region complexity
Hides more data in textured/complex areas (less noticeable)
Preserves image quality in smooth areas

2. DCT-based Steganography

Operates in frequency domain using Discrete Cosine Transform
Embeds data in mid-frequency DCT coefficients (less perceptible)
Converts image to YCbCr color space and modifies Y channel
More resistant to statistical analysis than spatial domain methods
"""

import sys
import os
from PIL import Image, ExifTags
from PIL.ExifTags import TAGS
import numpy as np
from scipy.fftpack import dct, idct

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QRadioButton, QGroupBox, QLineEdit, QPushButton, QComboBox,
    QLabel, QTextEdit, QFileDialog, QStatusBar, QMessageBox, QProgressBar
)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt
from PyQt6.QtCore import QObject, pyqtSignal, QRunnable, QThreadPool

debug_extract = True
debug_embed = True
debug_gui = True


class WorkerSignals(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(int)  # Percentage (0-100)
    status = pyqtSignal(str)    # Text updates ("Encrypting...")
    result = pyqtSignal(object) # Return value (e.g., output path)

class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    def run(self):
        try:
            self.signals.status.emit("Working... wait ...")
            result = self.fn(*self.args, **self.kwargs)
            self.signals.result.emit(result)
        except Exception as e:
            self.signals.status.emit(f"Error: {str(e)}")
        finally:
            self.signals.finished.emit()

class StegaMachine:
    START_CHAR = '#'  # Choose unique characters not likely to appear in the image or message
    STOP_CHAR = '$'

    def __init__(self):
        pass

    # ------------------ Adaptive LSB Implementation ------------------

    def pixel_complexity(self, pixel_region):
        """Calculate local complexity for adaptive LSB."""
        # Calculate standard deviation as complexity measure
        return np.std(pixel_region)

    def get_embedding_capacity(self, complexity, threshold_low=5, threshold_high=15):
        # Determine number of LSBs to use based on complexity.
        if complexity < threshold_low:
            return 1  # Low complexity - use only 1 LSB
        elif complexity < threshold_high:
            return 2  # Medium complexity - use 2 LSBs
        else:
            return 3  # High complexity - use 3 LSBs

    def adaptive_lsb_embed(self, img, binary_message):
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
                complexity = self.pixel_complexity(region)

                # Determine embedding capacity
                capacity = self.get_embedding_capacity(complexity)

                # Embed in center pixel with determined capacity
                pixel = list(img.getpixel((x + 1, y + 1)))

                if debug_embed:
                    print(f"Adaptive LSB embed Complexity: {complexity}")
                    print(f"Adaptive LSB embed Capacity: {capacity}")
                    print(f"Adaptive LSB embed Center pixel: {pixel}")


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

    def adaptive_lsb_extract(self, img, message_length):
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
                complexity = self.pixel_complexity(region)

                # Determine embedding capacity
                capacity = self.get_embedding_capacity(complexity)

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

        if debug_extract:
            print(f"Adaptive LSB extract Binary message: {binary_message}")

        return binary_message

    # ------------------ DCT Implementation ------------------

    def rgb_to_ycbcr(self, img):
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

    def ycbcr_to_rgb(self, ycbcr):
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

    def embed_in_dct_block(self, block, bits, alpha=5):
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

        if debug_embed:
            print(f"Embed in DCT_block: {idct_block}")

        return idct_block

    def extract_from_dct_block(self, block):
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

    def dct_embed(self, img, binary_message):
        """Embed message using DCT-based steganography."""
        # Convert to YCbCr
        ycbcr = self.rgb_to_ycbcr(img)

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
                modified_block = self.embed_in_dct_block(block, bits_to_embed)

                # Update block
                ycbcr[y:y + 8, x:x + 8, 0] = modified_block

        # Convert back to RGB
        rgb = self.ycbcr_to_rgb(ycbcr)

        # Create new image
        dct_img = Image.fromarray(rgb)
        return dct_img, data_index

    def dct_extract(self, img, message_length):
        """Extract message using DCT-based steganography."""
        # Convert to YCbCr
        ycbcr = self.rgb_to_ycbcr(img)

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
                bits = self.extract_from_dct_block(block)

                # Add bits to message
                remaining = bits_needed - len(binary_message)
                binary_message += bits[:min(5, remaining)]

        if debug_extract:
            print(f"DCT Extract binary message first 30 bits {binary_message[:30]}")

        return binary_message

    # ------------------ Hybrid Steganography Functions ------------------

    def hybrid_embed_message(self, image_path, message, progress_callback=None):
        """Embeds a message into an image, preserving original format when possible."""
        try:
            # Convert to absolute path
            abs_path = os.path.abspath(image_path)
            if not os.path.exists(abs_path):
                raise FileNotFoundError(f"Image file not found at {abs_path}")

            img = Image.open(abs_path)
            img = img.convert("RGB")

            if progress_callback:
                progress_callback(10)

            # Process message and embed (existing code)
            full_message = self.START_CHAR + message + self.STOP_CHAR
            binary_message = ''.join(format(ord(char), '08b') for char in full_message)
            message_length = len(binary_message)
            length_binary = format(message_length, '032b')

            img_copy = img.copy()
            img_with_length, _ = self.adaptive_lsb_embed(img_copy, length_binary)
            modified_img, embedded_bits = self.dct_embed(img_with_length, binary_message)

            if progress_callback:
                progress_callback(50)

            if embedded_bits < message_length:
                raise ValueError("Could not embed entire message.")

            # --- Enhanced Save Logic ---
            original_dir = os.path.dirname(abs_path)
            original_name = os.path.basename(abs_path)
            base_name, original_ext = os.path.splitext(original_name)
            original_ext = original_ext.lower()

            if progress_callback:
                progress_callback(70)

            # Always embed to PNG first (temporary if original isn't PNG)
            temp_png_path = os.path.join(original_dir, f"temp_embedded_{base_name}.png")
            modified_img.save(temp_png_path, format='PNG')

            # Case 1: Original is PNG -> Keep PNG output
            if original_ext == '.png':
                final_path = os.path.join(original_dir, f"encrypted_{base_name}.png")
                os.replace(temp_png_path, final_path)  # Atomic rename
                if progress_callback:
                    progress_callback(100)
                return final_path

            # Case 2: Original is JPEG/BMP -> Convert back to original format
            else:
                final_path = os.path.join(original_dir, f"encrypted_{base_name}{original_ext}")

                # High-quality conversion for JPEG
                if original_ext in ('.jpg', '.jpeg'):
                    Image.open(temp_png_path).save(
                        final_path,
                        format='JPEG',
                        quality=95,  # Minimize compression artifacts
                        subsampling=0  # 4:4:4 chroma (no subsampling)
                    )
                # Lossless conversion for other formats (BMP, etc.)
                else:
                    Image.open(temp_png_path).save(final_path, format=original_ext[1:].upper())

                os.remove(temp_png_path)  # Clean up temporary PNG
                if progress_callback:
                    progress_callback(100)

                return final_path

        except Exception as e:
            # Clean up temp file if something failed
            if 'temp_png_path' in locals() and os.path.exists(temp_png_path):
                os.remove(temp_png_path)
            raise e

        except FileNotFoundError:
            print(f"Error: Image file not found at {image_path}")
            return None
        except ValueError as e:
            print(f"ValueError: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None

    def hybrid_extract_message(self, image_path, progress_callback=None):
        """Extracts a message from an image using the hybrid steganography method."""
        try:
            img = Image.open(image_path)
            img = img.convert("RGB")

            # First extract the message length using Adaptive LSB
            length_binary = self.adaptive_lsb_extract(img, 32)
            message_length = int(length_binary, 2)

            if progress_callback:
                progress_callback(10)

            # Extract main message using DCT
            binary_message = self.dct_extract(img, message_length)

            if progress_callback:
                progress_callback(30)  # After DCT extraction

            # Convert binary to characters
            chars = []
            for i in range(0, len(binary_message), 8):
                byte = binary_message[i:i + 8]
                if len(byte) == 8:  # Ensure we have a full byte
                    chars.append(chr(int(byte, 2)))
            extracted = ''.join(chars)

            if progress_callback:
                progress_callback(50)

            # Find start and stop markers
            start_index = extracted.find(self.START_CHAR)
            if start_index == -1:
                print("Start marker not found.")
                return None

            if progress_callback:
                progress_callback(80)

            stop_index = extracted.find(self.STOP_CHAR, start_index + 1)
            if stop_index == -1:
                print("Stop marker not found.")
                return None

            # Extract message between markers

            if progress_callback:
                progress_callback(100)
            return extracted[start_index + 1:stop_index]

        except FileNotFoundError:
            print(f"Error: Image file not found at {image_path}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
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

        # Message input
        self.message_label = QLabel("Secret Message:")
        self.message_input = QLineEdit()
        layout.addWidget(self.message_label)
        layout.addWidget(self.message_input)

        # Action button
        self.action_btn = QPushButton("Encrypt")
        layout.addWidget(self.action_btn)

        # Preview area - modified with file info
        preview_layout = QHBoxLayout()

        # Image preview (left side)
        self.preview_label = QLabel()
        self.preview_label.setFixedSize(400, 400)
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_layout.addWidget(self.preview_label)

        # Right side container (file info + EXIF)
        right_layout = QVBoxLayout()

        # File information group
        file_info_group = QGroupBox("File Information")
        file_info_layout = QVBoxLayout()
        self.file_size_label = QLabel("Size: N/A")
        self.file_type_label = QLabel("Type: N/A")
        file_info_layout.addWidget(self.file_size_label)
        file_info_layout.addWidget(self.file_type_label)
        file_info_group.setLayout(file_info_layout)
        right_layout.addWidget(file_info_group)

        # EXIF data (existing)
        self.exif_text = QTextEdit()
        self.exif_text.setReadOnly(True)
        right_layout.addWidget(self.exif_text)

        preview_layout.addLayout(right_layout)
        layout.addLayout(preview_layout)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Connect signals
        self.encrypt_radio.toggled.connect(self.update_ui_mode)
        self.browse_btn.clicked.connect(self.handle_browse)
        self.action_btn.clicked.connect(self.handle_action)
        self.file_radio.toggled.connect(self.update_path_field)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(True)
        # Add under progress bar setup
        self.progress_label = QLabel("Ready")
        layout.insertWidget(6, self.progress_label)
        layout.addWidget(self.progress_bar)
        self.threadpool = QThreadPool()  # Initialize thread pool

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

    # Update the update_preview() method:
    def update_preview(self, path):
        # Update image preview
        pixmap = QPixmap(path)
        if not pixmap.isNull():
            scaled = pixmap.scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio)
            self.preview_label.setPixmap(scaled)

            # Update file info
            try:
                # File size
                size_bytes = os.path.getsize(path)
                size_kb = size_bytes / 1024
                self.file_size_label.setText(f"Size: {size_kb:.2f} KB")

                # File type
                file_ext = os.path.splitext(path)[1].upper().replace('.', '') or 'Unknown'
                self.file_type_label.setText(f"Type: {file_ext}")
            except Exception as e:
                self.file_size_label.setText("Size: Error")
                self.file_type_label.setText("Type: Error")
        else:
            self.preview_label.clear()
            self.file_size_label.setText("Size: N/A")
            self.file_type_label.setText("Type: N/A")

        # Update EXIF data (existing)
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
        message = self.message_input.text()

        if not path:
            self.status_bar.showMessage("Please select an input file or folder!")
            return
        if not message:
            self.status_bar.showMessage("Message is required!")
            return

        # Disable UI during operation
        self.action_btn.setEnabled(False)
        self.progress_bar.setValue(0)

        def on_progress(percent):
            phases = {
                5: "Starting...",
                10: "Loading image...",
                30: "Embedding LSB...",
                50: "Embedding DCT...",
                70: "Saving temporary file...",
                90: "Final conversion...",
                100: "Done"
            }
            self.progress_bar.setValue(percent)
            self.progress_label.setText(phases.get(percent, "Working..."))
            self.progress_bar.setValue(percent)
            if percent == 100:
                self.status_bar.showMessage("Encryption complete!")
            elif percent == -1:
                self.status_bar.showMessage("Encryption failed")

        def on_result(output_path):
            self.action_btn.setEnabled(True)
            if output_path:
                self.update_preview(output_path)
                QMessageBox.information(self, "Success",
                                        f"Saved to {output_path}")

        def on_error(e):
            self.action_btn.setEnabled(True)
            QMessageBox.critical(self, "Error", str(e))

        # Create and start worker
        worker = Worker(
            lambda: machine.hybrid_embed_message(
                path,
                message,
                progress_callback=on_progress
            )
        )
        worker.signals.result.connect(on_result)
        worker.signals.status.connect(self.status_bar.showMessage)
        worker.signals.finished.connect(lambda: self.action_btn.setEnabled(True))
        self.threadpool.start(worker)

    def on_encrypt_finished(self, output_path):
        self.action_btn.setEnabled(True)
        self.update_preview(output_path)
        QMessageBox.information(self, "Success", f"Saved to {output_path}")


    def handle_decrypt(self):
        path = self.path_edit.text()  # Get the input path from the UI
        if not path:
            self.status_bar.showMessage("Please select an input file or folder!")
            return

        # Disable UI during operation
        self.action_btn.setEnabled(False)
        self.progress_bar.setValue(0)  # Initialize progress bar

        def on_progress(percent):
            """Callback function to update the progress bar."""
            phases = {
                10: "Extracting length...",
                30: "Analyzing DCT...",
                50: "Decoding message...",
                80: "Verifying markers...",
                100: "Done"
            }
            self.progress_bar.setValue(percent)
            self.progress_label.setText(phases.get(percent, "Processing..."))
            self.progress_bar.setValue(percent)
            if percent == 100:
                self.status_bar.showMessage("Decryption complete!")
            elif percent == -1:  # Error indicator
                self.status_bar.showMessage("Decryption failed")

        def on_result(message):
            """Callback function to handle the decryption result."""
            self.action_btn.setEnabled(True) # Re-enable UI
            if message:
                QMessageBox.information(self, "Decrypted Message", message)
                self.status_bar.showMessage("Message extracted successfully")
            else:
                self.status_bar.showMessage("No message found or extraction failed")
                QMessageBox.warning(self, "Warning", "No message found or extraction failed")

        def on_error(e):
            """Callback function to handle errors during decryption."""
            self.action_btn.setEnabled(True) # Re-enable UI
            self.status_bar.showMessage(f"Decryption Error: {str(e)}")
            QMessageBox.critical(self, "Error", str(e))

        # Create and start worker thread
        worker = Worker(
            lambda: machine.hybrid_extract_message(path, progress_callback=on_progress) # Pass the callback to the function
        )
        worker.signals.status.connect(self.status_bar.showMessage)  # Connect status signal
        worker.signals.result.connect(on_result) #Connect result signal
        worker.signals.finished.connect(lambda: self.action_btn.setEnabled(True)) # Re-enable UI when finished
        self.threadpool.start(worker)

    def process_folder(self, folder_path, message):
        supported_ext = ('.png', '.jpg', '.jpeg', '.bmp')
        count = 0

        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(supported_ext):
                    file_path = os.path.join(root, file)
                    try:
                        # Force output to PNG
                        base_name = os.path.splitext(file)[0]
                        output_path = os.path.join(root, f"encrypted_{base_name}.png")
                        machine.hybrid_embed_message(file_path, message)
                        count += 1
                    except Exception as e:
                        print(f"Error processing {file}: {str(e)}")

        self.status_bar.showMessage(f"Processed {count} files successfully")

    def show_help(self):
        help_text = """Steganography Tool Help
This tool will embed or extract message. New file will be
named encrypted_[original filename]. Process will always save png file
temporarily and convert file to jpg, jpeg or bmp if needed. This way
embedding is not tampered with lossy formatting.

Usage Tips:
1. For encryption: Select file/folder, enter message, choose method
2. For decryption: Select encrypted file (method detection is automatic)
3. DCT method survives LinkedIn-style JPEG conversion and resizing
4. EXIF data is preserved during encryption

Note: Always keep original files as some platforms may alter images"""
        QMessageBox.information(self, "Help", help_text)

if __name__ == "__main__":
    machine = StegaMachine()
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
