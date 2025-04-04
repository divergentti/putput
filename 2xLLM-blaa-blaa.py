"""
This script makes two LLM discuss together any topic you like. Qt6 implementation.
"""

import sys
import requests
import json
import time
from typing import Generator, Union
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget,
                             QTextEdit, QPushButton, QLineEdit, QLabel, QSpinBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QTextCursor, QColor, QTextCharFormat, QFont


class WorkerThread(QThread):
    update_signal = pyqtSignal(str, str)  # message, sender

    def __init__(self, conversation, initial_prompt, turns):
        super().__init__()
        self.conversation = conversation
        self.initial_prompt = initial_prompt
        self.turns = turns

    def run(self):
        current_prompt = self.initial_prompt

        for turn in range(self.turns):
            # LM Studio turn
            lm_response = self.conversation.query_lm_studio(current_prompt, stream=False)
            self.update_signal.emit(lm_response, "LM Studio")

            # Mistral turn
            mistral_response = self.conversation.query_mistral(lm_response, stream=False)
            self.update_signal.emit(mistral_response, "Mistral")

            current_prompt = mistral_response


class LLMConversation:
    def __init__(self, mistral_api_key: str, lm_studio_url: str = "http://localhost:1234/v1/chat/completions",
                 debug: bool = False):
        self.MISTRAL_API_KEY = mistral_api_key
        self.LM_STUDIO_URL = lm_studio_url
        self.MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"
        self.debug = debug

        self.mistral_headers = {
            "Authorization": f"Bearer {self.MISTRAL_API_KEY}",
            "mistral-version": "2025-01-01",
            "Content-Type": "application/json",
        }

    def query_lm_studio(self, prompt: str, stream: bool = False) -> Union[str, Generator[str, None, None]]:
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 500,
            "stream": stream
        }

        try:
            response = requests.post(self.LM_STUDIO_URL, json=payload, timeout=60)
            if response.status_code != 200:
                raise Exception(f"HTTP Error {response.status_code}: {response.text}")
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error: {str(e)}"

    def query_mistral(self, prompt: str, stream: bool = False) -> Union[str, Generator[str, None, None]]:
        payload = {
            "model": "mistral-large-latest",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 500,
            "stream": stream
        }

        try:
            response = requests.post(self.MISTRAL_URL, headers=self.mistral_headers, json=payload, timeout=30)
            if response.status_code != 200:
                raise Exception(f"HTTP Error {response.status_code}: {response.text}")
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error: {str(e)}"


class ConversationWindow(QMainWindow):
    def __init__(self, conversation):
        super().__init__()
        self.conversation = conversation
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle("LLM Conversation")
        self.setGeometry(100, 100, 800, 600)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()

        # Conversation display
        self.conversation_display = QTextEdit()
        self.conversation_display.setReadOnly(True)
        self.conversation_display.setStyleSheet("""
            QTextEdit {
                background-color: #f5f5f5;
                border: 1px solid #ccc;
                padding: 10px;
            }
        """)
        layout.addWidget(self.conversation_display)

        # Controls
        control_layout = QVBoxLayout()

        # Initial prompt
        self.prompt_label = QLabel("Initial Prompt:")
        self.prompt_input = QLineEdit(
            "Hello! Let's discuss the future of AI. What do you think will be the most significant advancement in the next 5 years?")
        control_layout.addWidget(self.prompt_label)
        control_layout.addWidget(self.prompt_input)

        # Turns selection
        self.turns_label = QLabel("Conversation Turns:")
        self.turns_input = QSpinBox()
        self.turns_input.setRange(1, 20)
        self.turns_input.setValue(4)
        control_layout.addWidget(self.turns_label)
        control_layout.addWidget(self.turns_input)

        # Start button
        self.start_button = QPushButton("Start Conversation")
        self.start_button.clicked.connect(self.start_conversation)
        control_layout.addWidget(self.start_button)

        layout.addLayout(control_layout)
        central_widget.setLayout(layout)

    def start_conversation(self):
        self.conversation_display.clear()
        self.start_button.setEnabled(False)

        # Create and start worker thread
        self.worker = WorkerThread(
            self.conversation,
            self.prompt_input.text(),
            self.turns_input.value()
        )
        self.worker.update_signal.connect(self.update_conversation)
        self.worker.finished.connect(lambda: self.start_button.setEnabled(True))
        self.worker.start()

    def update_conversation(self, message, sender):
        cursor = self.conversation_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        # Create format based on sender
        format = QTextCharFormat()
        font = QFont()
        font.setBold(True)
        format.setFont(font)

        if sender == "Mistral":
            format.setForeground(QColor(0, 0, 255))  # Blue
            prefix = "Mistral: "
        else:
            format.setForeground(QColor(255, 0, 0))  # Red
            prefix = "LM Studio: "

        # Insert sender prefix
        cursor.insertText(prefix, format)

        # Insert message (normal text)
        normal_format = QTextCharFormat()
        normal_format.setForeground(QColor(0, 0, 0))  # Black
        cursor.insertText(message + "\n\n", normal_format)

        # Scroll to bottom
        self.conversation_display.ensureCursorVisible()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Initialize conversation with your Mistral API key
    conversation = LLMConversation(
        mistral_api_key="create at https://console.mistral.ai/api-keys",
        debug=True
    )

    window = ConversationWindow(conversation)
    window.show()
    sys.exit(app.exec())
