"""
The interface for Bob's hearing for the computer.
"""

from transformers import WhisperProcessor, WhisperForConditionalGeneration

import sounddevice as sd
import numpy as np
import logging  # Add logging for debugging
import threading  # Add threading for handling callbacks
import queue  # Add queue to communicate between threads
import collections  # {{ Added for buffering audio }}
import time  # {{ Added import for time handling }}
from app.env.audio import Audio  # Updated import

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class Hearing:
    def __init__(self, audio_system):
        self.audio = audio_system
        self.active = False
        self.audio_buffer = []
        
    def get_audio_input(self):
        """Get current audio input"""
        try:
            # Return the most recent audio buffer
            if self.audio_buffer:
                return self.audio_buffer[-1]
            return None
        except Exception as e:
            logging.error(f"Error getting audio input: {e}")
            return None
            
    def audio_callback(self, audio_data, *args):
        """Handle incoming audio data"""
        if self.active:
            self.audio_buffer.append(audio_data)
            # Keep buffer at reasonable size
            if len(self.audio_buffer) > 100:
                self.audio_buffer.pop(0)