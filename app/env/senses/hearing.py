"""
The interface for Bob's hearing for the computer.
"""
from app.agents.audio import AudioAgent
import logging
import numpy as np
from app.env.computer.audio import Audio

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class Hearing:
    def __init__(self, browser):
        self.browser = browser
        self.agent = AudioAgent()
        self.audio_buffer = []
        self.active = False
        self.is_listening = False
        logging.debug("Hearing system initialized with AudioAgent")

    def start_listening(self):
        """Start listening to browser audio stream"""
        try:
            if not self.is_listening:
                self.is_listening = True
                self.browser.start_audio_capture(self.audio_callback)
                logging.debug("Started listening to browser audio stream")
        except Exception as e:
            logging.error(f"Error starting audio listening: {e}")

    def stop_listening(self):
        """Stop listening to browser audio stream"""
        try:
            if self.is_listening:
                self.is_listening = False
                self.browser.stop_audio_capture()
                logging.debug("Stopped listening to browser audio stream")
        except Exception as e:
            logging.error(f"Error stopping audio listening: {e}")

    def get_audio_input(self):
        """Get current audio input"""
        try:
            if self.audio_buffer:
                return self.audio_buffer[-1]
            return None
        except Exception as e:
            logging.error(f"Error getting audio input: {e}")
            return None

    def audio_callback(self, audio_chunk):
        """Process incoming audio through the AudioAgent"""
        try:
            if self.is_listening:
                self.audio_buffer.append(audio_chunk)
                if len(self.audio_buffer) > 100:  # Keep last 100 chunks
                    self.audio_buffer.pop(0)
                self.agent.process_audio_chunk(audio_chunk)
        except Exception as e:
            logging.error(f"Error in audio callback: {e}")

    def get_transcription(self):
        """Get next transcription from AudioAgent"""
        return self.agent.get_next_transcription()

    def listen_and_transcribe(self):
        """Listen for speech and get transcription"""
        return self.agent.listen_and_transcribe()