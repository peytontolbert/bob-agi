"""
The interface for Bob's hearing for the computer.
"""
from app.agents.audio import AudioAgent
import logging
import numpy as np
from app.env.audio import Audio

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class Hearing:
    def __init__(self, audio: Audio):
        self.audio = audio
        self.agent = AudioAgent()
        self.audio.set_hearing(self)
        self.audio_buffer = []
        self.active = False
        logging.debug("Hearing system initialized with AudioAgent")

    def get_audio_input(self):
        """Get current audio input"""
        try:
            if self.audio_buffer:
                return self.audio_buffer[-1]
            return None
        except Exception as e:
            logging.error(f"Error getting audio input: {e}")
            return None

    def receive_audio(self, audio_stream):
        """Receives audio data from the Computer's Audio system."""
        logging.debug("Receiving audio from Computer's Audio system.")
        for audio_chunk in audio_stream:
            self.audio_callback(audio_chunk, None, None, None)

    def audio_callback(self, audio, frames, time_info, status):
        """Process incoming audio through the AudioAgent"""
        self.agent.process_audio_chunk(audio)

    def get_transcription(self):
        """Get next transcription from AudioAgent"""
        return self.agent.get_next_transcription()

    def listen_and_transcribe(self):
        """Listen for speech and get transcription"""
        return self.agent.listen_and_transcribe()