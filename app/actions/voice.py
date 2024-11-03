"""
The interface that Bob uses to interact with the computer.
"""

import logging
from app.agents.speech import SpeechAgent
import wave
import pyaudio
from pathlib import Path

class Voice:
    def __init__(self, computer_microphone):
        self.microphone = computer_microphone
        self.speech_agent = SpeechAgent()
        self.on_speak_callback = None
        # Connect to microphone upon initialization
        self.connect_microphone()

    def connect_microphone(self):
        """
        Sets up the microphone for interaction and registers the voice with it.
        """
        if self.microphone:
            stream_id = self.microphone.connect_application("voice")
            self.microphone.register_voice(self)
            logging.info(f"Voice connected to Microphone with stream ID: {stream_id}")
        else:
            logging.error("No microphone available to connect to")

    def on_speak(self, callback):
        """
        Registers a callback to be called when speaking.
        """
        self.on_speak_callback = callback

    def start_speaking(self, message: str):
        """
        Starts the speaking process by streaming audio data to the Microphone.
        """
        # Convert the message to audio and get the file path
        audio_file_path = self.text_to_speech(message)
        
        # Stream the audio file to the microphone
        chunk_size = 1024
        
        try:
            with wave.open(str(audio_file_path), 'rb') as audio_file:
                audio_data = audio_file.readframes(chunk_size)
                while audio_data:
                    self.microphone.receive_audio(audio_data)
                    audio_data = audio_file.readframes(chunk_size)
                    
            logging.debug(f"Audio streamed to Microphone: {message}")
        except Exception as e:
            logging.error(f"Error streaming audio: {e}")

    def text_to_speech(self, text: str):
        """
        Converts text to audio data and returns the path to the audio file.
        """
        return self.speech_agent.complete_task(text)