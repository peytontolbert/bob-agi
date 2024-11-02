"""
The interface that Bob uses to interact with the computer.
"""

import logging
from app.agents.speech import SpeechAgent

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
            self.microphone.register_voice(self)
            logging.info("Voice connected to Microphone")
        else:
            logging.error("No microphone available to connect to")

    def on_speak(self, callback):
        """
        Registers a callback to be called when speaking.
        """
        self.on_speak_callback = callback

    def start_speaking(self, message: str):
        """
        Starts the speaking process by sending audio data to the Microphone.
        """
        # Convert the message to audio data
        audio_data = self.text_to_speech(message)
        # Send directly to microphone instead of using callback
        self.microphone.receive_audio(audio_data)
        logging.debug(f"Audio data sent to Microphone: {message}")

    def text_to_speech(self, text: str):
        """
        Converts text to audio data.
        """
        return self.speech_agent.complete_task(text)