"""
This is the audio agent that completes audio tasks for the input
"""

from app.agents.base import BaseAgent
from pathlib import Path

class SpeechAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.speech_file_path = Path(__file__).parent / "speech.mp3"

    def complete_task(self, input):
        response = self.client.audio.speech.create(
            model="tts-1",
            input=input,
            voice="alloy",
        )
        # Save the file and return the path
        response.stream_to_file(self.speech_file_path)
        return self.speech_file_path
