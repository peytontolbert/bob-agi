import logging
from app.env.audio import Audio
import asyncio
import websockets
import time
import pyaudio

class Microphone:
    def __init__(self):
        self.active = False
        self.connected_apps = {}
        self.audio_callback = None
        self.stream = None
        self.registered_voice = None
        self.pyaudio = pyaudio.PyAudio()
        
    def initialize(self):
        """Initialize microphone system"""
        self.active = True
        # Initialize audio output stream
        self.stream = self.pyaudio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=44100,
            output=True
        )
        
    def connect_application(self, app_name):
        """Connect an application to receive audio"""
        if app_name not in self.connected_apps:
            self.connected_apps[app_name] = {
                'active': True,
                'stream_id': f"{app_name}_mic_{int(time.time())}"
            }
            return self.connected_apps[app_name]['stream_id']
        return None

    def receive_audio(self, audio_data):
        """Receive audio data chunks and play them through the system"""
        if not self.active or not self.stream:
            return
            
        try:
            self.stream.write(audio_data)
        except Exception as e:
            logging.error(f"Error playing audio: {e}")

    def start_sending_audio(self):
        """Start sending audio stream"""
        self.active = True
        while self.active:
            # Simulate audio capture
            time.sleep(0.1)

    def register_voice(self, voice):
        """Register a voice instance to handle speech output"""
        self.registered_voice = voice
        logging.info("Voice registered with microphone")

    def __del__(self):
        """Cleanup audio resources"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.pyaudio:
            self.pyaudio.terminate()