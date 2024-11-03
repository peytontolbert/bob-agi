import logging
from app.env.audio import Audio
import asyncio
import websockets
import time

class Microphone:
    def __init__(self):
        self.active = False
        self.connected_apps = {}
        self.audio_callback = None
        self.stream = None
        self.registered_voice = None
        
    def initialize(self):
        """Initialize microphone system"""
        self.active = True
        
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
        """Receive audio data and route to connected applications"""
        if not self.active:
            return
            
        for app_name, app_info in self.connected_apps.items():
            if app_info['active']:
                try:
                    # Route audio to connected application
                    if hasattr(app_name, 'receive_audio'):
                        app_name.receive_audio(audio_data)
                except Exception as e:
                    logging.error(f"Error routing audio to {app_name}: {e}")

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