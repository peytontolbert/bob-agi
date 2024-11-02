import logging
from app.env.audio import Audio
import asyncio
import websockets

class Microphone:
    def __init__(self):
        self.audio_stream = Audio()
        self.voice = None
        self.active = False
        self.connected_apps = []
        self.initialize()

    def initialize(self):
        """Initializes microphone and starts listening"""
        self.active = True
        logging.info("Microphone initialized and ready for input")

    def register_voice(self, voice):
        """Registers the voice component with the microphone"""
        self.voice = voice
        logging.info("Voice component registered with Microphone")

    def receive_audio(self, audio_data):
        """Receives audio data and routes it to connected applications"""
        if self.active:
            logging.debug(f"Routing audio data to {len(self.connected_apps)} connected applications")
            for app in self.connected_apps:
                try:
                    app.receive_audio(audio_data)
                    logging.debug(f"Audio routed to application: {app}")
                except Exception as e:
                    logging.error(f"Failed to route audio to application {app}: {str(e)}")
        else:
            logging.warning("Microphone is not active. Audio data not routed.")

    def connect_application(self, app):
        """Connects an application to the audio stream."""
        if app not in self.connected_apps:
            self.connected_apps.append(app)
            logging.info(f"Application {app} connected to microphone audio stream.")
        else:
            logging.warning(f"Application {app} is already connected.")

    def disconnect_application(self, app):
        """Disconnects an application from the audio stream."""
        if app in self.connected_apps:
            self.connected_apps.remove(app)
            logging.info(f"Application {app} disconnected from microphone audio stream.")
        else:
            logging.warning(f"Attempted to disconnect unknown application {app}.")

    async def send_audio_stream(self):
        """Streams audio data to websocket clients"""
        async with websockets.connect("ws://localhost:6789") as websocket:
            while self.active:
                audio_data = self.audio_stream.get_audio_data()
                await websocket.send(audio_data)
                await asyncio.sleep(0.1)

    def start_sending_audio(self):
        """Starts the audio streaming process"""
        asyncio.run(self.send_audio_stream())