import logging
import time
import numpy as np

class Audio:
    def __init__(self):
        self.volume = 50
        self.muted = False
        self.active_streams = {}
        self.audio_callbacks = {}  # Store callbacks for different streams
        self.voice_client = None
        self.hearing = None  # Reference to Hearing class

    def set_hearing(self, hearing):
        """
        Sets the Hearing instance to receive audio callbacks.
        """
        self.hearing = hearing
        logging.debug("Hearing system connected to Audio.")

    def initialize(self):
        """Initializes audio system"""
        logging.info("Audio system initialized")

    def register_audio_callback(self, stream_id, callback):
        """Register a callback for processing audio data"""
        self.audio_callbacks[stream_id] = callback
        logging.debug(f"Registered audio callback for stream {stream_id}")
        
    def process_audio_data(self, stream_id, audio_data):
        """Process incoming audio data and route it to the appropriate callback"""
        if stream_id in self.audio_callbacks:
            # Convert audio data to numpy array if it isn't already
            if not isinstance(audio_data, np.ndarray):
                audio_data = np.frombuffer(audio_data, dtype=np.float32)
            
            # Apply volume adjustments
            if not self.muted:
                audio_data = audio_data * (self.volume / 100.0)
            else:
                audio_data = np.zeros_like(audio_data)
                
            # Send to callback
            self.audio_callbacks[stream_id](audio_data, None, None, None)
        else:
            logging.warning(f"No callback registered for stream {stream_id}")
            
    def route_output(self, app_name):
        """
        Routes audio from the application to the hearing system.
        """
        stream_id = f"{app_name}_audio_{int(time.time())}"
        self.active_streams[stream_id] = {
            "app": app_name,
            "volume": self.volume
        }
        if self.hearing:
            self.register_audio_callback(stream_id, self.hearing.audio_callback)
            logging.debug(f"Audio routed from {app_name} with stream id {stream_id}")
            return stream_id
        else:
            logging.error("Hearing system not set. Cannot route audio.")
            return None

