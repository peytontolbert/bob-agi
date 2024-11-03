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
    def __init__(self, audio: Audio, threshold=0.05):
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
        self.threshold = threshold
        self.audio_queue = queue.Queue()  # Queue to handle incoming audio
        self.buffer_duration = 10  # seconds
        self.sample_rate = 16000
        self.channels = 1
        self.is_buffering = False
        self.last_speech_time = None
        self.silence_duration = 1.0
        self.buffer = collections.deque(maxlen=int(self.buffer_duration * self.sample_rate))
        self.audio = audio
        self.audio.set_hearing(self)  # Register Hearing with Audio
        logging.debug("Hearing connected to Audio system.")
        print("audio connected")
        self.audio_buffer = []
        self.active = False
        
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
    def receive_audio(self, audio_stream):
        """
        Receives audio data from the Computer's Audio system.
        """
        logging.debug("Receiving audio from Computer's Audio system.")
        # Assume audio_stream is an iterable of audio chunks
        for audio_chunk in audio_stream:
            self.audio_callback(audio_chunk, None, None, None)
    def transcribe_audio(self, audio_input):
        try:
            input_features = self.processor(audio_input["array"], sampling_rate=audio_input["sampling_rate"], return_tensors="pt").input_features
            predicted_ids = self.model.generate(input_features)
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            logging.debug(f"Transcription: {transcription}")
            return transcription
        except Exception as e:
            logging.error(f"Error during transcription: {e}")
            return "Transcription failed."
    def audio_callback(self, audio, frames, time_info, status):
        # Ensure audio is numpy array
        if not isinstance(audio, np.ndarray):
            audio = np.frombuffer(audio, dtype=np.float32)
            
        rms = np.sqrt(np.mean(audio**2))
        
        current_time = time.time()
        if rms > self.threshold:
            if not self.is_buffering:
                logging.debug("Speech started, initiating buffering.")
                self.is_buffering = True
            self.buffer.append(audio.copy())
            self.last_speech_time = current_time
        else:
            if self.is_buffering:
                if self.last_speech_time and (current_time - self.last_speech_time) > self.silence_duration:
                    logging.debug("Speech ended after silence, processing buffered audio.")
                    self.is_buffering = False
                    full_audio = np.concatenate(list(self.buffer))
                    self.buffer.clear()
                    self.audio_queue.put(full_audio)
    def get_transcription(self):
        try:
            if not self.audio_queue.empty():
                audio = self.audio_queue.get(timeout=1.0)
                return self.transcribe_audio({"array": audio, "sampling_rate": self.sample_rate})
        except queue.Empty:
            return "No speech detected."
        except Exception as e:
            logging.error(f"Error getting transcription: {e}")
            return "Error getting transcription."
    def listen_and_transcribe(self):
        logging.debug("Processing incoming audio for transcription.")
        while self.audio_queue.empty():
            time.sleep(0.1)
        transcription = self.get_transcription()
        logging.debug("Transcription completed.")
        return transcription