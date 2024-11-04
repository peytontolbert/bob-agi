"""
This class is for the audio agent which handles speech-to-text and audio processing tasks
"""
from app.agents.base import BaseAgent
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import numpy as np
import logging
import time
from typing import Optional, Dict, Any
import queue
import collections

class AudioAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize Whisper models
        self._initialize_models()
        
        # Audio processing settings
        self.threshold = 0.05
        self.sample_rate = 16000
        self.channels = 1
        self.buffer_duration = 10  # seconds
        self.silence_duration = 1.0
        
        # Audio processing state
        self.is_buffering = False
        self.last_speech_time = None
        self.buffer = collections.deque(maxlen=int(self.buffer_duration * self.sample_rate))
        self.audio_queue = queue.Queue()
        
        logging.debug("AudioAgent initialized successfully")

    def _initialize_models(self):
        """Initialize Whisper models with error handling"""
        try:
            self.processor = WhisperProcessor.from_pretrained("openai/whisper-small")
            self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
            self.model_healthy = True
        except Exception as e:
            logging.error(f"Failed to initialize Whisper models: {e}")
            self.model_healthy = False
            raise RuntimeError("Audio system initialization failed")

    def transcribe_audio(self, audio_input: Dict[str, Any]) -> str:
        """
        Transcribe audio using Whisper model
        
        Args:
            audio_input: Dictionary containing audio array and sampling rate
            
        Returns:
            str: Transcribed text
        """
        try:
            input_features = self.processor(
                audio_input["array"], 
                sampling_rate=audio_input["sampling_rate"], 
                return_tensors="pt"
            ).input_features
            
            predicted_ids = self.model.generate(input_features)
            transcription = self.processor.batch_decode(
                predicted_ids, 
                skip_special_tokens=True
            )[0]
            
            logging.debug(f"Transcription completed: {transcription}")
            return transcription
            
        except Exception as e:
            logging.error(f"Error during transcription: {e}")
            return "Transcription failed."

    def process_audio_chunk(self, audio):
        """
        Process incoming audio chunk and add to buffer if speech detected
        
        Args:
            audio: Audio data as bytes or numpy array
        """
        # Ensure audio is numpy array
        if not isinstance(audio, np.ndarray):
            audio = np.frombuffer(audio, dtype=np.float32)
            
        rms = np.sqrt(np.mean(audio**2))
        current_time = time.time()
        
        if rms > self.threshold:
            if not self.is_buffering:
                logging.debug("Speech detected, starting buffer")
                self.is_buffering = True
            self.buffer.append(audio.copy())
            self.last_speech_time = current_time
        else:
            if self.is_buffering:
                if self.last_speech_time and (current_time - self.last_speech_time) > self.silence_duration:
                    logging.debug("Speech ended, processing buffer")
                    self.is_buffering = False
                    full_audio = np.concatenate(list(self.buffer))
                    self.buffer.clear()
                    self.audio_queue.put(full_audio)

    def get_next_transcription(self) -> str:
        """
        Get transcription of next audio segment in queue
        
        Returns:
            str: Transcribed text or status message
        """
        try:
            if not self.audio_queue.empty():
                audio = self.audio_queue.get(timeout=1.0)
                return self.transcribe_audio({
                    "array": audio, 
                    "sampling_rate": self.sample_rate
                })
        except queue.Empty:
            return "No speech detected."
        except Exception as e:
            logging.error(f"Error getting transcription: {e}")
            return "Error getting transcription."

    def listen_and_transcribe(self) -> str:
        """
        Wait for and transcribe next speech segment
        
        Returns:
            str: Transcribed text
        """
        logging.debug("Waiting for speech to transcribe...")
        while self.audio_queue.empty():
            time.sleep(0.1)
        return self.get_next_transcription()
