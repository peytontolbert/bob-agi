"""
The interface that Bob uses to interact with the computer through browser audio.
"""

import logging
from app.agents.speech import SpeechAgent
import wave
from pathlib import Path

class Voice:
    def __init__(self, browser=None):
        self.browser = browser
        self.speech_agent = SpeechAgent()
        self.on_speak_callback = None
        # Connect to browser upon initialization
        self.connect_audio_output()

    def connect_audio_output(self):
        """
        Sets up the browser for audio output.
        """
        if self.browser:
            try:
                # Initialize browser audio output
                self.browser.driver.execute_script("""
                    if (!window.audioContext) {
                        window.audioContext = new AudioContext();
                    }
                """)
                logging.info("Voice connected to Browser audio output")
            except Exception as e:
                logging.error(f"Error connecting to browser audio: {e}")
        else:
            logging.warning("No browser available to connect to")

    def on_speak(self, callback):
        """
        Registers a callback to be called when speaking.
        """
        self.on_speak_callback = callback

    def start_speaking(self, message: str):
        """
        Starts the speaking process by streaming audio data to the browser.
        """
        # Convert the message to audio and get the file path
        audio_file_path = self.text_to_speech(message)
        
        # Stream to browser
        self._stream_to_browser(audio_file_path)
        
        logging.debug(f"Started speaking message: {message}")

    def _stream_to_browser(self, audio_file_path):
        """
        Stream audio to the browser output.
        """
        if not self.browser:
            return

        try:
            # Read the audio file and convert to base64
            with open(audio_file_path, 'rb') as audio_file:
                audio_data = audio_file.read()
            
            # JavaScript to play the audio in the browser
            script = """
                async function playAudio(audioData) {
                    try {
                        const arrayBuffer = await audioData.arrayBuffer();
                        const audioContext = window.audioContext || new AudioContext();
                        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
                        const source = audioContext.createBufferSource();
                        source.buffer = audioBuffer;
                        source.connect(audioContext.destination);
                        source.start(0);
                    } catch (error) {
                        console.error('Error playing audio:', error);
                    }
                }
                
                const blob = new Blob([new Uint8Array(arguments[0])], { type: 'audio/wav' });
                playAudio(blob);
            """
            
            self.browser.driver.execute_script(script, list(audio_data))
            logging.debug("Audio streamed to Browser")
        except Exception as e:
            logging.error(f"Error streaming audio to browser: {e}")

    def text_to_speech(self, text: str):
        """
        Converts text to audio data and returns the path to the audio file.
        """
        return self.speech_agent.complete_task(text)