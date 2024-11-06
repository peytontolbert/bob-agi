import unittest
from unittest.mock import patch, MagicMock
from app.env.computer.microphone import Microphone
import pyaudio

class TestMicrophone(unittest.TestCase):
    @patch('pyaudio.PyAudio')  # Patch PyAudio before setUp
    def setUp(self, mock_pyaudio):
        self.mock_pyaudio = mock_pyaudio
        self.mock_stream = MagicMock()
        self.mock_pyaudio_instance = self.mock_pyaudio.return_value
        self.mock_pyaudio_instance.open.return_value = self.mock_stream

        self.microphone = Microphone()
    
    def test_initialize(self):
        self.microphone.initialize()
        
        self.assertTrue(self.microphone.active)
        self.mock_pyaudio_instance.open.assert_called_with(
            format=pyaudio.paFloat32,
            channels=1,
            rate=44100,
            output=True
        )
        self.assertEqual(self.microphone.stream, self.mock_stream)
    
    def test_connect_application(self):
        stream_id = self.microphone.connect_application("test_app")
        self.assertIsNotNone(stream_id)
        self.assertIn("test_app", self.microphone.connected_apps)
        self.assertTrue(self.microphone.connected_apps["test_app"]['active'])

    def test_receive_audio_not_active(self):
        self.microphone.active = False
        self.microphone.stream = None
        self.microphone.receive_audio(b'sample data')  # Should do nothing

    @patch('app.env.microphone.logging')
    def test_receive_audio_exception(self, mock_logging):
        self.microphone.active = True
        mock_stream = MagicMock()
        mock_stream.write.side_effect = Exception("Test exception")
        self.microphone.stream = mock_stream
        self.microphone.receive_audio(b'sample data')
        mock_logging.error.assert_called_with("Error playing audio: Test exception")
    
    def tearDown(self):
        del self.microphone

if __name__ == '__main__':
    unittest.main()
