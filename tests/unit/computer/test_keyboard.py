import unittest
from unittest.mock import MagicMock
from app.env.keyboard import Keyboard
import threading
import time

class TestKeyboard(unittest.TestCase):
    def setUp(self):
        # Create a mock target with a receive_input method
        self.mock_target = MagicMock()
        self.keyboard = Keyboard(target=self.mock_target, typing_speed=10)
    
    def test_send_input(self):
        # Test that send_input adds text to the input_queue
        self.keyboard.send_input("Hello")
        self.assertIn("Hello", self.keyboard.input_queue)
    
    def test_process_queue_sends_input(self):
        # Start the keyboard processing
        self.keyboard.start()
        self.keyboard.send_input("A")
        
        # Allow some time for the process_queue to process input
        time.sleep(0.2)
        
        # Check that receive_input was called with 'A'
        self.mock_target.receive_input.assert_called_with('A')
        
        # Stop the keyboard processing
        self.keyboard.stop()
    
    def test_start_and_stop(self):
        # Test starting and stopping the keyboard thread
        self.keyboard.start()
        self.assertTrue(self.keyboard.is_running())
        self.keyboard.stop()
        self.assertFalse(self.keyboard.is_running())

if __name__ == '__main__':
    unittest.main()
