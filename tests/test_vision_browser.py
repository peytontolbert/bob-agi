import unittest
from unittest.mock import MagicMock, patch
from app.actions.eyesight import Eyesight
from app.env.screen import Screen
from app.env.browser import Browser
from app.agents.vision import VisionAgent
from PIL import Image
import numpy as np
import time

class TestVisionBrowserIntegration(unittest.TestCase):
    def setUp(self):
        # Initialize the mocked Screen
        self.mock_screen = MagicMock(spec=Screen)
        
        # Initialize the mocked Browser
        self.mock_browser = MagicMock(spec=Browser)
        
        # Initialize the Eyesight with mocked Screen
        self.eyesight = Eyesight(screen=self.mock_screen)
        
        # Mock the VisionAgent within Eyesight
        self.eyesight.vision_agent = MagicMock(spec=VisionAgent)
        
        # Create a sample image for testing
        self.sample_image = Image.new('RGB', (800, 600), color = 'blue')
        self.sample_image_np = np.array(self.sample_image)
        
        # Mock the screen's get_current_frame method
        self.mock_screen.get_current_frame.return_value = self.sample_image

    @patch('app.actions.eyesight.VisionAgent.complete_task')
    def test_complete_task_with_browser_image(self, mock_complete_task):
        # Define what the mocked VisionAgent should return
        mock_complete_task.return_value = {
            'status': 'success',
            'description': 'A blue screen with a browser window.'
        }
        
        # Simulate Browser capturing an image
        self.eyesight.process_visual_input("Describe the current screen", self.sample_image)
        
        # Assert that VisionAgent.complete_task was called with correct parameters
        mock_complete_task.assert_called_with("Describe the current screen", self.sample_image)
        
        # Retrieve the result from the vision queue
        result = self.eyesight.vision_queue.get()
        self.assertEqual(result['status'], 'success')
        self.assertIn('description', result)
        self.assertEqual(result['description'], 'A blue screen with a browser window.')

    @patch('app.actions.eyesight.VisionAgent.understand_scene')
    def test_understand_scene_with_valid_image(self, mock_understand_scene):
        # Define the mocked response for understand_scene
        mock_understand_scene.return_value = {
            'status': 'success',
            'description': 'A blue screen displaying a browser.'
        }
        
        # Call the understand_scene method
        response = self.eyesight.vision_agent.understand_scene(self.sample_image, "Describe the screen")
        
        # Assert that understand_scene was called correctly
        mock_understand_scene.assert_called_with(self.sample_image, "Describe the screen")
        
        # Assert the response
        self.assertEqual(response['status'], 'success')
        self.assertEqual(response['description'], 'A blue screen displaying a browser.')

    @patch('app.actions.eyesight.Eyesight.process_visual_input')
    def test_find_element_in_browser_screen(self, mock_process_visual_input):
        # Define the mocked response for finding an element
        mock_process_visual_input.return_value = {
            'type': 'button',
            'text': 'Submit',
            'coordinates': (400, 300),
            'confidence': 0.95
        }
        
        # Simulate finding an element
        element = self.eyesight.find_element("Submit Button")
        
        # Assert that process_visual_input was called correctly
        mock_process_visual_input.assert_called_with("Submit Button", self.sample_image)
        
        # Assert the returned element
        self.assertIsNotNone(element)
        self.assertEqual(element['type'], 'button')
        self.assertEqual(element['text'], 'Submit')
        self.assertEqual(element['coordinates'], (400, 300))
        self.assertGreaterEqual(element['confidence'], 0.95)

    def test_save_image_with_embedding(self):
        with patch.object(self.eyesight, 'generate_embedding', return_value=np.random.rand(512)) as mock_generate_embedding:
            # Save the image with an optional description
            filepath = self.eyesight.save_image_with_embedding(self.sample_image, description="Test Image")
            
            # Assert that the file was saved
            self.assertIsNotNone(filepath)
            
            # Assert that generate_embedding was called
            mock_generate_embedding.assert_called_with(self.sample_image)
            
            # Assert that embedding and metadata are stored
            self.assertEqual(len(self.eyesight.image_embeddings), 1)
            self.assertEqual(len(self.eyesight.image_metadata), 1)
            self.assertEqual(self.eyesight.image_metadata[0]['description'], "Test Image")

    @patch('app.actions.eyesight.Eyesight.find_element')
    def test_wait_for_element_timeout(self, mock_find_element):
        # Configure the mock to always return None, simulating the element not being found
        mock_find_element.return_value = None
        
        # Start time
        start_time = time.time()
        
        # Attempt to wait for an element with a short timeout
        element = self.eyesight.wait_for_element("Nonexistent Element", timeout=2)
        
        # End time
        end_time = time.time()
        
        # Assert that the method waited approximately the timeout duration
        self.assertTrue(2 <= end_time - start_time < 3)
        
        # Assert that the result is None
        self.assertIsNone(element)
        
        # Assert that find_element was called multiple times
        self.assertGreaterEqual(mock_find_element.call_count, 4)  # Rough estimate

if __name__ == '__main__':
    unittest.main()
