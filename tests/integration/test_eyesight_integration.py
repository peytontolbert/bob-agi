import pytest
from PIL import Image
import numpy as np
from app.env.senses.eyesight import Eyesight
from app.env.computer.screen import Screen

class TestEyesightIntegration:
    """Integration tests for Eyesight with actual dependencies"""
    
    @pytest.fixture
    def screen(self):
        return Screen()  # Real screen instance
        
    @pytest.fixture
    def eyesight(self, screen):
        eye = Eyesight(screen)
        yield eye
        eye.is_running = False  # Cleanup
        
    def test_vision_agent_integration(self, eyesight):
        """Test integration with VisionAgent"""
        result = eyesight.process_visual_input(
            "What's on screen?", 
            Image.new('RGB', (100, 100))
        )
        assert result is not None
        assert isinstance(result, dict)
        
    def test_clip_model_integration(self, eyesight):
        """Test integration with CLIP model"""
        test_image = Image.new('RGB', (100, 100))
        embedding = eyesight.generate_embedding(test_image)
        assert embedding is not None
        assert isinstance(embedding, np.ndarray)
        
    def test_screen_capture_integration(self, eyesight):
        """Test integration with screen capture"""
        state = eyesight.capture_screen()
        assert state is not None
        assert 'frame' in state
        assert isinstance(state['frame'], Image.Image)
        
    def test_continuous_perception(self, eyesight):
        """Test the continuous perception thread"""
        # Let it run for a short time
        import time
        time.sleep(2)  # Increased wait time to ensure perception occurs
        
        # Check that perceptions are being generated
        assert len(eyesight.perception_stream) > 0
        
        # Verify perception data structure
        perception = eyesight.perception_stream[0]
        assert isinstance(perception, dict)
        assert 'perception' in perception
        assert 'timestamp' in perception
        assert 'context' in perception