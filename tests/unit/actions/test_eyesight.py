import pytest
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import numpy as np
import torch
from app.actions.eyesight import Eyesight
import time

@pytest.fixture
def mock_screen():
    screen = Mock()
    screen.get_current_frame.return_value = Image.new('RGB', (100, 100))
    screen.frame_buffer = []
    screen.get_frame_buffer.return_value = []
    screen.capture = Mock(return_value=Image.new('RGB', (100, 100)))
    return screen

@pytest.fixture
def mock_vision_agent():
    agent = Mock()
    agent.understand_scene.return_value = {"test": "scene"}
    agent.find_element.return_value = {
        "type": "button",
        "coordinates": (100, 100),
        "confidence": 0.9
    }
    agent.prepare_for_action.return_value = [{
        "type": "button",
        "coordinates": (100, 100),
        "confidence": 0.9
    }]
    return agent

@pytest.fixture
def mock_clip():
    clip_processor = Mock()
    clip_processor.return_value = {'pixel_values': torch.randn(1, 3, 224, 224)}
    
    clip_model = Mock()
    clip_model.get_image_features = Mock(return_value=torch.randn(1, 512))
    
    return clip_processor, clip_model

@pytest.fixture
def eyesight(mock_screen, mock_vision_agent, mock_clip):
    mock_processor, mock_model = mock_clip
    
    with patch('app.actions.eyesight.VisionAgent', return_value=mock_vision_agent):
        with patch('app.actions.eyesight.CLIPModel.from_pretrained', return_value=mock_model):
            with patch('app.actions.eyesight.CLIPProcessor.from_pretrained', return_value=mock_processor):
                eye = Eyesight(mock_screen)
                eye.is_running = False  # Prevent background thread
                return eye

class TestEyesight:
    """Unit tests focusing on Eyesight's core functionality"""
    
    def test_initialization(self, eyesight, mock_screen):
        """Test proper initialization of core attributes"""
        assert eyesight.screen == mock_screen
        assert eyesight.last_processed_frame_index == -1
        assert len(eyesight.image_buffer) == 0
        assert len(eyesight.embedding_buffer) == 0
        assert eyesight.confidence_threshold == 0.7

    def test_find_element_basic(self, eyesight, mock_vision_agent):
        """Test basic element finding without external dependencies"""
        element = eyesight.find_element("test button")
        assert element is not None
        assert "coordinates" in element
        assert element["coordinates"] == (100, 100)
        assert element["confidence"] == 0.9
        mock_vision_agent.prepare_for_action.assert_called_once()

    def test_wait_for_element_timeout(self, eyesight):
        """Test element wait timeout behavior"""
        eyesight.find_element = Mock(return_value=None)
        result = eyesight.wait_for_element("nonexistent", timeout=0.1)
        assert result is None

    def test_wait_for_element_success(self, eyesight):
        """Test successful element wait"""
        mock_element = {"type": "button", "coordinates": (100, 100)}
        eyesight.find_element = Mock(return_value=mock_element)
        result = eyesight.wait_for_element("test button", timeout=1)
        assert result == mock_element

    def test_get_screen_state_structure(self, eyesight, mock_vision_agent):
        """Test screen state dictionary structure"""
        state = eyesight.get_screen_state()
        assert state is not None
        assert all(key in state for key in ['current', 'timestamp', 'elements', 'similar_contexts'])

    @patch('time.sleep', return_value=None)  # Prevent actual sleeping
    def test_process_screen_buffer(self, mock_sleep, eyesight):
        """Test screen buffer processing logic"""
        mock_frame = Image.new('RGB', (100, 100))
        eyesight.screen.get_frame_buffer.return_value = [mock_frame]
        eyesight.process_screen_buffer()
        assert eyesight.last_processed_frame_index == 0
        assert len(eyesight.image_buffer) == 1

    def test_generate_embedding(self, eyesight, mock_clip):
        """Test embedding generation for different image formats"""
        test_image = Image.new('RGB', (100, 100))
        embedding = eyesight.generate_embedding(test_image)
        assert embedding is not None
        assert isinstance(embedding, np.ndarray)

    def test_process_visual_input(self, eyesight, mock_vision_agent):
        """Test visual input processing"""
        test_image = Image.new('RGB', (100, 100))
        result = eyesight.process_visual_input("analyze this", test_image)
        assert result is not None
        mock_vision_agent.complete_task.assert_called_once()

    @pytest.mark.timeout(10)
    def test_continuous_perception_stream(self, eyesight, mock_screen):
        """Test continuous perception stream processing"""
        # Setup mock frames
        frames = [Image.new('RGB', (100, 100)) for _ in range(5)]
        mock_screen.get_current_frame.side_effect = frames
        
        # Let perception run
        time.sleep(2)
        
        # Verify perceptions are generated
        assert len(eyesight.perception_stream) > 0
        
        # Check perception data structure
        perception = eyesight.perception_stream[0]
        assert 'timestamp' in perception
        assert 'perception' in perception
        assert 'embedding' in perception
        assert isinstance(perception['embedding'], np.ndarray)

    @pytest.mark.timeout(5)
    def test_perception_error_handling(self, eyesight, mock_screen):
        """Test perception error recovery"""
        # Simulate errors
        mock_screen.get_current_frame.side_effect = [
            None,  # Missing frame
            Exception("Test error"),  # Error
            Image.new('RGB', (100, 100))  # Valid frame
        ]
        
        # Should recover and continue perception
        time.sleep(2)
        assert len(eyesight.perception_stream) > 0

    def test_perception_performance(self, eyesight, mock_screen):
        """Test perception performance metrics"""
        test_frame = Image.new('RGB', (100, 100))
        mock_screen.get_current_frame.return_value = test_frame
        
        # Process multiple frames
        start_time = time.time()
        for _ in range(10):
            eyesight.process_visual_input("analyze", test_frame)
        end_time = time.time()
        
        # Check processing time
        avg_time = (end_time - start_time) / 10
        assert avg_time < 0.5  # Each frame under 500ms

    def test_embedding_buffer_management(self, eyesight):
        """Test embedding buffer size management"""
        # Fill embedding buffer
        for _ in range(200):
            eyesight.embedding_buffer.append({
                'embedding': np.random.rand(512),
                'timestamp': time.time()
            })
            
        # Verify buffer size is maintained
        assert len(eyesight.embedding_buffer) <= eyesight.embedding_buffer.maxlen
