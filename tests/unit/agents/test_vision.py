import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from PIL import Image
import torch
from app.agents.vision import VisionAgent

@pytest.fixture
def vision_agent():
    with patch('app.agents.vision.AutoModel'), \
         patch('app.agents.vision.AutoTokenizer'), \
         patch('app.agents.vision.CLIPProcessor'):
        agent = VisionAgent()
        return agent

@pytest.fixture
def sample_image():
    # Create a simple test image
    return Image.new('RGB', (100, 100), color='red')

@pytest.fixture
def sample_numpy_image():
    # Create a sample numpy array image
    return np.zeros((100, 100, 3), dtype=np.uint8)

def test_init(vision_agent):
    assert vision_agent is not None
    assert vision_agent.model_healthy is True
    assert isinstance(vision_agent.processing_times, list)
    assert vision_agent.error_threshold == 50

def test_process_image_pil(vision_agent, sample_image):
    result = vision_agent.process_image(sample_image)
    assert isinstance(result, np.ndarray)
    assert result.shape == (100, 100, 3)

def test_process_image_numpy(vision_agent, sample_numpy_image):
    result = vision_agent.process_image(sample_numpy_image)
    assert isinstance(result, np.ndarray)
    assert result.shape == (100, 100, 3)

def test_process_image_invalid():
    with patch('app.agents.vision.AutoModel'), \
         patch('app.agents.vision.AutoTokenizer'), \
         patch('app.agents.vision.CLIPProcessor'):
        agent = VisionAgent()
        with pytest.raises(ValueError):
            agent.process_image(None)
        with pytest.raises(ValueError):
            agent.process_image(42)  # Invalid type

@patch('app.agents.vision.YOLO')
def test_find_element(mock_yolo, vision_agent, sample_image):
    # Mock YOLO detection results
    mock_result = MagicMock()
    mock_result.boxes.data = torch.tensor([
        [0, 0, 50, 50, 0.9, 0],  # button with high confidence
    ])
    mock_yolo.return_value.return_value = [mock_result]
    vision_agent._yolo_model = mock_yolo.return_value

    result = vision_agent.find_element(sample_image, "find button")
    
    assert result is not None
    assert 'coordinates' in result
    assert 'confidence' in result
    assert 'type' in result
    assert result['type'] == 'button'

def test_understand_scene(vision_agent, sample_image):
    # Mock the _generate_scene_understanding method
    vision_agent._generate_scene_understanding = Mock(return_value={
        'status': 'success',
        'description': 'A test image',
        'processing_time': 0.1
    })

    result = vision_agent.understand_scene(sample_image)
    
    assert result['status'] == 'success'
    assert 'description' in result
    assert len(vision_agent.processing_times) > 0

def test_validate_and_process_image(vision_agent, sample_image):
    result = vision_agent._validate_and_process_image(sample_image)
    assert isinstance(result, Image.Image)
    assert result.mode == 'RGB'

    # Test invalid image size
    tiny_image = Image.new('RGB', (5, 5), color='red')
    with pytest.raises(ValueError):
        vision_agent._validate_and_process_image(tiny_image)

def test_perceive_scene(vision_agent, sample_image):
    # Mock YOLO detection
    mock_result = MagicMock()
    mock_result.boxes.data = torch.tensor([
        [0, 0, 50, 50, 0.9, 0],  # Sample detection
    ])
    mock_yolo = Mock(return_value=[mock_result])
    mock_yolo.names = {0: 'button'}
    
    # Set the private _yolo_model attribute instead of the property
    vision_agent._yolo_model = mock_yolo

    # Mock scene understanding
    vision_agent._generate_scene_understanding = Mock(return_value={
        'description': 'A test scene',
        'status': 'success'
    })

    result = vision_agent.perceive_scene(sample_image)
    
    assert result['status'] == 'success'
    assert 'description' in result
    assert 'objects' in result
    assert len(result['objects']) == 1

def test_error_handling(vision_agent, sample_image):
    # Test error handling in understand_scene
    vision_agent._generate_scene_understanding = Mock(side_effect=Exception("Test error"))
    
    result = vision_agent.understand_scene(sample_image)
    assert result['status'] == 'error'
    assert 'message' in result
    assert vision_agent.error_counts['scene_understanding'] == 1

def test_performance_monitoring(vision_agent):
    test_time = 1.5
    vision_agent._update_performance_metrics(test_time)
    
    assert len(vision_agent.processing_times) == 1
    assert vision_agent.processing_times[0] == test_time

    # Test max processing times limit
    for _ in range(1000):
        vision_agent._update_performance_metrics(1.0)
    assert len(vision_agent.processing_times) == vision_agent.max_processing_times

def test_complete_task(vision_agent, sample_image):
    context = {'image': sample_image}
    
    # Mock find_element for find tasks
    vision_agent.find_element = Mock(return_value={'type': 'button', 'coordinates': (50, 50)})
    result = vision_agent.complete_task("find button", context)
    assert result['type'] == 'button'

    # Mock understand_scene for other tasks
    vision_agent.understand_scene = Mock(return_value={'status': 'success', 'description': 'test'})
    result = vision_agent.complete_task("describe image", context)
    assert result['status'] == 'success'

    # Test missing context
    result = vision_agent.complete_task("find button", {})
    assert result['status'] == 'error'
