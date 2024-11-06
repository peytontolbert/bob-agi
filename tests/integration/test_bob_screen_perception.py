import pytest
import time
from unittest.mock import Mock
from app.agents.vision import VisionAgent 
from app.env.senses.eyesight import Eyesight
from app.env.computer.computer import Computer
from app.env.computer.browser import Browser
from app.env.computer.screen import Screen
from app.env.computer.discord import Discord
from app.embeddings.embeddings import UnifiedEmbeddingSpace
from PIL import Image
import numpy as np
import logging

@pytest.fixture
def computer():
    """Fixture to create and initialize Computer instance"""
    computer = Computer()
    computer.start()
    yield computer
    computer.shutdown()

@pytest.fixture
def browser(computer):
    """Fixture to create and initialize Browser instance"""
    browser = Browser(
        audio=computer.audio,
        microphone=computer.microphone,
        screen=computer.screen
    )
    assert browser.launch(), "Browser failed to launch"
    yield browser
    browser.close()

@pytest.fixture
def vision_agent():
    """Fixture for VisionAgent with models"""
    agent = VisionAgent()
    return agent

@pytest.fixture
def bob_eyesight(computer, vision_agent):
    """Fixture to initialize Bob's eyesight with vision processing"""
    eyesight = Eyesight(computer.screen)
    eyesight.vision_agent = vision_agent
    eyesight.connect()  # Connect to screen's frame buffer
    return eyesight

def test_vision_agent_initialization(vision_agent):
    """Test that vision agent initializes with required models"""
    assert vision_agent.model is not None, "InternVL2 model not initialized"
    assert vision_agent.tokenizer is not None, "Tokenizer not initialized"
    assert vision_agent.model_healthy, "Vision model not healthy"
    assert vision_agent.clip_processor is not None, "CLIP processor not initialized"

def test_screen_frame_capture(computer, browser):
    """Test that screen properly captures browser frames"""
    browser.navigate("https://example.com")
    assert browser.wait_until_loaded(timeout=20), "Page failed to load"
    
    # Wait for frame capture
    max_retries = 10
    frame = None
    for _ in range(max_retries):
        frame = computer.screen.get_current_frame()
        if frame is not None:
            break
        time.sleep(0.5)
        
    assert frame is not None, "Failed to capture frame"
    assert isinstance(frame, Image.Image), "Frame not in PIL Image format"
    assert frame.size == (computer.screen.width, computer.screen.height)

def test_eyesight_frame_processing(bob_eyesight, browser):
    """Test that eyesight properly processes frames from screen"""
    browser.navigate("https://example.com")
    assert browser.wait_until_loaded(timeout=20)
    
    # Wait for frame processing
    time.sleep(1)
    
    # Check frame buffer
    assert len(bob_eyesight.image_buffer) > 0, "No frames in image buffer"
    
    # Check embedding generation
    latest_frame = bob_eyesight.image_buffer[-1]['frame']
    embedding = bob_eyesight.generate_embedding(latest_frame)
    assert embedding is not None, "Failed to generate embedding"
    assert isinstance(embedding, np.ndarray), "Embedding not in numpy array format"

def test_scene_understanding(vision_agent, computer):
    """Test vision agent's scene understanding capabilities"""
    # Create test image
    test_image = Image.new('RGB', (800, 600), color='white')
    
    # Get scene understanding
    understanding = vision_agent.understand_scene(test_image)
    
    assert understanding is not None, "No scene understanding generated"
    assert 'description' in understanding, "No description in understanding"
    assert 'status' in understanding, "No status in understanding"
    assert understanding['status'] == 'success', "Scene understanding failed"

def test_continuous_perception(bob_eyesight, browser):
    """Test continuous perception of screen changes"""
    browser.navigate("https://example.com")
    assert browser.wait_until_loaded(timeout=20)
    
    # Wait for initial perception
    time.sleep(1)
    
    # Get initial perception state
    initial_perceptions = len(bob_eyesight.perception_stream)
    
    # Modify page content
    browser.webdriver.execute_script(
        "document.body.innerHTML += '<div id=\"test\">Dynamic Content</div>'"
    )
    
    # Wait for perception update
    time.sleep(1)
    
    # Verify new perceptions were added
    assert len(bob_eyesight.perception_stream) > initial_perceptions
    
    # Check latest perception
    latest_perception = bob_eyesight.perception_stream[-1]
    assert 'perception' in latest_perception
    assert 'timestamp' in latest_perception
    assert 'embedding' in latest_perception

def test_element_detection(vision_agent, computer):
    """Test detection of UI elements"""
    # Create test image with UI elements
    test_image = Image.new('RGB', (800, 600), color='white')
    
    # Add some test elements
    elements = vision_agent.find_element(test_image, "button")
    
    if elements:  # Some elements detected
        assert isinstance(elements, dict), "Element detection result should be a dictionary"
        assert 'coordinates' in elements, "No coordinates in detected element"
        assert 'confidence' in elements, "No confidence score in detected element"
        assert 'type' in elements, "No type in detected element"
        assert elements['confidence'] >= 0.3, "Confidence too low"

def test_visual_embedding_pipeline(bob_eyesight, browser):
    """Test the full visual embedding pipeline"""
    browser.navigate("https://example.com")
    assert browser.wait_until_loaded(timeout=20)
    
    # Wait for processing
    time.sleep(1)
    
    # Check embedding buffer
    assert len(bob_eyesight.embedding_buffer) > 0, "No embeddings generated"
    
    # Get latest embedding
    latest_embedding = bob_eyesight.embedding_buffer[-1]
    assert 'embedding' in latest_embedding, "No embedding in buffer entry"
    assert 'timestamp' in latest_embedding, "No timestamp in buffer entry"
    
    # Test embedding similarity search
    current_frame = bob_eyesight.screen.get_current_frame()
    similar_images = bob_eyesight.find_similar_images(current_frame, top_k=3)
    assert isinstance(similar_images, list), "Similar images result should be a list"

def test_error_handling(bob_eyesight):
    """Test error handling in perception pipeline"""
    # Test with invalid frame
    result = bob_eyesight.vision_agent.understand_scene(None)
    assert result['status'] == 'error', "Should handle None input"
    assert 'error' in result, "Should include error message"
    
    # Test with invalid screen state
    bob_eyesight.screen = None
    perception = bob_eyesight.capture_screen()
    assert perception is None, "Should handle missing screen"
    
    # Test with mock screen returning invalid frame
    mock_screen = Mock()
    mock_screen.get_current_frame.return_value = None
    bob_eyesight.screen = mock_screen
    perception = bob_eyesight.capture_screen()
    assert perception is None, "Should handle invalid frame"
