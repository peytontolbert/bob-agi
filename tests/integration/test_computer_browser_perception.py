import pytest
from PIL import Image
from app.env.computer import Computer
from app.agents.vision import VisionAgent
from app.embeddings.embeddings import UnifiedEmbeddingSpace
from app.actions.eyesight import Eyesight
import asyncio

@pytest.fixture
def setup_computer():
    """Fixture to set up real computer with actual components"""
    computer = Computer()
    computer.startup()
    return computer

@pytest.fixture
def vision_agent():
    """Fixture to create vision agent"""
    return VisionAgent()

@pytest.fixture
def eyesight(setup_computer):
    """Fixture to create eyesight system connected to computer screen"""
    return Eyesight(setup_computer.screen)

def test_browser_screen_capture(setup_computer, eyesight):
    """Test capturing actual browser screen content"""
    computer = setup_computer
    
    # Navigate browser to a test page
    computer.apps['browser'].navigate('https://example.com')
    
    # Get screen state through eyesight system
    screen_state = eyesight.capture_screen()
    
    assert screen_state is not None
    assert 'frame' in screen_state
    assert isinstance(screen_state['frame'], Image.Image)
    assert screen_state['frame'].size == (800, 600)  # Match browser viewport

def test_vision_perception_of_browser(setup_computer, vision_agent, eyesight):
    """Test vision agent's perception of actual browser content"""
    computer = setup_computer
    
    # Navigate to test page
    computer.apps['browser'].navigate('https://example.com')
    
    # Get screen state through eyesight
    screen_state = eyesight.capture_screen()
    
    # Test vision agent's perception
    scene_understanding = vision_agent.perceive_scene(screen_state['frame'])
    
    assert scene_understanding is not None
    assert isinstance(scene_understanding, dict)
    assert 'description' in scene_understanding
    assert 'status' in scene_understanding
    assert scene_understanding['status'] == 'success'

def test_browser_element_detection(setup_computer, vision_agent, eyesight):
    """Test detection of UI elements in actual browser"""
    computer = setup_computer
    
    # Navigate to test page with known content
    computer.apps['browser'].navigate('https://example.com')
    
    # Get screen state through eyesight
    screen_state = eyesight.capture_screen()
    
    # Test element detection using eyesight's find_element
    element = eyesight.find_element("More information...")  # Text commonly found on example.com
    
    assert element is not None
    assert isinstance(element, dict)
    assert 'coordinates' in element
    assert 'confidence' in element
    assert 'type' in element

@pytest.mark.asyncio
async def test_continuous_browser_perception(setup_computer, vision_agent, eyesight):
    """Test continuous perception of browser content over time"""
    computer = setup_computer
    
    test_urls = [
        'https://example.com',
        'https://example.org',
        'https://example.net'
    ]
    
    perceptions = []
    for url in test_urls:
        # Navigate to different pages
        computer.apps['browser'].navigate(url)
        
        # Get screen state through eyesight
        screen_state = eyesight.capture_screen()
        
        # Get perception
        perception = vision_agent.perceive_scene(screen_state['frame'])
        perceptions.append(perception)
        
        await asyncio.sleep(0.1)
    
    assert len(perceptions) == 3
    assert all(p['status'] == 'success' for p in perceptions)
    # Verify perceptions are different
    descriptions = [p['description'] for p in perceptions]
    assert len(set(descriptions)) == 3  # All descriptions should be unique

def test_browser_perception_error_handling(setup_computer, eyesight):
    """Test handling of perception errors"""
    computer = setup_computer
    
    # Force an error state by navigating to invalid URL
    computer.apps['browser'].navigate('https://invalid.url.that.does.not.exist')
    
    # Get screen state through eyesight
    screen_state = eyesight.capture_screen()
    
    # Verify error handling in screen state
    assert screen_state is not None
    assert 'error' in screen_state or screen_state['frame'] is not None
