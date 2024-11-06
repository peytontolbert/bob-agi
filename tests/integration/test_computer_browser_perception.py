import pytest
from PIL import Image
from app.env.computer.computer import Computer
from app.agents.vision import VisionAgent
from app.env.senses.eyesight import Eyesight
import asyncio
import logging

@pytest.fixture
def setup_computer():
    """Fixture to set up real computer with actual components"""
    computer = Computer()
    computer.startup()  # This will launch browser and navigate to Discord
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
    """Test capturing browser screen content showing Discord"""
    # Get screen state through eyesight system
    screen_state = eyesight.capture_screen()
    
    assert screen_state is not None
    assert 'frame' in screen_state
    assert isinstance(screen_state['frame'], Image.Image)
    assert screen_state['frame'].size == (800, 600)  # Match browser viewport
    assert 'perception' in screen_state
    assert 'timestamp' in screen_state

def test_vision_perception_of_browser(setup_computer, vision_agent, eyesight):
    """Test vision agent's perception of Discord in browser"""
    # Get screen state through eyesight
    screen_state = eyesight.capture_screen()
    
    # Test vision agent's perception
    scene_understanding = vision_agent.perceive_scene(screen_state['frame'])
    
    assert scene_understanding is not None
    assert isinstance(scene_understanding, dict)
    assert 'description' in scene_understanding
    assert 'status' in scene_understanding
    assert scene_understanding['status'] == 'success'

def test_browser_element_detection(setup_computer, eyesight):
    """Test detection of Continue link in Discord interface"""
    # Get screen state through eyesight
    screen_state = eyesight.capture_screen()
    
    # Test element detection for Continue link
    element = eyesight.find_element("Continue")
    
    assert element is not None, "Continue link not detected"
    assert isinstance(element, dict)
    assert 'coordinates' in element
    assert 'confidence' in element
    assert element['confidence'] > 0.7

@pytest.mark.asyncio
async def test_continuous_browser_perception(setup_computer, vision_agent, eyesight):
    """Test continuous perception of Discord interface over time"""
    perceptions = []
    
    # Collect multiple perceptions over time
    for _ in range(3):
        screen_state = eyesight.capture_screen()
        perception = vision_agent.perceive_scene(screen_state['frame'])
        perceptions.append(perception)
        await asyncio.sleep(0.5)  # Wait between perceptions
    
    assert len(perceptions) == 3
    assert all(p['status'] == 'success' for p in perceptions)
    
    # Verify at least one perception contains the Continue link
    continue_found = False
    for perception in perceptions:
        if 'continue' in perception['description'].lower():
            continue_found = True
            break
    assert continue_found, "Continue link not found in any perception"

def test_browser_perception_error_handling(setup_computer, eyesight):
    """Test handling of perception errors"""
    # Get screen state through eyesight
    screen_state = eyesight.capture_screen()
    
    # Test error handling
    assert screen_state is not None
    if 'error' in screen_state:
        assert isinstance(screen_state['error'], str)
        logging.warning(f"Expected error occurred: {screen_state['error']}")
    else:
        assert 'frame' in screen_state
        assert screen_state['frame'] is not None
