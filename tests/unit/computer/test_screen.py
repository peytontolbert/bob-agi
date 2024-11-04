import pytest
import numpy as np
from PIL import Image
import threading
from unittest.mock import Mock, patch, MagicMock
from app.env.screen import Screen

@pytest.fixture
def screen():
    with patch('tkinter.Tk'), patch('tkinter.Canvas'):
        screen = Screen()
        screen.initialize()
        return screen

def test_screen_initialization():
    """Test screen initialization with default values"""
    with patch('tkinter.Tk'), patch('tkinter.Canvas'):
        screen = Screen()
        assert screen.width == 800
        assert screen.height == 600
        assert screen.current_state is None
        assert screen.ui_elements == []
        assert screen.mouse_position == (0, 0)
        assert screen.resolution == (800, 600)

def test_update_mouse_position(screen):
    """Test mouse position updates correctly"""
    mock_event = Mock()
    mock_event.x = 100
    mock_event.y = 200
    
    screen.update_mouse_position(mock_event)
    assert screen.get_mouse_position() == (100, 200)

def test_update_frame_numpy(screen):
    """Test updating frame with numpy array"""
    # Create test frame
    test_frame = np.zeros((600, 800, 3), dtype=np.uint8)
    test_frame[:, :, 0] = 255  # Add some red color
    
    screen.update_frame(test_frame)
    
    # Get current frame and verify
    current = screen.get_current_frame()
    assert isinstance(current, Image.Image)
    assert current.size == (800, 600)

def test_update_frame_pil(screen):
    """Test updating frame with PIL Image"""
    test_frame = Image.new('RGB', (800, 600), color='red')
    
    screen.update_frame(test_frame)
    
    current = screen.get_current_frame()
    assert isinstance(current, Image.Image)
    assert current.size == (800, 600)

def test_frame_buffer_management(screen):
    """Test frame buffer size management"""
    test_frame = Image.new('RGB', (800, 600), color='red')
    
    # Add multiple frames
    for _ in range(100):
        screen.update_frame(test_frame)
    
    # Buffer should maintain reasonable size
    assert len(screen.frame_buffer) <= 90

def test_capture_state(screen):
    """Test screen state capture"""
    state = screen.capture()
    
    assert isinstance(state, dict)
    assert 'timestamp' in state
    assert 'resolution' in state
    assert 'ui_elements' in state
    assert 'frame' in state
    assert 'mouse_position' in state
    assert 'quality_metrics' in state

def test_quality_metrics(screen):
    """Test frame quality metrics calculation"""
    # Create test frame with known properties
    test_frame = np.ones((600, 800, 3), dtype=np.uint8) * 128  # Mid-gray frame
    
    screen.update_frame(test_frame)
    state = screen.capture()
    
    metrics = state['quality_metrics']
    assert 'brightness' in metrics
    assert 'contrast' in metrics
    assert 'sharpness' in metrics
    
    # Basic validation of metrics
    assert metrics['brightness'] > 0
    assert metrics['contrast'] >= 0
    assert metrics['sharpness'] >= 0

def test_container_mode():
    """Test screen behavior in container mode"""
    with patch.dict('os.environ', {'IS_CONTAINER': 'True'}):
        screen = Screen()
        screen.initialize()
        
        # Verify no tkinter window is created
        assert screen.window is None
        assert screen.canvas is None
        
        # Basic functionality should still work
        test_frame = Image.new('RGB', (800, 600), color='red')
        screen.update_frame(test_frame)
        
        current = screen.get_current_frame()
        assert isinstance(current, Image.Image)

def test_error_handling_invalid_frame(screen):
    """Test error handling for invalid frame input"""
    # Test with invalid frame type
    screen.update_frame("invalid_frame")
    
    # Should not crash and return blank frame
    current = screen.get_current_frame()
    assert isinstance(current, Image.Image)
    assert current.size == (800, 600)

def test_display_elements(screen):
    """Test UI elements display"""
    elements = [{'type': 'button', 'position': (100, 100)}]
    screen.display_elements(elements)
    assert screen.ui_elements == elements

@pytest.mark.asyncio
async def test_thread_safety():
    """Test thread safety of frame updates"""
    with patch('tkinter.Tk'), patch('tkinter.Canvas'):
        screen = Screen()
        screen.initialize()
        
        test_frame = Image.new('RGB', (800, 600), color='red')
        
        # Simulate multiple threads updating frames
        def update_frame():
            for _ in range(50):
                screen.update_frame(test_frame)
        
        threads = [threading.Thread(target=update_frame) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Verify frame buffer integrity
        assert len(screen.frame_buffer) <= 90
        assert isinstance(screen.get_current_frame(), Image.Image)
