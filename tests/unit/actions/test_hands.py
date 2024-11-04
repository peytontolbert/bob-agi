"""Unit tests for Bob's hands system focusing on accuracy and reliability"""
import pytest
from unittest.mock import Mock, patch
import time
from PIL import Image
import numpy as np

class MockMouse:
    def __init__(self):
        self.position = (0, 0)
        self.action_queue = []
        self._running = False
        
    def move_to(self, x, y, smooth=True):
        self.position = (x, y)
        
    def click(self, button='left', double=False):
        self.action_queue.append(('click', button, double))
        
    def get_position(self):
        return self.position
        
    def is_running(self):
        return self._running
        
    def start(self):
        self._running = True

class MockKeyboard:
    def __init__(self):
        self.typed_text = []
        self._running = False
        
    def send_input(self, text):
        self.typed_text.append(text)
        
    def is_running(self):
        return self._running
        
    def start(self):
        self._running = True

@pytest.fixture
def mock_mouse():
    return MockMouse()

@pytest.fixture
def mock_keyboard():
    return MockKeyboard()

@pytest.fixture
def hands(mock_mouse, mock_keyboard):
    from app.actions.hands import Hands
    return Hands(mouse=mock_mouse, keyboard=mock_keyboard)

class TestHandsInitialization:
    def test_initialization(self, hands):
        """Test hands system initializes correctly"""
        assert hands.initialized
        assert hands.mouse is not None
        assert hands.keyboard is not None
        assert hands.last_action is None
        assert hands.action_cooldown == 0.1

    def test_failed_initialization(self, mock_mouse):
        """Test handling of failed initialization"""
        bad_keyboard = Mock()
        bad_keyboard.start.side_effect = Exception("Failed to start")
        
        from app.actions.hands import Hands
        hands = Hands(mouse=mock_mouse, keyboard=bad_keyboard)
        assert not hands.initialized

class TestMouseMovement:
    def test_move_mouse_basic(self, hands):
        """Test basic mouse movement"""
        success = hands.move_mouse(100, 200)
        assert success
        assert hands.mouse.position == (100, 200)
        assert hands.last_action == ('move', (100, 200))

    def test_move_mouse_invalid_coords(self, hands):
        """Test movement with invalid coordinates"""
        success = hands.move_mouse(-100, 5000)
        assert not success
        assert hands.mouse.position == (0, 0)

    def test_move_mouse_cooldown(self, hands):
        """Test movement respects cooldown period"""
        hands.move_mouse(100, 100)
        immediate_move = hands.move_mouse(200, 200)
        assert not immediate_move
        
        time.sleep(0.15)  # Wait past cooldown
        delayed_move = hands.move_mouse(200, 200)
        assert delayed_move

class TestElementClicking:
    @pytest.fixture
    def sample_element(self):
        return {
            'coordinates': (150, 150),
            'type': 'button',
            'confidence': 0.95
        }

    def test_click_element_basic(self, hands, sample_element):
        """Test basic element clicking"""
        success = hands.click_element(sample_element)
        assert success
        assert hands.mouse.position == (150, 150)
        assert ('click', 'left', False) in hands.mouse.action_queue

    def test_click_element_validation(self, hands):
        """Test element validation"""
        invalid_element = {'type': 'button'}  # Missing coordinates
        success = hands.click_element(invalid_element)
        assert not success

    def test_click_element_position_verification(self, hands):
        """Test position verification before clicking"""
        element = {
            'coordinates': (2000, 2000),  # Out of bounds
            'type': 'button'
        }
        success = hands.click_element(element)
        assert not success

    def test_double_click_element(self, hands, sample_element):
        """Test double-click functionality"""
        success = hands.double_click_element(sample_element)
        assert success
        assert hands.mouse.position == (150, 150)
        assert ('click', 'left', True) in hands.mouse.action_queue

class TestKeyboardInput:
    def test_type_text_basic(self, hands):
        """Test basic text typing"""
        success = hands.type_text("Hello World")
        assert success
        assert "Hello World" in hands.keyboard.typed_text

    def test_type_text_empty(self, hands):
        """Test handling empty text input"""
        success = hands.type_text("")
        assert not success
        assert len(hands.keyboard.typed_text) == 0

    def test_type_text_special_chars(self, hands):
        """Test typing special characters"""
        special_text = "Hello!@#$%^&*()"
        success = hands.type_text(special_text)
        assert success
        assert special_text in hands.keyboard.typed_text

class TestDragOperations:
    @pytest.fixture
    def drag_element(self):
        return {
            'coordinates': (100, 100),
            'type': 'draggable',
            'confidence': 0.9
        }

    def test_drag_element_basic(self, hands, drag_element):
        """Test basic drag operation"""
        success = hands.drag_element(drag_element, (200, 200))
        assert success
        assert ('click_down', 'left') in hands.mouse.action_queue
        assert ('click_up', 'left') in hands.mouse.action_queue
        assert hands.mouse.position == (200, 200)

    def test_drag_invalid_target(self, hands, drag_element):
        """Test drag with invalid target coordinates"""
        success = hands.drag_element(drag_element, (-100, 5000))
        assert not success

class TestErrorHandling:
    def test_handle_mouse_failure(self, hands):
        """Test handling of mouse operation failure"""
        hands.mouse.move_to = Mock(side_effect=Exception("Mouse failure"))
        success = hands.move_mouse(100, 100)
        assert not success

    def test_handle_keyboard_failure(self, hands):
        """Test handling of keyboard operation failure"""
        hands.keyboard.send_input = Mock(side_effect=Exception("Keyboard failure"))
        success = hands.type_text("test")
        assert not success

class TestCoordinateValidation:
    @pytest.mark.parametrize("coords,expected", [
        ((100, 100), True),
        ((-1, 100), False),
        ((100, -1), False),
        ((2000, 100), False),
        ((100, 2000), False),
        ((0, 0), True),
        ((1919, 1079), True)
    ])
    def test_coordinate_validation(self, hands, coords, expected):
        """Test coordinate validation with various inputs"""
        assert hands._validate_coordinates(*coords) == expected

