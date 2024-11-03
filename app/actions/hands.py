"""
This interface is for Bob's hands. He can interact with the mouse or keyboard.
"""
import logging
from typing import Dict, Tuple, Optional, Union
import time
from PIL import Image
import numpy as np

class Hands:
    def __init__(self, mouse, keyboard):
        self.mouse = mouse
        self.keyboard = keyboard
        self.last_action = None
        self.last_action_time = None
        self.action_cooldown = 0.1  # 100ms minimum between actions
        self.initialized = False
        self.initialize()

    def initialize(self):
        """Initialize the hands system."""
        try:
            # Only start mouse/keyboard threads if not already running
            if not self.mouse.is_running():
                self.mouse.start()
            if not self.keyboard.is_running():
                self.keyboard.start()
                
            self.initialized = True
            logging.info("Hands system initialized successfully")
            
        except Exception as e:
            logging.error(f"Failed to initialize hands: {e}")
            self.initialized = False

    def move_mouse(self, x: int, y: int, smooth: bool = True) -> bool:
        """
        Moves the mouse to specific coordinates with improved accuracy.
        
        Args:
            x: Target X coordinate
            y: Target Y coordinate
            smooth: Whether to move smoothly (True) or instantly (False)
            
        Returns:
            bool: True if movement was successful
        """
        try:
            if not self._check_action_cooldown():
                return False

            # Validate coordinates
            if not self._validate_coordinates(x, y):
                logging.warning(f"Invalid coordinates: ({x}, {y})")
                return False

            self.mouse.move_to(x, y, smooth=smooth)
            self.last_action = ('move', (x, y))
            self.last_action_time = time.time()
            return True

        except Exception as e:
            logging.error(f"Mouse movement error: {e}")
            return False

    def click_element(self, element: Dict) -> bool:
        """
        Clicks on a specific UI element with validation and retry logic.
        
        Args:
            element: Dictionary containing element information
                    Required keys: 'coordinates', 'type'
                    Optional: 'confidence', 'text'
                    
        Returns:
            bool: True if click was successful
        """
        try:
            if not self._validate_element(element):
                return False

            if not self._check_action_cooldown():
                return False

            x, y = element['coordinates']
            
            # Move to element first
            self.move_mouse(x, y, smooth=True)
            time.sleep(0.2)  # Small delay between move and click
            
            # Verify position before clicking
            current_x, current_y = self.mouse.get_position()
            if abs(current_x - x) > 5 or abs(current_y - y) > 5:
                logging.warning("Mouse position verification failed")
                return False

            # Perform click
            self.mouse.click(button='left')
            self.last_action = ('click', element)
            self.last_action_time = time.time()
            
            return True

        except Exception as e:
            logging.error(f"Click element error: {e}")
            return False

    def type_text(self, text: str, delay: float = 0.1) -> bool:
        """
        Types text with natural timing and error checking.
        
        Args:
            text: Text to type
            delay: Delay between keystrokes for natural typing
            
        Returns:
            bool: True if typing was successful
        """
        try:
            if not text:
                logging.warning("Empty text provided for typing")
                return False

            if not self._check_action_cooldown():
                return False

            # Send text to keyboard with natural timing
            self.keyboard.send_input(text)
            self.last_action = ('type', text)
            self.last_action_time = time.time()
            
            return True

        except Exception as e:
            logging.error(f"Text typing error: {e}")
            return False

    def drag_element(self, start_element: Dict, end_coordinates: Tuple[int, int]) -> bool:
        """
        Performs a drag operation from one element to target coordinates.
        
        Args:
            start_element: Dictionary containing starting element info
            end_coordinates: (x, y) coordinates of destination
            
        Returns:
            bool: True if drag was successful
        """
        try:
            if not self._validate_element(start_element):
                return False

            if not self._validate_coordinates(*end_coordinates):
                return False

            if not self._check_action_cooldown():
                return False

            # Execute drag operation
            start_x, start_y = start_element['coordinates']
            end_x, end_y = end_coordinates

            # Move to start position
            self.move_mouse(start_x, start_y, smooth=True)
            time.sleep(0.2)

            # Perform drag
            self.mouse.action_queue.append(('click_down', 'left'))
            time.sleep(0.1)
            self.move_mouse(end_x, end_y, smooth=True)
            time.sleep(0.1)
            self.mouse.action_queue.append(('click_up', 'left'))

            self.last_action = ('drag', (start_element, end_coordinates))
            self.last_action_time = time.time()
            
            return True

        except Exception as e:
            logging.error(f"Drag operation error: {e}")
            return False

    def double_click_element(self, element: Dict) -> bool:
        """
        Performs a double-click on a specific element.
        
        Args:
            element: Dictionary containing element information
            
        Returns:
            bool: True if double-click was successful
        """
        try:
            if not self._validate_element(element):
                return False

            if not self._check_action_cooldown():
                return False

            x, y = element['coordinates']
            
            # Move and double-click
            self.move_mouse(x, y, smooth=True)
            time.sleep(0.2)
            self.mouse.click(button='left', double=True)
            
            self.last_action = ('double_click', element)
            self.last_action_time = time.time()
            
            return True

        except Exception as e:
            logging.error(f"Double-click error: {e}")
            return False

    def _validate_element(self, element: Dict) -> bool:
        """
        Validates element dictionary has required information.
        """
        if not isinstance(element, dict):
            logging.error("Invalid element type")
            return False
            
        required_keys = ['coordinates', 'type']
        if not all(key in element for key in required_keys):
            logging.error(f"Missing required element keys: {required_keys}")
            return False
            
        if not self._validate_coordinates(*element['coordinates']):
            return False
            
        return True

    def _validate_coordinates(self, x: int, y: int) -> bool:
        """
        Validates coordinates are within screen bounds.
        """
        try:
            if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                return False
                
            # Assuming standard screen bounds
            if x < 0 or x > 1920 or y < 0 or y > 1080:
                return False
                
            return True
        except Exception:
            return False

    def _check_action_cooldown(self) -> bool:
        """
        Ensures minimum time between actions for stability.
        """
        if self.last_action_time is None:
            return True
            
        time_since_last = time.time() - self.last_action_time
        return time_since_last >= self.action_cooldown

    def get_last_action(self) -> Optional[Tuple]:
        """
        Returns the last action performed.
        """
        return self.last_action
