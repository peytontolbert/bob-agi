"""
This interface is for Bob's hands. He can interact with the mouse or keyboard.
"""
import logging
from typing import Dict, Tuple, Optional
import time

class Hands:
    def __init__(self, mouse, keyboard):
        self.mouse = mouse
        self.keyboard = keyboard
        self.initialize()

    def initialize(self):
        """
        Initializes the hands.
        """
        logging.info("Hands initialized")

    def move_mouse(self, x: int, y: int, smooth: bool = True):
        """
        Moves the mouse to the given coordinates.
        
        Args:
            x: Target X coordinate
            y: Target Y coordinate
            smooth: Whether to move the mouse smoothly
        """
        self.mouse.move_to(x, y, smooth=smooth)

    def click_mouse(self, button: str = 'left'):
        """
        Clicks the mouse.
        
        Args:
            button: Mouse button to click ('left', 'right', 'middle')
        """
        self.mouse.click(button=button)

    def double_click_mouse(self, button: str = 'left'):
        """
        Double clicks the mouse.
        
        Args:
            button: Mouse button to double click
        """
        self.mouse.click(button=button, double=True)

    def type_text(self, text: str):
        """
        Types the given text.
        
        Args:
            text: Text to type
        """
        self.keyboard.type_text(text)

    def click_element(self, element: Dict) -> bool:
        """
        Clicks on a visually identified element.
        
        Args:
            element: Dictionary containing element info including coordinates
                    Expected format: {
                        'type': str,  # Type of element (button, text, etc)
                        'text': str,  # Text content if any
                        'coordinates': Tuple[int, int],  # (x, y) position
                        'confidence': float  # Confidence score of detection
                    }
        
        Returns:
            bool: True if click was successful
        """
        if not element or 'coordinates' not in element:
            logging.error("Invalid element provided for clicking")
            return False

        try:
            x, y = element['coordinates']
            self.move_mouse(x, y)
            time.sleep(0.1)  # Small delay between move and click
            self.click_mouse()
            return True
        except Exception as e:
            logging.error(f"Error clicking element: {e}")
            return False

    def drag_element(self, start_element: Dict, end_coordinates: Tuple[int, int]) -> bool:
        """
        Drags an element from its location to target coordinates.
        
        Args:
            start_element: Dictionary containing starting element info
            end_coordinates: (x, y) coordinates of destination
        
        Returns:
            bool: True if drag was successful
        """
        if not start_element or 'coordinates' not in start_element:
            logging.error("Invalid start element for drag operation")
            return False

        try:
            start_x, start_y = start_element['coordinates']
            end_x, end_y = end_coordinates

            # Move to start position
            self.move_mouse(start_x, start_y)
            time.sleep(0.1)

            # Press and hold
            self.mouse.action_queue.append(('click_down', 'left'))
            time.sleep(0.1)

            # Drag to end position
            self.move_mouse(end_x, end_y, smooth=True)
            time.sleep(0.1)

            # Release
            self.mouse.action_queue.append(('click_up', 'left'))
            return True
        except Exception as e:
            logging.error(f"Error during drag operation: {e}")
            return False

    def hover_over_element(self, element: Dict, duration: float = 0.5) -> bool:
        """
        Hovers the mouse over an element for a specified duration.
        
        Args:
            element: Dictionary containing element info
            duration: How long to hover in seconds
        
        Returns:
            bool: True if hover was successful
        """
        if not element or 'coordinates' not in element:
            logging.error("Invalid element for hover operation")
            return False

        try:
            x, y = element['coordinates']
            self.move_mouse(x, y, smooth=True)
            time.sleep(duration)
            return True
        except Exception as e:
            logging.error(f"Error during hover operation: {e}")
            return False
