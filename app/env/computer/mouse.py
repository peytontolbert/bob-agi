import threading
import time
import logging
from selenium.webdriver.common.by import By
import numpy as np
from typing import Tuple, Optional
from PIL import Image, ImageDraw, ImageFont

class Mouse:
    """
    Simulates mouse controls with enhanced position tracking and movement validation.
    """
    def __init__(self, driver, viewport_width, viewport_height, movement_speed=1.0):
        self.driver = driver
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.movement_speed = movement_speed
        self.action_queue = []
        self.running = False
        self.thread = None
        self.current_position = (0, 0)
        self.is_clicking = False
        self.last_click = None
        self.position_lock = threading.Lock()
        self.queue_lock = threading.Lock()
        self.movement_tolerance = 2  # pixels
        self.screenshot_width = 1008    
        self.screenshot_height = 1008
        
        # Movement constraints
        self.max_speed = 2000  # pixels per second
        self.min_move_time = 0.05  # minimum seconds for any movement
        
        # Initialize logging
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging for mouse actions"""
        self.logger = logging.getLogger('mouse')
        self.logger.setLevel(logging.DEBUG)
        
        # Add handler if none exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def start(self):
        """Start mouse thread if not already running."""
        if not self.is_running():
            self.running = True
            self.thread = threading.Thread(target=self.process_queue, daemon=True)
            self.thread.start()
            self.logger.info("Mouse thread started")
        else:
            self.logger.debug("Mouse thread already running")

    def stop(self):
        """Safely stop mouse thread."""
        if self.is_running():
            self.running = False
            self.thread.join(timeout=1.0)
            self.logger.info("Mouse thread stopped")

    def is_running(self) -> bool:
        """Check if mouse thread is running."""
        return bool(self.thread and self.thread.is_alive())

    def move_mouse_to(self, x, y):
        """Move the virtual mouse to the specified coordinates within the browser."""
        print(f" window dimensions: {self.viewport_width}x{self.viewport_height}")
        print(f" last mouse position: {self.last_mouse_position}")
        if self.last_mouse_position is None:
            # If this is the first movement, set the initial position as the current (0, 0)
            self.last_mouse_position = (0, 0)
        if 0 <= x <= self.viewport_width and 0 <= y <= self.viewport_height:
            # Move to the coordinates within the browser window
            offset_x = x - self.last_mouse_position[0]
            offset_y = y - self.last_mouse_position[1]
            self.actions.move_by_offset(offset_x, offset_y).perform()
            self.last_mouse_position = (x, y)
            self.take_screenshot(f"images/screenshot_{x}_{y}.png")
            print(f"Moved mouse to ({x}, {y}) within the browser window.")
            self.last_mouse_position = (x, y)  # Update mouse position
        else:
            print(f"Coordinates ({x}, {y}) are out of browser bounds.")

    def click_at(self, x, y):
        """Move the virtual mouse to (x, y) and perform a click."""
        self.move_mouse_to(x, y)
        self.actions.click().perform()
        print(f"Clicked at ({x}, {y}) within the browser window.")

    def scroll_down(self, amount=300):
        """
        Scroll the page down by the specified amount of pixels.
        Args:
            amount (int): Number of pixels to scroll down. Default is 300.
        """
        try:
            self.driver.execute_script(f"window.scrollBy(0, {amount});")
            print(f"Scrolled down {amount} pixels")
            time.sleep(0.5)  # Small delay after scrolling
            self.take_screenshot()  # Take a screenshot after scrolling
        except Exception as e:
            print(f"Error scrolling down: {e}")

    def scroll_up(self, amount=300):
        """
        Scroll the page up by the specified amount of pixels.
        Args:
            amount (int): Number of pixels to scroll up. Default is 300.
        """
        try:
            self.driver.execute_script(f"window.scrollBy(0, -{amount});")
            print(f"Scrolled up {amount} pixels")
            time.sleep(0.5)  # Small delay after scrolling
            self.take_screenshot()  # Take a screenshot after scrolling
        except Exception as e:
            print(f"Error scrolling up: {e}")

    def scroll_to_element(self, element_text):
        """
        Scroll to make an element with specific text visible.
        Args:
            element_text (str): The text of the element to scroll to
        """
        try:
            element = self.driver.find_element(By.LINK_TEXT, element_text)
            self.driver.execute_script("arguments[0].scrollIntoView(true);", element)
            print(f"Scrolled to element with text: {element_text}")
            time.sleep(0.5)  # Small delay after scrolling
            self.take_screenshot()  # Take a screenshot after scrolling
        except Exception as e:
            print(f"Error scrolling to element: {e}")



    def take_screenshot(self, filename="images/screenshot.png"):
        """Take a screenshot and overlay coordinate system scaled to 1000x1000."""
        # Take the screenshot
        self.driver.save_screenshot(filename)
        
        try:
            # Open and resize the screenshot to 1000x1000
            image = Image.open(filename)
            draw = ImageDraw.Draw(image)

            try:
                font = ImageFont.truetype("arial.ttf", 15)
            except IOError:
                font = None
            
            # Overlay the mouse position if available
            if self.last_mouse_position:
                # Draw viewport coordinates in red
                viewport_x, viewport_y = self.last_mouse_position
                mouse_size = 10
                draw.ellipse(
                    (viewport_x - mouse_size, viewport_y - mouse_size, 
                     viewport_x + mouse_size, viewport_y + mouse_size),
                    fill='red',
                    outline='black'
                )
                draw.text((viewport_x + 15, viewport_y), 
                         f"Viewport: ({int(viewport_x)}, {int(viewport_y)})", 
                         fill="red", 
                         font=font)
                
                # Draw screenshot coordinates in blue
                screenshot_x, screenshot_y = self.normalize_coordinates(
                    viewport_x, 
                    viewport_y, 
                    from_screenshot=False
                )
                draw.ellipse(
                    (screenshot_x - mouse_size, screenshot_y - mouse_size, 
                     screenshot_x + mouse_size, screenshot_y + mouse_size),
                    fill='blue',
                    outline='black'
                )
                draw.text((screenshot_x + 15, screenshot_y + 25), 
                         f"Screenshot: ({int(screenshot_x)}, {int(screenshot_y)})", 
                         fill="blue", 
                         font=font)

            image = image.resize((self.screenshot_width, self.screenshot_height))
            # Save the modified screenshot
            image.save(filename)
            print(f"Enhanced screenshot saved with viewport and screenshot coordinates at {filename}")
        except Exception as e:
            print(f"Error processing screenshot: {e}")



    def normalize_coordinates(self, x, y, from_screenshot=True):
        """
        Convert coordinates between screenshot (1000x1000) and viewport spaces.
        
        Args:
            x (float): X coordinate
            y (float): Y coordinate
            from_screenshot (bool): If True, convert from screenshot to viewport.
                                  If False, convert from viewport to screenshot.
        
        Returns:
            tuple: (normalized_x, normalized_y)
        """
        if from_screenshot:
            # Convert from 1000x1000 screenshot space to viewport space
            normalized_x = (x * self.viewport_width) / self.screenshot_width
            normalized_y = (y * self.viewport_height) / self.screenshot_height
            print(f"Converting screenshot ({x}, {y}) -> viewport ({normalized_x}, {normalized_y})")
        else:
            # Convert from viewport space to 1000x1000 screenshot space
            normalized_x = (x * self.screenshot_width) / self.viewport_width
            normalized_y = (y * self.screenshot_height) / self.viewport_height
            print(f"Converting viewport ({x}, {y}) -> screenshot ({normalized_x}, {normalized_y})")
        
        return normalized_x, normalized_y
