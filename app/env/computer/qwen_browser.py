from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
import time
from PIL import Image, ImageDraw, ImageFont
import os

class BrowserController:
    def __init__(self, window_width=800, window_height=600):
        # Configure Edge WebDriver
        edge_options = Options()
        edge_options.add_argument(f"--window-size={window_width},{window_height}")
        self.driver = webdriver.Edge(options=edge_options)
        
        # Wait for the browser to open
        time.sleep(2)
        
        # Store both viewport and screenshot dimensions
        self.viewport_width = self.driver.execute_script("return window.innerWidth")
        self.viewport_height = self.driver.execute_script("return window.innerHeight")
        # Calculate the difference between window and viewport size
        width_diff = window_width - self.viewport_width
        height_diff = window_height - self.viewport_height
        
        # Adjust window size to account for the difference
        self.driver.set_window_size(window_width + width_diff, window_height + height_diff)
        
        self.screenshot_width = 1008    
        self.screenshot_height = 1008
        
        self.actions = ActionChains(self.driver)
        self.last_mouse_position = None
        
        print(f"Initialized browser with viewport dimensions: {self.viewport_width}x{self.viewport_height}")

    def navigate(self, url):
        """Navigate to a specified URL."""
        self.driver.get(url)
        print(f"Navigated to {url}")
        time.sleep(2)  # Wait for the page to load

    def locate_element_by_text(self, text):
        """Locate an element by link text and return its center coordinates."""
        try:
            element = self.driver.find_element(By.LINK_TEXT, text)
            location = element.location
            size = element.size
            # Calculate center coordinates within the browser
            center_x = location['x'] + (size['width'] / 2)
            center_y = location['y'] + (size['height'] / 2)
            print(f"Located element '{text}' at ({center_x}, {center_y})")
            return center_x, center_y
        except Exception as e:
            print(f"Error locating element by text '{text}': {e}")
            return None, None

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

    def close(self):
        """Close the browser."""
        self.driver.quit()
        print("Browser closed.")

    def type_text(self, text):
        """Type text into the currently focused element."""
        try:
            actions = ActionChains(self.driver)
            actions.send_keys(text)
            actions.perform()
            print(f"Typed text")
            time.sleep(0.5)  # Small delay after typing
        except Exception as e:
            print(f"Error typing text: {e}")

    def press_key(self, key):
        """Press a specific key (e.g., Enter, Tab, etc.)."""
        try:
            actions = ActionChains(self.driver)
            actions.send_keys(getattr(Keys, key.upper()))
            actions.perform()
            print(f"Pressed key: {key}")
            time.sleep(0.5)  # Small delay after key press
        except Exception as e:
            print(f"Error pressing key: {e}")

    def click_and_type(self, x, y, text):
        """Click at coordinates and type text."""
        self.click_at(x, y)
        time.sleep(0.5)  # Wait for click to register
        self.type_text(text)

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

