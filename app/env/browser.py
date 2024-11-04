import logging
from selenium import webdriver
from selenium.webdriver.edge.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from app.env.audio import Audio
from app.env.microphone import Microphone
import threading
from PIL import Image
import io
import tkinter as tk
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException, WebDriverException
import time

class Browser:
    def __init__(self, audio: Audio = None, microphone: Microphone = None, screen=None):
        self.webdriver = None
        self.audio = audio
        self.microphone = microphone
        self.screen = screen
        self.update_interval = 100  # Update screen every 100ms
        self.is_capturing = False
        self.wait = None  # Will store WebDriverWait instance
        self._default_timeout = 10  # Add default timeout setting
        self.target_fps = 30  # Target frames per second for screen capture
        
    def launch(self):
        """
        Launches the browser and initializes all components.
        Returns True if successful, False otherwise.
        """
        logging.info("Starting browser launch...")
        try:
            # Initialize webdriver
            self.setup_webdriver()
            if not self.webdriver:
                logging.error("WebDriver is None after setup")
                raise Exception("WebDriver initialization failed")
                
            # Initialize WebDriverWait
            self.wait = WebDriverWait(self.webdriver, timeout=10)
            logging.info("WebDriverWait initialized")
                
            # Start screen capture in main thread
            self.is_capturing = True
            logging.info("Starting screen capture...")
            self.start_screen_capture()
            
            # Setup audio routing if available
            #if self.audio:
            #    logging.info("Setting up audio routing...")
            #    self.audio.route_output("browser")
            #if self.microphone:
            #    logging.info("Setting up microphone routing...")
            #    self.microphone.route_input("browser")
                
            logging.info("Browser launched successfully")
            return True
            
        except Exception as e:
            logging.error(f"Failed to launch browser: {e}")
            self.close()  # Cleanup on failure
            return False

    def setup_webdriver(self):
        """
        Sets up the Edge WebDriver with appropriate options.
        """
        try:
            edge_options = Options()
            edge_options.use_chromium = True
            edge_options.add_argument("--no-sandbox")
            edge_options.add_argument("--window-size=800,600")
            edge_options.add_argument("--disable-dev-shm-usage")  # Helps with container stability
            edge_options.headless = True  # Run in headless mode for container
            
            self.webdriver = webdriver.Edge(options=edge_options)
            logging.info("WebDriver initialized successfully")
            
        except Exception as e:
            logging.error(f"WebDriver setup failed: {e}")
            raise

    def start_screen_capture(self):
        """Starts a thread to continuously capture and update the browser screen"""
        def capture_loop():
            consecutive_failures = 0
            max_consecutive_failures = 5
            retry_delay = 0.1
            frame_interval = 1.0 / self.target_fps
            
            while self.is_capturing:
                try:
                    if not self.webdriver:
                        logging.error("WebDriver not available")
                        break
                        
                    # Capture screenshot with retry mechanism
                    screenshot = None
                    for attempt in range(3):  # Max 3 retries
                        try:
                            screenshot = self.webdriver.get_screenshot_as_png()
                            break
                        except Exception as e:
                            if attempt == 2:  # Last attempt
                                raise
                            time.sleep(retry_delay)
                        
                    if screenshot is None:
                        raise Exception("Failed to capture screenshot after retries")
                        
                    # Process screenshot
                    image = Image.open(io.BytesIO(screenshot))
                    
                    if self.screen:
                        # Validate image before updating
                        if image.size != (self.screen.width, self.screen.height):
                            image = image.resize((self.screen.width, self.screen.height), 
                                              Image.Resampling.LANCZOS)
                        
                        # Update screen with new frame
                        self.screen.update_frame(image)
                        consecutive_failures = 0  # Reset failure counter on success
                    else:
                        logging.warning("Screen not set, cannot update frame")
                    
                    # Frame rate control
                    time.sleep(max(0.001, frame_interval))
                    
                except Exception as e:
                    consecutive_failures += 1
                    logging.error(f"Error in screen capture loop (attempt {consecutive_failures}): {e}")
                    
                    if consecutive_failures >= max_consecutive_failures:
                        logging.critical("Too many consecutive failures in screen capture, stopping...")
                        self.is_capturing = False
                        break
                        
                    # Exponential backoff for retry delays
                    time.sleep(min(30, retry_delay * (2 ** consecutive_failures)))
                    
        # Start capture thread
        self.capture_thread = threading.Thread(target=capture_loop, daemon=True)
        self.capture_thread.start()
        logging.info("Screen capture thread started")

    def set_screen(self, screen):
        """Sets the screen instance for displaying browser content"""
        self.screen = screen

    def close(self):
        """Closes the browser and stops screen capture"""
        self.is_capturing = False
        if self.webdriver:
            try:
                self.webdriver.quit()
            except Exception as e:
                logging.error(f"Error closing webdriver: {e}")

    def navigate(self, url: str):
        if self.webdriver:
            self.webdriver.get(url)
            logging.info(f"Navigated to {url}")

    def move_mouse(self, x, y, speed=1.0):
        """
        Moves the mouse to the specified coordinates.
        
        :param x: X-coordinate.
        :param y: Y-coordinate.
        :param speed: Movement speed factor.
        """
        logging.debug(f"Moving mouse to ({x}, {y}) with speed {speed}).")
        if self.webdriver:
            script = f"window.moveMouse({x}, {y}, {speed})"
            self.webdriver.execute_script(script)  # Execute the mouse move script

    def click_mouse(self, button='left'):
        """
        Performs a mouse click.
        
        :param button: The mouse button to click ('left', 'right', 'middle').
        """
        valid_buttons = ['left', 'right', 'middle']
        if button not in valid_buttons:
            raise ValueError(f"Invalid mouse button: {button}")  # Raise ValueError for invalid buttons
        logging.debug(f"Clicking mouse '{button}' button.")
        if self.webdriver:
            script = f"window.clickMouse('{button}')"
            self.webdriver.execute_script(script)  # Execute the mouse click script

    def wait_for_element(self, by, value, timeout=10):
        """
        Waits for an element to be present on the page.
        
        :param by: Locator strategy (e.g., By.ID, By.CSS_SELECTOR)
        :param value: Locator value
        :param timeout: Maximum time to wait in seconds
        :return: The found element
        """
        try:
            return self.wait.until(
                EC.presence_of_element_located((by, value))
            )
        except Exception as e:
            logging.error(f"Element not found: {by}={value}, Error: {e}")
            raise

    def find_element(self, selector, timeout=None):
        """
        Finds an element using CSS selector with timeout.
        
        Args:
            selector (str): CSS selector
            timeout (int, optional): Timeout in seconds. Uses default if None.
            
        Returns:
            WebElement or None: The found element or None if not found
        """
        try:
            timeout = timeout or self._default_timeout
            return self.wait_for_element(By.CSS_SELECTOR, selector, timeout)
        except (TimeoutException, WebDriverException) as e:
            logging.debug(f"Element not found: {selector}")
            return None

    def wait_until_loaded(self, timeout=None):
        """
        Waits until the page is fully loaded.
        
        Returns:
            bool: True if page loaded, False if timeout
        """
        try:
            timeout = timeout or self._default_timeout
            self.wait.until(lambda d: d.execute_script('return document.readyState') == 'complete')
            return True
        except TimeoutException:
            return False
