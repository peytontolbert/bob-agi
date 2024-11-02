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

class Browser:
    def __init__(self, audio: Audio = None, microphone: Microphone = None):
        self.webdriver = None
        self.audio = audio
        self.microphone = microphone
        self.screen = None
        self.update_interval = 100  # Update screen every 100ms
        self.is_capturing = False
        self.wait = None  # Will store WebDriverWait instance
        
        # Initialize Tk root for proper threading
        self.root = tk.Tk()
        self.root.withdraw()  # Hide the window

    def launch(self):
        """
        Launches the browser and initializes all components.
        Returns True if successful, False otherwise.
        """
        try:
            # Initialize webdriver
            self.setup_webdriver()
            if not self.webdriver:
                raise Exception("WebDriver initialization failed")
                
            # Initialize WebDriverWait
            self.wait = WebDriverWait(self.webdriver, timeout=10)
                
            # Start screen capture in main thread
            self.is_capturing = True
            self.start_screen_capture()
            
            # Setup audio routing if available
            if self.audio:
                self.audio.route_output("browser")
            if self.microphone:
                self.microphone.route_input("browser")
                
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
            while self.is_capturing:
                try:
                    if not self.webdriver:
                        break
                        
                    screenshot = self.webdriver.get_screenshot_as_png()
                    image = Image.open(io.BytesIO(screenshot))
                    
                    if self.screen:
                        # Use root.after to schedule screen updates
                        self.root.after(0, lambda: self.screen.update_frame(image))
                    
                    threading.Event().wait(self.update_interval / 1000)
                except Exception as e:
                    logging.error(f"Error capturing browser screen: {e}")
                    break

        capture_thread = threading.Thread(target=capture_loop, daemon=True)
        capture_thread.start()

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
        if self.root:
            try:
                self.root.destroy()
            except Exception as e:
                logging.error(f"Error destroying root window: {e}")

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
        logging.debug(f"Moving mouse to ({x}, {y}) with speed {speed}.")

    def click_mouse(self, button='left'):
        """
        Performs a mouse click.
        
        :param button: The mouse button to click ('left', 'right', 'middle').
        """
        logging.debug(f"Clicking mouse '{button}' button.")

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
