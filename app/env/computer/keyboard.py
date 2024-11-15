import threading
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
import time
import logging

class Keyboard:
    """
    Simulates a keyboard for textual input.
    Streams input at an average typing speed to the target application or screen.
    """
    def __init__(self, driver, typing_speed=5):
        """
        Initializes the Keyboard.

        :param driver: The browser driver instance.
        :param typing_speed: Average typing speed in characters per second.
        """
        self.driver = driver
        self.typing_speed = typing_speed  # characters per second
        self.input_queue = []
        self.running = False
        self.thread = threading.Thread(target=self.process_queue, daemon=True)

    def is_running(self):
        """Check if keyboard thread is running."""
        return hasattr(self, 'thread') and self.thread.is_alive()

    def start(self):
        """Start keyboard thread if not already running."""
        if not self.is_running():
            self.running = True
            self.thread = threading.Thread(target=self.process_queue, daemon=True)
            self.thread.start()
        else:
            logging.debug("Keyboard thread already running")

    def stop(self):
        """
        Stops the keyboard input processing.
        """
        self.running = False
        self.thread.join()
        logging.debug("Keyboard input processing stopped.")

    def send_input(self, text):
        """
        Adds text to the input queue to be sent to the target.

        :param text: The text to send.
        """
        self.input_queue.append(text)
        logging.debug(f"Input added to queue: {text}")

    def process_queue(self):
        """
        Processes the input queue, sending inputs at the specified typing speed.
        """
        while self.running:
            if self.input_queue:
                text = self.input_queue.pop(0)
                for char in text:
                    self.target.receive_input(char)
                    logging.debug(f"Sent character: {char}")
                    time.sleep(1 / self.typing_speed)
            else:
                time.sleep(0.1)  # Wait before checking the queue again

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
