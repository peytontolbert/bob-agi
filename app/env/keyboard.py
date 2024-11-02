import threading
import time
import logging

class Keyboard:
    """
    Simulates a keyboard for textual input.
    Streams input at an average typing speed to the target application or screen.
    """
    def __init__(self, target, typing_speed=5):
        """
        Initializes the Keyboard.

        :param target: The target application or screen to receive inputs.
        :param typing_speed: Average typing speed in characters per second.
        """
        self.target = target
        self.typing_speed = typing_speed  # characters per second
        self.input_queue = []
        self.running = False
        self.thread = threading.Thread(target=self.process_queue, daemon=True)

    def start(self):
        """
        Starts the keyboard input processing.
        """
        self.running = True
        self.thread.start()
        logging.debug("Keyboard input processing started.")

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
