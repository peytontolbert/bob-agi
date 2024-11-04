import unittest
import time  # Add import for time
from app.env.mouse import Mouse
from app.env.screen import Screen

class TestMouse(unittest.TestCase):
    def setUp(self):
        self.screen = Screen()
        self.mouse = Mouse(target=self.screen)
        self.mouse.start()

    def wait_for_position(self, expected_position, timeout=1.0):
        """
        Waits until the mouse reaches the expected position or the timeout is exceeded.

        :param expected_position: Tuple of (x, y) coordinates.
        :param timeout: Maximum time to wait in seconds.
        :return: True if the position is reached, False otherwise.
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.mouse.position == expected_position:
                return True
            time.sleep(0.01)  # Sleep briefly to allow processing
        return False

    def test_mouse_movement(self):
        self.mouse.move(x=100, y=200)
        movement_completed = self.wait_for_position((100, 200))
        self.assertTrue(movement_completed, f"Mouse did not move to (100, 200) within the timeout.")

    def test_mouse_click(self):
        self.mouse.click(button='left')
        # Optionally, you can add a wait here if necessary
        self.assertTrue(self.mouse.last_click == 'left')

