import threading
import time
import logging
import numpy as np

class Mouse:
    """
    Simulates mouse controls for navigating applications such as a browser.
    """
    def __init__(self, target, movement_speed=1.0):
        """
        Initializes the Mouse.

        :param target: The target application or screen to control.
        :param movement_speed: Speed factor for mouse movements.
        """
        self.target = target
        self.movement_speed = movement_speed  # multiplier for movement speed
        self.action_queue = []
        self.running = False
        self.thread = threading.Thread(target=self.process_queue, daemon=True)
        self.current_position = (0, 0)
        self.is_clicking = False

    def start(self):
        """
        Starts the mouse action processing.
        """
        if not self.thread.is_alive():
            self.running = True
            self.thread = threading.Thread(target=self.process_queue, daemon=True)
            self.thread.start()
            logging.debug("Mouse action processing started.")
        else:
            logging.warning("Mouse thread is already running")

    def stop(self):
        """
        Stops the mouse action processing.
        """
        if self.thread.is_alive():
            self.running = False
            self.thread.join()
            logging.debug("Mouse action processing stopped.")
        else:
            logging.warning("Mouse thread is not running")

    def move_to(self, x, y, smooth=True):
        """
        Queues a mouse movement with optional smooth motion.

        Args:
            x: Target X coordinate
            y: Target Y coordinate
            smooth: If True, creates intermediate points for smooth movement
        """
        if smooth:
            # Generate path points for smooth movement
            start_x, start_y = self.current_position
            steps = max(abs(x - start_x), abs(y - start_y)) // 10
            if steps > 0:
                for i in range(steps):
                    intermediate_x = start_x + ((x - start_x) * i / steps)
                    intermediate_y = start_y + ((y - start_y) * i / steps)
                    self.action_queue.append(('move', intermediate_x, intermediate_y))
                    
        self.action_queue.append(('move', x, y))
        logging.debug(f"Queued mouse movement to ({x}, {y})")

    def click(self, button='left', double=False):
        """
        Queues a mouse click action.

        Args:
            button: Mouse button to click ('left', 'right', 'middle')
            double: If True, performs a double click
        """
        self.action_queue.append(('click_down', button))
        self.action_queue.append(('click_up', button))
        
        if double:
            self.action_queue.append(('click_down', button))
            self.action_queue.append(('click_up', button))
            
        logging.debug(f"Queued mouse {button} button {'double ' if double else ''}click")

    def process_queue(self):
        """
        Processes the mouse action queue with improved movement interpolation.
        """
        while self.running:
            if self.action_queue:
                action = self.action_queue.pop(0)
                action_type = action[0]
                
                if action_type == 'move':
                    _, x, y = action
                    self.current_position = (x, y)
                    self.target.update_mouse_position(type('Event', (), {'x': x, 'y': y}))
                    time.sleep(0.01 / self.movement_speed)  # Smoother movement
                    
                elif action_type.startswith('click'):
                    _, button = action
                    self.is_clicking = action_type == 'click_down'
                    time.sleep(0.05)  # Click duration
                    
            else:
                time.sleep(0.01)

    def get_position(self):
        """
        Returns current mouse position.
        """
        return self.current_position

    def is_button_pressed(self, button='left'):
        """
        Returns whether the specified mouse button is currently pressed.
        """
        return self.is_clicking
