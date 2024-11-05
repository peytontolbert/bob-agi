import threading
import time
import logging
import numpy as np
from typing import Tuple, Optional

class Mouse:
    """
    Simulates mouse controls with enhanced position tracking and movement validation.
    """
    def __init__(self, target, movement_speed=1.0):
        self.target = target
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
        
        # Movement constraints
        self.max_speed = 2000  # pixels per second
        self.min_move_time = 0.05  # minimum seconds for any movement
        self.screen_bounds = (1920, 1080)  # default bounds
        
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

    def move_to(self, x: float, y: float, smooth: bool = True) -> bool:
        """
        Queue mouse movement with enhanced validation and smoothing.
        
        Args:
            x: Target X coordinate
            y: Target Y coordinate
            smooth: Whether to use smooth movement
            
        Returns:
            bool: True if movement was queued successfully
        """
        try:
            # Validate coordinates
            if not self._validate_coordinates(x, y):
                self.logger.warning(f"Invalid coordinates: ({x}, {y})")
                return False

            # Round coordinates to integers to avoid floating point issues
            x = round(x)
            y = round(y)

            with self.queue_lock:
                if smooth:
                    # Generate smooth movement path with more points for precision
                    path = self._generate_movement_path(
                        self.current_position, 
                        (x, y),
                        steps=30  # Increased steps for smoother movement
                    )
                    for px, py in path:
                        self.action_queue.append(('move', round(px), round(py)))
                else:
                    self.action_queue.append(('move', x, y))

            return True

        except Exception as e:
            self.logger.error(f"Error queueing movement: {e}")
            return False

    def _generate_movement_path(
        self, 
        start: Tuple[float, float], 
        end: Tuple[float, float],
        steps: int = 30
    ) -> list:
        """
        Generate smooth movement path using improved easing.
        """
        path = []
        for i in range(steps + 1):
            t = i / steps
            # Improved easing function for more natural movement
            t = t * t * (3 - 2 * t)  # Smoothstep interpolation
            x = start[0] + (end[0] - start[0]) * t
            y = start[1] + (end[1] - start[1]) * t
            path.append((x, y))
        return path

    def _validate_coordinates(self, x: float, y: float) -> bool:
        """
        Validate coordinates are within screen bounds.
        """
        return (0 <= x <= self.screen_bounds[0] and 
                0 <= y <= self.screen_bounds[1])

    def _update_position(self, x: float, y: float):
        """Thread-safe position update."""
        with self.position_lock:
            self.current_position = (x, y)
            # Update target's mouse position if method exists
            if hasattr(self.target, 'update_mouse_position'):
                self.target.update_mouse_position(
                    type('Event', (), {'x': x, 'y': y})
                )

    @property
    def position(self) -> Tuple[float, float]:
        """Thread-safe position access."""
        with self.position_lock:
            return self.current_position

    def get_position(self) -> Tuple[float, float]:
        """
        Thread-safe method to get current mouse position.
        
        Returns:
            Tuple[float, float]: Current (x, y) coordinates
        """
        with self.position_lock:
            return self.current_position

    def process_queue(self):
        """
        Process queued mouse actions with improved timing and verification.
        """
        while self.running:
            try:
                action = None
                with self.queue_lock:
                    if self.action_queue:
                        action = self.action_queue.pop(0)

                if action:
                    action_type = action[0]
                    
                    if action_type == 'move':
                        _, x, y = action
                        self._update_position(x, y)
                        # Add small delay based on movement speed
                        time.sleep(0.005 / self.movement_speed)
                        
                    elif action_type == 'click_down':
                        button = action[1]
                        self.is_clicking = True
                        # Notify target of mouse down if method exists
                        if hasattr(self.target, 'mouse_down'):
                            self.target.mouse_down(button)
                        time.sleep(0.05)  # Click timing delay
                        
                    elif action_type == 'click_up':
                        button = action[1]
                        self.is_clicking = False
                        # Notify target of mouse up if method exists
                        if hasattr(self.target, 'mouse_up'):
                            self.target.mouse_up(button)
                        time.sleep(0.05)  # Click timing delay
                        
                    elif action_type == 'delay':
                        delay_time = action[1]
                        time.sleep(delay_time)
                else:
                    time.sleep(0.005)

            except Exception as e:
                self.logger.error(f"Error processing mouse action: {e}")
                time.sleep(0.1)

    def click(self, button: str = 'left', double: bool = False):
        """
        Perform mouse click action.
        
        Args:
            button: Which mouse button to click ('left' or 'right')
            double: Whether to perform a double click
        """
        try:
            with self.queue_lock:
                self.action_queue.append(('click_down', button))
                self.action_queue.append(('click_up', button))
                
                if double:
                    # Add small delay between clicks for double-click
                    self.action_queue.append(('delay', 0.1))
                    self.action_queue.append(('click_down', button))
                    self.action_queue.append(('click_up', button))
                    
            self.last_click = {
                'button': button,
                'time': time.time(),
                'position': self.current_position,
                'double': double
            }
            
            self.logger.debug(
                f"{'Double ' if double else ''}Click queued at {self.current_position}"
            )
            
        except Exception as e:
            self.logger.error(f"Error queueing click: {e}")
