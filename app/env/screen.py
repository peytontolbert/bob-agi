import logging
import tkinter as tk
import os
from PIL import Image, ImageTk
import numpy as np
import time
import cv2

class Screen:
    def __init__(self):
        self.width = 1920
        self.height = 1080
        self.current_state = None
        self.ui_elements = []
        self.window = None
        self.canvas = None
        self.current_frame = None
        self.mouse_position = (0, 0)
        self.is_container = os.getenv("IS_CONTAINER", "False") == "True"
        self.resolution = (800, 600)  # Default resolution
        self.frame_buffer = []  # Store screen frames for Bob's vision

    def initialize(self):
        """
        Initializes the simulated screen with improved container support.
        """
        if not self.is_container:
            self.window = tk.Tk()
            self.window.title("Bob's View")
            self.window.geometry(f"{self.resolution[0]}x{self.resolution[1]}")
            
            # Create canvas for drawing
            self.canvas = tk.Canvas(
                self.window, 
                width=self.resolution[0], 
                height=self.resolution[1]
            )
            self.canvas.pack()
            
            # Bind mouse movement
            self.canvas.bind('<Motion>', self.update_mouse_position)
            
        logging.debug("Screen initialized in %s mode", 
                     "container" if self.is_container else "window")

        self.current_state = self.capture()

    def update_mouse_position(self, event):
        """
        Updates the current mouse position.
        """
        self.mouse_position = (event.x, event.y)
        logging.debug(f"Mouse position updated to {self.mouse_position}")

    def get_mouse_position(self):
        """
        Returns the current mouse position.
        """
        return self.mouse_position

    def update_frame(self, frame):
        """
        Updates the current frame buffer with new screen content.
        
        Args:
            frame: numpy array or PIL Image representing the screen content
        """
        self.current_frame = frame
        self.frame_buffer.append(frame)
        if len(self.frame_buffer) > 10:  # Keep last 10 frames
            self.frame_buffer.pop(0)
            
        if not self.is_container:
            # Convert frame to PhotoImage and display
            if isinstance(frame, np.ndarray):
                image = Image.fromarray(frame)
            else:
                image = frame
            photo = ImageTk.PhotoImage(image)
            self.canvas.create_image(0, 0, image=photo, anchor='nw')
            self.canvas.image = photo  # Keep reference
            self.window.update()

    def get_current_frame(self):
        """
        Returns the current frame with enhanced error handling.
        
        Returns:
            PIL.Image or None: The current screen frame as a PIL Image
        """
        try:
            if self.current_frame is not None:
                if isinstance(self.current_frame, np.ndarray):
                    return Image.fromarray(self.current_frame)
                elif isinstance(self.current_frame, Image.Image):
                    return self.current_frame.copy()
                else:
                    logging.warning(f"Unexpected frame type: {type(self.current_frame)}")
                    return None
                    
            # If no frame exists, create blank frame
            blank_frame = Image.new('RGB', (self.width, self.height), 'black')
            logging.debug("Created blank frame due to no current frame")
            return blank_frame
            
        except Exception as e:
            logging.error(f"Error getting current frame: {e}", exc_info=True)
            return None

    def get_frame_buffer(self):
        """
        Returns the recent frame history buffer.
        
        Returns:
            list: List of recent frames, each as PIL Image
        """
        try:
            return [
                Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame 
                for frame in self.frame_buffer
            ]
        except Exception as e:
            logging.error(f"Error getting frame buffer: {e}")
            return []

    def capture(self):
        """
        Captures current screen state with enhanced metadata, quality control and error handling.
        
        Returns:
            dict: Screen state including timestamp, resolution, UI elements, and quality metrics
            None: If capture fails
        """
        try:
            # First try to get current frame with error handling
            current_frame = self.get_current_frame()
            if current_frame is None:
                # Try to create a blank frame as fallback
                current_frame = Image.new('RGB', (self.width, self.height), 'black')
                logging.warning("Created blank frame as fallback after capture failure")
                
            # Enhanced screen state with quality metrics
            screen_state = {
                'timestamp': time.time(),
                'resolution': (self.width, self.height),
                'ui_elements': self.ui_elements.copy(),
                'frame': current_frame,
                'mouse_position': self.mouse_position,
                'quality_metrics': {
                    'brightness': self._calculate_brightness(current_frame),
                    'contrast': self._calculate_contrast(current_frame),
                    'sharpness': self._calculate_sharpness(current_frame)
                }
            }
            
            # Add frame to buffer with quality check
            if self._check_frame_quality(current_frame):
                self.frame_buffer.append({
                    'frame': current_frame,
                    'timestamp': time.time(),
                    'metadata': screen_state
                })
                
                # Maintain buffer size
                while len(self.frame_buffer) > 100:
                    self.frame_buffer.pop(0)
                    
            return screen_state
            
        except Exception as e:
            logging.error(f"Error capturing screen state: {e}", exc_info=True)
            return None

    def _calculate_brightness(self, frame):
        """Calculate average brightness of frame."""
        try:
            if isinstance(frame, Image.Image):
                frame = np.array(frame)
            return np.mean(frame)
        except Exception as e:
            logging.error(f"Error calculating brightness: {e}")
            return 0

    def _calculate_contrast(self, frame):
        """Calculate RMS contrast of frame."""
        try:
            if isinstance(frame, Image.Image):
                frame = np.array(frame)
            return np.std(frame)
        except Exception as e:
            logging.error(f"Error calculating contrast: {e}")
            return 0

    def _calculate_sharpness(self, frame):
        """Calculate image sharpness using Laplacian variance."""
        try:
            if isinstance(frame, Image.Image):
                frame = np.array(frame)
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            return cv2.Laplacian(frame, cv2.CV_64F).var()
        except Exception as e:
            logging.error(f"Error calculating sharpness: {e}")
            return 0

    def _check_frame_quality(self, frame, min_brightness=10, min_contrast=5, min_sharpness=100):
        """
        Check if frame meets minimum quality requirements.
        """
        try:
            brightness = self._calculate_brightness(frame)
            contrast = self._calculate_contrast(frame)
            sharpness = self._calculate_sharpness(frame)
            
            return (brightness >= min_brightness and 
                    contrast >= min_contrast and 
                    sharpness >= min_sharpness)
        except Exception as e:
            logging.error(f"Error checking frame quality: {e}")
            return False

    def display_elements(self, elements):
        """Update UI elements on screen"""
        self.ui_elements = elements
