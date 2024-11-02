import logging
import tkinter as tk
import os
from PIL import Image, ImageTk
import numpy as np

class Screen:
    def __init__(self):
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
        Returns the current frame being displayed on screen.
        
        Returns:
            PIL.Image or None: The current screen frame as a PIL Image, or None if no frame is available
        """
        try:
            if self.current_frame is not None:
                if isinstance(self.current_frame, np.ndarray):
                    return Image.fromarray(self.current_frame)
                elif isinstance(self.current_frame, Image.Image):
                    return self.current_frame
            return None
        except Exception as e:
            logging.error(f"Error getting current frame: {e}")
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
