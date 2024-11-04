import logging
import tkinter as tk
import os
from PIL import Image, ImageTk
import numpy as np
import time
import cv2
import threading

class Screen:
    def __init__(self):
        self.width = 800  # Match browser viewport
        self.height = 600  # Match browser viewport
        self.current_state = None
        self.ui_elements = []
        self.window = None
        self.canvas = None
        self.current_frame = None
        self.mouse_position = (0, 0)
        self.is_container = os.getenv("IS_CONTAINER", "False") == "True"
        self.resolution = (800, 600)  # Match browser viewport
        self.frame_buffer = []
        self.frame_lock = threading.Lock()  # Add thread safety
        self.last_frame_time = 0
        self.frame_count = 0
        self.fps = 0

    def initialize(self):
        """
        Initializes the simulated screen with improved container support.
        """
        try:
            if not self.is_container:
                # Create Tk instance in main thread
                if not hasattr(self, '_tk_ready'):
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
                    
                    # Mark Tk as initialized
                    self._tk_ready = True
                    
                    # Start Tk event loop in separate thread if not running
                    if not hasattr(self, '_tk_thread'):
                        self._tk_thread = threading.Thread(target=self._run_tk_loop, daemon=True)
                        self._tk_thread.start()
                
            logging.debug("Screen initialized in %s mode", 
                         "container" if self.is_container else "window")

            self.current_state = self.capture()
            
        except Exception as e:
            logging.error(f"Error initializing screen: {e}")
            raise

    def _run_tk_loop(self):
        """Run Tkinter event loop in separate thread"""
        try:
            self.window.mainloop()
        except Exception as e:
            logging.error(f"Error in Tk event loop: {e}")

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
        Thread-safe implementation.
        
        Args:
            frame: numpy array or PIL Image representing the screen content
        """
        try:
            with self.frame_lock:
                current_time = time.time()
                
                # Calculate FPS
                if current_time - self.last_frame_time >= 1.0:
                    self.fps = self.frame_count
                    self.frame_count = 0
                    self.last_frame_time = current_time
                self.frame_count += 1
                
                # Validate and convert frame if needed
                if frame is None:
                    logging.warning("Received None frame")
                    return
                    
                if isinstance(frame, np.ndarray):
                    frame = self._process_numpy_frame(frame)
                elif not isinstance(frame, Image.Image):
                    raise ValueError(f"Unsupported frame type: {type(frame)}")

                # Ensure correct size
                if frame.size != (self.width, self.height):
                    frame = frame.resize((self.width, self.height), Image.Resampling.LANCZOS)

                self.current_frame = frame
                
                # Update frame buffer
                self._update_frame_buffer(frame)
                    
                if not self.is_container and self.canvas:
                    try:
                        # Convert to PhotoImage and display using thread-safe method
                        photo = ImageTk.PhotoImage(frame)
                        self.canvas.after(0, lambda: self._update_canvas(photo))
                    except Exception as e:
                        logging.error(f"Error updating canvas: {e}")

        except Exception as e:
            logging.error(f"Error updating frame: {e}", exc_info=True)

    def _process_numpy_frame(self, frame):
        """Process numpy array frames to correct format"""
        if len(frame.shape) == 2:  # Grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:  # RGBA
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        elif frame.shape[2] == 3 and frame.dtype == np.uint8:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame)

    def _update_frame_buffer(self, frame):
        """Update frame buffer with timestamp and maintain size"""
        self.frame_buffer.append({
            'frame': frame,
            'timestamp': time.time()
        })
        
        # Keep last 3 seconds of frames (assuming 30 FPS)
        while len(self.frame_buffer) > 90:
            self.frame_buffer.pop(0)

    def get_current_frame(self):
        """
        Returns the current frame with enhanced error handling and format consistency.
        
        Returns:
            PIL.Image or None: The current screen frame as a PIL Image in RGB format
        """
        try:
            if self.current_frame is not None:
                # Convert numpy array to PIL Image if needed
                if isinstance(self.current_frame, np.ndarray):
                    if len(self.current_frame.shape) == 2:  # Grayscale
                        return Image.fromarray(self.current_frame, 'L').convert('RGB')
                    elif len(self.current_frame.shape) == 3:
                        if self.current_frame.shape[2] == 4:  # RGBA
                            return Image.fromarray(self.current_frame, 'RGBA').convert('RGB')
                        else:  # Assume BGR/RGB
                            return Image.fromarray(cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB))
                elif isinstance(self.current_frame, Image.Image):
                    return self.current_frame.convert('RGB')  # Ensure RGB mode
                
            # If no frame exists, create blank RGB frame
            blank_frame = Image.new('RGB', (self.width, self.height), 'black')
            logging.debug("Created blank RGB frame due to no current frame")
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

    def _update_canvas(self, photo):
        """Thread-safe method to update canvas"""
        if not self.canvas:
            return
        
        try:
            # Store reference to photo to prevent garbage collection
            if not hasattr(self, '_current_photo'):
                self._current_photo = None
                
            self._current_photo = photo  # Keep strong reference
            
            def update():
                try:
                    if self.canvas.winfo_exists():
                        self.canvas.delete("all")
                        self.canvas.create_image(0, 0, image=self._current_photo, anchor='nw')
                except Exception as e:
                    logging.error(f"Error in canvas update: {e}")
                    
            # Schedule update in main thread
            if self.window and self.window.winfo_exists():
                self.window.after(0, update)
                
        except Exception as e:
            logging.error(f"Error preparing canvas update: {e}")
