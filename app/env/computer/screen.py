import logging
import tkinter as tk
import os
from PIL import Image, ImageTk, ImageDraw, ImageFont
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
        self.last_frame_time = time.time()  # Initialize to current time
        self.frame_count = 0
        self.fps = 0
        self._tk_ready = False

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
    def normalize_coordinates(self, x, y, from_screenshot=True):
        """
        Convert coordinates between screenshot (1000x1000) and viewport spaces.
        
        Args:
            x (float): X coordinate
            y (float): Y coordinate
            from_screenshot (bool): If True, convert from screenshot to viewport.
                                  If False, convert from viewport to screenshot.
        
        Returns:
            tuple: (normalized_x, normalized_y)
        """
        if from_screenshot:
            # Convert from 1000x1000 screenshot space to viewport space
            normalized_x = (x * self.viewport_width) / self.screenshot_width
            normalized_y = (y * self.viewport_height) / self.screenshot_height
            print(f"Converting screenshot ({x}, {y}) -> viewport ({normalized_x}, {normalized_y})")
        else:
            # Convert from viewport space to 1000x1000 screenshot space
            normalized_x = (x * self.screenshot_width) / self.viewport_width
            normalized_y = (y * self.screenshot_height) / self.viewport_height
            print(f"Converting viewport ({x}, {y}) -> screenshot ({normalized_x}, {normalized_y})")
        
        return normalized_x, normalized_y

    def take_screenshot(self, filename="images/screenshot.png"):
        """Take a screenshot and overlay coordinate system scaled to 1000x1000."""
        # Take the screenshot
        self.driver.save_screenshot(filename)
        
        try:
            # Open and resize the screenshot to 1000x1000
            image = Image.open(filename)
            draw = ImageDraw.Draw(image)

            try:
                font = ImageFont.truetype("arial.ttf", 15)
            except IOError:
                font = None
            
            # Overlay the mouse position if available
            if self.last_mouse_position:
                # Draw viewport coordinates in red
                viewport_x, viewport_y = self.last_mouse_position
                mouse_size = 10
                draw.ellipse(
                    (viewport_x - mouse_size, viewport_y - mouse_size, 
                     viewport_x + mouse_size, viewport_y + mouse_size),
                    fill='red',
                    outline='black'
                )
                draw.text((viewport_x + 15, viewport_y), 
                         f"Viewport: ({int(viewport_x)}, {int(viewport_y)})", 
                         fill="red", 
                         font=font)
                
                # Draw screenshot coordinates in blue
                screenshot_x, screenshot_y = self.normalize_coordinates(
                    viewport_x, 
                    viewport_y, 
                    from_screenshot=False
                )
                draw.ellipse(
                    (screenshot_x - mouse_size, screenshot_y - mouse_size, 
                     screenshot_x + mouse_size, screenshot_y + mouse_size),
                    fill='blue',
                    outline='black'
                )
                draw.text((screenshot_x + 15, screenshot_y + 25), 
                         f"Screenshot: ({int(screenshot_x)}, {int(screenshot_y)})", 
                         fill="blue", 
                         font=font)

            image = image.resize((self.screenshot_width, self.screenshot_height))
            # Save the modified screenshot
            image.save(filename)
            print(f"Enhanced screenshot saved with viewport and screenshot coordinates at {filename}")
        except Exception as e:
            print(f"Error processing screenshot: {e}")



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
                time_diff = current_time - self.last_frame_time
                
                # Update frame count
                self.frame_count += 1
                
                # Calculate FPS every second
                if time_diff >= 1.0:
                    self.fps = int(self.frame_count / time_diff)
                    self.frame_count = 0
                    self.last_frame_time = current_time
                
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
                    
                # Only try to update canvas if we're not in test mode
                if not self.is_container and self.canvas and self._tk_ready:
                    try:
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
        Returns the current frame in a consistent RGB PIL Image format.
        """
        try:
            if not self.current_frame:
                # Create blank frame if none exists
                blank = Image.new('RGB', (self.width, self.height), 'black')
                return blank
                
            # If numpy array, convert to PIL Image
            if isinstance(self.current_frame, np.ndarray):
                if len(self.current_frame.shape) == 2:  # Grayscale
                    return Image.fromarray(self.current_frame, 'L').convert('RGB')
                elif len(self.current_frame.shape) == 3:
                    if self.current_frame.shape[2] == 4:  # RGBA
                        return Image.fromarray(self.current_frame, 'RGBA').convert('RGB')
                    else:  # Assume BGR
                        rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
                        return Image.fromarray(rgb)
                        
            # If already PIL Image, ensure RGB
            elif isinstance(self.current_frame, Image.Image):
                return self.current_frame.convert('RGB')
                
            return self.current_frame
            
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

    def get_screen_image(self):
        """
        Returns the current screen image as a numpy array.
        """
        try:
            current_frame = self.get_current_frame()
            if current_frame is None:
                logging.warning("No current frame available")
                return None
                
            # Convert PIL Image to numpy array if needed
            if isinstance(current_frame, Image.Image):
                return np.array(current_frame)
                
            return current_frame
            
        except Exception as e:
            logging.error(f"Error getting screen image: {e}")
            return None
