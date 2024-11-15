"""
This class is for the vision agent which completes vision tasks for the input
"""
from app.agents.base import BaseAgent
import torch
import numpy as np
from PIL import Image
import cv2
import logging
import os
import time
from collections import defaultdict
from functools import wraps
import tempfile
from typing import Optional, Dict, Any
from app.models.qwen2vl import Qwen2VL

def timeout(seconds):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            if end_time - start_time > seconds:
                raise TimeoutError(f"Function {func.__name__} took {end_time - start_time:.2f} seconds, which exceeds the timeout of {seconds} seconds")
            return result
        return wrapper
    return decorator

class VisionAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize models with error handling
        try:
            # Initialize Qwen2VL
            self.qwen = Qwen2VL()
            
            # Performance monitoring
            self.processing_times = []
            self.max_processing_times = 1000
            self.error_counts = defaultdict(int)
            self.last_error_reset = time.time()
            self.error_threshold = 50
            
        except Exception as e:
            logging.error(f"Error initializing VisionAgent: {e}")
            raise

    def perceive_scene(self, image, context=None):
        """
        Main entry point for scene perception.
        
        Args:
            image: Input image (PIL Image, numpy array, or dict with 'frame' key)
            context: Optional context for scene understanding
            
        Returns:
            dict: Scene perception results
        """
        try:
            # Validate and process image
            processed_image = self._validate_and_process_image(image)
            
            # Save image to temp file for Qwen2VL
            image_path = self._save_image_to_temp(processed_image)
            
            # Prepare input for Qwen2VL
            input_dict = {
                "image": image_path,
                "query": context if context else "Describe what you see in this image in detail."
            }
            
            # Get response from Qwen2VL
            response = self.qwen.chat(input_dict)
            
            # Clean up temp file
            os.remove(image_path)
            
            return {
                'status': 'success',
                'description': response[0] if isinstance(response, list) else response,
                'timestamp': time.time()
            }
                
        except Exception as e:
            logging.error(f"Error in scene perception: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'timestamp': time.time()
            }

    def _save_image_to_temp(self, image: Image.Image) -> str:
        """Save PIL Image to temporary file and return path"""
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"temp_image_{time.time()}.jpg")
        image.save(temp_path)
        return temp_path

    def understand_scene(self, image, context=None):
        """Direct scene understanding using Qwen2VL"""
        return self.perceive_scene(image, context)

    def _validate_and_process_image(self, image):
        """Validate and process image with quality checks and format conversion"""
        if image is None:
            raise ValueError("Image input cannot be None")
            
        # Handle dictionary input
        if isinstance(image, dict) and 'frame' in image:
            return self._validate_and_process_image(image['frame'])
            
        # Convert numpy array to PIL Image
        if isinstance(image, np.ndarray):
            # Validate array properties
            if not np.isfinite(image).all():
                raise ValueError("Image contains invalid values")
            if image.min() < 0 or image.max() > 255:
                raise ValueError("Image values out of valid range")
                
            # Convert based on channels
            if len(image.shape) == 2:  # Grayscale
                image = Image.fromarray(image, 'L').convert('RGB')
            elif len(image.shape) == 3:
                if image.shape[2] == 4:  # RGBA
                    image = Image.fromarray(image, 'RGBA').convert('RGB')
                else:  # Assume BGR
                    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    
        # Load from path
        elif isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image file not found: {image}")
            image = Image.open(image).convert('RGB')
            
        # Ensure we have a PIL Image
        elif not isinstance(image, Image.Image):
            raise ValueError(f"Unsupported image type: {type(image)}")
            
        # Ensure RGB mode
        image = image.convert('RGB')
            
        # Validate image size
        if image.size[0] < 10 or image.size[1] < 10:
            raise ValueError("Image too small")
        if image.size[0] > 4096 or image.size[1] > 4096:
            raise ValueError("Image too large")
            
        return image

    def find_element(self, image, target_description: str):
        """
        Find a specific UI element in the image.
        
        Args:
            image: Input image (PIL Image or numpy array)
            target_description: Text description of element to find
        
        Returns:
            dict: Detection results with element details or None if not found
        """
        try:
            # Process image to correct format
            processed_image = self._validate_and_process_image(image)
            
            # Save image to temp file
            image_path = self._save_image_to_temp(processed_image)
            
            # Prepare input for Qwen2VL
            prompt = f"""Find and locate this specific UI element: {target_description}
            If you find it, provide its location using coordinates in this exact format:
            coordinates: (x1, y1, x2, y2)
            Also describe where the element is on the screen."""
            
            input_dict = {
                "image": image_path,
                "query": prompt
            }
            
            # Get response from Qwen2VL
            response = self.qwen.chat(input_dict)
            response_text = response[0] if isinstance(response, list) else response
            
            # Clean up temp file
            os.remove(image_path)
            
            # Try to extract coordinates from response
            coordinates = self._try_direct_coordinate_parse(response_text)
            
            if coordinates:
                x1, y1, x2, y2 = coordinates
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                element_dict = {
                    'type': 'ui_element',
                    'coordinates': (center_x, center_y),
                    'bbox': (x1, y1, x2, y2),
                    'description': target_description,
                    'width': x2 - x1,
                    'height': y2 - y1
                }
                
                return {
                    'element_found': True,
                    'element_details': element_dict
                }
                
            return {
                'element_found': False,
                'message': 'Element not found'
            }
            
        except Exception as e:
            logging.error(f"Error finding element: {e}")
            return {
                'element_found': False,
                'error': str(e)
            }

    def _try_direct_coordinate_parse(self, text: str) -> Optional[tuple]:
        """
        Try to directly parse coordinates from text in format (x1, y1, x2, y2).
        """
        try:
            import re
            
            # Look for coordinates in format: coordinates: (x1, y1, x2, y2)
            pattern = r'coordinates:?\s*\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)'
            match = re.search(pattern, text, re.IGNORECASE)
            
            if match:
                coords = tuple(map(int, match.groups()))
                if self._validate_coordinates(coords):
                    return coords
                    
            return None
            
        except Exception as e:
            logging.error(f"Error in direct coordinate parsing: {e}")
            return None

    def _validate_coordinates(self, coords: tuple) -> bool:
        """Validate that coordinates form a valid bounding box."""
        try:
            if not coords or len(coords) != 4:
                return False
                
            x1, y1, x2, y2 = coords
            
            # Basic range checks (adjusted for typical screen resolutions)
            if any(c < 0 or c > 3840 for c in coords):  # Support up to 4K resolution
                return False
                
            # Ensure second coordinates are larger than first
            if x2 <= x1 or y2 <= y1:
                return False
                
            # Check for reasonable box size
            width = x2 - x1
            height = y2 - y1
            if width < 2 or height < 2:  # Allow smaller elements
                return False
            if width > 2000 or height > 2000:  # Maximum reasonable size
                return False
                
            return True
            
        except Exception:
            return False

    def complete_task(self, task_description: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Complete a vision-related task"""
        try:
            if not context or 'image' not in context:
                return {'status': 'error', 'message': 'No image provided in context'}
                
            image = context['image']
            if task_description.lower().startswith('find'):
                return self.find_element(image, task_description)
            else:
                return self.understand_scene(image, task_description)
                
        except Exception as e:
            logging.error(f"Error completing vision task: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
