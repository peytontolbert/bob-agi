"""
The interface for Bob's vision for the computer with image embedding capabilities.
"""
from app.agents.vision import VisionAgent
import logging
import queue
from PIL import Image
import numpy as np
from datetime import datetime
import os
import torch
from transformers import CLIPProcessor, CLIPModel
from typing import Dict, Optional, List
import time
import cv2
import collections
import json
import threading

class Eyesight:
    def __init__(self, browser, vision_agent: Optional[VisionAgent] = None):
        self.browser = browser
        self.vision_agent = vision_agent
        self.is_running = True
        
        # Initialize CLIP model for embeddings
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Enhanced image buffer with timestamps and metadata
        self.image_buffer = collections.deque(maxlen=100)  # Store last 100 frames
        self.embedding_buffer = collections.deque(maxlen=100)  # Store corresponding embeddings
        
        # Create directory for storing images
        self.image_dir = "app/data/vision_memory"
        os.makedirs(self.image_dir, exist_ok=True)
        
        # Initialize element detection confidence threshold
        self.confidence_threshold = 0.7
        
        # Add perception stream buffer
        self.perception_stream = collections.deque(maxlen=300)  # Last 30 seconds at 10fps
        self.last_perception_time = 0
        self.PERCEPTION_INTERVAL = 0.1  # 100ms between perceptions

        # Start perception thread
        self.perception_thread = threading.Thread(
            target=self._continuous_perception,
            daemon=True
        )
        self.perception_thread.start()

    def generate_embedding(self, image):
        """
        Generates CLIP embedding for an image with proper format handling.
        """
        try:
            # Convert to PIL Image if needed
            if isinstance(image, np.ndarray):
                if len(image.shape) == 2:  # Grayscale
                    image = Image.fromarray(image, 'L').convert('RGB')
                elif len(image.shape) == 3:
                    if image.shape[2] == 4:  # RGBA
                        image = Image.fromarray(image, 'RGBA').convert('RGB')
                    else:  # Assume BGR
                        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            elif not isinstance(image, Image.Image):
                raise ValueError(f"Unsupported image type: {type(image)}")
                
            # Ensure RGB mode
            image = image.convert('RGB')
            
            # Generate embedding
            with torch.no_grad():
                inputs = self.clip_processor(images=image, return_tensors="pt")
                image_features = self.clip_model.get_image_features(**inputs)
                return image_features.detach().numpy()
            
        except Exception as e:
            logging.error(f"Error generating embedding: {e}")
            return np.zeros((1, 512))  # Return zero embedding as fallback

    def get_screen_state(self):
        """
        Returns the current screen state and recent visual memory.
        """
        try:
            # Take screenshot using browser's function
            screenshot_path = os.path.join(self.image_dir, "current_screen.png")
            self.browser.take_screenshot(screenshot_path)
            
            # Load the screenshot as PIL Image
            current_screen = Image.open(screenshot_path)
            
            # Generate embedding
            embedding = self.generate_embedding(current_screen)
            
            # Store in buffers
            screen_data = {
                'image': current_screen,
                'embedding': embedding,
                'timestamp': datetime.now(),
                'path': screenshot_path
            }
            
            self.image_buffer.append(screen_data)
            
            return screen_data
            
        except Exception as e:
            logging.error(f"Error getting screen state: {e}")
            return None

    def find_element(self, description: str) -> Optional[Dict]:
        """
        Uses vision agent to find specific UI element matching description.
        Returns element with coordinates and confidence score.
        """
        try:
            # Get current screen
            screen_state = self.get_screen_state()
            if not screen_state:
                return None
            
            # Have vision agent find matching elements
            elements = self.vision_agent.detect_elements(
                screen_state['image'],
                description
            )
            
            if not elements:
                return None
            
            # Return element with highest confidence above threshold
            best_match = max(elements, key=lambda x: x['confidence'])
            if best_match['confidence'] >= self.confidence_threshold:
                return {
                    'coordinates': best_match['coordinates'],
                    'confidence': best_match['confidence'],
                    'description': description,
                    'type': best_match['type']
                }
            
            return None
            
        except Exception as e:
            logging.error(f"Error finding element: {e}")
            return None

    def _continuous_perception(self):
        """
        Continuously processes visual input based on current thoughts.
        """
        while self.is_running:
            try:
                current_time = time.time()
                
                if current_time - self.last_perception_time >= self.PERCEPTION_INTERVAL:
                    # Get current screen state
                    screen_state = self.get_screen_state()
                    if screen_state:
                        # Get scene perception from vision agent
                        perception = self.vision_agent.perceive_scene(screen_state['image'])
                        
                        if perception:
                            perception_data = {
                                'perception': perception,
                                'embedding': screen_state['embedding'],
                                'timestamp': current_time,
                                'image_path': screen_state['path']
                            }
                            
                            self.perception_stream.append(perception_data)
                            self.last_perception_time = current_time
                        
            except Exception as e:
                logging.error(f"Error in continuous perception: {e}")
                time.sleep(0.1)  # Avoid tight loop on errors
            
            time.sleep(0.01)  # Small sleep to prevent CPU overuse
