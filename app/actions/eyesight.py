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
import collections
import json
import threading

class Eyesight:
    def __init__(self, screen):
        self.screen = screen
        self.vision_queue = queue.Queue()
        self.vision_agent = VisionAgent()
        
        # Initialize CLIP model for embeddings
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Enhanced image buffer with timestamps and metadata
        self.image_buffer = collections.deque(maxlen=100)  # Store last 100 frames
        self.embedding_buffer = collections.deque(maxlen=100)  # Store corresponding embeddings
        
        # Storage for image embeddings and metadata
        self.image_embeddings = []
        self.image_metadata = []
        
        # Create directory for storing images
        self.image_dir = "app/data/vision_memory"
        os.makedirs(self.image_dir, exist_ok=True)
        
        # Initialize element detection confidence threshold
        self.confidence_threshold = 0.7
        
        # Connect to screen's frame buffer
        self.screen.frame_buffer = []  # Initialize frame buffer in screen if not exists
        self.last_processed_frame_index = -1
        
        self.start_seeing()

    def generate_embedding(self, image):
        """
        Generates CLIP embedding for an image.
        """
        try:
            inputs = self.clip_processor(images=image, return_tensors="pt", padding=True)
            image_features = self.clip_model.get_image_features(**inputs)
            return image_features.detach().numpy()
        except Exception as e:
            logging.error(f"Error generating embedding: {e}")
            return None

    def save_image_with_embedding(self, image, description=None):
        """
        Saves image and its embedding with metadata.
        """
        try:
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"vision_{timestamp}.png"
            filepath = os.path.join(self.image_dir, filename)
            
            # Save image
            image.save(filepath)
            
            # Generate and store embedding
            embedding = self.generate_embedding(image)
            
            if embedding is not None:
                metadata = {
                    'filepath': filepath,
                    'timestamp': timestamp,
                    'description': description
                }
                
                self.image_embeddings.append(embedding)
                self.image_metadata.append(metadata)
                
                return filepath
            return None
        except Exception as e:
            logging.error(f"Error saving image and embedding: {e}")
            return None

    def find_similar_images(self, query_image, top_k=5):
        """
        Finds similar images based on embedding similarity.
        """
        try:
            query_embedding = self.generate_embedding(query_image)
            
            if query_embedding is None or not self.image_embeddings:
                return []
            
            # Calculate similarities
            similarities = [
                np.dot(query_embedding.flatten(), emb.flatten()) / 
                (np.linalg.norm(query_embedding) * np.linalg.norm(emb))
                for emb in self.image_embeddings
            ]
            
            # Get top-k similar images
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            return [self.image_metadata[i] for i in top_indices]
            
        except Exception as e:
            logging.error(f"Error finding similar images: {e}")
            return []

    def process_visual_input(self, input_text, image_path):
        """
        Processes visual input using the vision agent and stores embeddings.
        """
        try:
            # Process with vision agent
            result = self.vision_agent.complete_task(
                input_text,
                image_path
            )
            
            # Save image with embedding if it's not already saved
            if isinstance(image_path, Image.Image):
                saved_path = self.save_image_with_embedding(image_path, description=result)
                if saved_path:
                    result = {'description': result, 'image_path': saved_path}
            
            self.vision_queue.put(result)
            return result
        except Exception as e:
            logging.error(f"Error processing visual input: {e}")
            return None

    def process_screen_buffer(self):
        """
        Processes new frames from screen's frame buffer.
        """
        try:
            current_buffer = self.screen.get_frame_buffer()
            if not current_buffer:
                return
                
            # Process any new frames
            for i in range(self.last_processed_frame_index + 1, len(current_buffer)):
                frame = current_buffer[i]
                
                # Generate embedding
                embedding = self.generate_embedding(frame)
                
                # Store frame and embedding in buffers
                self.image_buffer.append({
                    'frame': frame,
                    'timestamp': time.time(),
                    'index': i
                })
                
                if embedding is not None:
                    self.embedding_buffer.append({
                        'embedding': embedding,
                        'timestamp': time.time(),
                        'index': i
                    })
                
                # Process frame for visual information
                self.process_visual_input("Analyze current screen state", frame)
                
            self.last_processed_frame_index = len(current_buffer) - 1
                
        except Exception as e:
            logging.error(f"Error processing screen buffer: {e}")

    def capture_screen(self):
        """
        Captures and processes the current screen state.
        """
        try:
            screenshot = self.screen.get_current_frame()
            if screenshot is None:
                return None
                
            # Find similar past images
            similar_images = self.find_similar_images(screenshot)
            
            # Create context-aware prompt using similar images
            context = "Previous similar observations:\n"
            for img_data in similar_images:
                if img_data.get('description'):
                    context += f"- {img_data['description']}\n"
            
            input_text = f"{context}\nGiven this context and the current image, describe what you see in detail."
            
            # Process the screenshot
            result = self.process_visual_input(input_text, screenshot)
            
            # Update buffers
            self.process_screen_buffer()
            
            return result
            
        except Exception as e:
            logging.error(f"Error capturing screen: {e}")
            return None

    def connect(self):
        """
        Sets up the eyesight for interaction with screen.
        """
        try:
            # Register with screen's frame buffer
            if hasattr(self.screen, 'frame_buffer'):
                self.screen.frame_buffer = []
            
            # Load existing embeddings if available
            embedding_path = os.path.join(self.image_dir, "embeddings.npy")
            metadata_path = os.path.join(self.image_dir, "metadata.json")
            
            if os.path.exists(embedding_path) and os.path.exists(metadata_path):
                self.image_embeddings = np.load(embedding_path)
                with open(metadata_path, 'r') as f:
                    self.image_metadata = json.load(f)
                    
            logging.info("Eyesight connected to screen system")
            
        except Exception as e:
            logging.error(f"Error connecting eyesight: {e}")

    def start_seeing(self):
        """
        Starts the eyesight system and begins processing visual input.
        """
        try:
            self.connect()
            
            # Start processing thread for screen buffer
            def process_buffer_loop():
                while True:
                    self.process_screen_buffer()
                    time.sleep(0.1)  # Process every 100ms
                    
            threading.Thread(target=process_buffer_loop, daemon=True).start()
            
        except Exception as e:
            logging.error(f"Error starting eyesight: {e}")
    
    def find_element(self, description: str) -> Optional[Dict]:
        """
        Finds an element on the screen based on description.
        
        Args:
            description: Text description of the element to find (e.g. "blue button", "search box")
            
        Returns:
            Dict containing element info or None if not found:
            {
                'type': str,  # Type of element (button, text, link, etc)
                'text': str,  # Text content if any
                'coordinates': tuple[int, int],  # (x, y) center position
                'confidence': float  # Detection confidence score
            }
        """
        try:
            screenshot = self.screen.capture()
            
            # Create a focused prompt for element detection
            prompt = f"""Find and locate the following element: {description}
            Provide the element type, any text content, and precise x,y coordinates."""
            
            # Process with vision agent
            result = self.vision_agent.complete_task(prompt, screenshot)
            
            # Parse the vision agent's response to extract element info
            # Note: Assuming vision agent returns structured data with coordinates
            if result and isinstance(result, dict):
                if result.get('confidence', 0) >= self.confidence_threshold:
                    return {
                        'type': result.get('type', 'unknown'),
                        'text': result.get('text', ''),
                        'coordinates': result.get('coordinates'),
                        'confidence': result.get('confidence', 0)
                    }
            return None
            
        except Exception as e:
            logging.error(f"Error finding element: {e}")
            return None

    def find_all_elements(self, description: str) -> List[Dict]:
        """
        Finds all elements matching the description on the screen.
        
        Args:
            description: Text description of elements to find
            
        Returns:
            List of dictionaries containing element info
        """
        try:
            screenshot = self.screen.capture()
            
            prompt = f"""Find all elements matching: {description}
            For each element, provide its type, text content, and precise x,y coordinates."""
            
            result = self.vision_agent.complete_task(prompt, screenshot)
            
            elements = []
            if result and isinstance(result, list):
                for elem in result:
                    if elem.get('confidence', 0) >= self.confidence_threshold:
                        elements.append({
                            'type': elem.get('type', 'unknown'),
                            'text': elem.get('text', ''),
                            'coordinates': elem.get('coordinates'),
                            'confidence': elem.get('confidence', 0)
                        })
            return elements
            
        except Exception as e:
            logging.error(f"Error finding elements: {e}")
            return []

    def wait_for_element(self, description: str, timeout: int = 10) -> Optional[Dict]:
        """
        Waits for an element to appear on screen up to timeout seconds.
        
        Args:
            description: Text description of the element to wait for
            timeout: Maximum time to wait in seconds
            
        Returns:
            Dict containing element info or None if timeout occurs
        """
        try:
            start_time = time.time()
            while time.time() - start_time < timeout:
                element = self.find_element(description)
                if element:
                    return element
                time.sleep(0.5)
            return None
            
        except Exception as e:
            logging.error(f"Error waiting for element: {e}")
            return None

    def get_screen_state(self):
        """
        Returns the current screen state and recent visual memory.
        """
        try:
            current_screen = self.screen.capture()
            screen_data = {
                'current': current_screen,
                'timestamp': datetime.now(),
                'similar_contexts': self.find_similar_images(current_screen),
                'elements': self.find_all_elements("*")  # Get all visible elements
            }
            return screen_data
        except Exception as e:
            logging.error(f"Error getting screen state: {e}")
            return None
