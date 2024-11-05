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
import cv2

class Eyesight:
    def __init__(self, screen):
        self.screen = screen
        self.vision_queue = queue.Queue()
        self.vision_agent = VisionAgent()
        self.is_running = True
        
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
        
        # Add perception stream buffer
        self.perception_stream = collections.deque(maxlen=300)  # Last 30 seconds at 10fps
        self.last_perception_time = 0
        self.PERCEPTION_INTERVAL = 0.1  # 100ms between perceptions
        
        # Add embedding storage
        self.perception_embeddings = []
        self.embedding_timestamps = []
        
        # Start perception thread
        self.perception_thread = threading.Thread(
            target=self._continuous_perception,
            daemon=True
        )
        self.perception_thread.start()

        self.start_seeing()

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
            elif isinstance(image, dict) and 'frame' in image:
                # Handle frame dictionary format
                return self.generate_embedding(image['frame'])
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

    def process_visual_input(self, input_text, image):
        """
        Processes visual input using the vision agent and stores embeddings.
        """
        try:
            # Validate image input
            if image is None:
                raise ValueError("Image input cannot be None")
                
            # Process image to correct format
            if isinstance(image, np.ndarray):
                if len(image.shape) == 2:  # Grayscale
                    image = Image.fromarray(image, 'L').convert('RGB')
                elif len(image.shape) == 3:
                    if image.shape[2] == 4:  # RGBA
                        image = Image.fromarray(image, 'RGBA').convert('RGB')
                    else:  # Assume BGR
                        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            elif isinstance(image, dict) and 'frame' in image:
                return self.process_visual_input(input_text, image['frame'])
            
            if not isinstance(image, Image.Image):
                raise ValueError(f"Unsupported image type: {type(image)}")
                
            # Ensure RGB mode
            image = image.convert('RGB')
            
            # Generate embedding first
            embedding = self.generate_embedding(image)
            
            # Process with vision agent
            result = self.vision_agent.understand_scene(image, input_text)
            
            if not isinstance(result, dict):
                result = {'description': str(result)}
                
            # Add embedding and metadata
            result.update({
                'embedding': embedding,
                'timestamp': time.time(),
                'status': 'success'
            })
            
            # Save to perception stream
            self.perception_stream.append(result)
            
            return result
            
        except Exception as e:
            logging.error(f"Error processing visual input: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'description': 'Failed to process visual input'
            }

    def process_screen_buffer(self):
        """
        Processes new frames from screen's frame buffer with enhanced error handling.
        """
        try:
            current_buffer = self.screen.get_frame_buffer()
            if not current_buffer:
                logging.debug("No new frames in buffer to process")
                return
                
            # Process any new frames
            for i in range(self.last_processed_frame_index + 1, len(current_buffer)):
                frame = current_buffer[i]
                
                if frame is None:
                    logging.warning(f"Skipping null frame at index {i}")
                    continue
                    
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
            logging.error(f"Error processing screen buffer: {e}", exc_info=True)

    def capture_screen(self):
        """
        Captures and processes the current screen state with proper image format handling.
        """
        try:
            # Get screenshot with retry logic
            max_retries = 3
            screenshot = None
            
            for attempt in range(max_retries):
                screenshot = self.screen.get_current_frame()
                if screenshot is not None and isinstance(screenshot, Image.Image):
                    break
                logging.warning(f"Screen capture attempt {attempt + 1} failed, retrying...")
                time.sleep(0.1)
                    
            if screenshot is None:
                logging.error("All screen capture attempts failed")
                return None
                
            # Ensure RGB format
            screenshot = screenshot.convert('RGB')
            
            # Get current thought context
            context = self._get_thought_context()
            
            # Use vision agent for scene understanding
            perception = self.vision_agent.understand_scene(screenshot, context)
            
            screen_state = {
                'timestamp': time.time(),
                'perception': perception,
                'context': context,
                'frame': screenshot
            }
            
            return screen_state
            
        except Exception as e:
            logging.error(f"Error capturing screen: {e}", exc_info=True)
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
        Finds specific element for action using precise detection.
        """
        try:
            current_frame = self.screen.capture()
            
            # Use YOLO+SAM for precise element detection
            elements = self.vision_agent.prepare_for_action(
                current_frame, 
                description
            )
            
            if elements:
                # Return most confident match
                best_match = max(elements, key=lambda x: x['confidence'])
                return best_match
            
            return None
            
        except Exception as e:
            logging.error(f"Error finding element: {e}")
            return None

    def _find_in_perception_stream(self, description: str) -> Optional[Dict]:
        """
        Searches recent perception stream for relevant information.
        """
        try:
            # Get last 30 seconds of perceptions
            recent_perceptions = list(self.perception_stream)
            
            if not recent_perceptions:
                return None
                
            # Find most relevant perception
            best_match = None
            best_score = 0
            
            for perception in recent_perceptions:
                # Calculate relevance score
                score = self._calculate_relevance(
                    description,
                    perception['perception']['description']
                )
                
                if score > best_score:
                    best_score = score
                    best_match = perception
            
            if best_score > 0.7:  # Threshold for relevance
                return best_match
                
            return None
            
        except Exception as e:
            logging.error(f"Error searching perception stream: {e}")
            return None

    def _calculate_relevance(self, query: str, perception: str) -> float:
        """
        Calculates semantic relevance between query and perception.
        """
        # Implement semantic similarity calculation
        # Could use cosine similarity between embeddings
        return 0.0  # Placeholder

    def _continuous_perception(self):
        """
        Continuously processes visual input based on current thoughts.
        """
        while self.is_running:
            try:
                current_time = time.time()
                
                if current_time - self.last_perception_time >= self.PERCEPTION_INTERVAL:
                    # Get current frame
                    frame = self.screen.get_current_frame()
                    if frame is None:
                        time.sleep(0.1)
                        continue
                        
                    # Get current context from Bob's thoughts
                    context = self._get_thought_context()
                    
                    # Get scene perception
                    perception = self.vision_agent.perceive_scene(frame)
                    
                    if perception and perception.get('status') == 'success':
                        # Generate embedding
                        embedding = self.generate_embedding(frame)
                        
                        # Store perception and embedding
                        perception_data = {
                            'perception': perception,
                            'embedding': embedding,
                            'timestamp': current_time,
                            'context': context
                        }
                        
                        self.perception_stream.append(perception_data)
                        self.last_perception_time = current_time
                        
            except Exception as e:
                logging.error(f"Error in continuous perception: {e}")
                time.sleep(0.1)  # Avoid tight loop on errors
            
            time.sleep(0.01)  # Small sleep to prevent CPU overuse

    def _get_thought_context(self):
        """Gets current thought context from Bob."""
        # This will be connected to Bob's thought system
        return {
            'thought_focus': None,
            'current_goal': None
        }

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

    def get_screen_image(self):
        """
        Gets the current screen image in the correct format.
        
        Returns:
            PIL.Image: Current screen image in RGB format
        """
        try:
            # Get current frame from screen
            screenshot = self.screen.get_current_frame()
            
            # Validate and convert image format
            if screenshot is None:
                self.logger.error("Failed to capture screen image")
                return None
                
            # Ensure we have a PIL Image
            if isinstance(screenshot, np.ndarray):
                if len(screenshot.shape) == 2:  # Grayscale
                    screenshot = Image.fromarray(screenshot, 'L').convert('RGB')
                elif len(screenshot.shape) == 3:
                    if screenshot.shape[2] == 4:  # RGBA
                        screenshot = Image.fromarray(screenshot, 'RGBA').convert('RGB')
                    else:  # Assume BGR
                        screenshot = Image.fromarray(cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB))
            elif not isinstance(screenshot, Image.Image):
                raise ValueError(f"Unsupported image type: {type(screenshot)}")
                
            # Ensure RGB mode
            screenshot = screenshot.convert('RGB')
            
            return screenshot
            
        except Exception as e:
            self.logger.error(f"Error getting screen image: {e}")
            return None
