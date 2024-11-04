"""
This class is for the vision agent which completes vision tasks for the input
"""
from app.agents.base import BaseAgent
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
import cv2
import logging
from transformers import AutoTokenizer, AutoModel
import os
import time
from collections import defaultdict
from functools import wraps
import contextlib
from typing import Optional, Dict, Any
from transformers import CLIPProcessor

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
        
        # Performance monitoring
        self.processing_times = []
        self.max_processing_times = 1000  # Keep last 1000 measurements
        self.error_counts = defaultdict(int)
        self.last_error_reset = time.time()
        self.error_threshold = 50  # Max errors before requiring intervention
        
        # Initialize models with proper error handling
        self._initialize_models()
        
        # Add YOLO model initialization
        self._yolo_model = None
        self.ui_classes = {
            0: "button",
            1: "text",
            2: "input",
            3: "link",
            4: "image",
            5: "icon"
            # Add more UI element classes as needed
        }
        
        # Initialize CLIPProcessor for image processing
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def _initialize_models(self):
        """Initialize vision models with fallback options"""
        try:
            # Initialize InternVL2 model
            path = "OpenGVLab/InternVL2-2B"
            self.model = AutoModel.from_pretrained(
                path,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                use_flash_attn=False,
                trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                path, 
                trust_remote_code=True, 
                use_fast=False
            )
            
            # Initialize model states
            self.model_healthy = True
            self.last_model_check = time.time()
            
        except Exception as e:
            logging.error(f"Failed to initialize vision models: {e}")
            self.model_healthy = False
            raise RuntimeError("Vision system initialization failed")

    def _check_model_health(self):
        """Periodic health check of vision models"""
        current_time = time.time()
        if current_time - self.last_model_check > 3600:  # Check every hour
            try:
                # Run test inference
                test_image = Image.new('RGB', (100, 100), color='red')
                self.understand_scene(test_image, "Test prompt")
                
                # Reset error counts periodically
                if current_time - self.last_error_reset > 86400:  # Daily reset
                    self.error_counts.clear()
                    self.last_error_reset = current_time
                    
                self.model_healthy = True
                
            except Exception as e:
                logging.error(f"Model health check failed: {e}")
                self.model_healthy = False
                
            self.last_model_check = current_time

    @property 
    def yolo_model(self):
        if self._yolo_model is None:
            self._yolo_model = YOLO('yolov8x.pt')
        return self._yolo_model

    def process_image(self, image):
        """Convert various image inputs to numpy array."""
        try:
            if image is None:
                raise ValueError("Image input cannot be None")
                
            if isinstance(image, str):
                # If image path
                if not os.path.exists(image):
                    raise FileNotFoundError(f"Image file not found: {image}")
                return cv2.imread(image)
                
            elif isinstance(image, Image.Image):
                # If PIL Image
                return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
            elif isinstance(image, np.ndarray):
                # Ensure correct number of channels
                if len(image.shape) == 2:  # Grayscale
                    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                elif image.shape[2] == 4:  # RGBA
                    return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
                elif image.shape[2] == 3:  # RGB/BGR
                    return image
                else:
                    raise ValueError(f"Unsupported number of channels: {image.shape[2]}")
                    
            else:
                raise ValueError(f"Unsupported image format: {type(image)}")
                
        except Exception as e:
            logging.error(f"Error processing image: {str(e)}")
            raise

    def _generate_scene_understanding(self, image: Any, question: Optional[str] = None) -> Dict[str, Any]:
        """Generate scene understanding response"""
        try:
            # Handle dictionary input by extracting the frame
            if isinstance(image, dict) and 'frame' in image:
                image = image['frame']
            
            if isinstance(image, Image.Image):
                # Convert PIL Image to tensor with correct dtype
                image = self.clip_processor(images=image, return_tensors="pt")
                image = {k: v.to(torch.float32) for k, v in image.items()}  # Ensure dtype matches model
            
            # Format question
            if question:
                prompt = f"<image>\n{question}"
            else:
                prompt = "<image>\nDescribe what you see in this image in detail."
                
            # Generate response with timeout handling
            start_time = time.time()
            response = self.model.chat(
                self.tokenizer,
                image,
                prompt,
                generation_config={
                    'max_new_tokens': 256,
                    'do_sample': True,
                    'temperature': 0.7
                }
            )
            
            # Check timeout
            if time.time() - start_time > 30:
                raise TimeoutError("Scene understanding took too long")
                
            return {
                'status': 'success',
                'description': response,
                'processing_time': time.time() - start_time
            }
                
        except Exception as e:
            logging.error(f"Error generating scene understanding: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'fallback_description': 'Unable to analyze image content'
            }

    def understand_scene(self, image, question=None):
        """Enhanced scene understanding with performance monitoring and fallbacks"""
        start_time = time.time()
        
        try:
            self._check_model_health()
            if not self.model_healthy:
                raise RuntimeError("Vision model is unhealthy")
                
            # Process image with validation
            processed_image = self._validate_and_process_image(image)
            
            # Convert PIL Image to tensor with correct dtype
            if isinstance(processed_image, Image.Image):
                processed_image = self.clip_processor(images=processed_image, return_tensors="pt")
                processed_image = {k: v.to(torch.float32) for k, v in processed_image.items()}  # Ensure dtype matches model
            
            # Generate response with timeout handling
            response = self._generate_scene_understanding(processed_image, question)
            
            # Monitor processing time
            processing_time = time.time() - start_time
            self._update_performance_metrics(processing_time)
            
            return response
            
        except Exception as e:
            self.error_counts['scene_understanding'] += 1
            logging.error(f"Scene understanding error: {e}")
            
            # Check if error threshold exceeded
            if self.error_counts['scene_understanding'] > self.error_threshold:
                self.model_healthy = False
                logging.critical("Vision system error threshold exceeded")
                
            return self._get_fallback_response()

    def _validate_and_process_image(self, image):
        """Validate and process image with quality checks and format conversion"""
        if image is None:
            raise ValueError("Image input cannot be None")
            
        # Convert to PIL Image with validation
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
        elif isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image file not found: {image}")
            image = Image.open(image).convert('RGB')
        elif isinstance(image, dict) and 'frame' in image:
            return self._validate_and_process_image(image['frame'])
        elif not isinstance(image, Image.Image):
            raise ValueError(f"Unsupported image type: {type(image)}")
            
        # Ensure RGB mode
        image = image.convert('RGB')
            
        # Validate image size and content
        if image.size[0] < 10 or image.size[1] < 10:
            raise ValueError("Image too small")
        if image.size[0] > 4096 or image.size[1] > 4096:
            raise ValueError("Image too large")
            
        return image

    def _update_performance_metrics(self, processing_time):
        """Update performance monitoring metrics"""
        self.processing_times.append(processing_time)
        if len(self.processing_times) > self.max_processing_times:
            self.processing_times.pop(0)
            
        # Calculate performance statistics
        avg_time = np.mean(self.processing_times)
        if avg_time > 5.0:  # Alert if average processing time exceeds 5 seconds
            logging.warning(f"High average processing time: {avg_time:.2f}s")

    def _get_fallback_response(self):
        """Provide fallback response when vision system fails"""
        return {
            'status': 'error',
            'message': 'Vision system temporarily unavailable',
            'fallback_description': 'Unable to analyze image content'
        }

    def find_element(self, image, target_description, confidence_threshold=0.3):
        """
        Uses YOLO to find precise coordinates of UI elements.
        
        Args:
            image: Input image
            target_description: Description of element to find
            confidence_threshold: Minimum confidence score
            
        Returns:
            dict: Element info including coordinates and confidence
        """
        try:
            img_array = self.process_image(image)
            
            # Run YOLO detection
            results = self.yolo_model(img_array)
            
            detected_elements = []
            for result in results[0].boxes.data:
                x1, y1, x2, y2, conf, class_id = result
                
                if conf < confidence_threshold:
                    continue
                    
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                element_info = {
                    'coordinates': (center_x, center_y),
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': float(conf),
                    'type': self.ui_classes[int(class_id)]
                }
                
                detected_elements.append(element_info)
            
            # Find best match for target description
            best_match = None
            best_score = 0
            
            for element in detected_elements:
                # Calculate match score based on description and element type
                score = self._calculate_match_score(target_description, element)
                if score > best_score:
                    best_score = score
                    best_match = element
                    
            return best_match if best_match and best_score > 0.5 else None

        except Exception as e:
            logging.error(f"Error finding element: {e}")
            return None

    def _calculate_match_score(self, description, element):
        """Calculate how well element matches description."""
        # Simple matching based on element type and keywords
        # This could be enhanced with better NLP matching
        description = description.lower()
        element_type = element['type'].lower()
        
        base_score = 0.0
        if element_type in description:
            base_score += 0.6
            
        # Add points for common UI terms
        ui_terms = ['button', 'link', 'input', 'text', 'image', 'icon']
        for term in ui_terms:
            if term in description and term in element_type:
                base_score += 0.2
                
        # Weight by confidence
        return base_score * element['confidence']

    def perceive_scene(self, image, include_details=True):
        """
        Perceives and analyzes the current scene using InternVL2.
        """
        try:
            # Process image if needed
            if isinstance(image, Image.Image):
                processed_image = np.array(image)
            elif isinstance(image, np.ndarray):
                processed_image = image
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")
                
            # Get basic scene understanding
            scene_description = self.understand_scene(processed_image)
            
            if not include_details:
                return {'description': scene_description}
                
            # Get detailed object detection
            results = self.yolo_model(processed_image)
            
            # Extract detected objects
            objects = []
            for result in results[0].boxes.data:
                x1, y1, x2, y2, conf, class_id = result
                objects.append({
                    'type': self.yolo_model.names[int(class_id)],
                    'confidence': float(conf),
                    'bbox': (int(x1), int(y1), int(x2), int(y2))
                })
                
            # Analyze spatial relationships
            spatial_relations = self._analyze_spatial_relations(objects)
            
            # Extract scene attributes
            attributes = self._extract_scene_attributes(processed_image)
            
            return {
                'description': scene_description,
                'objects': objects,
                'spatial_relations': spatial_relations,
                'attributes': attributes
            }
            
        except Exception as e:
            logging.error(f"Error in scene perception: {e}")
            return {'error': str(e)}
            
    def _analyze_spatial_relations(self, objects):
        """Analyzes spatial relationships between detected objects"""
        relations = []
        for i, obj1 in enumerate(objects):
            for obj2 in objects[i+1:]:
                relation = self._get_spatial_relation(obj1['bbox'], obj2['bbox'])
                relations.append({
                    'object1': obj1['type'],
                    'object2': obj2['type'],
                    'relation': relation
                })
        return relations
        
    def _get_spatial_relation(self, bbox1, bbox2):
        """Determines spatial relation between two bounding boxes"""
        x1_center = (bbox1[0] + bbox1[2]) / 2
        y1_center = (bbox1[1] + bbox1[3]) / 2
        x2_center = (bbox2[0] + bbox2[2]) / 2
        y2_center = (bbox2[1] + bbox2[3]) / 2
        
        dx = x2_center - x1_center
        dy = y2_center - y1_center
        
        if abs(dx) > abs(dy):
            return 'right of' if dx > 0 else 'left of'
        else:
            return 'below' if dy > 0 else 'above'
            
    def _extract_scene_attributes(self, image):
        """Extracts visual attributes from the scene"""
        # Convert to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
            
        # Calculate basic image statistics
        brightness = np.mean(image)
        contrast = np.std(image)
        
        # Dominant colors
        pixels = image.reshape(-1, 3)
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=5, n_init=10)
        kmeans.fit(pixels)
        colors = kmeans.cluster_centers_.astype(int)
        
        return {
            'brightness': float(brightness),
            'contrast': float(contrast),
            'dominant_colors': colors.tolist()
        }

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


