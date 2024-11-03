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

class VisionAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize InternVL2 model for visual understanding
        path = "OpenGVLab/InternVL2-2B"
        self.model = AutoModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=False,
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
        
        # Lazy load YOLO only when needed for precise coordinate detection
        self._yolo_model = None

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

    def understand_scene(self, image, question=None):
        """
        Uses InternVL2 for high-level visual understanding.
        
        Args:
            image: Input image (path, PIL Image, or numpy array)
            question: Optional specific question about the image
            
        Returns:
            str: Description or answer about the image
        """
        try:
            # Process image
            if isinstance(image, np.ndarray):
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            elif isinstance(image, str):
                image = Image.open(image).convert('RGB')
                
            # Format question
            if question:
                prompt = f"<image>\n{question}"
            else:
                prompt = "<image>\nDescribe what you see in this image in detail."
                
            # Generate response
            generation_config = {
                'max_new_tokens': 256,
                'do_sample': True,
                'temperature': 0.7
            }
            
            response = self.model.chat(
                self.tokenizer,
                image,
                prompt,
                generation_config
            )
            
            return response

        except Exception as e:
            logging.error(f"Error in scene understanding: {e}")
            return None

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
        
        Args:
            image: Input image (path, PIL Image, or numpy array)
            include_details: Whether to include detailed analysis
        
        Returns:
            dict: Scene perception results including:
                - description: General scene description
                - objects: List of detected objects
                - spatial_relations: Spatial relationships between objects
                - attributes: Scene attributes (colors, lighting, etc)
        """
        try:
            # Process image if needed
            processed_image = self.process_image(image)
            
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


