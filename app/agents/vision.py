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
from transformers import CLIPProcessor, CLIPModel
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

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
        
        # Initialize InternVL2 for direct image+text understanding
        self._initialize_models()
        
        # Initialize CLIP for multimodal embeddings only
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        
        # Add transform for InternVL2
        self.transform = self._build_transform(input_size=448)
        
        # Performance monitoring
        self.processing_times = []
        self.max_processing_times = 1000
        self.error_counts = defaultdict(int)
        self.last_error_reset = time.time()
        self.error_threshold = 50

    def _build_transform(self, input_size):
        """Build transform pipeline for InternVL2"""
        return T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

    def _initialize_models(self):
        """Initialize InternVL2 model for direct image+text understanding"""
        try:
            path = "OpenGVLab/InternVL2-2B"
            self.model = AutoModel.from_pretrained(
                path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                use_flash_attn=True,
                trust_remote_code=True
            ).eval()
            
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                
            self.tokenizer = AutoTokenizer.from_pretrained(
                path, 
                trust_remote_code=True,
                use_fast=False
            )
            self.model_healthy = True
            self.last_model_check = time.time()
            
        except Exception as e:
            logging.error(f"Failed to initialize vision models: {e}")
            self.model_healthy = False
            raise RuntimeError("Vision system initialization failed")

    def _preprocess_image(self, image, max_num=12):
        """Preprocess image for InternVL2"""
        try:
            # Ensure we have a PIL Image
            if isinstance(image, np.ndarray):
                if len(image.shape) == 2:  # Grayscale
                    image = Image.fromarray(image, 'L').convert('RGB')
                elif len(image.shape) == 3:
                    if image.shape[2] == 4:  # RGBA
                        image = Image.fromarray(image, 'RGBA').convert('RGB')
                    else:  # Assume BGR
                        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            elif isinstance(image, str):
                image = Image.open(image).convert('RGB')
            elif not isinstance(image, Image.Image):
                raise ValueError(f"Unsupported image type: {type(image)}")
                
            image = image.convert('RGB')
            
            # Dynamic preprocessing
            processed_images = self._dynamic_preprocess(image, image_size=448, max_num=max_num)
            pixel_values = [self.transform(img) for img in processed_images]
            pixel_values = torch.stack(pixel_values)
            
            if torch.cuda.is_available():
                pixel_values = pixel_values.to(torch.bfloat16).cuda()
            else:
                pixel_values = pixel_values.to(torch.bfloat16)
                
            return pixel_values
            
        except Exception as e:
            logging.error(f"Error preprocessing image: {e}")
            raise

    def _dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
        """Dynamic preprocessing for InternVL2"""
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height
        
        # Calculate target ratios
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) 
            for i in range(1, n + 1) 
            for j in range(1, n + 1) 
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
        
        # Find closest aspect ratio
        best_ratio = self._find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )
        
        # Calculate target dimensions
        target_width = image_size * best_ratio[0]
        target_height = image_size * best_ratio[1]
        blocks = best_ratio[0] * best_ratio[1]
        
        # Resize and split image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
            
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
            
        return processed_images

    def perceive_scene(self, image, context=None):
        """
        Main entry point for scene perception using InternVL2.
        
        Args:
            image: Input image (PIL Image, numpy array, or dict with 'frame' key)
            context: Optional context for scene understanding
            
        Returns:
            dict: Scene perception results
        """
        try:
            # Validate and process image
            processed_image = self._validate_and_process_image(image)
            
            # Generate base scene understanding
            scene_result = self.understand_scene(processed_image, context)
            
            # Get visual embedding for multimodal processing if needed
            if context and context.get('need_embedding', False):
                embedding = self.get_visual_embedding(processed_image)
                scene_result['embedding'] = embedding
                
            return {
                'status': 'success',
                'scene': scene_result,
                'timestamp': time.time()
            }
            
        except Exception as e:
            logging.error(f"Error in scene perception: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'timestamp': time.time()
            }

    def understand_scene(self, image, context=None):
        """Direct scene understanding using InternVL2"""
        try:
            # Ensure image is in correct format
            if isinstance(image, dict) and 'frame' in image:
                image = image['frame']
                
            # Convert numpy array to PIL Image if needed
            if isinstance(image, np.ndarray):
                if len(image.shape) == 2:  # Grayscale
                    image = Image.fromarray(image, 'L').convert('RGB')
                elif len(image.shape) == 3:
                    if image.shape[2] == 4:  # RGBA
                        image = Image.fromarray(image, 'RGBA').convert('RGB')
                    else:  # Assume BGR
                        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            elif isinstance(image, str):
                image = Image.open(image).convert('RGB')
            elif not isinstance(image, Image.Image):
                raise ValueError(f"Unsupported image type: {type(image)}")

            # Ensure image is RGB
            image = image.convert('RGB')

            # Preprocess image
            pixel_values = self._preprocess_image(image)
            
            # Format prompt based on context
            if context and isinstance(context, str):
                prompt = f"<image>\n{context}"
            elif context and isinstance(context, dict):
                context_str = context.get('question', '')
                focus = context.get('focus', '')
                prompt = f"<image>\n{context_str}\nFocus on: {focus}" if focus else f"<image>\n{context_str}"
            else:
                prompt = "<image>\nDescribe what you see in this image in detail."
                
            # Generate response with InternVL2
            generation_config = {
                'max_new_tokens': 256,
                'do_sample': True,
                'temperature': 0.7
            }
            
            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                prompt,
                generation_config
            )
            
            return {
                'status': 'success',
                'description': response,
                'timestamp': time.time()
            }
            
        except Exception as e:
            logging.error(f"Scene understanding error: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'fallback_description': 'Unable to analyze image content'
            }

    def get_visual_embedding(self, image):
        """Get CLIP embedding for multimodal processing"""
        try:
            # Process image for CLIP
            inputs = self.clip_processor(images=image, return_tensors="pt")
            
            with torch.no_grad():
                embedding = self.clip_model.get_image_features(**inputs)
            return embedding.numpy()
            
        except Exception as e:
            logging.error(f"Error getting visual embedding: {e}")
            return None

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
        Uses InternVL2 to find precise coordinates of UI elements.
        
        Args:
            image: Input image
            target_description: Description of element to find
            confidence_threshold: Minimum confidence score
            
        Returns:
            dict: Element info including coordinates and confidence
        """
        try:
            # Process image
            processed_image = self._validate_and_process_image(image)
            pixel_values = self._preprocess_image(processed_image)
            
            # Format specific prompt to get bounding box coordinates
            prompt = f"""<image>
Find the exact bounding box coordinates of: {target_description}
Return only the coordinates in format:
x1: [left coordinate], y1: [top coordinate], x2: [right coordinate], y2: [bottom coordinate]
If the element is not found, return "not found"."""
            
            # Generate response with InternVL2
            generation_config = {
                'max_new_tokens': 128,
                'do_sample': False,  # We want deterministic output for coordinates
                'temperature': 0.1   # Low temperature for precise answers
            }
            
            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                prompt,
                generation_config
            )
            
            # Parse coordinates from response
            coords = self._parse_coordinates(response)
            if coords:
                x1, y1, x2, y2 = coords
                
                # Validate coordinates are within image bounds
                img_width, img_height = processed_image.size
                if not (0 <= x1 < x2 <= img_width and 0 <= y1 < y2 <= img_height):
                    logging.warning(f"Invalid coordinates detected: {coords}")
                    return None
                
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                # Calculate confidence based on response clarity
                confidence = 1.0 if "not found" not in response.lower() else 0.0
                
                return {
                    'coordinates': (center_x, center_y),
                    'bbox': (x1, y1, x2, y2),
                    'confidence': confidence,
                    'description': target_description,
                    'width': x2 - x1,
                    'height': y2 - y1,
                    'area': (x2 - x1) * (y2 - y1),
                    'timestamp': time.time()
                }
                    
            return None

        except Exception as e:
            logging.error(f"Error finding element: {e}")
            return None

    def _parse_coordinates(self, response):
        """
        Parse coordinates from InternVL2 response with enhanced error handling.
        
        Args:
            response: String response from InternVL2
            
        Returns:
            tuple: (x1, y1, x2, y2) coordinates or None if parsing fails
        """
        try:
            import re
            
            # Look for coordinate patterns in the response
            # Match both "x1: 100" and "left: 100" style formats
            patterns = [
                r'(?:x1:|left:)\s*(\d+).*?(?:y1:|top:)\s*(\d+).*?(?:x2:|right:)\s*(\d+).*?(?:y2:|bottom:)\s*(\d+)',
                r'coordinates.*?(\d+)\D+(\d+)\D+(\d+)\D+(\d+)',
                r'bbox.*?(\d+)\D+(\d+)\D+(\d+)\D+(\d+)'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
                if match:
                    coords = tuple(map(int, match.groups()))
                    
                    # Basic validation
                    x1, y1, x2, y2 = coords
                    if x1 < x2 and y1 < y2:  # Ensure valid box dimensions
                        return coords
                        
            if "not found" in response.lower():
                logging.info("Element explicitly marked as not found in response")
            else:
                logging.warning(f"Could not parse coordinates from response: {response}")
                
            return None
            
        except Exception as e:
            logging.error(f"Error parsing coordinates: {e}")
            return None

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

    def _find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        """
        Finds the closest aspect ratio from target ratios that best matches the input image.
        
        Args:
            aspect_ratio (float): Original image aspect ratio
            target_ratios (list): List of (width, height) ratio tuples to choose from
            width (int): Original image width
            height (int): Original image height
            image_size (int): Target size for image patches
            
        Returns:
            tuple: Best matching (width, height) ratio
        """
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                # If same ratio difference, prefer the one that better preserves image area
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
                    
        return best_ratio

    def prepare_for_action(self, image, description):
        """
        Prepares visual input for action by detecting and analyzing UI elements.
        
        Args:
            image: Input image
            description: Description of elements to find
            
        Returns:
            list: Detected elements with coordinates and confidence scores
        """
        try:
            # Process image to correct format
            processed_image = self._validate_and_process_image(image)
            
            # Get scene understanding for context
            scene_context = self.understand_scene(processed_image, description)
            
            # Find the element using InternVL2
            element = self.find_element(processed_image, description)
            
            if element:
                # Calculate relevance based on scene understanding
                relevance = self._calculate_semantic_relevance(
                    description, 
                    scene_context.get('description', '')
                )
                element['relevance'] = relevance
                
                return [element]  # Return as list for compatibility
                
            return []
                
        except Exception as e:
            logging.error(f"Error preparing for action: {e}")
            return []

    def _calculate_semantic_relevance(self, target_desc, scene_desc):
        """Calculate semantic relevance between target and scene description"""
        # Simple keyword matching for now
        target_words = set(target_desc.lower().split())
        scene_words = set(scene_desc.lower().split())
        
        common_words = target_words.intersection(scene_words)
        if not target_words:
            return 0.0
            
        return len(common_words) / len(target_words)


