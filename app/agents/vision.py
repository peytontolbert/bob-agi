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
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import io
import base64
from transformers import AutoModelForCausalLM
from transformers import DetrImageProcessor, DetrForObjectDetection
import traceback

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
        
        # Initialize models
        self._initialize_models()
        
        # Initialize CLIP for multimodal embeddings only
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        
        if torch.cuda.is_available():
            self.clip_model = self.clip_model.cuda()
        
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
        """Initialize vision models including InternVL2 and DETR"""
        try:
            # Initialize InternVL2 on GPU if available
            model_name = "OpenGVLab/InternVL2-2B"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,  # Use float16 instead of bfloat16
                device_map="cuda:0",
                trust_remote_code=True
            )
            
            # Move InternVL2 to GPU if available
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                logging.info("InternVL2 initialized on GPU")
            
            # Initialize DETR for UI element detection
            self.detr_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
            self.detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
            
            if torch.cuda.is_available():
                self.detr_model = self.detr_model.cuda()
                logging.info("DETR initialized on GPU")
                
            logging.info("Vision models initialized successfully")
            self.model_healthy = True
            
        except Exception as e:
            logging.error(f"Failed to initialize vision models: {e}")
            self.model_healthy = False
            raise RuntimeError(f"Vision system initialization failed: {str(e)}")

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
                pixel_values = pixel_values.to(torch.float16).cuda()  # Use float16 instead of bfloat16
            else:
                pixel_values = pixel_values.to(torch.float16)  # Use float16 instead of bfloat16
                
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
            # Process image
            pixel_values = self._preprocess_image(image)
            
            # Format prompt based on context
            if context and isinstance(context, str):
                prompt = f"<image>\n{context}"
            elif context and isinstance(context, dict):
                context_str = context.get('question', '')
                focus = context.get('focus', '')
                prompt = f"<image>\n{context_str}\nFocus on: {focus}" if focus else f"<image>\n{context_str}"
            else:
                prompt = "<image>\nDescribe what you see in this image in detail, focusing on UI elements, buttons, and interactive components."
                
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
        Enhanced element finding using InternVL2 understanding and YOLO detection.
        """
        try:
            # First get scene understanding to help identify the target
            scene_context = self.understand_scene(image, {
                'question': f"Describe the visual appearance and location of: {target_description}",
                'focus': 'UI elements and interactive components'
            })
            
            if scene_context['status'] != 'success':
                logging.warning("Scene understanding failed, proceeding with direct detection")
            else:
                # Use scene understanding to enhance target description
                enhanced_description = scene_context['description']
                logging.info(f"Enhanced understanding: {enhanced_description}")
            
            # Run YOLO detection
            yolo_results = self.detect_ui_elements(image)
            
            if yolo_results['status'] == 'success' and yolo_results['detections']:
                # Use scene understanding to help filter detections
                relevant_detections = []
                
                for detection in yolo_results['detections']:
                    # Calculate base confidence from YOLO
                    base_confidence = detection['confidence']
                    
                    # Check if detection matches target description
                    match_score = self._calculate_match_score(
                        detection,
                        target_description,
                        scene_context.get('description', '')
                    )
                    
                    if match_score > 0:
                        detection['match_score'] = match_score
                        detection['final_confidence'] = base_confidence * match_score
                        relevant_detections.append(detection)
                
                # Sort by final confidence
                relevant_detections.sort(key=lambda x: x['final_confidence'], reverse=True)
                
                if relevant_detections and relevant_detections[0]['final_confidence'] >= confidence_threshold:
                    best_match = relevant_detections[0]
                    return {
                        'element_found': True,
                        'element_details': {
                            'coordinates': best_match['center'],
                            'bbox': best_match['bbox'],
                            'confidence': best_match['final_confidence'],
                            'description': target_description,
                            'width': best_match['width'],
                            'height': best_match['height'],
                            'area': best_match['area'],
                            'class': best_match['class']
                        },
                        'scene_understanding': scene_context.get('description', ''),
                        'timestamp': time.time(),
                        'detector': 'yolo_with_internvl2'
                    }
            
            # If no good matches found
            return {
                'element_found': False,
                'timestamp': time.time(),
                'error_message': 'No matching elements found',
                'scene_understanding': scene_context.get('description', '')
            }
            
        except Exception as e:
            logging.error(f"Error finding element: {e}")
            return {
                'element_found': False,
                'timestamp': time.time(),
                'error_message': str(e)
            }

    def _calculate_match_score(self, detection, target_description, scene_understanding):
        """
        Calculate how well a detection matches the target description.
        
        Args:
            detection: Dictionary containing detection details
            target_description: Original target description
            scene_understanding: InternVL2's scene understanding
            
        Returns:
            float: Match score between 0 and 1
        """
        score = 0.0
        
        # Convert all text to lowercase for comparison
        target_words = set(target_description.lower().split())
        class_words = set(detection['class'].lower().split())
        scene_words = set(scene_understanding.lower().split())
        
        # Check direct class match
        common_class_words = target_words.intersection(class_words)
        if common_class_words:
            score += len(common_class_words) / len(target_words) * 0.6
        
        # Check scene understanding match
        target_in_scene = any(word in scene_understanding.lower() for word in target_words)
        if target_in_scene:
            score += 0.4
            
            # Check if detection location matches scene description
            location_keywords = ['top', 'bottom', 'left', 'right', 'center', 'middle']
            for keyword in location_keywords:
                if keyword in scene_understanding.lower() and self._check_position_match(detection, keyword):
                    score += 0.2
                    break
        
        return min(score, 1.0)  # Cap score at 1.0

    def _check_position_match(self, detection, position_keyword):
        """Check if detection position matches the described position"""
        center_x, center_y = detection['center']
        width, height = detection['width'], detection['height']
        
        # Get image dimensions from detection bbox
        x1, y1, x2, y2 = detection['bbox']
        img_width = x2 - x1 + width
        img_height = y2 - y1 + height
        
        # Check position match
        if position_keyword == 'top' and center_y < img_height * 0.33:
            return True
        elif position_keyword == 'bottom' and center_y > img_height * 0.67:
            return True
        elif position_keyword == 'left' and center_x < img_width * 0.33:
            return True
        elif position_keyword == 'right' and center_x > img_width * 0.67:
            return True
        elif position_keyword in ['center', 'middle']:
            return (0.33 < center_x/img_width < 0.67) and (0.33 < center_y/img_height < 0.67)
        
        return False

    def detect_ui_elements(self, image, query=None, similarity_threshold=0.5):
        """
        Detect UI elements using DETR with CLIP-based query filtering.
        
        Args:
            image: Input image
            query: Optional text query to filter elements
            similarity_threshold: Threshold for CLIP similarity (default 0.5)
        """
        try:
            # Convert image to PIL if needed
            image_pil = self._validate_and_process_image(image)

            # Process image with DETR
            inputs = self.detr_processor(images=image_pil, return_tensors="pt")
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
                self.detr_model = self.detr_model.cuda()

            # Get DETR predictions
            with torch.no_grad():
                outputs = self.detr_model(**inputs)
                
                # Process DETR outputs
                probas = outputs.logits.softmax(-1)[0, :, :-1]
                keep = probas.max(-1).values > 0.5  # Confidence threshold
                
                # Convert boxes to image coordinates
                boxes = outputs.pred_boxes[0, keep].cpu()
                scores = probas[keep].max(-1).values.cpu()

            # If no query, return all detected elements
            if not query:
                return self._process_detections(image_pil, boxes, scores)

            # Process query with CLIP
            text_inputs = self.clip_processor(
                text=[query], 
                return_tensors="pt", 
                padding=True
            )
            if torch.cuda.is_available():
                text_inputs = {k: v.cuda() for k, v in text_inputs.items()}
                
            with torch.no_grad():
                text_features = self.clip_model.get_text_features(**text_inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Process each region with CLIP
            matched_elements = []
            for box, score in zip(boxes, scores):
                # Extract region
                x1, y1, x2, y2 = box.tolist()
                region = image_pil.crop((x1, y1, x2, y2))
                
                # Get CLIP embedding for region
                region_inputs = self.clip_processor(
                    images=region, 
                    return_tensors="pt"
                )
                if torch.cuda.is_available():
                    region_inputs = {k: v.cuda() for k, v in region_inputs.items()}
                    
                with torch.no_grad():
                    image_features = self.clip_model.get_image_features(**region_inputs)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    
                    # Calculate similarity
                    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                    
                    if similarity.item() > similarity_threshold:
                        matched_elements.append({
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'confidence': score.item() * similarity.item(),
                            'similarity': similarity.item(),
                            'center': (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                            'width': int(x2 - x1),
                            'height': int(y2 - y1),
                            'area': int((x2 - x1) * (y2 - y1))
                        })

            # Sort by combined confidence
            matched_elements.sort(key=lambda x: x['confidence'], reverse=True)

            return {
                'status': 'success',
                'detections': matched_elements,
                'query': query,
                'timestamp': time.time(),
                'image_size': image_pil.size
            }

        except Exception as e:
            logging.error(f"Error in UI element detection: {e}")
            logging.debug(f"Stack trace: {traceback.format_exc()}")
            return {
                'status': 'error',
                'message': str(e),
                'timestamp': time.time()
            }

    def _process_detections(self, image, boxes, scores):
        """Process DETR detections into standardized format"""
        elements = []
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box.tolist()
            elements.append({
                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                'confidence': score.item(),
                'center': (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                'width': int(x2 - x1),
                'height': int(y2 - y1),
                'area': int((x2 - x1) * (y2 - y1))
            })
        
        return {
            'status': 'success',
            'detections': elements,
            'timestamp': time.time(),
            'image_size': image.size
        }

    def _check_position_match(self, detection, scene_desc):
        """Enhanced position matching using scene description"""
        desc_lower = scene_desc.lower()
        center_x, center_y = detection['center']
        width, height = detection['width'], detection['height']
        
        # Get image dimensions from detection bbox
        x1, y1, x2, y2 = detection['bbox']
        img_width = x2 - x1 + width
        img_height = y2 - y1 + height
        
        # Check various position descriptions
        if 'top' in desc_lower and center_y < img_height * 0.33:
            return True
        if 'bottom' in desc_lower and center_y > img_height * 0.67:
            return True
        if 'left' in desc_lower and center_x < img_width * 0.33:
            return True
        if 'right' in desc_lower and center_x > img_width * 0.67:
            return True
        if any(pos in desc_lower for pos in ['center', 'middle', 'middle of']):
            return (0.33 < center_x/img_width < 0.67) and (0.33 < center_y/img_height < 0.67)
            
        return False

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
