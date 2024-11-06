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
import os
import time
from collections import defaultdict
from functools import wraps
import contextlib
from typing import Optional, Dict, Any, Tuple
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import tempfile
from app.models.qwenvl import QwenVL
from app.models.samtwo import SAM2, SAM2Config

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

WEIGHTS_PATH = "weights/best.pt"
BOX_THRESHOLD = 0.03
DRAW_BBOX_CONFIG = {
    'text_scale': 0.8,
    'text_thickness': 2,
    'text_padding': 3,
    'thickness': 3,
}

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
            # Initialize SAM2 instead of OmniParser
            self.sam2 = SAM2()
            
            # Initialize QwenVL with retries
            max_retries = 3
            self.qwenvl = None  # Initialize to None first
            for attempt in range(max_retries):
                try:
                    self.qwenvl = QwenVL()
                    if hasattr(self.qwenvl, 'model_healthy') and self.qwenvl.model_healthy:
                        break
                    raise RuntimeError("QwenVL model initialization incomplete")
                except Exception as e:
                    if attempt == max_retries - 1:
                        logging.error(f"Failed to initialize QwenVL after {max_retries} attempts: {e}")
                        raise
                    logging.warning(f"QwenVL initialization attempt {attempt + 1} failed: {e}")
                    time.sleep(1)  # Wait before retry
            
            # Get references to models from QwenVL
            if self.qwenvl and self.qwenvl.model_healthy:
                self.clip_processor = self.qwenvl.clip_processor
                self.clip_model = self.qwenvl.clip_model
            else:
                raise RuntimeError("QwenVL initialization failed")
            
            # Add transform for InternVL2
            self.transform = self._build_transform(input_size=448)
            
            # Performance monitoring
            self.processing_times = []
            self.max_processing_times = 1000
            self.error_counts = defaultdict(int)
            self.last_error_reset = time.time()
            self.error_threshold = 50
            
        except Exception as e:
            logging.error(f"Error initializing VisionAgent: {e}")
            raise

    def _build_transform(self, input_size):
        """Build transform pipeline for InternVL2"""
        return T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

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
        Main entry point for scene perception.
        
        Args:
            image: Input image (PIL Image, numpy array, or dict with 'frame' key)
            context: Optional context for scene understanding
            
        Returns:
            dict: Scene perception results
        """
        try:
            # Handle case where QwenVL failed to initialize
            if not hasattr(self, 'qwenvl') or self.qwenvl is None:
                return {
                    'status': 'error',
                    'message': 'Vision system not properly initialized',
                    'timestamp': time.time()
                }

            # Use OmniParser as fallback if QwenVL fails
            try:
                return self.qwenvl.perceive_scene(image, context)
            except Exception as qwen_error:
                logging.warning(f"QwenVL perception failed, falling back to OmniParser: {qwen_error}")
                try:
                    return self.omniparser.detect_elements(image, context)
                except Exception as omni_error:
                    logging.error(f"Both perception systems failed: {omni_error}")
                    return {
                        'status': 'error',
                        'message': 'All perception systems failed',
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
        """Direct scene understanding using Qwen-VL"""
        return self.qwenvl.understand_scene(image, context)

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

    def detect_ui_elements(self, image, query=None):
        """Detect UI elements using SAM2"""
        return self.sam2.detect_elements(image, query)

    def _generate_detection_prompt(self, query):
        """
        Generate a focused prompt for element detection.
        
        Args:
            query: Description of the element to detect
            
        Returns:
            str: Formatted prompt for InternVL2
        """
        # Extract key characteristics from query
        characteristics = self._extract_element_characteristics(query)
        
        # Build the prompt template
        prompt = f"""<image>
Please help me locate this specific UI element:
{query}

Key characteristics to look for:
{characteristics}

Instructions:
1. Look for the exact element described
2. If found, provide its precise location using coordinates
3. Format the coordinates as: coordinates: (x1, y1, x2, y2)
4. Also describe where the element is on the screen
5. If the element is not found, explicitly state that

Remember to be precise with coordinate values and ensure they define a valid bounding box.
"""
        return prompt

    def _extract_element_characteristics(self, query):
        """
        Extract key characteristics from the element query to enhance detection.
        
        Args:
            query: Element description
            
        Returns:
            str: Formatted characteristics
        """
        # Common UI element types to look for
        ui_types = ['button', 'link', 'text', 'input', 'icon', 'menu', 'dropdown']
        visual_cues = ['color', 'blue', 'red', 'green', 'white', 'black', 'gray']
        positions = ['top', 'bottom', 'left', 'right', 'center']
        
        characteristics = []
        query_lower = query.lower()
        
        # Check for UI element type
        for ui_type in ui_types:
            if ui_type in query_lower:
                characteristics.append(f"- Type: {ui_type}")
                break
                
        # Check for visual cues
        for cue in visual_cues:
            if cue in query_lower:
                characteristics.append(f"- Visual cue: {cue}")
                
        # Check for position information
        for pos in positions:
            if pos in query_lower:
                characteristics.append(f"- Position: {pos}")
                
        # Add any specific text content
        if '"' in query:
            text = query[query.find('"')+1:query.rfind('"')]
            characteristics.append(f"- Exact text: \"{text}\"")
            
        # If no characteristics found, add the raw query
        if not characteristics:
            characteristics.append(f"- Description: {query}")
            
        return "\n".join(characteristics)

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
            
            # Generate focused prompt
            prompt = self.qwenvl.generate_detection_prompt(target_description)
            
            # Get scene understanding
            detection_result = self.qwenvl.understand_scene(processed_image, prompt)
            
            # Parse coordinates from response
            coordinates = self.qwenvl.parse_coordinates(detection_result.get('description', ''))
            
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

    def _get_coordinate_clarification(self, previous_response: str) -> Optional[str]:
        """
        Get clarification for coordinate extraction using multi-turn conversation.
        """
        try:
            clarification_prompt = f"""I see this description of coordinates:
            "{previous_response}"
            
            Please reformat these coordinates into a single line using exactly this format:
            coordinates: (x1, y1, x2, y2)
            
            For example: coordinates: (100, 200, 300, 400)
            
            Extract the numbers from the description and format them correctly.
            If you see multiple coordinates, use the most specific ones for the element."""
            
            # Get clarification from model
            result = self.understand_scene(None, clarification_prompt)
            if result and 'description' in result:
                return result['description']
                
            return None
            
        except Exception as e:
            logging.error(f"Error getting coordinate clarification: {e}")
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

    def _try_direct_coordinate_parse(self, text: str) -> Optional[Tuple[int, int, int, int]]:
        """
        Try to directly parse coordinates from text in format (x1, y1, x2, y2).
        
        Args:
            text: Text containing coordinate information
            
        Returns:
            Optional[Tuple[int, int, int, int]]: Parsed coordinates or None
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

    def _validate_coordinates(self, coords: Tuple[int, int, int, int]) -> bool:
        """
        Validate that coordinates form a valid bounding box.
        
        Args:
            coords: Tuple of (x1, y1, x2, y2) coordinates
            
        Returns:
            bool: True if coordinates are valid
        """
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
