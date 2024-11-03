"""
This class is for the vision agent which completes vision tasks for the input
"""
from app.agents.base import BaseAgent
import base64
import torch
import numpy as np
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
from ultralytics import YOLO
import cv2
import logging
from transformers import (
    Qwen2VLForConditionalGeneration, 
    AutoProcessor,
    AutoTokenizer
)
import os
import time
import collections

class VisionAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize Qwen VL model for high-level scene understanding
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
            min_pixels=256*28*28,
            max_pixels=1280*28*28
        )
        
        # Lazy load YOLO and SAM only when needed for precise actions
        self._yolo_model = None
        self._sam_predictor = None
        
        # Add separate buffers for scene understanding and action detection
        self.scene_memory = collections.deque(maxlen=50)  # Last 50 scene understandings
        self.action_memory = collections.deque(maxlen=20)  # Last 20 precise detections

    @property
    def yolo_model(self):
        if self._yolo_model is None:
            self._yolo_model = YOLO('yolov8x.pt')
        return self._yolo_model

    @property
    def sam_predictor(self):
        if self._sam_predictor is None:
            self._sam_model = sam_model_registry["vit_h"](checkpoint="models/sam_vit_h_4b8939.pth")
            self._sam_model.to("cuda" if torch.cuda.is_available() else "cpu")
            self._sam_predictor = SamPredictor(self._sam_model)
        return self._sam_predictor

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

    def detect_elements(self, image, confidence_threshold=0.3):
        """
        Enhanced UI element detection with improved accuracy and error handling.
        
        Args:
            image: Input image (path, PIL Image, or numpy array)
            confidence_threshold: Minimum confidence score for detections
            
        Returns:
            List of detected elements with detailed metadata
        """
        try:
            img_array = self.process_image(image)
            
            # Apply image preprocessing
            img_array = self._preprocess_image(img_array)
            
            # Run YOLO detection with NMS
            results = self.yolo_model(img_array)
            
            # Set image for SAM predictor
            self.sam_predictor.set_image(img_array)
            
            detected_elements = []
            
            # Track overlapping detections
            processed_regions = []
            
            for result in results[0].boxes.data:
                x1, y1, x2, y2, conf, class_id = result
                
                if conf < confidence_threshold:
                    continue
                    
                # Check for overlapping detections
                current_box = [x1, y1, x2, y2]
                if self._check_overlap(current_box, processed_regions):
                    continue
                    
                processed_regions.append(current_box)
                
                # Enhanced element analysis
                element_info = self._analyze_element(
                    img_array, 
                    current_box,
                    int(class_id),
                    float(conf)
                )
                
                if element_info:
                    detected_elements.append(element_info)
            
            # Post-process detections
            detected_elements = self._post_process_detections(detected_elements)
            
            return detected_elements
            
        except Exception as e:
            logging.error(f"Error in element detection: {e}", exc_info=True)
            return []

    def _preprocess_image(self, image):
        """Apply preprocessing to improve detection quality."""
        try:
            # Convert to RGB if needed
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                
            # Enhance contrast
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            lab = cv2.merge((l,a,b))
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            return image
        except Exception as e:
            logging.error(f"Error in image preprocessing: {e}")
            return image

    def _analyze_element(self, image, bbox, class_id, confidence):
        """Detailed analysis of detected UI element."""
        try:
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            
            # Extract element region
            element_region = image[y1:y2, x1:x2]
            
            # Generate SAM mask
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            masks, scores, _ = self.sam_predictor.predict(
                point_coords=np.array([[center_x, center_y]]),
                point_labels=np.array([1]),
                multimask_output=True
            )
            
            # Get best mask
            best_mask_idx = np.argmax(scores)
            
            # Extract text using OCR if applicable
            text = self._extract_text(element_region)
            
            return {
                'type': self.ui_classes[class_id],
                'coordinates': (center_x, center_y),
                'bbox': (x1, y1, x2, y2),
                'confidence': float(confidence),
                'mask': masks[best_mask_idx].tolist(),
                'mask_score': float(scores[best_mask_idx]),
                'text': text,
                'attributes': self._get_element_attributes(element_region),
                'state': self._determine_element_state(element_region)
            }
            
        except Exception as e:
            logging.error(f"Error analyzing element: {e}")
            return None

    def _check_overlap(self, box, processed_boxes, overlap_threshold=0.5):
        """Check if box overlaps significantly with any processed boxes."""
        def calculate_iou(box1, box2):
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            
            if x2 < x1 or y2 < y1:
                return 0.0
                
            intersection = (x2 - x1) * (y2 - y1)
            box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
            box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
            
            return intersection / float(box1_area + box2_area - intersection)
        
        return any(calculate_iou(box, proc_box) > overlap_threshold 
                  for proc_box in processed_boxes)

    def complete_task(self, input_text, image):
        """
        Process vision tasks with Qwen VL model.
        """
        try:
            if image is None:
                raise ValueError("Image input cannot be None")
                
            # First detect elements using ML models
            detected_elements = self.detect_elements(image)
            
            # Prepare image for Qwen
            if isinstance(image, str):
                image = Image.open(image)
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            elif not isinstance(image, Image.Image):
                raise ValueError(f"Unsupported image type: {type(image)}")
            
            # Create messages format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": self._create_vision_prompt(input_text, detected_elements)}
                    ]
                }
            ]
            
            # Process with Qwen
            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Prepare inputs
            inputs = self.processor(
                text=[text],
                images=[image],
                videos=None,
                padding=True,
                return_tensors="pt"
            )
            
            # Move inputs to same device as model
            inputs = inputs.to(self.model.device)
            
            # Generate response
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
            
            # Decode response
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            # Parse response for coordinates if needed
            if "find" in input_text.lower() or "locate" in input_text.lower():
                return self._parse_location_response(response, detected_elements)
                
            return response

        except Exception as e:
            logging.error(f"Error in vision task completion: {e}", exc_info=True)
            return None

    def _create_vision_prompt(self, input_text, detected_elements):
        """
        Creates a focused prompt for the vision model.
        """
        if "find" in input_text.lower() or "locate" in input_text.lower():
            return f"""Please analyze the image and:
1. Locate the specific element: {input_text}
2. Return its precise coordinates (x,y)
3. Describe its appearance and type
4. Provide confidence score (0-1)

Format: "coordinates: (x,y), type: element_type, confidence: score, description: brief_desc"
"""
        else:
            return f"""Please analyze the image and answer: {input_text}
Focus on:
- Visible UI elements and their layout
- Text content and labels
- Interactive elements (buttons, links, etc.)
- Visual hierarchy and relationships

Provide a clear, concise response."""

    def _parse_location_response(self, response, detected_elements):
        """
        Parses location information from model response.
        """
        try:
            # Extract coordinates using regex or string parsing
            import re
            coords_match = re.search(r'coordinates: \((\d+),(\d+)\)', response)
            if coords_match:
                x, y = map(int, coords_match.groups())
                
                # Match with detected elements for additional info
                matched_element = None
                for element in detected_elements:
                    element_x, element_y = element['coordinates']
                    if abs(x - element_x) < 20 and abs(y - element_y) < 20:
                        matched_element = element
                        break
                
                return {
                    'coordinates': (x, y),
                    'type': matched_element['type'] if matched_element else 'unknown',
                    'confidence': matched_element['confidence'] if matched_element else 0.5,
                    'description': response
                }
                
            return None
            
        except Exception as e:
            logging.error(f"Error parsing location response: {e}")
            return None

    def encode_image(self, image_path):
        """Encode image to base64 string."""
        if isinstance(image_path, str):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        elif isinstance(image_path, Image.Image):
            import io
            buffer = io.BytesIO()
            image_path.save(buffer, format="JPEG")
            return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def perceive_scene(self, image, context=None):
        """
        High-level scene understanding using QwenVL.
        Used for general perception and thinking.
        """
        try:
            # Create focused prompt based on context
            base_prompt = "Analyze this screen and describe:"
            if context and context.get('thought_focus'):
                base_prompt += f"\nFocusing on: {context['thought_focus']}"
            if context and context.get('current_goal'):
                base_prompt += f"\nIn relation to: {context['current_goal']}"
                
            messages = [
                {
                    "role": "user", 
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": base_prompt}
                    ]
                }
            ]

            # Process with Qwen
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = self.processor(
                text=[text],
                images=[image],
                return_tensors="pt"
            ).to(self.model.device)
            
            # Generate perception
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7
            )
            
            perception = self.processor.batch_decode(
                output_ids[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )[0]
            
            # Store in scene memory
            self.scene_memory.append({
                'perception': perception,
                'timestamp': time.time(),
                'context': context
            })
            
            return perception

        except Exception as e:
            logging.error(f"Error in scene perception: {e}")
            return None

    def prepare_for_action(self, image, target_description):
        """
        Precise element detection using YOLO+SAM.
        Used specifically when Bob needs to take actions.
        """
        try:
            img_array = self.process_image(image)
            
            # Run YOLO detection
            results = self.yolo_model(img_array)
            
            # Set image for SAM
            self.sam_predictor.set_image(img_array)
            
            detected_elements = []
            for result in results[0].boxes.data:
                x1, y1, x2, y2, conf, class_id = result
                
                # Get precise mask using SAM
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                masks, scores, _ = self.sam_predictor.predict(
                    point_coords=np.array([[center_x, center_y]]),
                    point_labels=np.array([1])
                )
                
                element_info = {
                    'coordinates': (center_x, center_y),
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'mask': masks[0].tolist(),
                    'confidence': float(conf),
                    'type': self.ui_classes[int(class_id)]
                }
                
                detected_elements.append(element_info)
            
            # Store in action memory
            self.action_memory.append({
                'elements': detected_elements,
                'target': target_description,
                'timestamp': time.time()
            })
            
            return detected_elements
            
        except Exception as e:
            logging.error(f"Error preparing for action: {e}")
            return None


