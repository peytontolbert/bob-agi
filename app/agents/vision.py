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

class VisionAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize YOLO model for object detection
        self.yolo_model = YOLO('yolov8x.pt')
        
        # Initialize SAM for precise segmentation
        self.sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
        self.sam.to("cuda" if torch.cuda.is_available() else "cpu")
        self.sam_predictor = SamPredictor(self.sam)
        
        # Define common UI element classes
        self.ui_classes = [
            'button', 'text', 'textbox', 'checkbox', 'dropdown',
            'icon', 'image', 'link', 'menu', 'slider'
        ]

    def process_image(self, image):
        """Convert various image inputs to numpy array."""
        if isinstance(image, str):
            # If image path
            return cv2.imread(image)
        elif isinstance(image, Image.Image):
            # If PIL Image
            return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        elif isinstance(image, np.ndarray):
            return image
        else:
            raise ValueError("Unsupported image format")

    def detect_elements(self, image, confidence_threshold=0.3):
        """
        Detect UI elements using YOLO and refine with SAM.
        
        Args:
            image: Input image (path, PIL Image, or numpy array)
            confidence_threshold: Minimum confidence score for detections
            
        Returns:
            List of detected elements with coordinates and metadata
        """
        try:
            # Process image to numpy array
            img_array = self.process_image(image)
            
            # Run YOLO detection
            results = self.yolo_model(img_array)
            
            # Set image for SAM predictor
            self.sam_predictor.set_image(img_array)
            
            detected_elements = []
            
            for result in results[0].boxes.data:
                x1, y1, x2, y2, conf, class_id = result
                
                if conf < confidence_threshold:
                    continue
                
                # Get center coordinates
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                # Generate SAM mask for precise boundaries
                input_point = np.array([[center_x, center_y]])
                input_label = np.array([1])
                
                masks, scores, _ = self.sam_predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=False
                )
                
                # Get class name
                class_name = self.ui_classes[int(class_id)]
                
                element_info = {
                    'type': class_name,
                    'coordinates': (int(center_x), int(center_y)),
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': float(conf),
                    'mask': masks[0].tolist() if len(masks) > 0 else None,
                    'mask_score': float(scores[0]) if len(scores) > 0 else None
                }
                
                detected_elements.append(element_info)
            
            return detected_elements
            
        except Exception as e:
            logging.error(f"Error in element detection: {e}")
            return []

    def complete_task(self, input_text, image):
        """
        Process vision tasks with both ML models and GPT-4 Vision.
        """
        try:
            # First detect elements using ML models
            detected_elements = self.detect_elements(image)
            
            # For tasks requiring element locations, return ML results directly
            if "find" in input_text.lower() or "locate" in input_text.lower():
                return detected_elements
            
            # For general vision tasks, use GPT-4 Vision
            base64_string = self.encode_image(image)
            response = self.client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": input_text},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_string}",
                                },
                            },
                        ],
                    }
                ],
                max_tokens=5000,
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logging.error(f"Error in vision task completion: {e}")
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


