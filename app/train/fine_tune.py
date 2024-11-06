"""
Live fine-tuning script for OmniParser models to improve GUI element detection
"""
import logging
import os
import time
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from ultralytics import YOLO
import supervision
from utils import get_yolo_model

class LiveFineTuner:
    def __init__(self, 
                 base_model_path: str = "OmniParser/icon_detect/best.pt",
                 fine_tune_dir: str = "fine_tune_data",
                 confidence_threshold: float = 0.3,
                 min_samples_for_tuning: int = 10,
                 max_samples_per_class: int = 100):
        """
        Initialize the live fine-tuning system.
        
        Args:
            base_model_path: Path to base YOLO model
            fine_tune_dir: Directory for storing fine-tuning data
            confidence_threshold: Minimum confidence for good detections
            min_samples_for_tuning: Minimum samples needed before fine-tuning
            max_samples_per_class: Maximum samples to keep per class
        """
        self.base_model_path = base_model_path
        self.fine_tune_dir = Path(fine_tune_dir)
        self.confidence_threshold = confidence_threshold
        self.min_samples_for_tuning = min_samples_for_tuning
        self.max_samples_per_class = max_samples_per_class
        
        # Create directories
        self.fine_tune_dir.mkdir(exist_ok=True)
        (self.fine_tune_dir / "images").mkdir(exist_ok=True)
        (self.fine_tune_dir / "labels").mkdir(exist_ok=True)
        
        # Initialize model and training stats
        self.model = self._load_model()
        self.training_stats = self._load_training_stats()
        self.samples_since_tuning = 0
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Configure logging for fine-tuning process"""
        log_file = self.fine_tune_dir / "fine_tune.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _load_model(self) -> YOLO:
        """Load the YOLO model with error handling"""
        try:
            model = get_yolo_model(self.base_model_path)
            self.logger.info(f"Loaded base model from {self.base_model_path}")
            return model
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise

    def _load_training_stats(self) -> Dict:
        """Load or initialize training statistics"""
        stats_file = self.fine_tune_dir / "training_stats.json"
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                return json.load(f)
        return {
            'samples_collected': 0,
            'fine_tune_rounds': 0,
            'class_counts': {},
            'performance_history': [],
            'last_fine_tune': None
        }

    def _save_training_stats(self):
        """Save current training statistics"""
        stats_file = self.fine_tune_dir / "training_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(self.training_stats, f, indent=2)

    def collect_sample(self, 
                      image: Image.Image,
                      detection_results: Dict,
                      correct_annotations: Optional[List[Dict]] = None) -> bool:
        """
        Collect a training sample with its annotations.
        
        Args:
            image: Input image
            detection_results: Original detection results
            correct_annotations: Optional manual annotations for correction
            
        Returns:
            bool: Whether sample was collected successfully
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = self.fine_tune_dir / "images" / f"sample_{timestamp}.jpg"
            label_path = self.fine_tune_dir / "labels" / f"sample_{timestamp}.txt"
            
            # Save image
            image.save(image_path)
            
            # Process annotations
            if correct_annotations:
                annotations = correct_annotations
            else:
                # Filter confident detections
                annotations = [
                    det for det in detection_results['detections']
                    if det['confidence'] >= self.confidence_threshold
                ]
            
            # Convert to YOLO format and save labels
            with open(label_path, 'w') as f:
                for ann in annotations:
                    # Convert bbox to YOLO format (normalized)
                    x1, y1, x2, y2 = ann['bbox']
                    width = image.width
                    height = image.height
                    
                    # Calculate normalized center coordinates and dimensions
                    x_center = ((x1 + x2) / 2) / width
                    y_center = ((y1 + y2) / 2) / height
                    w = (x2 - x1) / width
                    h = (y2 - y1) / height
                    
                    # Get class index
                    class_idx = self._get_class_index(ann['type'])
                    
                    # Write YOLO format line
                    f.write(f"{class_idx} {x_center} {y_center} {w} {h}\n")
                    
                    # Update class counts
                    class_name = ann['type']
                    self.training_stats['class_counts'][class_name] = \
                        self.training_stats['class_counts'].get(class_name, 0) + 1
            
            self.training_stats['samples_collected'] += 1
            self.samples_since_tuning += 1
            self._save_training_stats()
            
            self.logger.info(f"Collected sample: {image_path.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error collecting sample: {e}")
            return False

    def _get_class_index(self, class_name: str) -> int:
        """Get class index from name, adding new classes if needed"""
        class_file = self.fine_tune_dir / "classes.txt"
        
        # Load existing classes
        if class_file.exists():
            with open(class_file, 'r') as f:
                classes = f.read().splitlines()
        else:
            classes = []
            
        # Add new class if needed
        if class_name not in classes:
            classes.append(class_name)
            with open(class_file, 'w') as f:
                f.write('\n'.join(classes))
                
        return classes.index(class_name)

    def should_fine_tune(self) -> bool:
        """Determine if fine-tuning should be triggered"""
        if self.samples_since_tuning < self.min_samples_for_tuning:
            return False
            
        # Check if we have enough samples per class
        class_counts = self.training_stats['class_counts']
        if not all(count >= 5 for count in class_counts.values()):
            return False
            
        # Check if performance has degraded
        if self.training_stats['performance_history']:
            recent_performance = np.mean(self.training_stats['performance_history'][-5:])
            if recent_performance < 0.7:  # Threshold for retraining
                return True
                
        return self.samples_since_tuning >= self.min_samples_for_tuning

    async def fine_tune(self, focus_classes=None, class_weights=None):
        """
        Run fine-tuning with optional focus on specific classes
        
        Args:
            focus_classes: List of classes to focus training on
            class_weights: Dict of class weights for training
        """
        try:
            # Configure training parameters
            train_args = {
                'data': self._prepare_data_yaml(),
                'epochs': 10,
                'imgsz': 640,
                'batch': 16,
                'patience': 5
            }
            
            # Add class weights if provided
            if class_weights:
                train_args['class_weights'] = class_weights
                
            # Filter training data if focus classes specified
            if focus_classes:
                self._filter_training_data(focus_classes)
            
            # Start training
            results = await self.model.train(**train_args)
            
            return True
            
        except Exception as e:
            logging.error(f"Error in fine-tuning: {e}")
            return False

    def _filter_training_data(self, focus_classes):
        """Filter training data to focus on specific classes"""
        try:
            filtered_samples = []
            for sample in self.training_samples:
                # Keep samples that contain focus classes
                if any(det['type'] in focus_classes for det in sample['detections']):
                    filtered_samples.append(sample)
            
            self.training_samples = filtered_samples
            
        except Exception as e:
            logging.error(f"Error filtering training data: {e}")

    def _prepare_data_yaml(self) -> str:
        """Prepare YAML configuration for training"""
        yaml_path = self.fine_tune_dir / "data.yaml"
        
        # Load class names
        class_file = self.fine_tune_dir / "classes.txt"
        with open(class_file, 'r') as f:
            classes = f.read().splitlines()
            
        # Create YAML content
        yaml_content = {
            'path': str(self.fine_tune_dir),
            'train': 'images',
            'val': 'images',  # Using same data for validation
            'names': {i: name for i, name in enumerate(classes)}
        }
        
        # Save YAML file
        with open(yaml_path, 'w') as f:
            import yaml
            yaml.dump(yaml_content, f)
            
        return str(yaml_path)

    def evaluate_performance(self, 
                           image: Image.Image,
                           ground_truth: List[Dict]) -> float:
        """
        Evaluate model performance on a sample.
        
        Args:
            image: Input image
            ground_truth: List of ground truth annotations
            
        Returns:
            float: Performance score (0-1)
        """
        try:
            # Run detection
            results = self.model(image)[0]
            
            # Convert detections to same format as ground truth
            detections = []
            for i in range(len(results.boxes)):
                box = results.boxes[i]
                detections.append({
                    'bbox': box.xyxy[0].tolist(),
                    'type': results.names[int(box.cls[0])],
                    'confidence': float(box.conf[0])
                })
            
            # Calculate metrics
            matches = 0
            for gt in ground_truth:
                for det in detections:
                    if self._calculate_iou(gt['bbox'], det['bbox']) > 0.5 and \
                       gt['type'] == det['type']:
                        matches += 1
                        break
                        
            # Calculate performance score
            score = matches / len(ground_truth) if ground_truth else 0
            
            # Update performance history
            self.training_stats['performance_history'].append(score)
            self._save_training_stats()
            
            return score
            
        except Exception as e:
            self.logger.error(f"Error evaluating performance: {e}")
            return 0.0

    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
            
        intersection = (x2 - x1) * (y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        return intersection / (box1_area + box2_area - intersection)

