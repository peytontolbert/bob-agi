"""
Pipeline for fine-tuning OmniParser model on Discord browser interface,
specifically focused on improving "Continue in Browser" link detection
"""
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import logging
import os
from datetime import datetime
from PIL import Image
import time
from app.env.computer.computer import Computer
from app.train.fine_tune import LiveFineTuner
from app.agents.vision import VisionAgent

@dataclass
class DiscordElement:
    """Represents a Discord UI element for training"""
    element_type: str
    bbox: tuple
    text: Optional[str] = None
    description: Optional[str] = None
    confidence: Optional[float] = None

class DiscordFineTunePipeline:
    def __init__(self, 
                 output_dir: str = "debug_output/discord_finetune",
                 confidence_threshold: float = 0.7,
                 target_element: str = "continue_browser_link"):
        """
        Initialize Discord fine-tuning pipeline
        
        Args:
            output_dir: Directory for saving debug output
            confidence_threshold: Confidence threshold for detections
            target_element: Target element type to focus training on
        """
        self.output_dir = output_dir
        self.confidence_threshold = confidence_threshold
        self.target_element = target_element
        self.initialize_computer()
        self.vision_agent = VisionAgent()
        self.screen = self.computer.screen

        # Create all required directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "positive_samples"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "negative_samples"), exist_ok=True)
        os.makedirs("fine_tune_data/discord", exist_ok=True)
        
        # Initialize components
        self.omniparser = self.vision_agent.omniparser
        self.fine_tuner = LiveFineTuner(
            fine_tune_dir="fine_tune_data/discord",
            min_samples_for_tuning=3  # Lower threshold for focused fine-tuning
        )
        
        # Setup logging
        self._setup_logging()

        # Initialize training stats
        self.training_stats = {
            'target_detections': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'total_samples': 0
        }
    
    def initialize_computer(self):
        """Initialize computer and ensure screen access"""
        self.computer = Computer()
        self.computer.run()

    async def capture_discord_screen(self) -> Optional[Image.Image]:
        """Capture Discord login screen"""
        try:
            # Ensure computer is initialized
            if not hasattr(self, 'computer') or not self.computer:
                self.logger.error("Computer not initialized")
                return None
                
            if not hasattr(self, 'screen') or not self.screen:
                self.logger.error("Screen not initialized")
                return None

            # Capture screen with retry
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    screen_image = self.screen.capture()
                    if screen_image is None:
                        raise ValueError("Screen capture returned None")
                    
                    # Validate image
                    if not isinstance(screen_image, Image.Image):
                        raise ValueError(f"Invalid image type: {type(screen_image)}")
                    
                    # Save raw capture for debugging
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    debug_path = os.path.join(self.output_dir, f"raw_capture_{timestamp}.png")
                    screen_image.save(debug_path)
                    
                    self.logger.info(f"Screen captured successfully: {debug_path}")
                    return screen_image
                    
                except Exception as e:
                    self.logger.warning(f"Screen capture attempt {attempt + 1} failed: {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1)  # Wait before retry
                    else:
                        raise
                        
        except Exception as e:
            self.logger.error(f"Error capturing screen: {e}")
            return None

    async def collect_discord_samples(self, screen_image: Image.Image) -> Dict[str, Any]:
        """
        Collect training samples from Discord interface
        
        Args:
            screen_image: Screenshot of Discord interface
            
        Returns:
            Dict containing collection results
        """
        try:
            # Run detection
            detection_results = await self.omniparser.detect_elements(screen_image)
            
            if detection_results['status'] != 'success':
                self.logger.error("Detection failed")
                return {'status': 'error', 'message': 'Detection failed'}
            
            # Process detections
            target_found = False
            for det in detection_results['detections']:
                if det['type'] == 'link' and 'continue' in det.get('text', '').lower():
                    target_found = True
                    self.training_stats['target_detections'] += 1
                    
                    # Save positive sample
                    self._save_training_sample(screen_image, det, is_positive=True)
                    break
            
            if not target_found:
                self.training_stats['false_negatives'] += 1
                # Save negative sample for analysis
                self._save_training_sample(screen_image, None, is_positive=False)
            
            # Save sample for fine-tuning
            collected = self.fine_tuner.collect_sample(
                screen_image,
                detection_results,
                focus_element=self.target_element
            )
            
            self.training_stats['total_samples'] += 1
            
            return {
                'status': 'success',
                'target_found': target_found,
                'timestamp': datetime.now().isoformat(),
                'detections': detection_results['detections']
            }
            
        except Exception as e:
            self.logger.error(f"Error collecting Discord sample: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }

    def _save_training_sample(self, image: Image.Image, detection: Optional[Dict], is_positive: bool):
        """Save training sample with annotations"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        subdir = "positive_samples" if is_positive else "negative_samples"
        
        # Save image
        image_path = os.path.join(self.output_dir, subdir, f"sample_{timestamp}.png")
        image.save(image_path)
        
        # Save annotation if available
        if detection:
            annotation_path = os.path.join(self.output_dir, subdir, f"sample_{timestamp}.json")
            import json
            with open(annotation_path, 'w') as f:
                json.dump(detection, f, indent=2)

    async def fine_tune_on_discord(self, num_samples: int = 5) -> Dict[str, Any]:
        """
        Run focused fine-tuning process
        
        Args:
            num_samples: Number of samples to collect before fine-tuning
            
        Returns:
            Dict containing fine-tuning results
        """
        try:
            self.logger.info(f"Starting focused Discord fine-tuning with {num_samples} samples")
            
            # Check if we have enough samples
            if self.training_stats['total_samples'] < num_samples:
                return {
                    'status': 'pending',
                    'message': f"Need {num_samples - self.training_stats['total_samples']} more samples"
                }
            
            # Calculate current performance
            detection_rate = self.training_stats['target_detections'] / self.training_stats['total_samples']
            
            # Run fine-tuning with focus on target element
            fine_tune_success = await self.fine_tuner.fine_tune(
                focus_classes=['link'],
                class_weights={'link': 2.0}  # Increase weight for link class
            )
            
            if not fine_tune_success:
                return {
                    'status': 'error',
                    'message': 'Fine-tuning failed'
                }
            
            return {
                'status': 'success',
                'message': 'Fine-tuning completed successfully',
                'detection_rate': detection_rate,
                'model_version': self.fine_tuner.training_stats['fine_tune_rounds']
            }
            
        except Exception as e:
            self.logger.error(f"Error in Discord fine-tuning: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }

    def evaluate_discord_detection(self, 
                                 screen_image: Image.Image,
                                 ground_truth: List[DiscordElement]) -> Dict[str, Any]:
        """
        Evaluate detection performance specifically for target element
        
        Args:
            screen_image: Screenshot of Discord interface
            ground_truth: List of ground truth Discord elements
            
        Returns:
            Dict containing evaluation results
        """
        try:
            # Convert ground truth to format for evaluation
            gt_annotations = [
                {
                    'bbox': element.bbox,
                    'type': element.element_type,
                    'text': element.text,
                    'confidence': element.confidence
                }
                for element in ground_truth
            ]
            
            # Run evaluation with focus on target element
            score = self.fine_tuner.evaluate_performance(
                screen_image,
                gt_annotations,
                focus_element=self.target_element
            )
            
            return {
                'status': 'success',
                'score': score,
                'detection_rate': self.training_stats['target_detections'] / max(1, self.training_stats['total_samples']),
                'false_negative_rate': self.training_stats['false_negatives'] / max(1, self.training_stats['total_samples']),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error evaluating Discord detection: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }

    def _setup_logging(self):
        """Configure logging for the pipeline"""
        log_file = os.path.join(self.output_dir, "discord_finetune.log")
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Get logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Initial log message
        self.logger.info("Discord fine-tuning pipeline initialized")

async def main():
    """Main execution function"""
    try:
        # Initialize pipeline
        pipeline = DiscordFineTunePipeline()
        logging.info("Pipeline initialized")

        # Capture initial screen
        screen_image = await pipeline.capture_discord_screen()
        if screen_image is None:
            logging.error("Failed to capture screen")
            return

        # Collect samples
        collection_result = await pipeline.collect_discord_samples(screen_image)
        if collection_result['status'] != 'success':
            logging.error(f"Sample collection failed: {collection_result['message']}")
            return

        # Run fine-tuning if enough samples
        fine_tune_result = await pipeline.fine_tune_on_discord(num_samples=5)
        logging.info(f"Fine-tuning result: {fine_tune_result}")

    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}")

if __name__ == "__main__":
    import asyncio
    import argparse

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Discord Fine-tuning Pipeline')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()

    # Configure logging based on verbosity
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run the async main function
    asyncio.run(main())
