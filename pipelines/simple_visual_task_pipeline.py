"""
Simple visual task pipeline for testing Discord's "Continue in Browser" link click.
"""
import logging
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import cv2
import numpy as np

@dataclass
class PipelineResult:
    """Represents the result of the pipeline execution"""
    success: bool
    error_message: Optional[str] = None
    execution_time: float = 0.0
    debug_image_path: Optional[str] = None

class SimpleVisualTaskPipeline:
    def __init__(self, vision_agent, eyes, hands):
        self.vision_agent = vision_agent
        self.eyes = eyes
        self.hands = hands
        
        # Setup debug output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.debug_dir = Path(f"debug_output/continue_browser_test_{timestamp}")
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging for the pipeline"""
        log_path = Path("logs/simple_pipeline")
        log_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_path / f"continue_browser_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    def _save_debug_image(self, image, bbox: tuple, filename: str, original_bbox: tuple = None) -> str:
        """
        Save debug image with detection visualization, handling both processed and original coordinates.
        
        Args:
            image: The screen image (PIL Image or numpy array)
            bbox: Bounding box coordinates (x1, y1, x2, y2) for display
            filename: Name for the debug image file
            original_bbox: Original bounding box from InternVL2 (if different from display bbox)
        
        Returns:
            Path to the saved debug image
        """
        # Convert PIL Image to numpy array if needed
        if hasattr(image, 'convert'):  # Check if it's a PIL Image
            image = np.array(image.convert('RGB'))
        
        debug_image = image.copy()
        x1, y1, x2, y2 = bbox
        
        # Draw main bounding box in green
        cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # If we have original InternVL2 coordinates, draw them in blue
        if original_bbox:
            ox1, oy1, ox2, oy2 = original_bbox
            cv2.rectangle(debug_image, (ox1, oy1), (ox2, oy2), (255, 0, 0), 1)
            cv2.putText(debug_image, "InternVL2 Box", (ox1, oy1-25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Draw center point in red
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        cv2.circle(debug_image, (center_x, center_y), 5, (0, 0, 255), -1)
        
        # Add text labels
        cv2.putText(debug_image, "Display Box", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Save the image
        debug_path = self.debug_dir / filename
        cv2.imwrite(str(debug_path), debug_image)
        return str(debug_path)

    def execute(self) -> PipelineResult:
        """
        Execute the simple visual task pipeline to click "Continue in Browser".
        
        Returns:
            PipelineResult containing success status and execution details
        """
        start_time = time.time()
        
        try:
            # Get initial screen state
            screen_image = self.eyes.get_screen_image()
            if screen_image is None:
                return PipelineResult(
                    success=False,
                    error_message="Failed to capture screen image",
                    execution_time=time.time() - start_time
                )

            # Save initial screen image
            cv2.imwrite(str(self.debug_dir / "01_initial_screen.png"), screen_image)

            # Search for the Continue in Browser link with retries
            button_query = "'Continue in Browser'"
            max_retries = 3
            detection_result = None

            logging.info("Searching for Continue in Browser link...")
            
            for attempt in range(max_retries):
                detection_result = self.vision_agent.detect_ui_elements(
                    screen_image,
                    query=button_query
                )
                
                # Save raw vision agent response for debugging
                if detection_result:
                    with open(str(self.debug_dir / f"vision_response_{attempt}.txt"), 'w') as f:
                        f.write(str(detection_result))
                
                    # Save processed image if available
                    if 'processed_image' in detection_result:
                        processed_image = detection_result['processed_image']
                        if processed_image is not None:
                            processed_image_path = str(self.debug_dir / f"processed_image_{attempt}.png")
                            if hasattr(processed_image, 'convert'):  # PIL Image
                                processed_image = np.array(processed_image.convert('RGB'))
                            cv2.imwrite(processed_image_path, processed_image)
                
                if detection_result.get('status') == 'success' and detection_result.get('detections'):
                    break
                
                logging.warning(f"Attempt {attempt + 1}: No link detected, retrying...")
                time.sleep(0.5)

            if not detection_result or detection_result.get('status') != 'success':
                return PipelineResult(
                    success=False,
                    error_message="Failed to detect Continue in Browser link",
                    execution_time=time.time() - start_time
                )

            # Get button coordinates
            if not detection_result.get('detections'):
                return PipelineResult(
                    success=False,
                    error_message="No link detected after retries",
                    execution_time=time.time() - start_time
                )

            button = detection_result['detections'][0]
            bbox = button.get('bbox')
            original_bbox = button.get('original_bbox')  # Get original InternVL2 coordinates if available
            
            if not bbox:
                return PipelineResult(
                    success=False,
                    error_message="No bounding box found for link",
                    execution_time=time.time() - start_time
                )

            # Save detection visualization with both coordinate systems using processed image
            if 'processed_image' in detection_result and detection_result['processed_image'] is not None:
                debug_image_path = self._save_debug_image(
                    detection_result['processed_image'], 
                    bbox,
                    "02_detection_result.png",
                    original_bbox=original_bbox
                )
            else:
                debug_image_path = self._save_debug_image(
                    screen_image, 
                    bbox,
                    "02_detection_result.png",
                    original_bbox=original_bbox
                )

            # Calculate center of bounding box for more accurate clicking
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Create element dict with more precise information
            element = {
                'coordinates': (center_x, center_y),
                'bbox': bbox,
                'type': 'link',
                'width': x2 - x1,
                'height': y2 - y1
            }

            # Click the button with improved accuracy
            logging.info(f"Attempting to click link at center coordinates ({center_x}, {center_y})")
            logging.info(f"Link bounding box: {bbox}")
            click_success = self.hands.click_element(element)

            if not click_success:
                return PipelineResult(
                    success=False,
                    error_message="Failed to click link",
                    execution_time=time.time() - start_time
                )

            # Verify the click worked by checking for page change
            time.sleep(1.0)  # Wait for page transition
            
            # Get new screen state
            new_screen = self.eyes.get_screen_image()
            
            # Verify we're on a new page
            verification_result = self.vision_agent.detect_ui_elements(
                new_screen,
                query="Look for Discord login page or login form elements"
            )

            success = verification_result.get('status') == 'success' and \
                     len(verification_result.get('detections', [])) > 0

            # Save verification screen
            if new_screen is not None:
                cv2.imwrite(str(self.debug_dir / "03_verification_screen.png"), new_screen)

            return PipelineResult(
                success=success,
                error_message=None if success else "Failed to verify page transition",
                execution_time=time.time() - start_time,
                debug_image_path=debug_image_path  # Add debug image path to result
            )

        except Exception as e:
            logging.error(f"Pipeline execution failed: {e}", exc_info=True)
            return PipelineResult(
                success=False,
                error_message=str(e),
                execution_time=time.time() - start_time
            )

if __name__ == "__main__":
    from app.agents.vision import VisionAgent
    from app.env.screen import Screen
    from app.env.computer import Computer
    from app.actions.eyesight import Eyesight
    from app.actions.hands import Hands
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Initialize components
        computer = Computer()
        computer.run()
        eyes = Eyesight(computer.screen)
        hands = Hands(computer.mouse, computer.keyboard)
        vision_agent = VisionAgent()
        
        # Create and execute pipeline
        pipeline = SimpleVisualTaskPipeline(vision_agent, eyes, hands)
        
        logging.info("Starting Continue in Browser test...")
        result = pipeline.execute()
        
        # Log results
        logging.info("\nTest Results:")
        logging.info(f"Success: {result.success}")
        logging.info(f"Execution time: {result.execution_time:.2f}s")
        
        if result.error_message:
            logging.error(f"Error: {result.error_message}")
            
    except Exception as e:
        logging.error(f"Test execution failed: {e}", exc_info=True)
    finally:
        logging.info("Test execution completed")
