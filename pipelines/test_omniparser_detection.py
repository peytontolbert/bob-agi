"""
Test pipeline for OmniParser detection of Discord UI elements using Computer and Eyesight
"""
import logging
from pathlib import Path
import time
import sys
from typing import List, Dict, Optional
import traceback
from PIL import Image
import numpy as np

from app.env.computer.computer import Computer
from app.env.senses.eyesight import Eyesight
from app.agents.vision import VisionAgent
from app.models.omniparser import OmniParser

class OmniParserTest:
    def __init__(self):
        self.parser = None
        self.computer = None
        self.eyesight = None
        self.vision_agent = VisionAgent()
        self.debug_dir = Path("debug_output")
        self.setup_logging()
        self.initialize_components()

    def setup_logging(self):
        """Configure logging for test pipeline"""
        self.debug_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.debug_dir / "test.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def initialize_components(self):
        """Initialize all required components with error handling"""
        try:
            # Initialize OmniParser
            self.logger.info("Initializing OmniParser...")
            self.parser = OmniParser()
            if not self.parser:
                raise RuntimeError("Failed to initialize OmniParser")

            # Initialize computer environment
            self.logger.info("Initializing Computer environment...")
            self.computer = Computer()
            self.computer.run()
            # Initialize eyesight with computer's screen
            self.logger.info("Initializing Eyesight...")
            if not self.computer.screen:
                raise RuntimeError("Computer screen not initialized")
            self.eyesight = Eyesight(self.computer.screen, vision_agent=self.vision_agent)
            if not self.eyesight:
                raise RuntimeError("Failed to initialize Eyesight")

            self.logger.info("All components initialized successfully")

        except Exception as e:
            self.logger.error(f"Initialization error: {e}")
            self.logger.error(traceback.format_exc())
            raise

    def validate_screen_image(self, image) -> bool:
        """Validate screen capture image"""
        if image is None:
            return False
            
        if isinstance(image, np.ndarray):
            if image.size == 0 or not np.any(image):
                return False
        elif isinstance(image, Image.Image):
            if image.size[0] == 0 or image.size[1] == 0:
                return False
        else:
            return False
            
        return True

    def detect_ui_elements(self, 
                         query: Optional[str] = None, 
                         confidence_threshold: float = 0.3,
                         max_retries: int = 3) -> List[Dict]:
        """
        Detect UI elements in current screen using OmniParser with retries
        
        Args:
            query: Optional text query to filter elements
            confidence_threshold: Minimum confidence score for detections
            max_retries: Maximum number of retry attempts
            
        Returns:
            List of detected elements above threshold
        """
        for attempt in range(max_retries):
            try:
                # Get current screen image
                screen_image = self.computer.screen.get_current_frame()
                if not self.validate_screen_image(screen_image):
                    raise ValueError("Invalid screen capture")

                # Run detection
                results = self.parser.detect_elements(screen_image, query)
                
                if results['status'] != 'success':
                    raise RuntimeError(f"Detection failed: {results.get('message')}")
                    
                # Filter by confidence
                detections = [
                    det for det in results['detections']
                    if det['confidence'] >= confidence_threshold
                ]
                
                # Save debug visualization if available
                if results.get('labeled_image'):
                    try:
                        debug_path = self.debug_dir / f"detection_{int(time.time())}.png"
                        # Write bytes directly to file
                        with open(debug_path, 'wb') as f:
                            f.write(results['labeled_image'])
                        self.logger.info(f"Saved detection visualization to {debug_path}")
                    except Exception as e:
                        self.logger.warning(f"Failed to save debug image: {e}")
                
                return detections

            except Exception as e:
                self.logger.warning(f"Detection attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    self.logger.error("All detection attempts failed")
                    self.logger.error(traceback.format_exc())
                    return []
                time.sleep(1)  # Wait before retry

    def click_element(self, element: Dict) -> bool:
        """
        Click detected UI element using computer mouse and save visual click location
        
        Args:
            element: Dictionary containing element detection info
            
        Returns:
            bool: Whether click was successful
        """
        try:
            if not element or not isinstance(element, dict):
                self.logger.error("Invalid element data")
                return False

            # Check for both coordinate formats
            coords = element.get('coordinates')
            bbox = element.get('bbox')
            
            if coords and len(coords) == 2:
                center_x, center_y = coords
            elif bbox and len(bbox) == 4:
                # Calculate center from bbox
                x1, y1, x2, y2 = bbox
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
            else:
                self.logger.error("No valid coordinates found in element detection")
                return False
                
            if not isinstance(center_x, (int, float)) or not isinstance(center_y, (int, float)):
                self.logger.error("Invalid coordinate types")
                return False

            self.logger.info(f"Clicking at ({center_x}, {center_y})")
            
            # Get current frame to check bounds
            frame = self.computer.screen.get_current_frame()
            if frame is None:
                self.logger.error("Could not get current frame")
                return False
                
            # Get screen size from PIL Image
            screen_size = frame.size  # PIL Image size is (width, height)

            # Validate coordinates are within screen bounds
            if not (0 <= center_x <= screen_size[0] and 0 <= center_y <= screen_size[1]):
                self.logger.error(f"Coordinates ({center_x}, {center_y}) outside screen bounds {screen_size}")
                return False

            # Convert coordinates to integers
            center_x = int(center_x)
            center_y = int(center_y)

            # Save visual representation of click location
            try:
                from PIL import ImageDraw
                click_viz = frame.copy()
                draw = ImageDraw.Draw(click_viz)
                
                # Draw a red circle at click location
                radius = 10
                draw.ellipse(
                    [(center_x - radius, center_y - radius), 
                     (center_x + radius, center_y + radius)], 
                    outline='red', 
                    width=2
                )
                
                # Draw crosshair
                line_length = 20
                draw.line(
                    [(center_x - line_length, center_y), 
                     (center_x + line_length, center_y)], 
                    fill='red', 
                    width=2
                )
                draw.line(
                    [(center_x, center_y - line_length), 
                     (center_x, center_y + line_length)], 
                    fill='red', 
                    width=2
                )
                
                # Add coordinates text
                draw.text(
                    (center_x + radius + 5, center_y - radius - 5),
                    f"({center_x}, {center_y})",
                    fill='red'
                )
                
                # Save the visualization
                timestamp = int(time.time())
                viz_path = self.debug_dir / f"click_location_{timestamp}.png"
                click_viz.save(viz_path)
                self.logger.info(f"Saved click visualization to {viz_path}")
                
            except Exception as e:
                self.logger.warning(f"Failed to save click visualization: {e}")

            self.computer.mouse.move_to(center_x, center_y)
            time.sleep(0.1)  # Small delay to ensure move completes
            self.computer.mouse.click()
            return True
            
        except Exception as e:
            self.logger.error(f"Error clicking element: {e}")
            self.logger.error(traceback.format_exc())
            return False

    def cleanup(self):
        """Cleanup test resources"""
        try:
            if self.computer:
                self.computer.shutdown()
            self.logger.info("Cleanup completed successfully")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            self.logger.error(traceback.format_exc())

def main():
    test = None
    try:
        test = OmniParserTest()
        test.logger.info("Starting Discord UI element detection test...")
        
        # Wait for Discord to load
        time.sleep(2)
        
        # Try detecting and clicking "Continue in browser" button
        thresholds = [0.3, 0.2, 0.1]  # Try progressively lower thresholds
        success = False
        
        for threshold in thresholds:
            test.logger.info(f"Attempting detection with threshold {threshold}")
            
            # Detect UI elements
            elements = test.detect_ui_elements(
                query="Continue in browser button",
                confidence_threshold=threshold
            )
            
            # Try to click highest confidence match
            if elements:
                best_match = max(elements, key=lambda x: x['confidence'])
                if test.click_element(best_match):
                    test.logger.info(
                        f"Successfully clicked element with confidence {best_match['confidence']}"
                    )
                    success = True
                    break
            
            time.sleep(1)
            
        if not success:
            test.logger.warning("Failed to detect and click button after all attempts")
            
    except Exception as e:
        logging.error(f"Test failed: {e}")
        logging.error(traceback.format_exc())
        sys.exit(1)
    finally:
        if test:
            test.cleanup()

if __name__ == "__main__":
    main()
