"""
Test pipeline for YOLOv8 detection of Discord UI elements using Computer and Eyesight
"""
import logging
from pathlib import Path
import time
from app.env.computer import Computer
from app.actions.eyesight import Eyesight

class YOLODetectionTest:
    def __init__(self):
        self.model_path = Path("weights/yolov8n.pt")
        if not self.model_path.exists():
            raise FileNotFoundError(f"YOLOv8 weights not found at {self.model_path}")
        
        # Initialize computer environment
        self.computer = Computer()
        self.computer.startup()
        
        # Initialize eyesight with computer's screen
        self.eyesight = Eyesight(self.computer.screen)
        
        # Setup debug directory
        self.debug_dir = Path("debug_output")
        self.debug_dir.mkdir(exist_ok=True)
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def detect_and_click_button(self, description="Continue in browser button", confidence_threshold=0.3):
        """
        Detect and click the 'Continue in browser' button using Eyesight
        """
        try:
            # Use eyesight to find the button element
            button = self.eyesight.find_element(description)
            
            if button and button.get('confidence', 0) >= confidence_threshold:
                # Get coordinates from detection
                coords = button.get('coordinates')
                if coords:
                    center_x, center_y = coords.get('center_x'), coords.get('center_y')
                    
                    # Save debug image with detection
                    debug_frame = self.computer.screen.get_current_frame()
                    if debug_frame is not None:
                        self.eyesight.vision_agent.draw_detection(
                            debug_frame, 
                            button,
                            self.debug_dir / "detection.png"
                        )
                    
                    # Click using computer's mouse
                    logging.info(f"Clicking at ({center_x}, {center_y})")
                    self.computer.mouse.move_to(center_x, center_y)
                    self.computer.mouse.click()
                    return True
                    
            logging.info("No suitable button detected")
            return False

        except Exception as e:
            logging.error(f"Error during detection: {e}")
            return False

    def cleanup(self):
        """Cleanup resources"""
        try:
            self.computer.shutdown()
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")

def main():
    detector = None
    try:
        detector = YOLODetectionTest()
        logging.info("Starting Discord button detection test...")
        
        # Wait for Discord to load
        time.sleep(2)
        
        # Multiple detection attempts with different thresholds
        thresholds = [0.3, 0.2, 0.1]
        for threshold in thresholds:
            logging.info(f"Attempting detection with threshold {threshold}")
            success = detector.detect_and_click_button(
                description="Continue in browser button", 
                confidence_threshold=threshold
            )
            if success:
                logging.info(f"Successfully detected and clicked button with threshold {threshold}")
                break
            time.sleep(1)
        else:
            logging.warning("Failed to detect button after all attempts")
            
    except Exception as e:
        logging.error(f"Test failed: {e}")
    finally:
        if detector:
            detector.cleanup()

if __name__ == "__main__":
    main()
