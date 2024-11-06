"""
Test script for SAM2 detection on computer screen captures
"""
import logging
from pathlib import Path
import time
import asyncio
from PIL import Image
from app.models.samtwo import SAM2, SAM2Config
from app.env.computer.computer import Computer

class ScreenSAM2Test:
    def __init__(self):
        self.sam2 = None
        self.computer = None
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
                logging.FileHandler(self.debug_dir / "screen_sam2_test.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def initialize_components(self):
        """Initialize SAM2 model and Computer environment"""
        try:
            # Initialize SAM2
            self.logger.info("Initializing SAM2...")
            self.sam2 = SAM2(SAM2Config())
            if not self.sam2:
                raise RuntimeError("Failed to initialize SAM2")
            self.logger.info("SAM2 initialized successfully")

            # Initialize Computer
            self.logger.info("Initializing Computer environment...")
            self.computer = Computer()
            self.computer.start()  # Start the computer process
            time.sleep(2)  # Give it time to initialize
            self.logger.info("Computer environment initialized")

        except Exception as e:
            self.logger.error(f"Initialization error: {e}")
            raise

    async def test_screen_detection(self, query="Continue in Browser"):
        """
        Test SAM2 detection on screen captures
        
        Args:
            query: Text query to guide detection (default: "Continue in Browser")
        """
        try:
            self.logger.info(f"Testing screen detection with query: {query}")

            # Capture screen state
            screen_state = self.computer.screen.capture()
            if not screen_state:
                raise RuntimeError("Failed to capture screen state")

            # Get the frame from screen state
            frame = screen_state['frame']
            if not isinstance(frame, Image.Image):
                raise ValueError(f"Invalid frame type: {type(frame)}")

            # Save captured frame
            timestamp = int(time.time())
            frame_path = self.debug_dir / f"screen_capture_{timestamp}.png"
            frame.save(frame_path)
            self.logger.info(f"Saved screen capture to {frame_path}")

            # Run SAM2 detection
            results = await self.sam2.detect_elements(frame, query)
            
            if results['status'] != 'success':
                raise RuntimeError(f"Detection failed: {results.get('message')}")

            # Save detection visualization
            vis_path = self.debug_dir / f"detection_results_{timestamp}.png"
            results['annotated_image'].save(vis_path)
            self.logger.info(f"Saved detection visualization to {vis_path}")

            # Log detection results
            self.logger.info("\nDetection Results:")
            self.logger.info(f"Total detections: {len(results['detections'])}")
            
            if results['detections']:
                # Print details of all detections
                for i, det in enumerate(results['detections'], 1):
                    self.logger.info(
                        f"\nDetection {i}:"
                        f"\n  Confidence: {det['confidence']:.3f}"
                        f"\n  Position: ({det['coordinates'][0]:.0f}, {det['coordinates'][1]:.0f})"
                        f"\n  Size: {det['width']:.0f}x{det['height']:.0f}"
                        f"\n  Area: {det['area']:.0f}"
                    )

            return results

        except Exception as e:
            self.logger.error(f"Error in screen detection test: {e}")
            raise

    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.computer:
                self.computer.shutdown()
            self.logger.info("Cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

def main():
    test = ScreenSAM2Test()
    
    try:
        # Run detection test
        asyncio.run(test.test_screen_detection())
    finally:
        test.cleanup()

if __name__ == "__main__":
    main()
