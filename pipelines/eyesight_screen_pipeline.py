"""
Pipeline for testing screen capture and eyesight integration.
"""
import logging
import time
from PIL import Image
import numpy as np
import cv2
from app.env.computer import Computer
from app.env.senses.eyesight import Eyesight
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EyesightScreenPipeline:
    def __init__(self):
        self.computer = Computer()
        self.computer.run()
        self.screen = self.computer.screen
        self.eyesight = Eyesight(self.screen)
        
        # Test metrics
        self.metrics = {
            'capture_times': [],
            'frame_rates': [],
            'successful_captures': 0,
            'failed_captures': 0,
            'format_errors': 0
        }
        
        # Create output directory for debug images
        self.debug_dir = "debug/screen_captures"
        os.makedirs(self.debug_dir, exist_ok=True)

    def run_pipeline(self):
        """Execute screen capture and eyesight integration tests"""
        try:
            logger.info("Starting eyesight-screen pipeline...")
            
            # Test basic screen capture
            self.test_screen_capture()
            
            # Test frame buffer
            self.test_frame_buffer()
            
            # Test eyesight processing
            self.test_eyesight_processing()
            
            # Test real-time performance
            self.test_realtime_performance()
            
            # Report results
            self.report_results()
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
        finally:
            self.cleanup()

    def test_screen_capture(self):
        """Test basic screen capture functionality"""
        logger.info("Testing screen capture...")
        
        try:
            # Test multiple captures
            for i in range(5):
                start_time = time.time()
                
                # Get frame from screen
                frame = self.screen.get_current_frame()
                
                if frame is None:
                    logger.error("Failed to capture frame")
                    self.metrics['failed_captures'] += 1
                    continue
                
                # Validate frame format
                if not isinstance(frame, Image.Image):
                    logger.error(f"Incorrect frame format: {type(frame)}")
                    self.metrics['format_errors'] += 1
                    continue
                
                # Verify RGB mode
                if frame.mode != 'RGB':
                    logger.error(f"Incorrect image mode: {frame.mode}")
                    self.metrics['format_errors'] += 1
                    continue
                
                # Save debug image
                debug_path = os.path.join(self.debug_dir, f"capture_{i}.png")
                frame.save(debug_path)
                
                capture_time = time.time() - start_time
                self.metrics['capture_times'].append(capture_time)
                self.metrics['successful_captures'] += 1
                
                logger.info(f"Capture {i+1} completed in {capture_time:.3f}s")
                time.sleep(1)  # Wait between captures
                
        except Exception as e:
            logger.error(f"Screen capture test failed: {e}")
            raise

    def test_frame_buffer(self):
        """Test frame buffer functionality"""
        logger.info("Testing frame buffer...")
        
        try:
            # Monitor buffer for 5 seconds
            start_time = time.time()
            frames_received = 0
            
            while time.time() - start_time < 5:
                buffer = self.screen.get_frame_buffer()
                if buffer:
                    frames_received = len(buffer)
                    
                    # Validate each frame
                    for frame_data in buffer:
                        if not isinstance(frame_data, dict):
                            logger.error("Invalid frame buffer format")
                            continue
                            
                        frame = frame_data.get('frame')
                        timestamp = frame_data.get('timestamp')
                        
                        if not frame or not timestamp:
                            logger.error("Missing frame data")
                            continue
                            
                        if not isinstance(frame, Image.Image):
                            logger.error(f"Invalid frame type in buffer: {type(frame)}")
                            continue
                
                time.sleep(0.1)
            
            fps = frames_received / 5  # Calculate average FPS
            self.metrics['frame_rates'].append(fps)
            logger.info(f"Frame buffer test completed. Average FPS: {fps:.1f}")
            
        except Exception as e:
            logger.error(f"Frame buffer test failed: {e}")
            raise

    def test_eyesight_processing(self):
        """Test eyesight processing of screen captures"""
        logger.info("Testing eyesight processing...")
        
        try:
            # Capture and process multiple frames
            for i in range(3):
                # Get screen state through eyesight
                screen_state = self.eyesight.capture_screen()
                
                if not screen_state:
                    logger.error("Failed to get screen state")
                    continue
                
                # Validate screen state
                if not isinstance(screen_state, dict):
                    logger.error(f"Invalid screen state type: {type(screen_state)}")
                    continue
                
                # Check required components
                required_keys = ['frame', 'timestamp', 'perception']
                missing_keys = [key for key in required_keys if key not in screen_state]
                if missing_keys:
                    logger.error(f"Missing keys in screen state: {missing_keys}")
                    continue
                
                # Save processed frame
                frame = screen_state['frame']
                if isinstance(frame, Image.Image):
                    debug_path = os.path.join(self.debug_dir, f"processed_{i}.png")
                    frame.save(debug_path)
                
                logger.info(f"Processing test {i+1} completed")
                time.sleep(1)
                
        except Exception as e:
            logger.error(f"Eyesight processing test failed: {e}")
            raise

    def test_realtime_performance(self):
        """Test real-time performance metrics"""
        logger.info("Testing real-time performance...")
        
        try:
            start_time = time.time()
            frames_processed = 0
            test_duration = 10  # Test for 10 seconds
            
            while time.time() - start_time < test_duration:
                # Get and process frame
                frame = self.screen.get_current_frame()
                if frame is not None:
                    # Test eyesight processing
                    result = self.eyesight.process_visual_input(
                        "Monitor screen activity",
                        frame
                    )
                    
                    if result and result.get('status') == 'success':
                        frames_processed += 1
                        
                time.sleep(0.1)  # Limit to ~10 FPS for testing
                
            fps = frames_processed / test_duration
            self.metrics['frame_rates'].append(fps)
            logger.info(f"Real-time test completed. Average FPS: {fps:.1f}")
            
        except Exception as e:
            logger.error(f"Real-time performance test failed: {e}")
            raise

    def report_results(self):
        """Generate and display test results"""
        logger.info("\nTest Results:")
        logger.info(f"Successful captures: {self.metrics['successful_captures']}")
        logger.info(f"Failed captures: {self.metrics['failed_captures']}")
        logger.info(f"Format errors: {self.metrics['format_errors']}")
        
        if self.metrics['capture_times']:
            avg_capture_time = np.mean(self.metrics['capture_times'])
            logger.info(f"Average capture time: {avg_capture_time:.3f}s")
            
        if self.metrics['frame_rates']:
            avg_fps = np.mean(self.metrics['frame_rates'])
            logger.info(f"Average frame rate: {avg_fps:.1f} FPS")

    def cleanup(self):
        """Clean up test artifacts and resources"""
        try:
            self.computer.shutdown()
            logger.info("Cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

if __name__ == "__main__":
    pipeline = EyesightScreenPipeline()
    pipeline.run_pipeline()
