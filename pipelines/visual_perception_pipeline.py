"""
Pipeline for testing Bob's visual perception capabilities.
"""
import logging
import time
from datetime import datetime
import numpy as np
from PIL import Image
import os
from app.env.computer import Computer
from app.env.screen import Screen
from app.env.senses.eyesight import Eyesight
from app.agents.vision import VisionAgent
import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VisualPerceptionPipeline:
    def __init__(self):
        self.computer = Computer()
        self.computer.run()
        self.screen = self.computer.screen
        self.eyesight = Eyesight(self.screen)
        self.vision_agent = VisionAgent()
        
        # Test metrics
        self.metrics = {
            'perception_times': [],
            'embedding_times': [],
            'detection_success': 0,
            'detection_total': 0
        }
        
        # Add image validation settings
        self.min_image_size = (100, 100)
        self.max_image_size = (4096, 4096)
        
    def validate_image(self, image):
        """Validate image format and properties"""
        if image is None:
            raise ValueError("Image is None")
            
        if isinstance(image, np.ndarray):
            if not np.isfinite(image).all():
                raise ValueError("Image contains invalid values")
            if image.min() < 0 or image.max() > 255:
                raise ValueError("Image values out of valid range")
                
        if isinstance(image, Image.Image):
            if image.size[0] < self.min_image_size[0] or image.size[1] < self.min_image_size[1]:
                raise ValueError(f"Image too small: {image.size}")
            if image.size[0] > self.max_image_size[0] or image.size[1] > self.max_image_size[1]:
                raise ValueError(f"Image too large: {image.size}")
                
        return True
        
    def run_pipeline(self):
        """Execute full visual perception test pipeline"""
        try:
            logger.info("Starting visual perception pipeline...")
            
            # Initialize computer environment
            self.computer.startup()
            time.sleep(2)  # Allow time for initialization
            
            # Run test scenarios
            self.test_basic_perception()
            self.test_element_detection()
            self.test_visual_memory()
            self.test_realtime_processing()
            
            # Report results
            self.report_results()
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
        finally:
            self.cleanup()
            
    def test_basic_perception(self):
        """Test basic screen perception capabilities"""
        logger.info("Testing basic perception...")
        
        try:
            # Capture current screen with validation
            start_time = time.time()
            screen_state = self.eyesight.capture_screen()
            
            if not screen_state:
                logger.error("Failed to capture screen state")
                return
                
            # Validate frame format
            frame = screen_state.get('frame')
            if not frame:
                logger.error("No frame in screen state")
                return
                
            try:
                self.validate_image(frame)
            except ValueError as e:
                logger.error(f"Image validation failed: {e}")
                return
                
            # Use batched processing for faster scene understanding
            batch_size = 4
            frame_batches = [frame] * batch_size  # Create multiple views of same frame
            
            try:
                perceptions = []
                for i in range(0, len(frame_batches), batch_size):
                    batch = frame_batches[i:i + batch_size]
                    batch_results = self.vision_agent.understand_scene(
                        batch,
                        ["Describe the current screen content"] * len(batch)
                    )
                    perceptions.extend(batch_results)
                
                # Use most confident perception
                perception = max(perceptions, key=lambda x: x.get('confidence', 0))
                
                if not perception:
                    logger.error("Scene understanding returned None")
                    return
                    
                if perception.get('status') != 'success':
                    logger.error(f"Scene understanding failed: {perception.get('message', 'Unknown error')}")
                    return
                    
                processing_time = time.time() - start_time
                self.metrics['perception_times'].append(processing_time)
                
                logger.info(f"Basic perception test completed in {processing_time:.2f}s")
                logger.info(f"Scene description: {perception.get('description', '')[:200]}...")
                
            except Exception as e:
                logger.error(f"Scene understanding failed with exception: {e}")
                
        except Exception as e:
            logger.error(f"Basic perception test failed: {e}")
            
    def test_element_detection(self):
        """Test UI element detection capabilities"""
        logger.info("Testing element detection...")
        
        test_elements = [
            "button",
            "text input",
            "link",
            "image"
        ]
        
        # Get initial frame for batch processing
        frame = self.screen.get_current_frame()
        if not frame:
            logger.error("Failed to capture frame for element detection")
            return
            
        for element in test_elements:
            try:
                start_time = time.time()
                
                # Use new prepare_for_action method
                results = self.vision_agent.prepare_for_action(frame, element)
                
                self.metrics['detection_total'] += 1
                if results and len(results) > 0:
                    self.metrics['detection_success'] += 1
                    
                processing_time = time.time() - start_time
                logger.info(f"Element '{element}' detection took {processing_time:.2f}s")
                
                # Log detection details
                if results:
                    logger.info(f"Found {len(results)} potential {element}s")
                    for i, result in enumerate(results[:3]):  # Log top 3 matches
                        logger.info(f"Match {i+1}: confidence={result['confidence']:.2f}, " 
                                  f"relevance={result.get('relevance', 0):.2f}")
            
            except Exception as e:
                logger.error(f"Element detection failed for '{element}': {e}")
                
    def test_visual_memory(self):
        """Test visual memory and embedding capabilities"""
        logger.info("Testing visual memory...")
        
        try:
            # Generate and store embeddings
            start_time = time.time()
            current_frame = self.screen.get_current_frame()
            
            embedding = self.eyesight.generate_embedding(current_frame)
            if embedding is None:
                raise ValueError("Failed to generate embedding")
                
            # Test embedding storage
            filepath = self.eyesight.save_image_with_embedding(
                current_frame,
                "Test frame"
            )
            
            if not filepath or not os.path.exists(filepath):
                raise ValueError("Failed to save image with embedding")
                
            # Test similarity search
            similar_images = self.eyesight.find_similar_images(current_frame)
            
            processing_time = time.time() - start_time
            self.metrics['embedding_times'].append(processing_time)
            
            logger.info(f"Visual memory test completed in {processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Visual memory test failed: {e}")
            raise
            
    def test_realtime_processing(self):
        """Test real-time perception processing"""
        logger.info("Testing real-time processing...")
        
        try:
            start_time = time.time()
            frames_processed = 0
            test_duration = 5  # Test for 5 seconds
            
            # Create frame buffer for batch processing
            frame_buffer = []
            buffer_size = 4  # Process 4 frames at once
            
            while time.time() - start_time < test_duration:
                frame = self.screen.get_current_frame()
                if frame:
                    frame_buffer.append(frame)
                    
                    # Process batch when buffer is full
                    if len(frame_buffer) >= buffer_size:
                        # Process frame batch
                        self.vision_agent.understand_scene(
                            frame_buffer,
                            ["Monitor screen activity"] * len(frame_buffer)
                        )
                        frames_processed += len(frame_buffer)
                        frame_buffer = []  # Clear buffer
                    
                time.sleep(0.02)  # Aim for 50 FPS capture rate
                
            fps = frames_processed / test_duration
            logger.info(f"Real-time processing test: {fps:.1f} FPS")
            
        except Exception as e:
            logger.error(f"Real-time processing test failed: {e}")
            raise
            
    def report_results(self):
        """Generate and display test results"""
        results = {
            'avg_perception_time': np.mean(self.metrics['perception_times']),
            'avg_embedding_time': np.mean(self.metrics['embedding_times']),
            'detection_accuracy': (self.metrics['detection_success'] / 
                                 max(1, self.metrics['detection_total']))
        }
        
        logger.info("\nTest Results:")
        logger.info(f"Average perception time: {results['avg_perception_time']:.2f}s")
        logger.info(f"Average embedding time: {results['avg_embedding_time']:.2f}s")
        logger.info(f"Element detection accuracy: {results['detection_accuracy']:.1%}")
        
    def cleanup(self):
        """Clean up test artifacts and resources"""
        try:
            self.computer.shutdown()
            # Remove any temporary test files
            logger.info("Cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

if __name__ == "__main__":
    pipeline = VisualPerceptionPipeline()
    pipeline.run_pipeline()
