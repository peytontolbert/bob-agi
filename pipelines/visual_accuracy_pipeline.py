"""
Pipeline for accurately locating UI elements on screen and testing visual accuracy.
"""
import logging
import time
from typing import Optional, Dict, Tuple, List, Any
import numpy as np
from PIL import Image
import json
import os
from datetime import datetime
import cv2
import backoff  # For exponential backoff
from dataclasses import dataclass, asdict
import traceback

@dataclass
class DetectionResult:
    """Standardized detection result structure"""
    success: bool
    confidence: float
    bbox: Optional[Tuple[int, int, int, int]] = None
    error_message: Optional[str] = None
    processing_time: float = 0.0
    timestamp: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class VisualAccuracyPipeline:
    def __init__(self, vision_agent, eyes):
        self.vision_agent = vision_agent
        # Initialize Eyesight with screen
        self.eyes = eyes
        self.confidence_threshold = 0.7
        self.max_retries = 3
        self.retry_delay = 0.5
        
        # Setup results directory
        self.results_dir = "results/visual_accuracy"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize accuracy metrics
        self.accuracy_metrics = {
            'total_attempts': 0,
            'successful_detections': 0,
            'average_confidence': [],
            'detection_times': [],
            'element_types': {}
        }
        
        # Add performance monitoring
        self.performance_metrics = {
            'total_processing_time': 0,
            'calls_per_minute': [],
            'error_count': 0,
            'last_minute_timestamp': time.time()
        }
        
        # Add debug mode
        self.debug_mode = False
        
    def enable_debug_mode(self):
        """Enables detailed logging and saves intermediate results"""
        self.debug_mode = True
        logging.getLogger().setLevel(logging.DEBUG)

    def analyze_scene_and_find_element(self, target_description: str) -> Dict:
        """
        Analyzes the scene and finds a specific element using Eyesight.
        """
        try:
            start_time = time.time()
            
            # Use Eyesight to capture screen
            screen_state = self.eyes.capture_screen()
            if not screen_state:
                return self._create_error_result("Failed to capture screen")

            current_frame = screen_state.get('frame')
            
            # Get scene understanding through Eyesight
            perception = screen_state.get('perception', {})
            
            # Find specific element using Eyesight
            element = self.eyes.find_element(target_description)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Prepare result
            result = {
                'timestamp': datetime.now().isoformat(),
                'target_description': target_description,
                'scene_analysis': perception,
                'element_found': element is not None,
                'element_details': element if element else None,
                'processing_time': processing_time,
                'confidence': element.get('confidence', 0) if element else 0
            }
            
            # Update accuracy metrics
            self._update_accuracy_metrics(result)
            
            # Save result
            self._save_analysis_result(result, current_frame)
            
            return result

        except Exception as e:
            logging.error(f"Error in scene analysis: {e}")
            return self._create_error_result(str(e))

    def test_element_detection(self, element_description: str, num_attempts: int = 5) -> Dict:
        """
        Tests the accuracy of element detection over multiple attempts.
        
        Args:
            element_description: Description of element to find
            num_attempts: Number of detection attempts
            
        Returns:
            Dict containing test results and statistics
        """
        results = []
        successful_detections = 0
        total_confidence = 0
        detection_times = []

        for i in range(num_attempts):
            start_time = time.time()
            result = self.analyze_scene_and_find_element(element_description)
            detection_time = time.time() - start_time
            
            if result.get('element_found'):
                successful_detections += 1
                total_confidence += result.get('confidence', 0)
                
            detection_times.append(detection_time)
            results.append(result)
            
            # Add small delay between attempts
            time.sleep(0.5)

        # Calculate statistics
        success_rate = successful_detections / num_attempts
        avg_confidence = total_confidence / successful_detections if successful_detections > 0 else 0
        avg_detection_time = sum(detection_times) / len(detection_times)

        test_results = {
            'element_description': element_description,
            'num_attempts': num_attempts,
            'success_rate': success_rate,
            'average_confidence': avg_confidence,
            'average_detection_time': avg_detection_time,
            'detailed_results': results,
            'timestamp': datetime.now().isoformat()
        }

        # Save test results
        self._save_test_results(test_results)
        
        return test_results

    @backoff.on_exception(
        backoff.expo,
        (TimeoutError, ConnectionError),
        max_tries=3,
        max_time=30
    )
    def find_element_location(self, description: str) -> Optional[Dict]:
        """
        Enhanced element location using Eyesight with better error handling and retries.
        """
        start_time = time.time()
        try:
            # Use Eyesight's wait_for_element with retry logic built in
            element = self.eyes.wait_for_element(
                description,
                timeout=self.max_retries * self.retry_delay
            )
            
            if element:
                # Validate and refine the location
                refined = self._validate_and_refine_location(element, self.eyes.get_screen_state()['current'])
                if refined:
                    # Update performance metrics
                    processing_time = time.time() - start_time
                    self._update_performance_metrics(processing_time)
                    return refined

            self._log_error(f"Element not found after {self.max_retries} attempts: {description}")
            return None

        except Exception as e:
            self._log_error(f"Error in find_element_location: {str(e)}\n{traceback.format_exc()}")
            self.performance_metrics['error_count'] += 1
            return None

    def _capture_screen_with_retry(self, max_retries: int = 3) -> Optional[Image.Image]:
        """Captures screen with retry logic using Eyesight"""
        for attempt in range(max_retries):
            try:
                screen_state = self.eyes.capture_screen()
                if screen_state and 'frame' in screen_state:
                    return screen_state['frame']
                    
            except Exception as e:
                logging.error(f"Screen capture attempt {attempt + 1} failed: {e}")
                
            time.sleep(0.1)
            
        return None

    def _update_performance_metrics(self, processing_time: float):
        """Updates performance monitoring metrics"""
        current_time = time.time()
        self.performance_metrics['total_processing_time'] += processing_time
        
        # Update calls per minute
        if current_time - self.performance_metrics['last_minute_timestamp'] >= 60:
            self.performance_metrics['calls_per_minute'].append(len(self.accuracy_metrics['detection_times']))
            self.performance_metrics['last_minute_timestamp'] = current_time
            self.accuracy_metrics['detection_times'] = []

    def _save_debug_frame(self, frame: Image.Image, attempt: int, description: str):
        """Saves debug frames during detection attempts"""
        if not self.debug_mode:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_path = os.path.join(self.results_dir, "debug")
        os.makedirs(debug_path, exist_ok=True)
        
        filename = f"debug_frame_{timestamp}_attempt_{attempt}.png"
        filepath = os.path.join(debug_path, filename)
        frame.save(filepath)
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'attempt': attempt,
            'description': description,
            'confidence_threshold': self.confidence_threshold
        }
        
        meta_file = os.path.join(debug_path, f"debug_meta_{timestamp}.json")
        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _log_error(self, message: str):
        """Centralized error logging"""
        logging.error(message)
        if self.debug_mode:
            # Save error to debug log file
            debug_log_path = os.path.join(self.results_dir, "debug", "error_log.txt")
            with open(debug_log_path, 'a') as f:
                f.write(f"{datetime.now().isoformat()}: {message}\n")

    def _save_detection_image(self, frame: Image.Image, element: Dict, description: str):
        """Saves the frame with detected element highlighted and coordinates displayed"""
        try:
            # Convert PIL to OpenCV format
            frame_cv = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            
            # Draw bounding box
            bbox = element.get('bbox')
            if bbox:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add element description
                cv2.putText(frame_cv, f"{description[:30]}...", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Add coordinates
                coord_text = f"({x1},{y1}),({x2},{y2})"
                cv2.putText(frame_cv, coord_text, (x1, y2+20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                # Add center point and its coordinates
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.circle(frame_cv, (center_x, center_y), 4, (0, 0, 255), -1)
                center_text = f"Center: ({center_x},{center_y})"
                cv2.putText(frame_cv, center_text, (x1, y2+40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Save image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detection_{timestamp}.png"
            filepath = os.path.join(self.results_dir, filename)
            cv2.imwrite(filepath, frame_cv)
            
        except Exception as e:
            logging.error(f"Error saving detection image: {e}")

    def _save_analysis_result(self, result: Dict, frame: Image.Image):
        """Saves analysis result and corresponding image"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save result JSON
            result_file = os.path.join(self.results_dir, f"analysis_{timestamp}.json")
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
                
            # Save frame
            frame_file = os.path.join(self.results_dir, f"frame_{timestamp}.png")
            frame.save(frame_file)
            
        except Exception as e:
            logging.error(f"Error saving analysis result: {e}")

    def _save_test_results(self, results: Dict):
        """Saves test results to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_results_{timestamp}.json"
            filepath = os.path.join(self.results_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2)
                
        except Exception as e:
            logging.error(f"Error saving test results: {e}")

    def _update_accuracy_metrics(self, result: Dict):
        """Updates accuracy metrics based on detection result"""
        self.accuracy_metrics['total_attempts'] += 1
        
        if result.get('element_found'):
            self.accuracy_metrics['successful_detections'] += 1
            self.accuracy_metrics['average_confidence'].append(result.get('confidence', 0))
            self.accuracy_metrics['detection_times'].append(result.get('processing_time', 0))
            
            # Track element type statistics
            element_type = result.get('target_description', 'unknown')
            if element_type not in self.accuracy_metrics['element_types']:
                self.accuracy_metrics['element_types'][element_type] = {
                    'attempts': 0,
                    'successes': 0
                }
            self.accuracy_metrics['element_types'][element_type]['attempts'] += 1
            self.accuracy_metrics['element_types'][element_type]['successes'] += 1

    def _create_error_result(self, error_message: str) -> Dict:
        """Creates a standardized error result"""
        return {
            'status': 'error',
            'message': error_message,
            'timestamp': datetime.now().isoformat(),
            'element_found': False
        }

    def _validate_and_refine_location(self, element: Dict, frame: Image.Image) -> Optional[Dict]:
        """
        Validates and refines the detected element location.
        
        Args:
            element: Initial element detection
            frame: Current screen frame
            
        Returns:
            Refined element information or None if invalid
        """
        try:
            # Extract coordinates
            bbox = element.get('bbox')
            if not bbox or len(bbox) != 4:
                return None
                
            x1, y1, x2, y2 = bbox
            
            # Validate coordinates are within frame
            frame_width, frame_height = frame.size
            if not (0 <= x1 < x2 <= frame_width and 0 <= y1 < y2 <= frame_height):
                return None
                
            # Calculate center coordinates
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            # Add additional metadata
            element.update({
                'coordinates': (center_x, center_y),
                'width': x2 - x1,
                'height': y2 - y1,
                'area': (x2 - x1) * (y2 - y1),
                'aspect_ratio': (x2 - x1) / (y2 - y1) if y2 > y1 else 0,
                'timestamp': time.time()
            })
            
            return element

        except Exception as e:
            logging.error(f"Error validating element location: {e}")
            return None

    def get_element_region(self, description: str) -> Optional[Dict]:
        """Gets the region of the screen containing the described element using Eyesight"""
        try:
            # Use Eyesight's find_element with region info
            element = self.eyes.find_element(description)
            if not element:
                return None
                
            # Get region around element
            bbox = element.get('bbox')
            if not bbox:
                return None
                
            # Add padding around region
            x1, y1, x2, y2 = bbox
            padding = 10
            
            screen_state = self.eyes.get_screen_state()
            if not screen_state:
                return None
                
            frame = screen_state['current']
            frame_width, frame_height = frame.size
            
            # Ensure region stays within screen bounds
            region = {
                'x1': max(0, x1 - padding),
                'y1': max(0, y1 - padding),
                'x2': min(frame_width, x2 + padding),
                'y2': min(frame_height, y2 + padding),
                'element': element
            }
            
            return region

        except Exception as e:
            logging.error(f"Error getting element region: {e}")
            return None

    def get_performance_report(self) -> Dict[str, Any]:
        """Generates a performance report"""
        return {
            'total_processing_time': self.performance_metrics['total_processing_time'],
            'average_processing_time': (self.performance_metrics['total_processing_time'] / 
                                      max(1, self.accuracy_metrics['total_attempts'])),
            'error_rate': (self.performance_metrics['error_count'] / 
                          max(1, self.accuracy_metrics['total_attempts'])),
            'success_rate': (self.accuracy_metrics['successful_detections'] / 
                           max(1, self.accuracy_metrics['total_attempts'])),
            'calls_per_minute': self.performance_metrics['calls_per_minute']
        }

if __name__ == "__main__":
    from app.agents.vision import VisionAgent
    from app.env.screen import Screen
    from app.env.computer import Computer
    from app.actions.eyesight import Eyesight
    import logging
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        computer = Computer()
        computer.run()
        eyes = Eyesight(computer.screen)
        
        vision_agent = VisionAgent()
        
        # Create pipeline
        pipeline = VisualAccuracyPipeline(vision_agent, eyes)
        
        # Enable debug mode for testing
        pipeline.enable_debug_mode()
        
        logging.info("Starting visual accuracy test for Discord elements...")
        
        test_results = pipeline.test_element_detection(
            "Continue in browser link",
            num_attempts=5
        )
        
        # Print results
        logging.info("Test Results:")
        logging.info(f"Success rate: {test_results['success_rate']:.2%}")
        logging.info(f"Average confidence: {test_results['average_confidence']:.2f}")
        logging.info(f"Average detection time: {test_results['average_detection_time']:.2f}s")
        
        # Get performance report
        perf_report = pipeline.get_performance_report()
        logging.info("\nPerformance Report:")
        logging.info(f"Average processing time: {perf_report['average_processing_time']:.2f}s")
        logging.info(f"Error rate: {perf_report['error_rate']:.2%}")
        logging.info(f"Success rate: {perf_report['success_rate']:.2%}")
        
    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}", exc_info=True)
    finally:
        logging.info("Pipeline execution completed")

