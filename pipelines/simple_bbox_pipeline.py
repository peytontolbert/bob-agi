"""
Simple pipeline for testing bounding box accuracy of the vision system using screenshots.
"""
import logging
import time
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import Optional, Dict, List, Any, Union
import json
import glob
import shutil
from app.agents.text import TextAgent
import base64
import io

@dataclass
class BBoxTestResult:
    """Results from a bounding box detection test"""
    success: bool
    query: str
    detection_time: float
    bbox_coordinates: Optional[tuple] = None
    error_message: Optional[str] = None
    debug_image_path: Optional[str] = None
    original_image_path: Optional[str] = None
    llm_generated_query: Optional[str] = None

class SimpleBBoxPipeline:
    def __init__(self, vision_agent, text_agent):
        self.vision_agent = vision_agent
        self.text_agent = text_agent
        
        # Setup directories
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.debug_dir = Path(f"debug_output/bbox_test_{timestamp}")
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        
        self.screenshots_dir = Path("test_data/screenshots")
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)
        
        # Create default test screenshots if none exist
        self._ensure_test_screenshots()
        
        # Setup logging
        self._setup_logging()
        
        # Test results storage
        self.results = []
        
        # Default queries if LLM generation fails
        self.default_queries = [
            "Find and place boxes around the files",
            "Locate a function and put a box around it",
            "Frame the composer chatbox with bbox",
        ]

    def _ensure_test_screenshots(self):
        """Ensure test screenshots exist, create defaults if needed"""
        if not list(self.screenshots_dir.glob("*.png")):
            logging.info("No test screenshots found, creating defaults...")
            try:
                # Take a screenshot of current screen
                import pyautogui
                screen = pyautogui.screenshot()
                screen_path = self.screenshots_dir / "desktop_screen.png"
                screen.save(str(screen_path))
                logging.info(f"Created default screenshot: {screen_path}")
                
                # You could add more default screenshots here
                # For example, copying from a test_assets folder
                
            except Exception as e:
                logging.error(f"Failed to create default screenshots: {e}")
                raise RuntimeError("No test screenshots available")

    def _setup_logging(self):
        """Configure logging for the pipeline"""
        log_path = Path("logs/bbox_test")
        log_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_path / f"bbox_test_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        logging.info("BBox testing pipeline initialized")

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    def generate_test_queries(self, image_path: str) -> List[str]:
        """Generate test queries for the image using TextAgent"""
        try:
            # Load image
            image = Image.open(image_path)
            
            # Convert image to base64 for text agent
            image_base64 = self._image_to_base64(image)
            
            # Create prompt for text agent
            prompt = f"""Given this screenshot, generate 5 specific queries for testing UI element detection.
            Each query should:
            1. Target a single, clearly visible element in the image
            2. Be precise and unambiguous
            3. Include location context if possible (e.g., "in the bottom left", "at the top")
            4. Focus on interactive elements like buttons, icons, or UI components

            Format each query on a new line starting with 'QUERY: '

            <image>
            {image_base64}
            </image>"""
            
            # Get response from text agent
            response = self.text_agent.complete_task(prompt)
            
            # Extract queries from response
            queries = []
            for line in response.split('\n'):
                if line.strip().startswith('QUERY:'):
                    query = line.replace('QUERY:', '').strip()
                    if query:  # Only add non-empty queries
                        queries.append(query)
            
            if not queries:
                logging.warning("No queries generated by text agent, using default queries")
                return self.default_queries
                
            logging.info("Generated queries from text agent:")
            for query in queries:
                logging.info(f"- {query}")
                
            return queries
            
        except Exception as e:
            logging.error(f"Error generating test queries: {e}")
            return self.default_queries

    def run_single_test(self, image_path: Union[str, Path], query: str) -> BBoxTestResult:
        """Run a single bounding box detection test on a saved image"""
        try:
            # Load test image
            test_image = Image.open(str(image_path))
            
            # Save copy as original
            original_path = self.debug_dir / f"original_{int(time.time())}.png"
            test_image.save(str(original_path))

            # Time the detection
            start_time = time.time()
            detection_result = self.vision_agent.detect_ui_elements(
                test_image,
                query=query
            )
            detection_time = time.time() - start_time

            if detection_result['status'] != 'success' or not detection_result.get('detections'):
                return BBoxTestResult(
                    success=False,
                    query=query,
                    detection_time=detection_time,
                    error_message="No detection found",
                    original_image_path=str(original_path),
                    llm_generated_query=query
                )

            # Get first detection
            detection = detection_result['detections'][0]
            bbox = detection.get('bbox')

            # Save debug image with visualization
            debug_image_path = self._save_debug_image(
                test_image,
                bbox,
                query,
                f"detection_{int(time.time())}.png"
            )

            return BBoxTestResult(
                success=True,
                query=query,
                detection_time=detection_time,
                bbox_coordinates=bbox,
                debug_image_path=debug_image_path,
                original_image_path=str(original_path),
                llm_generated_query=query
            )

        except Exception as e:
            logging.error(f"Error in bbox test: {e}")
            return BBoxTestResult(
                success=False,
                query=query,
                detection_time=0,
                error_message=str(e),
                llm_generated_query=query
            )

    def _save_debug_image(self, image, bbox: tuple, query: str, filename: str) -> str:
        """Save debug image with detection visualization"""
        # Create a copy for drawing
        debug_image = image.copy()
        debug_array = np.array(debug_image)
        
        if bbox:
            x1, y1, x2, y2 = bbox
            # Draw bounding box in green
            cv2.rectangle(debug_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw center point in red
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.circle(debug_array, (center_x, center_y), 5, (0, 0, 255), -1)
            
            # Add text label
            cv2.putText(debug_array, query, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Convert back to PIL and save
        debug_image = Image.fromarray(cv2.cvtColor(debug_array, cv2.COLOR_BGR2RGB))
        debug_path = self.debug_dir / filename
        debug_image.save(str(debug_path))
        return str(debug_path)

    def run_tests(self, screenshot_path: Optional[str] = None) -> List[BBoxTestResult]:
        """Run all bounding box detection tests on screenshots"""
        if screenshot_path:
            screenshot_paths = [screenshot_path]
        else:
            # Get all PNG files in screenshots directory
            screenshot_paths = list(self.screenshots_dir.glob("*.png"))
            
        if not screenshot_paths:
            raise RuntimeError("No screenshots available for testing")
            
        for path in screenshot_paths:
            logging.info(f"\nTesting screenshot: {path}")
            
            if not Path(path).exists():
                logging.error(f"Screenshot not found: {path}")
                continue
                
            # Generate test queries for this specific screenshot
            queries = self.generate_test_queries(str(path))
            logging.info(f"Generated {len(queries)} test queries")
            
            for query in queries:
                logging.info(f"\nTesting query: {query}")
                result = self.run_single_test(path, query)
                self.results.append(result)
                
                # Log result
                if result.success:
                    logging.info(f"Detection successful:")
                    logging.info(f"- Time: {result.detection_time:.3f}s")
                    logging.info(f"- BBox: {result.bbox_coordinates}")
                    logging.info(f"- Debug image: {result.debug_image_path}")
                else:
                    logging.error(f"Detection failed: {result.error_message}")
                
                time.sleep(0.5)
        
        self._save_test_results()
        return self.results

    def _save_test_results(self):
        """Save test results to JSON file"""
        try:
            results_data = []
            for result in self.results:
                result_dict = {
                    'success': result.success,
                    'query': result.query,
                    'llm_generated_query': result.llm_generated_query,
                    'detection_time': result.detection_time,
                    'bbox_coordinates': result.bbox_coordinates,
                    'error_message': result.error_message,
                    'debug_image_path': result.debug_image_path,
                    'original_image_path': result.original_image_path
                }
                results_data.append(result_dict)

            results_path = self.debug_dir / 'test_results.json'
            with open(results_path, 'w') as f:
                json.dump(results_data, f, indent=2)
                
            logging.info(f"Test results saved to {results_path}")
            
        except Exception as e:
            logging.error(f"Failed to save test results: {e}")

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of test results"""
        if not self.results:
            return {"error": "No tests have been run"}
            
        successful_tests = len([r for r in self.results if r.success])
        total_tests = len(self.results)
        
        detection_times = [r.detection_time for r in self.results if r.success]
        avg_detection_time = sum(detection_times) / len(detection_times) if detection_times else 0
        
        return {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "success_rate": successful_tests / total_tests if total_tests > 0 else 0,
            "average_detection_time": avg_detection_time,
            "failed_queries": [r.query for r in self.results if not r.success]
        }

if __name__ == "__main__":
    from app.agents.vision import VisionAgent
    from app.agents.text import TextAgent
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Initialize agents
        vision_agent = VisionAgent()
        text_agent = TextAgent()
        
        # Create pipeline instance
        pipeline = SimpleBBoxPipeline(vision_agent, text_agent)
        
        # Run tests on all screenshots in the directory
        logging.info("Starting bounding box accuracy tests...")
        results = pipeline.run_tests()
        
        # Print summary
        summary = pipeline.get_summary()
        logging.info("\nTest Summary:")
        logging.info(f"Total tests: {summary.get('total_tests', 0)}")
        logging.info(f"Successful tests: {summary.get('successful_tests', 0)}")
        logging.info(f"Success rate: {summary.get('success_rate', 0)*100:.1f}%")
        logging.info(f"Average detection time: {summary.get('average_detection_time', 0):.3f}s")
        
        if summary.get('failed_queries'):
            logging.info("\nFailed queries:")
            for query in summary['failed_queries']:
                logging.info(f"- {query}")
                
    except Exception as e:
        logging.error(f"Test execution failed: {e}", exc_info=True)
    finally:
        logging.info("Test execution completed")
