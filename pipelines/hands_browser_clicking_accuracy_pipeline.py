"""
Pipeline for testing and debugging hand clicking accuracy on browser elements.
"""
import logging
import time
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from datetime import datetime
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from app.env.computer import Computer
from app.actions.hands import Hands

@dataclass
class ClickTestResult:
    """Stores results of a click accuracy test"""
    target_coordinates: Tuple[int, int]
    actual_coordinates: Tuple[int, int]
    offset: Tuple[float, float]
    success: bool
    time_taken: float
    element_type: str
    screenshot_path: Optional[str] = None
    error_message: Optional[str] = None

class HandClickingAccuracyPipeline:
    def __init__(self, hands: Hands, computer, config: Optional[Dict] = None):
        self.hands = hands
        self.computer = computer
        self.config = config or self._get_default_config()
        
        # Setup logging and debug directories
        self.debug_path = Path(self.config['debug_path'])
        self.debug_path.mkdir(parents=True, exist_ok=True)
        self._setup_logging()
        
        # Test metrics
        self.results: List[ClickTestResult] = []
        self.total_tests = 0
        self.successful_tests = 0

    def _get_default_config(self) -> Dict:
        """Get default configuration settings"""
        return {
            'debug_path': 'debug_output/click_accuracy',
            'accuracy_threshold': 5,  # pixels
            'test_iterations': 10,
            'movement_speed': 1.0,
            'save_screenshots': True,
            'logging_level': logging.INFO
        }

    def _setup_logging(self):
        """Configure logging for the pipeline"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.debug_path / f"click_accuracy_{timestamp}.log"
        
        logging.basicConfig(
            level=self.config['logging_level'],
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    def test_click_accuracy(self, element_type: str, coordinates: Tuple[int, int]) -> ClickTestResult:
        """
        Test clicking accuracy for specific coordinates.
        
        Args:
            element_type: Type of UI element being tested
            coordinates: Target coordinates (x, y)
        """
        start_time = time.time()
        
        try:
            # Get initial mouse position for movement calculation
            initial_pos = self.hands.position
            logging.info(f"Starting position: {initial_pos}")
            
            # Move to target with validation
            self.hands.move_mouse(coordinates[0], coordinates[1], smooth=True)
            time.sleep(0.2)  # Allow movement to complete
            
            # Get actual position after movement with validation
            actual_pos = self.hands.position
            if actual_pos == (0, 0):
                raise Exception("Failed to get valid mouse position")
                
            logging.info(f"Final position: {actual_pos}")
            
            # Calculate offset with improved precision
            offset = (
                round(actual_pos[0] - coordinates[0], 2),
                round(actual_pos[1] - coordinates[1], 2)
            )
            
            # Determine success based on accuracy threshold
            success = (abs(offset[0]) <= self.config['accuracy_threshold'] and
                      abs(offset[1]) <= self.config['accuracy_threshold'])
            
            # Save screenshot if enabled
            screenshot_path = None
            if self.config['save_screenshots']:
                screenshot_path = self._save_accuracy_visualization(
                    coordinates, actual_pos, offset, success
                )
            
            # Create and log result
            result = ClickTestResult(
                target_coordinates=coordinates,
                actual_coordinates=actual_pos,
                offset=offset,
                success=success,
                time_taken=time.time() - start_time,
                element_type=element_type,
                screenshot_path=screenshot_path
            )
            
            logging.info(f"Click test result: {result}")
            self.results.append(result)
            self.total_tests += 1
            if success:
                self.successful_tests += 1
                
            return result
            
        except Exception as e:
            logging.error(f"Click accuracy test failed: {e}", exc_info=True)
            return ClickTestResult(
                target_coordinates=coordinates,
                actual_coordinates=(0, 0),
                offset=(0, 0),
                success=False,
                time_taken=time.time() - start_time,
                element_type=element_type,
                error_message=str(e)
            )

    def _save_accuracy_visualization(
        self, target: Tuple[int, int], 
        actual: Tuple[int, int], 
        offset: Tuple[float, float],
        success: bool
    ) -> str:
        """
        Save visualization of click accuracy test.
        """
        try:
            # Get current screen image
            screen_image = self.eyes.get_screen_image()
            if screen_image is None:
                return None
                
            # Create visualization
            vis_image = screen_image.copy()
            draw = ImageDraw.Draw(vis_image)
            
            # Draw target point (green)
            draw.ellipse(
                [target[0]-5, target[1]-5, target[0]+5, target[1]+5],
                outline='green', width=2
            )
            
            # Draw actual point (red if failed, blue if success)
            color = 'blue' if success else 'red'
            draw.ellipse(
                [actual[0]-5, actual[1]-5, actual[0]+5, actual[1]+5],
                outline=color, width=2
            )
            
            # Draw line between points
            draw.line([target[0], target[1], actual[0], actual[1]], 
                     fill=color, width=1)
            
            # Add text with offset information
            font = ImageFont.load_default()
            text = f"Offset: ({offset[0]:.1f}, {offset[1]:.1f}) pixels"
            draw.text((10, 10), text, fill=color, font=font)
            
            # Save image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.debug_path / f"accuracy_{timestamp}.png"
            vis_image.save(filepath)
            
            return str(filepath)
            
        except Exception as e:
            logging.error(f"Error saving accuracy visualization: {e}")
            return None

    def run_test_suite(self, test_elements: List[Dict]) -> Dict:
        """
        Run a full suite of click accuracy tests.
        
        Args:
            test_elements: List of elements to test with coordinates and types
        """
        results = {
            'total_tests': 0,
            'successful_tests': 0,
            'average_offset': (0.0, 0.0),
            'average_confidence': 0.0,
            'average_time': 0.0,
            'results_by_type': {},
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            for element in test_elements:
                for _ in range(self.config['test_iterations']):
                    result = self.test_click_accuracy(
                        element['type'],
                        element['coordinates']
                    )
                    
                    # Update type-specific results
                    if element['type'] not in results['results_by_type']:
                        results['results_by_type'][element['type']] = {
                            'tests': 0,
                            'successes': 0,
                            'average_offset': (0.0, 0.0),
                            'average_confidence': 0.0
                        }
                    
                    type_results = results['results_by_type'][element['type']]
                    type_results['tests'] += 1
                    if result.success:
                        type_results['successes'] += 1
                    
                    # Update averages
                    type_results['average_offset'] = (
                        (type_results['average_offset'][0] * (type_results['tests'] - 1) + 
                         result.offset[0]) / type_results['tests'],
                        (type_results['average_offset'][1] * (type_results['tests'] - 1) + 
                         result.offset[1]) / type_results['tests']
                    )
                    type_results['average_confidence'] = (
                        (type_results['average_confidence'] * (type_results['tests'] - 1) + 
                         result.confidence) / type_results['tests']
                    )
            
            # Calculate overall results
            results['total_tests'] = self.total_tests
            results['successful_tests'] = self.successful_tests
            results['success_rate'] = self.successful_tests / max(1, self.total_tests)
            
            # Save detailed results
            self._save_test_results(results)
            
            return results
            
        except Exception as e:
            logging.error(f"Error running test suite: {e}")
            results['error'] = str(e)
            return results

    def _save_test_results(self, results: Dict):
        """Save test results to JSON file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.debug_path / f"results_{timestamp}.json"
            
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2)
                
            logging.info(f"Saved test results to {filepath}")
            
        except Exception as e:
            logging.error(f"Error saving test results: {e}")


if __name__ == "__main__":
    # Initialize pipeline
    computer = Computer()
    computer.run()
    hands = Hands(computer.mouse, computer.keyboard)
    pipeline = HandClickingAccuracyPipeline(
        hands=hands,
        computer=computer,
        config={
            'accuracy_threshold': 3,  # Stricter accuracy requirement
            'movement_speed': 1.0,
            'screen_bounds': (1920, 1080)
        }
    )

    # Run some tests
    result = pipeline.test_click_accuracy('button', (100, 200))

    # Get summary stats
    stats = pipeline.get_summary_stats()
    print(f"Success rate: {stats['success_rate']:.2%}")
    print(f"Average offset: {stats['average_offset']}")

    