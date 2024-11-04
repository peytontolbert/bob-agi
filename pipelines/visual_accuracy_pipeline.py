"""
Pipeline for visual accuracy testing and element detection
"""
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import time
import logging

@dataclass
class DetectionResult:
    """Result from visual element detection"""
    success: bool
    coordinates: Optional[Tuple[int, int]] = None
    confidence: float = 0.0
    bbox: Optional[Tuple[int, int, int, int]] = None
    error_message: str = ""

class VisualAccuracyPipeline:
    def __init__(self, vision_agent, eyes):
        self.vision_agent = vision_agent
        self.eyes = eyes
        self.results = []

    def find_element_with_retry(self, search_query: str, confidence_threshold: float = 0.7, max_retries: int = 3) -> Optional[DetectionResult]:
        """
        Find an element with retries and confidence threshold.
        
        Args:
            search_query: Description of element to find
            confidence_threshold: Minimum confidence threshold
            max_retries: Maximum number of retry attempts
            
        Returns:
            DetectionResult object or None if not found
        """
        for attempt in range(max_retries):
            try:
                # Get current screen image
                screen_image = self.eyes.get_screen_image()
                
                # Use vision agent to find element
                result = self.vision_agent.find_element(
                    screen_image,
                    search_query,
                    similarity_threshold=confidence_threshold
                )
                
                if result.get('element_found', False):
                    element = result['element_details']
                    return DetectionResult(
                        success=True,
                        coordinates=element['center'],
                        confidence=element['confidence'],
                        bbox=element['bbox']
                    )
                    
                if attempt < max_retries - 1:
                    time.sleep(1.0)  # Wait before retry
                    
            except Exception as e:
                logging.warning(f"Error in find_element_with_retry (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(1.0)
                continue
                
        return None

    def verify_screen_state(self, expected_state: str, confidence_threshold: float = 0.6) -> DetectionResult:
        """
        Verify if screen matches expected state
        
        Args:
            expected_state: Description of expected screen state
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            DetectionResult indicating verification success/failure
        """
        try:
            # Get current screen image
            screen_image = self.eyes.get_screen_image()
            
            # Use vision agent to understand scene
            scene_result = self.vision_agent.understand_scene(
                screen_image,
                f"Verify if the screen shows: {expected_state}"
            )
            
            if scene_result['status'] == 'success':
                description = scene_result['description'].lower()
                expected_state_lower = expected_state.lower()
                
                # Check if expected state is described in scene understanding
                if any(keyword in description for keyword in expected_state_lower.split()):
                    return DetectionResult(
                        success=True,
                        confidence=0.8  # Default confidence for successful verification
                    )
                    
            return DetectionResult(
                success=False,
                error_message=f"Screen state '{expected_state}' not verified"
            )
            
        except Exception as e:
            logging.error(f"Error verifying screen state: {e}")
            return DetectionResult(
                success=False,
                error_message=str(e)
            )

    def _save_test_results(self, results: Dict[str, Any]):
        """Save test results for analysis"""
        self.results.append(results)

