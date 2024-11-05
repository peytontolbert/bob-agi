"""
Pipeline for visual accuracy testing and element detection with multi-turn conversation and visualization
"""
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
import time
import logging
import os
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

@dataclass
class DetectionResult:
    """Result from visual element detection"""
    success: bool
    coordinates: Optional[Tuple[int, int]] = None
    confidence: float = 0.0
    bbox: Optional[Tuple[int, int, int, int]] = None
    error_message: str = ""
    description: str = ""

class VisualAccuracyPipeline:
    def __init__(self, vision_agent, eyes):
        self.vision_agent = vision_agent
        self.eyes = eyes
        self.results = []
        self.conversation_history = []
        
        # Create directory for saving visualizations
        self.output_dir = "debug_output/element_detection"
        os.makedirs(self.output_dir, exist_ok=True)

    def _save_detection_visualization(self, image, detection: DetectionResult, attempt: int):
        """
        Save visualization of detection result with bounding box and metadata.
        
        Args:
            image: PIL Image of the screen
            detection: DetectionResult containing coordinates and metadata
            attempt: Current attempt number
        """
        try:
            # Create copy of image for drawing
            vis_image = image.copy()
            draw = ImageDraw.Draw(vis_image)
            
            # Draw bounding box if available
            if detection.bbox:
                x1, y1, x2, y2 = detection.bbox
                # Draw red rectangle for bbox
                draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
                
                # Draw center point if available
                if detection.coordinates:
                    cx, cy = detection.coordinates
                    radius = 5
                    draw.ellipse([cx-radius, cy-radius, cx+radius, cy+radius], 
                               fill='blue')
                
                # Add text with detection info
                text_y = y1 - 20
                font = ImageFont.load_default()
                
                # Draw text with background for better visibility
                text = f"Confidence: {detection.confidence:.2f}"
                text_bbox = draw.textbbox((x1, text_y), text, font=font)
                draw.rectangle(text_bbox, fill='white')
                draw.text((x1, text_y), text, fill='black', font=font)
                
                # Add description
                if detection.description:
                    desc_y = text_y - 20
                    desc_bbox = draw.textbbox((x1, desc_y), detection.description, font=font)
                    draw.rectangle(desc_bbox, fill='white')
                    draw.text((x1, desc_y), detection.description, fill='black', font=font)
            
            # Save image with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detection_{timestamp}_attempt_{attempt}.png"
            filepath = os.path.join(self.output_dir, filename)
            vis_image.save(filepath)
            
            logging.info(f"Saved detection visualization to {filepath}")
            return filepath
            
        except Exception as e:
            logging.error(f"Error saving detection visualization: {e}")
            return None

    def _adjust_coordinates_with_feedback(self, image, detection: DetectionResult, click_result: bool) -> Optional[DetectionResult]:
        """
        Use multi-turn conversation to adjust coordinates after failed click.
        
        Args:
            image: Screen image
            detection: Previous detection result
            click_result: Whether the click succeeded
            
        Returns:
            Optional[DetectionResult]: Adjusted detection or None if adjustment fails
        """
        try:
            if click_result:
                return detection
                
            # Build adjustment query with previous coordinates
            x1, y1, x2, y2 = detection.bbox if detection.bbox else (0, 0, 0, 0)
            cx, cy = detection.coordinates if detection.coordinates else (0, 0)
            
            adjust_query = f"""The click at coordinates {(cx, cy)} failed for element: "{detection.description}"
            Previous bounding box was: ({x1}, {y1}, {x2}, {y2})
            
            Please analyze the element again and:
            1. Verify if the element is still visible
            2. Check if the element might be partially obscured or in a different state
            3. Provide adjusted coordinates that ensure clicking the interactive part of the element
            4. Consider if the element needs to be clicked in a specific location (e.g. center, edges)
            
            Provide new coordinates as: coordinates: (x1, y1, x2, y2)
            """
            
            # Get adjusted coordinates from vision agent
            result = self.vision_agent.understand_scene(image, adjust_query)
            
            # Parse new coordinates
            new_coords = self.vision_agent._parse_coordinates(result.get('description', ''))
            
            if new_coords:
                x1, y1, x2, y2 = new_coords
                return DetectionResult(
                    success=True,
                    coordinates=((x1 + x2) // 2, (y1 + y2) // 2),
                    bbox=(x1, y1, x2, y2),
                    confidence=detection.confidence * 0.9,  # Slightly reduce confidence
                    description=detection.description
                )
                
            return None
            
        except Exception as e:
            logging.error(f"Error adjusting coordinates: {e}")
            return None

    def find_element_with_retry(self, search_query: str, confidence_threshold: float = 0.7, max_retries: int = 3) -> Optional[DetectionResult]:
        """
        Find element using multi-turn conversation for improved accuracy.
        """
        previous_attempts = []
        last_detection = None
        click_attempted = False
        
        for attempt in range(max_retries):
            try:
                screen_image = self.eyes.get_screen_image()
                if screen_image is None:
                    logging.warning(f"Failed to get screen image on attempt {attempt + 1}")
                    continue

                # If we have a failed click, try to adjust coordinates
                if click_attempted and last_detection:
                    detection = self._adjust_coordinates_with_feedback(
                        screen_image, 
                        last_detection,
                        False  # Click failed
                    )
                    if detection:
                        # Save visualization of adjusted coordinates
                        vis_path = self._save_detection_visualization(
                            screen_image, detection, f"{attempt + 1}_adjusted"
                        )
                        if vis_path:
                            detection.visualization_path = vis_path
                        return detection
                else:
                    # Normal detection flow
                    refined_query = self._build_element_query(search_query, previous_attempts)
                    result = self.vision_agent.find_element(
                        image=screen_image,
                        target_description=refined_query,
                        confidence_threshold=confidence_threshold
                    )
                    
                    if result and result.get('element_found', False):
                        detection = DetectionResult(
                            success=True,
                            coordinates=result['element_details']['coordinates'],
                            confidence=result['element_details']['confidence'],
                            bbox=result['element_details'].get('bbox'),
                            description=search_query
                        )
                        
                        # Save visualization
                        vis_path = self._save_detection_visualization(
                            screen_image, detection, attempt + 1
                        )
                        if vis_path:
                            detection.visualization_path = vis_path
                        
                        # Store last detection for potential adjustment
                        last_detection = detection
                        
                        if self._verify_detection(screen_image, detection, search_query):
                            return detection
                
                # Store attempt information
                previous_attempts.append({
                    'description': result.get('message', 'Element not found'),
                    'attempt': attempt + 1,
                    'click_attempted': click_attempted
                })
                
                # Save failed attempt visualization
                if screen_image:
                    failed_detection = DetectionResult(
                        success=False,
                        description=f"Failed attempt {attempt + 1}: {result.get('message', '')}"
                    )
                    self._save_detection_visualization(screen_image, failed_detection, attempt + 1)
                
                if attempt < max_retries - 1:
                    time.sleep(1.0)
                    
            except Exception as e:
                logging.warning(f"Error in find_element_with_retry (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(1.0)
                continue
                
        return None

    def _verify_detection(self, image, detection: DetectionResult, search_query: str) -> bool:
        """
        Verify detection with enhanced click-point analysis.
        """
        try:
            # Build verification query with click-point analysis
            verify_query = f"""Verify if these coordinates: {detection.coordinates} 
            correspond to the element: "{search_query}"
            Bounding box: {detection.bbox}
            
            Check:
            1. Is this exactly the element we're looking for?
            2. Are the coordinates precise for clicking?
            3. Is the click point ({detection.coordinates[0]}, {detection.coordinates[1]}) 
               in the best location for interaction?
            4. Are there any overlapping elements that might interfere?
            
            Respond with 'VERIFIED' if correct, or explain what needs adjustment."""

            verification = self.vision_agent.understand_scene(image, verify_query)
            
            if verification and 'VERIFIED' in verification.get('description', '').upper():
                return True
                
            logging.info(f"Detection not verified: {verification.get('description', 'Unknown reason')}")
            return False
            
        except Exception as e:
            logging.error(f"Error in detection verification: {e}")
            return False

    def verify_screen_state(self, expected_state: str, confidence_threshold: float = 0.6) -> DetectionResult:
        """
        Verify screen state with improved accuracy.
        """
        try:
            screen_image = self.eyes.get_screen_image()
            
            # Build detailed verification query
            verify_query = f"""Analyze the current screen state:
            Expected state: {expected_state}
            
            Please verify:
            1. Does the screen match the expected state?
            2. What elements confirm this state?
            3. Are there any inconsistencies?
            
            Provide a detailed response about the screen state."""

            scene_result = self.vision_agent.understand_scene(screen_image, verify_query)
            
            if scene_result['status'] == 'success':
                description = scene_result['description'].lower()
                expected_state_lower = expected_state.lower()
                
                # Check for state confirmation
                state_confirmed = any(keyword in description for keyword in expected_state_lower.split())
                
                if state_confirmed:
                    return DetectionResult(
                        success=True,
                        confidence=0.8,
                        description=scene_result['description']
                    )
                    
            return DetectionResult(
                success=False,
                error_message=f"Screen state '{expected_state}' not verified",
                description=scene_result.get('description', '')
            )
            
        except Exception as e:
            logging.error(f"Error verifying screen state: {e}")
            return DetectionResult(
                success=False,
                error_message=str(e)
            )

    def _save_test_results(self, results: Dict[str, Any]):
        """Save test results with enhanced metadata and visualizations"""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        # Create test results directory
        test_dir = os.path.join(self.output_dir, f"test_{timestamp}")
        os.makedirs(test_dir, exist_ok=True)
        
        # Save metadata
        enhanced_results = {
            **results,
            'timestamp': timestamp,
            'conversation_history': self.conversation_history.copy()
        }
        
        # Save results to JSON
        import json
        results_path = os.path.join(test_dir, "results.json")
        with open(results_path, 'w') as f:
            json.dump(enhanced_results, f, indent=2)
        
        self.results.append(enhanced_results)

    def _build_element_query(self, search_query: str, previous_attempts: List[Dict] = None) -> str:
        """
        Builds a refined query based on search history and previous attempts.
        """
        if not previous_attempts:
            return f"""Find this specific UI element: {search_query}
            Describe its location and provide exact coordinates as: coordinates: (x1, y1, x2, y2)
            Focus on visual characteristics and precise positioning."""
            
        # Build context from previous attempts
        context = "\n".join([
            f"Previous attempt found: {attempt.get('description', 'unknown location')}"
            for attempt in previous_attempts
        ])
        
        return f"""Find this specific UI element: {search_query}
        Previous attempts:
        {context}
        
        Please provide:
        1. Exact coordinates in format: coordinates: (x1, y1, x2, y2)
        2. Clear description of element location
        3. Visual characteristics that identify this element
        
        Focus on finding the exact element, not similar ones."""

