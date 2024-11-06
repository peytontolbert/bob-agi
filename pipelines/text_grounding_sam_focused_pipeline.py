from app.agents.vision import VisionAgent
from app.agents.text import TextAgent
from app.env.computer.computer import Computer
from app.models.grounding_dino import GroundingDINO
import numpy as np
import torch
import time
from typing import Optional, Tuple, List
from pathlib import Path
import cv2
import os

class TextGroundingSAMFocusedPipeline:
    def __init__(self):
        self.computer = Computer()
        self.computer.run()
        self.screen = self.computer.screen
        self.mouse = self.computer.mouse
        
        # Initialize agents and models
        self.vision_agent = VisionAgent()
        self.text_agent = TextAgent()
        self.sam2 = self.vision_agent.sam2
        
        # Initialize GroundingDINO
        self.grounding_dino = GroundingDINO()

        # Create output directory for visualizations
        self.output_dir = Path("output/visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_initial_guidance(self, image: np.ndarray) -> dict:
        """Get text guidance from LLM about where to look for the button and expected visual properties."""
        prompt = """
        Analyze this screenshot to help find the 'Continue in Browser' button/link.
        Provide guidance in the following format:
        1. Location: Describe where the button is likely located (e.g., bottom right, center of page)
        2. Visual Description: Describe its appearance (color, shape, size, etc.)
        3. Surrounding Context: Describe nearby elements that can help identify it
        4. Suggested Search Terms: List 2-3 precise terms to help identify this element
        
        Format your response as a JSON with keys: location, visual_desc, context, search_terms
        """
        
        response = self.text_agent.complete_task({
            "query": prompt,
            "image": image
        })
        
        # Assuming response is in JSON format
        return response

    def save_visualization(self, image: np.ndarray, boxes: List[List[float]], filename: str):
        """Save visualization of bounding boxes."""
        vis_image = image.copy()
        for box in boxes:
            x1, y1, x2, y2 = [int(coord) for coord in box]
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        output_path = self.output_dir / filename
        cv2.imwrite(str(output_path), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        return output_path

    def save_mask_visualization(self, image: np.ndarray, mask: np.ndarray, filename: str):
        """Save visualization of SAM mask."""
        vis_image = image.copy()
        mask_overlay = np.zeros_like(vis_image)
        mask_overlay[mask] = [0, 255, 0]  # Green overlay
        vis_image = cv2.addWeighted(vis_image, 0.7, mask_overlay, 0.3, 0)
        
        output_path = self.output_dir / filename
        cv2.imwrite(str(output_path), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        return output_path

    def find_continue_button(self) -> Optional[Tuple[int, int]]:
        """Main pipeline to find the 'Continue in Browser' button."""
        screen_state = self.screen.capture()
        if not screen_state or 'frame' not in screen_state:
            return None
            
        image = np.array(screen_state['frame'])
        timestamp = int(time.time())
        
        # Get detailed guidance from LLM
        guidance = self.get_initial_guidance(image)
        
        # Use multiple search terms from the LLM's guidance
        all_boxes = []
        for search_term in guidance['search_terms']:
            results = self.grounding_dino.predict_with_caption(
                image=image,
                caption=search_term
            )
            if results:
                all_boxes.extend(results)
        
        if all_boxes:
            # Save Grounding DINO visualization
            self.save_visualization(
                image, 
                all_boxes, 
                f"dino_boxes_{timestamp}.png"
            )
        else:
            return None
        
        # Ask text agent to evaluate each candidate region
        prompt = f"""
        Analyze this region of the screenshot.
        Context: {guidance['location']}
        Expected appearance: {guidance['visual_desc']}
        Nearby elements: {guidance['context']}
        
        Is this the 'Continue in Browser' button/link? Rate confidence 0-100.
        """
        
        best_box = None
        highest_confidence = 0
        
        for box in all_boxes:
            x1, y1, x2, y2 = [int(coord) for coord in box]
            response = self.text_agent.complete_task({
                "query": prompt,
                "image": image[y1:y2, x1:x2]
            })
            
            # Extract confidence score from response
            try:
                confidence = float(response.strip().split()[-1])
                if confidence > highest_confidence:
                    highest_confidence = confidence
                    best_box = box
            except:
                continue
        
        if not best_box or highest_confidence < 50:  # Confidence threshold
            return None
            
        # Use SAM2 for precise segmentation
        center_x = int((best_box[0] + best_box[2]) / 2)
        center_y = int((best_box[1] + best_box[3]) / 2)
        
        with torch.inference_mode():
            self.sam2.set_image(image)
            masks, scores, _ = self.sam2.predict(
                point_coords=np.array([[center_x, center_y]]),
                point_labels=np.array([1]),
                box=np.array(best_box),
                multimask_output=True
            )
        
        if not masks:
            return None
            
        # Save SAM visualization
        best_mask = masks[0]
        self.save_mask_visualization(
            image,
            best_mask,
            f"sam_mask_{timestamp}.png"
        )
        
        # Get center of best mask
        indices = np.where(best_mask)
        if len(indices[0]) == 0:
            return None
            
        target_x = int(np.mean(indices[1]))
        target_y = int(np.mean(indices[0]))
        
        return target_x, target_y

    def run(self) -> bool:
        """Run the pipeline and click the button if found."""
        target_position = self.find_continue_button()
        
        if target_position is None:
            return False
            
        # Move to target and click
        self.mouse.move_to(*target_position)
        time.sleep(0.5)  # Brief pause before clicking
        self.mouse.click()
        
        return True

def main():
    pipeline = TextGroundingSAMFocusedPipeline()
    success = pipeline.run()
    
    if success:
        print("Successfully found and clicked 'Continue in Browser'")
    else:
        print("Failed to find 'Continue in Browser'")

if __name__ == "__main__":
    main()
