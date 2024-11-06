from app.agents.vision import VisionAgent
from app.agents.text import TextAgent
from app.env.computer.computer import Computer
from app.models.sam2.sam2.build_sam import build_sam2
from app.models.sam2.sam2.sam2_image_predictor import SAM2ImagePredictor
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
import cv2
from app.env.controllers.nlp_mouse_controller import NLPMouseController
from typing import List, Tuple, Optional
import time


np.random.seed(3)

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()
        plt.close()

class Test:
    def __init__(self):
        self.computer = Computer()
        self.computer.run()
        self.screen = self.computer.screen
        self.mouse = self.computer.mouse
        self.nlp_mouse = NLPMouseController(self.mouse, self.screen)
        self.vision_agent = VisionAgent()
        self.sam2 = self.vision_agent.sam2
        self.text_agent = TextAgent()
        sam2_checkpoint = "./weights/sam2.1_hiera_large.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

        self.predictor = SAM2ImagePredictor(build_sam2(model_cfg, sam2_checkpoint))
        self.last_successful_mask = None
        self.search_grid = self.generate_search_grid()
        
        # Move to center of screen on initialization
        self.move_to_screen_center()

        self.search_patterns = [
            self.systematic_grid_search,
            self.smart_region_search,
            self.vision_guided_search
        ]
        
    def generate_search_grid(self, step_size=50):
        """Generate a grid of coordinates to search"""
        coords = []
        for x in range(0, self.screen.width, step_size):
            for y in range(0, self.screen.height, step_size):
                coords.append((x, y))
        return coords
        
    def verify_mouse_movement(self, target_x, target_y, tolerance=5):
        """Verify that the mouse moved to the intended position"""
        current_x, current_y = self.mouse.get_position()
        distance = ((current_x - target_x) ** 2 + (current_y - target_y) ** 2) ** 0.5
        if distance > tolerance:
            print(f"Warning: Mouse position ({current_x}, {current_y}) differs from target ({target_x}, {target_y})")
            return False
        return True

    def move_to_screen_center(self):
        """Move the mouse to the center of the screen with verification"""
        center_x = self.screen.width // 2
        center_y = self.screen.height // 2
        print(f"Moving to screen center: ({center_x}, {center_y})")
        
        # Try direct movement first
        self.mouse.move_to(center_x, center_y)
        
        # Verify movement
        if not self.verify_mouse_movement(center_x, center_y):
            print("Attempting movement with NLP controller...")
            # Try using NLP controller as backup
            self.nlp_mouse.execute_command("move to center")
            
        current_pos = self.mouse.get_position()
        print(f"Final position: {current_pos}")

    def analyze_screen_layout(self, image) -> str:
        """Use LLM to analyze the screen layout and suggest search strategy."""
        prompt = """
        Analyze this screenshot and suggest the best strategy to find a 'Continue in Browser' link.
        Consider:
        1. Common UI patterns
        2. Typical button/link locations
        3. Visual hierarchy
        
        Reply with one of: 'top-down', 'bottom-up', 'center-out', 'edges-in'
        """
        
        response = self.text_agent.complete_task({
            "query": prompt,
            "image": image
        })
        return response.lower().strip()

    def smart_region_search(self) -> bool:
        """Use LLM-guided search to find the target."""
        screen_state = self.screen.capture()
        if not screen_state or 'frame' not in screen_state:
            return False
            
        image = np.array(screen_state['frame'])
        
        # Get search strategy from LLM
        strategy = self.analyze_screen_layout(image)
        
        # Define search regions based on strategy
        regions = {
            'top-down': [(0, 0, self.screen.width, self.screen.height//2),
                        (0, self.screen.height//2, self.screen.width, self.screen.height)],
            'bottom-up': [(0, self.screen.height//2, self.screen.width, self.screen.height),
                         (0, 0, self.screen.width, self.screen.height//2)],
            'center-out': [(self.screen.width//4, self.screen.height//4, 
                           3*self.screen.width//4, 3*self.screen.height//4),
                          (0, 0, self.screen.width, self.screen.height)],
            'edges-in': [(0, 0, self.screen.width//4, self.screen.height),
                        (3*self.screen.width//4, 0, self.screen.width, self.screen.height)]
        }
        
        for region in regions.get(strategy, []):
            if self.search_region(region, image):
                return True
        return False

    def vision_guided_search(self) -> bool:
        """Enhanced vision-guided search combining LLMs, SAM2, and NLP control."""
        screen_state = self.screen.capture()
        if not screen_state or 'frame' not in screen_state:
            return False
            
        image = np.array(screen_state['frame'])

        # Step 1: Get high-level visual analysis from LLM
        analysis_prompt = """
        Analyze this screenshot and describe where the 'Continue in Browser' link might be located.
        Consider:
        1. Visual layout and hierarchy
        2. Common UI patterns
        3. Distinctive visual elements
        
        Describe the location in terms of screen regions and visual landmarks.
        """
        
        initial_analysis = self.text_agent.complete_task({
            "query": analysis_prompt,
            "image": image
        })

        # Step 2: Generate region proposals based on LLM analysis
        region_prompt = f"""
        Based on this analysis: '{initial_analysis}'
        
        Give me a sequence of 3-4 screen regions to check, in order of likelihood.
        Format each region as: x1,y1,x2,y2 (as percentages of screen width/height)
        Example: "25,30,75,60" means check from 25% to 75% width and 30% to 60% height
        Respond with one region per line, nothing else.
        """
        
        region_response = self.text_agent.complete_task({
            "query": region_prompt,
            "image": image
        })

        # Parse and check each suggested region
        for region_str in region_response.strip().split('\n'):
            try:
                x1, y1, x2, y2 = map(int, region_str.split(','))
                region = (
                    int(x1 * self.screen.width / 100),
                    int(y1 * self.screen.height / 100),
                    int(x2 * self.screen.width / 100),
                    int(y2 * self.screen.height / 100)
                )
                if self.search_region_with_sam(region, image):
                    return True
            except ValueError:
                continue

        return False

    def search_region_with_sam(self, region: Tuple[int, int, int, int], image: np.ndarray) -> bool:
        """Search a specific region using SAM2 and LLM guidance."""
        x1, y1, x2, y2 = region
        
        # Move to region center first
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        self.mouse.move_to(center_x, center_y)

        # Generate masks for the region
        with torch.inference_mode():
            self.predictor.set_image(image)
            
            # Use point prompt at region center
            point_coords = np.array([[center_x, center_y]])
            point_labels = np.array([1])
            
            masks, scores, _ = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=np.array([x1, y1, x2, y2]),
                multimask_output=True
            )

        # Filter and analyze masks
        for mask, score in zip(masks, scores):
            if score < 0.7:  # Skip low confidence masks
                continue
                
            # Get mask properties
            mask_center = self.get_mask_center(mask)
            if not mask_center:
                continue

            # Ask LLM to analyze the masked region
            analysis_prompt = """
            Look at the masked region near the mouse cursor.
            Does this look like it could be a 'Continue in Browser' link?
            Consider:
            1. Size and shape (typical for a link/button)
            2. Position in layout
            3. Surrounding context
            
            Reply with: 'likely', 'possible', or 'unlikely'
            """
            
            # Move to mask center for better context
            self.mouse.move_to(*mask_center)
            
            # Capture current view for analysis
            current_view = self.screen.capture()
            if not current_view:
                continue

            response = self.text_agent.complete_task({
                "query": analysis_prompt,
                "image": current_view['frame']
            })

            if response.strip() in ['likely', 'possible']:
                # Try fine-tuning the position
                if self.refine_position(mask_center):
                    return True
                    
        return False

    def refine_position(self, initial_pos: Tuple[int, int]) -> bool:
        """Fine-tune the mouse position using small movements and verification."""
        x, y = initial_pos
        
        # Define search pattern (spiral out from center)
        offsets = [(0, 0), (0, -5), (5, 0), (0, 5), (-5, 0),
                  (-5, -5), (5, -5), (5, 5), (-5, 5)]
        
        for dx, dy in offsets:
            self.mouse.move_to(x + dx, y + dy)
            if self.verify_target():
                return True
                
            # Try using NLP controller for micro-adjustments
            for direction in ['up', 'down', 'left', 'right']:
                self.nlp_mouse.execute_command(f"move {direction} tiny")
                if self.verify_target():
                    return True
                    
        return False

    def verify_target(self) -> bool:
        """Enhanced target verification using multiple checks."""
        screen_state = self.screen.capture()
        if not screen_state or 'frame' not in screen_state:
            return False
            
        # Multiple verification prompts for robustness
        prompts = [
            "Is there a 'Continue in Browser' link or button exactly at the mouse cursor position?",
            "Would clicking at the current mouse position activate a 'Continue in Browser' action?",
            "Is the mouse cursor directly over text that says 'Continue in Browser'?"
        ]
        
        positive_count = 0
        for prompt in prompts:
            response = self.text_agent.complete_task({
                "query": prompt,
                "image": screen_state['frame']
            })
            if response.lower().strip() == 'yes':
                positive_count += 1
                
        # Require at least 2 positive verifications
        return positive_count >= 2

    def run_test(self) -> bool:
        """Try different search strategies until target is found."""
        for search_method in self.search_patterns:
            if search_method():
                return True
            time.sleep(0.5)  # Brief pause between strategies
        return False

    def systematic_grid_search(self) -> bool:
        """Perform a systematic grid-based search of the screen."""
        screen_state = self.screen.capture()
        if not screen_state or 'frame' not in screen_state:
            return False
            
        image = np.array(screen_state['frame'])
        
        # Create a more refined grid for systematic search
        grid_size = 30  # pixels between grid points
        rows = self.screen.height // grid_size
        cols = self.screen.width // grid_size
        
        # Define search patterns (spiral from center)
        center_row, center_col = rows // 2, cols // 2
        spiral_coords = self.generate_spiral_coordinates(rows, cols, center_row, center_col)
        
        for row, col in spiral_coords:
            x = col * grid_size
            y = row * grid_size
            
            # Skip points outside screen bounds
            if not (0 <= x < self.screen.width and 0 <= y < self.screen.height):
                continue
                
            # Move to grid position
            self.mouse.move_to(x, y)
            
            # Generate and analyze mask at current position
            with torch.inference_mode():
                self.predictor.set_image(image)
                point_coords = np.array([[x, y]])
                point_labels = np.array([1])
                
                masks, scores, _ = self.predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=True
                )
                
                # Check each mask
                for mask, score in zip(masks, scores):
                    if score < 0.7:
                        continue
                        
                    mask_center = self.get_mask_center(mask)
                    if not mask_center:
                        continue
                        
                    # Move to mask center and verify
                    self.mouse.move_to(*mask_center)
                    if self.verify_target():
                        return True
                        
            # Brief pause to prevent system overload
            time.sleep(0.1)
            
        return False

    def generate_spiral_coordinates(self, rows: int, cols: int, start_row: int, start_col: int) -> List[Tuple[int, int]]:
        """Generate coordinates in a spiral pattern from the center."""
        coordinates = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
        
        current_row, current_col = start_row, start_col
        step_size = 1
        direction_idx = 0
        
        coordinates.append((current_row, current_col))
        
        while len(coordinates) < rows * cols:
            # Take steps in current direction
            for _ in range(2):  # Do each direction length twice before increasing
                for _ in range(step_size):
                    dr, dc = directions[direction_idx]
                    current_row += dr
                    current_col += dc
                    
                    if 0 <= current_row < rows and 0 <= current_col < cols:
                        coordinates.append((current_row, current_col))
                        
                direction_idx = (direction_idx + 1) % 4
                
            step_size += 1
            
        return coordinates

    def get_mask_center(self, mask) -> Optional[Tuple[int, int]]:
        """Calculate the center of mass for a mask."""
        if not mask.any():
            return None
        indices = np.where(mask)
        return (int(np.mean(indices[1])), int(np.mean(indices[0])))

def main():
    test = Test()
    found = test.run_test()
    if found:
        print("Successfully found and verified the 'Continue in Browser' link")
    else:
        print("Could not find the 'Continue in Browser' link")

if __name__ == "__main__":
    main()