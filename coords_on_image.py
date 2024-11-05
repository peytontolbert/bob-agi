import cv2
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import json
import os
from datetime import datetime

class CoordinateVisualizer:
    def __init__(self):
        self.image = None
        self.original_image = None
        self.coords = (0, 0, 0, 0)  # (x1, y1, x2, y2)
        self.window_name = "Coordinate Visualizer"
        self.is_drawing = False
        self.current_coord_index = None
        self.coord_radius = 5
        
        # Create output directory
        self.output_dir = "coordinate_outputs"
        os.makedirs(self.output_dir, exist_ok=True)

    def load_image(self, image_path=None):
        """Load an image from file or file dialog"""
        if image_path is None:
            root = tk.Tk()
            root.withdraw()  # Hide the main window
            image_path = filedialog.askopenfilename(
                title="Select Image",
                filetypes=[
                    ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif"),
                    ("All files", "*.*")
                ]
            )
            if not image_path:
                return False

        try:
            # Load image with PIL first to handle different formats
            pil_image = Image.open(image_path)
            # Convert to RGB if necessary
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            # Convert to OpenCV format
            self.original_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            self.image = self.original_image.copy()
            return True
        except Exception as e:
            print(f"Error loading image: {e}")
            return False

    def draw_coordinates(self):
        """Draw the coordinate box and points on the image"""
        if self.image is None:
            return

        # Create a fresh copy of the original image
        self.image = self.original_image.copy()
        x1, y1, x2, y2 = self.coords

        # Draw the bounding box
        cv2.rectangle(self.image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw the coordinate points
        points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # Different colors for each corner
        
        for (x, y), color in zip(points, colors):
            cv2.circle(self.image, (x, y), self.coord_radius, color, -1)

        # Add coordinate text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.image, f"({x1}, {y1})", (x1, y1-10), font, 0.5, (255, 255, 255), 2)
        cv2.putText(self.image, f"({x2}, {y2})", (x2, y2+20), font, 0.5, (255, 255, 255), 2)

        # Display center point
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        cv2.circle(self.image, (center_x, center_y), 3, (0, 255, 255), -1)
        cv2.putText(self.image, f"Center: ({center_x}, {center_y})", 
                    (center_x + 10, center_y), font, 0.5, (0, 255, 255), 2)

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for coordinate adjustment"""
        if self.image is None:
            return

        x1, y1, x2, y2 = self.coords
        points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if clicking near any corner point
            for i, (px, py) in enumerate(points):
                if abs(x - px) < self.coord_radius and abs(y - py) < self.coord_radius:
                    self.is_drawing = True
                    self.current_coord_index = i
                    return

        elif event == cv2.EVENT_MOUSEMOVE and self.is_drawing:
            # Update coordinates based on which corner is being dragged
            if self.current_coord_index == 0:  # Top-left
                self.coords = (x, y, x2, y2)
            elif self.current_coord_index == 1:  # Top-right
                self.coords = (x1, y, x, y2)
            elif self.current_coord_index == 2:  # Bottom-right
                self.coords = (x1, y1, x, y)
            elif self.current_coord_index == 3:  # Bottom-left
                self.coords = (x, y1, x2, y)
            
            self.draw_coordinates()

        elif event == cv2.EVENT_LBUTTONUP:
            self.is_drawing = False
            self.current_coord_index = None

    def save_coordinates(self):
        """Save the current coordinates and annotated image"""
        if self.image is None:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save annotated image
        image_filename = os.path.join(self.output_dir, f"annotated_{timestamp}.png")
        cv2.imwrite(image_filename, self.image)

        # Save coordinates to JSON
        coords_data = {
            "coordinates": {
                "x1": self.coords[0],
                "y1": self.coords[1],
                "x2": self.coords[2],
                "y2": self.coords[3]
            },
            "center": {
                "x": (self.coords[0] + self.coords[2]) // 2,
                "y": (self.coords[1] + self.coords[3]) // 2
            },
            "timestamp": timestamp
        }

        json_filename = os.path.join(self.output_dir, f"coordinates_{timestamp}.json")
        with open(json_filename, 'w') as f:
            json.dump(coords_data, f, indent=2)

        print(f"Saved annotated image to: {image_filename}")
        print(f"Saved coordinates to: {json_filename}")

    def run(self, initial_coords=None):
        """Run the coordinate visualizer"""
        if not self.load_image():
            print("No image loaded. Exiting...")
            return

        if initial_coords:
            self.coords = initial_coords
        else:
            # Default to center rectangle
            height, width = self.image.shape[:2]
            center_x, center_y = width // 2, height // 2
            self.coords = (
                center_x - 50, center_y - 50,
                center_x + 50, center_y + 50
            )

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        print("\nControls:")
        print("- Drag corner points to adjust coordinates")
        print("- Press 's' to save current coordinates")
        print("- Press 'r' to reset coordinates")
        print("- Press 'q' to quit")

        while True:
            self.draw_coordinates()
            cv2.imshow(self.window_name, self.image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_coordinates()
            elif key == ord('r'):
                # Reset to center rectangle
                height, width = self.image.shape[:2]
                center_x, center_y = width // 2, height // 2
                self.coords = (
                    center_x - 50, center_y - 50,
                    center_x + 50, center_y + 50
                )

        cv2.destroyAllWindows()

if __name__ == "__main__":
    visualizer = CoordinateVisualizer()
    
    # Example usage with initial coordinates
    initial_coords = (508, 568, 708, 702)
    visualizer.run(initial_coords)
    
