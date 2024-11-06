"""
Simple test script for SAM2 image masking capabilities
"""
import logging
from pathlib import Path
import time
from typing import Optional
import traceback
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from app.models.samtwo import SAM2, SAM2Config

class SimpleSAM2Test:
    def __init__(self):
        self.sam2 = None
        self.debug_dir = Path("debug_output")
        self.setup_logging()
        self.initialize_sam()

    def setup_logging(self):
        """Configure logging for test pipeline"""
        self.debug_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.debug_dir / "sam2_test.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def initialize_sam(self):
        """Initialize SAM2 model"""
        try:
            self.logger.info("Initializing SAM2...")
            self.sam2 = SAM2(SAM2Config())
            if not self.sam2:
                raise RuntimeError("Failed to initialize SAM2")
            self.logger.info("SAM2 initialized successfully")

        except Exception as e:
            self.logger.error(f"Initialization error: {e}")
            self.logger.error(traceback.format_exc())
            raise

    def create_test_image(self) -> Image.Image:
        """Create a simple test image with some shapes"""
        # Create a white background
        image = Image.new('RGB', (512, 512), 'white')
        draw = ImageDraw.Draw(image)
        
        # Draw some shapes
        # Blue rectangle (simulating a button)
        draw.rectangle([50, 50, 200, 100], fill='blue')
        # Green circle (simulating an icon)
        draw.ellipse([250, 250, 350, 350], fill='green')
        # Red text box
        draw.rectangle([50, 400, 450, 450], fill='red')
        
        # Save the test image
        test_image_path = self.debug_dir / "test_image.png"
        image.save(test_image_path)
        self.logger.info(f"Created test image at {test_image_path}")
        
        return image

    def save_mask_visualization(self, image: np.ndarray, masks: np.ndarray, scores: np.ndarray, 
                              query: Optional[str], timestamp: int):
        """Save individual mask visualizations using show_masks"""
        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            self.sam2.show_mask(mask, plt.gca(), random_color=True)
            
            title = f"Mask {i+1}"
            if query:
                title += f" (Query: {query})"
            title += f"\nScore: {score:.3f}"
            plt.title(title)
            
            plt.axis('off')
            mask_path = self.debug_dir / f"mask_{timestamp}_{i+1}.png"
            plt.savefig(mask_path, bbox_inches='tight', pad_inches=0)
            plt.close()
            
            self.logger.info(f"Saved mask visualization to {mask_path}")

    async def test_image_masking(self, image: Image.Image, query: Optional[str] = None):
        """
        Test SAM2 masking on a single image
        
        Args:
            image: PIL Image to test
            query: Optional text query to guide detection
        """
        try:
            self.logger.info(f"Testing image masking" + (f" with query: {query}" if query else ""))

            # Run detection
            results = await self.sam2.detect_elements(image, query)
            
            if results['status'] != 'success':
                raise RuntimeError(f"Detection failed: {results.get('message')}")

            timestamp = int(time.time())

            # Save individual mask visualizations
            image_np = np.array(image)
            masks = np.array(results['raw_masks'])
            scores = np.array([d['confidence'] for d in results['detections']])
            self.save_mask_visualization(image_np, masks, scores, query, timestamp)

            # Plot combined results
            plt.figure(figsize=(15, 5))
            
            # Original image
            plt.subplot(131)
            plt.imshow(image)
            plt.title("Original Image")
            plt.axis('off')
            
            # Annotated image with all masks
            plt.subplot(132)
            plt.imshow(results['annotated_image'])
            plt.title(f"Detected Elements ({len(results['detections'])})")
            plt.axis('off')
            
            # Confidence distribution
            plt.subplot(133)
            confidences = [d['confidence'] for d in results['detections']]
            plt.hist(confidences, bins=20)
            plt.title("Confidence Distribution")
            plt.xlabel("Confidence Score")
            plt.ylabel("Count")
            
            # Save combined plot
            plot_path = self.debug_dir / f"sam2_results_{timestamp}.png"
            plt.savefig(plot_path)
            plt.close()
            
            self.logger.info(f"Saved combined results visualization to {plot_path}")
            
            # Print detection statistics
            self.logger.info("\nDetection Statistics:")
            self.logger.info(f"Total detections: {len(results['detections'])}")
            if results['detections']:
                conf_array = np.array(confidences)
                self.logger.info(f"Mean confidence: {conf_array.mean():.3f}")
                self.logger.info(f"Max confidence: {conf_array.max():.3f}")
                self.logger.info(f"Min confidence: {conf_array.min():.3f}")
                
                # Print details of top 3 detections
                self.logger.info("\nTop 3 Detections:")
                sorted_dets = sorted(results['detections'], 
                                   key=lambda x: x['confidence'], 
                                   reverse=True)[:3]
                for i, det in enumerate(sorted_dets, 1):
                    self.logger.info(
                        f"{i}. Confidence: {det['confidence']:.3f}, "
                        f"Size: {det['width']:.0f}x{det['height']:.0f}, "
                        f"Area: {det['area']:.0f}"
                    )

        except Exception as e:
            self.logger.error(f"Error in image masking test: {e}")
            self.logger.error(traceback.format_exc())

def main():
    import asyncio
    
    test = SimpleSAM2Test()
    
    # Create test image
    test_image = test.create_test_image()
    
    # Test queries
    test_queries = [
        None,  # Test without query first
        "button",
        "circle",
        "text box"
    ]
    
    async def run_tests():
        for query in test_queries:
            await test.test_image_masking(test_image, query)
    
    asyncio.run(run_tests())

if __name__ == "__main__":
    main()
