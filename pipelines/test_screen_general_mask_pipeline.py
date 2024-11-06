from app.agents.vision import VisionAgent
from app.env.computer.computer import Computer
from app.models.sam2.sam2.build_sam import build_sam2
from app.models.sam2.sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import matplotlib.pyplot as plt
import numpy as np

class Test:
    def __init__(self):
        self.computer = Computer()
        self.computer.run()
        self.screen = self.computer.screen
        self.vision_agent = VisionAgent()
        sam2_checkpoint = "./weights/sam2.1_hiera_large.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

        sam2 = build_sam2(model_cfg, sam2_checkpoint, device="cuda:0", apply_postprocessing=False)
        self.mask_generator = SAM2AutomaticMaskGenerator(sam2)
        
    def run_test(self):
        # Get the screen capture state
        screen_state = self.screen.capture()
        
        # Extract the frame from the screen state
        if screen_state and 'frame' in screen_state:
            image = screen_state['frame']
            # Convert PIL Image to numpy array
            image = np.array(image)
            
            # Generate masks
            masks = self.mask_generator.generate(image)
            if masks:
                print(masks[0].keys())
                plt.figure(figsize=(20,20))
                plt.imshow(image)
                self.show_anns(masks)
                plt.axis('off')
                plt.show()
            else:
                print("No masks were generated")
        else:
            print("Failed to capture screen state")

    def show_anns(self, anns, borders=True):
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)

        img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
        img[:, :, 3] = 0
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.5]])
            img[m] = color_mask 
            if borders:
                import cv2
                contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
                # Try to smooth contours
                contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
                cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 

        ax.imshow(img)
        
def main():
    test = Test()
    test.run_test()


main()