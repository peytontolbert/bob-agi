"""
This is the text Agent that completes text tasks for the input
"""
from app.agents.base import BaseAgent
import base64
from PIL import Image
import io

class TextAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def encode_image(self, image_path):
        """Encode image file to base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def encode_pil_image(self, pil_image):
        """Encode PIL Image to base64 string"""
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def complete_task(self, input):
        """Complete text task with optional image input"""
        # Check if input contains an image tag
        if "<image>" in input and "</image>" in input:
            # Extract base64 string between image tags
            start_idx = input.find("<image>") + 7
            end_idx = input.find("</image>")
            base64_str = input[start_idx:end_idx].strip()
            
            # Create message with image
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": input[:start_idx-7]},  # Text before image
                        {"type": "image_url", "image_url": f"data:image/png;base64,{base64_str}"},
                        {"type": "text", "text": input[end_idx+8:]}  # Text after image
                    ]
                }
            ]
        else:
            # Regular text-only message
            messages = [{"role": "user", "content": input}]

        # Make API call with proper model that supports vision
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",  # Use vision-capable model
            messages=messages,
            max_tokens=1000,
        )
        
        return response.choices[0].message.content

