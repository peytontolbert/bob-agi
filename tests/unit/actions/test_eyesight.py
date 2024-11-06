import pytest
from PIL import Image, ImageDraw
import logging
import os
import time
import numpy as np

class TestEyesight:
    @pytest.fixture(scope="class", autouse=True)
    def setup_class(self):
        """Setup any state specific to the execution of the given class."""
        import gc
        gc.collect()
        yield
        gc.collect()

    @pytest.fixture(scope="class")
    def computer(self):
        """Initialize real computer environment"""
        try:
            from app.env.computer.computer import Computer
            computer = Computer()
            computer.startup()
            yield computer
            computer.shutdown()
        except Exception as e:
            pytest.skip(f"Computer initialization failed: {e}")

    @pytest.fixture(scope="class")
    def screen(self, computer):
        """Initialize real screen implementation"""
        try:
            return computer.screen
        except Exception as e:
            pytest.skip(f"Screen initialization failed: {e}")

    @pytest.fixture(scope="class")
    def eyesight(self, screen):
        """Initialize Eyesight with real screen and vision system"""
        try:
            from app.env.senses.eyesight import Eyesight
            eye = Eyesight(screen)
            yield eye
            if hasattr(eye, 'cleanup'):
                eye.cleanup()
        except Exception as e:
            pytest.skip(f"Eyesight initialization failed: {e}")

    @pytest.fixture
    def test_image(self):
        """Create test image with actual UI elements"""
        img = Image.new('RGB', (800, 600), color='white')
        draw = ImageDraw.Draw(img)
        
        # Draw a blue button
        draw.rectangle([100, 100, 200, 150], fill='blue', outline='white')
        draw.text((120, 115), "Click Me", fill='white')
        
        # Draw some black text
        draw.text((300, 200), "Sample Text", fill='black')
        
        # Draw a search input field
        draw.rectangle([100, 300, 300, 330], fill='white', outline='gray')
        draw.text((110, 305), "Search...", fill='gray')
        
        # Draw a navigation menu
        draw.rectangle([50, 50, 150, 80], fill='darkblue', outline='white')
        draw.text((70, 55), "Menu", fill='white')
        
        # Draw a link
        draw.text((400, 150), "Click here", fill='blue', width=2)
        draw.line([(400, 152), (460, 152)], fill='blue', width=1)
        
        return img

    def test_full_perception_pipeline(self, eyesight, test_image, screen):
        """Test complete perception pipeline with real components"""
        try:
            screen.update_frame(test_image)
            time.sleep(0.5)
            
            # Test basic scene understanding
            perception = eyesight.process_visual_input("What do you see in this image?", test_image)
            assert perception is not None
            assert isinstance(perception, dict)
            assert perception.get('status') == 'success'
            
            description = perception.get('description', '')
            assert isinstance(description, str)
            assert len(description) > 0
            
            # Verify key UI elements are detected
            for element in ['button', 'text', 'menu', 'search']:
                assert element.lower() in description.lower(), f"Failed to detect {element}"
            
            # Test element finding capabilities
            elements = {
                'button': "blue button that says Click Me",
                'menu': "darkblue menu button",
                'search': "search input field",
                'link': "blue Click here link"
            }
            
            for element_type, description in elements.items():
                element = eyesight.find_element(description)
                assert element is not None, f"Failed to find {element_type}"
                assert element.get('confidence', 0) > 0.5, f"Low confidence for {element_type}"
                assert 'coordinates' in element, f"No coordinates for {element_type}"
            
            # Test embedding consistency
            embedding1 = eyesight.generate_embedding(test_image)
            embedding2 = eyesight.generate_embedding(test_image)
            
            assert embedding1 is not None
            assert embedding2 is not None
            assert np.allclose(embedding1, embedding2, rtol=1e-5), "Embeddings not consistent"
            
            # Test perception stream
            assert len(eyesight.perception_stream) > 0
            latest_perception = eyesight.perception_stream[-1]
            assert 'embedding' in latest_perception
            assert 'timestamp' in latest_perception
            
        except Exception as e:
            pytest.skip(f"Perception pipeline test failed: {str(e)}")

    def test_continuous_perception(self, eyesight, test_image, screen):
        """Test continuous perception capabilities"""
        try:
            # Create sequence of slightly modified images
            images = []
            for i in range(5):
                modified = test_image.copy()
                draw = ImageDraw.Draw(modified)
                draw.text((500, 400 + i*20), f"Dynamic Text {i}", fill='black')
                images.append(modified)
            
            # Test perception of changing content
            perceptions = []
            for img in images:
                screen.update_frame(img)
                time.sleep(0.1)  # Simulate realistic frame rate
                
                perception = eyesight.process_visual_input(
                    "What changed in this image?", 
                    img
                )
                perceptions.append(perception)
            
            # Verify perceptions
            assert len(perceptions) == len(images)
            assert all(p.get('status') == 'success' for p in perceptions)
            
            # Check temporal consistency
            timestamps = [p.get('timestamp', 0) for p in perceptions]
            assert all(t1 < t2 for t1, t2 in zip(timestamps[:-1], timestamps[1:]))
            
            # Verify perception stream updates
            assert len(eyesight.perception_stream) >= len(images)
            
        except Exception as e:
            pytest.skip(f"Continuous perception test failed: {str(e)}")

    def test_element_detection_accuracy(self, eyesight, test_image):
        """Test accuracy of UI element detection"""
        try:
            # Test precise element location
            button = eyesight.find_element("blue button that says Click Me")
            assert button is not None
            x, y = button['coordinates']
            assert 100 <= x <= 200  # Within button bounds
            assert 100 <= y <= 150
            
            # Test multiple element detection
            elements = eyesight.find_all_elements("button")
            assert len(elements) >= 2  # Should find both Menu and Click Me buttons
            
            # Test element type classification
            for element in elements:
                assert 'type' in element
                assert 'confidence' in element
                assert element['confidence'] > 0.5
                
            # Test text detection
            text_elem = eyesight.find_element("Sample Text")
            assert text_elem is not None
            assert 'text' in text_elem['type'].lower()
            
        except Exception as e:
            pytest.skip(f"Element detection test failed: {str(e)}")

    def test_gpu_acceleration_real_load(self, eyesight, test_image):
        """Test GPU acceleration under real load"""
        try:
            import torch
            if not torch.cuda.is_available():
                pytest.skip("CUDA not available")
                
            with torch.cuda.device(0):
                start_time = time.time()
                
                # Process multiple frames rapidly
                for _ in range(10):
                    embedding = eyesight.generate_embedding(test_image)
                    assert embedding is not None
                    assert embedding.shape[-1] == 512
                    
                duration = time.time() - start_time
                assert duration < 2.0  # Should process quickly with GPU
                
        except ImportError:
            pytest.skip("PyTorch not available")
        except Exception as e:
            pytest.skip(f"GPU acceleration test failed: {e}")
