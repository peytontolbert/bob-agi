from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
import time
from PIL import Image, ImageDraw, ImageFont
import os
from .keyboard import Keyboard
from .mouse import Mouse

class Browser:
    def __init__(self, window_width=800, window_height=600):
        # Configure Edge WebDriver
        edge_options = Options()
        edge_options.add_argument(f"--window-size={window_width},{window_height}")
        edge_options.add_argument("--use-fake-ui-for-media-stream")  # Allow audio access
        edge_options.add_argument("--use-fake-device-for-media-stream")
        
        self.driver = webdriver.Edge(options=edge_options)
        self.keyboard = Keyboard(self.driver)
        
        # Audio capture variables
        self.audio_callback = None
        self.is_capturing_audio = False
        
        # Wait for the browser to open
        time.sleep(2)
        
        # Store both viewport and screenshot dimensions
        self.viewport_width = self.driver.execute_script("return window.innerWidth")
        self.viewport_height = self.driver.execute_script("return window.innerHeight")
        self.mouse = Mouse(self.driver, self.viewport_width, self.viewport_height)
        # Calculate the difference between window and viewport size
        #width_diff = window_width - self.viewport_width
        #height_diff = window_height - self.viewport_height
        
        # Adjust window size to account for the difference
        #self.driver.set_window_size(window_width + width_diff, window_height + height_diff)
        
        self.screenshot_width = 1008    
        self.screenshot_height = 1008
        
        self.actions = ActionChains(self.driver)
        self.last_mouse_position = None
        
        print(f"Initialized browser with viewport dimensions: {self.viewport_width}x{self.viewport_height}")

    def navigate(self, url):
        """Navigate to a specified URL."""
        self.driver.get(url)
        print(f"Navigated to {url}")
        time.sleep(2)  # Wait for the page to load

    def locate_element_by_text(self, text):
        """Locate an element by link text and return its center coordinates."""
        try:
            element = self.driver.find_element(By.LINK_TEXT, text)
            location = element.location
            size = element.size
            # Calculate center coordinates within the browser
            center_x = location['x'] + (size['width'] / 2)
            center_y = location['y'] + (size['height'] / 2)
            print(f"Located element '{text}' at ({center_x}, {center_y})")
            return center_x, center_y
        except Exception as e:
            print(f"Error locating element by text '{text}': {e}")
            return None, None

    def close(self):
        """Close the browser."""
        self.driver.quit()
        print("Browser closed.")
 
    def click_and_type(self, x, y, text):
        """Click at coordinates and type text."""
        self.mouse.click_at(x, y)
        time.sleep(0.5)  # Wait for click to register
        self.keyboard.type_text(text)

    def start_audio_capture(self, callback):
        """Start capturing audio from the browser"""
        try:
            if not self.is_capturing_audio:
                self.audio_callback = callback
                
                # Execute JavaScript to start audio capture
                script = """
                navigator.mediaDevices.getUserMedia({ audio: true })
                    .then(function(stream) {
                        window.audioContext = new AudioContext();
                        var source = audioContext.createMediaStreamSource(stream);
                        var processor = audioContext.createScriptProcessor(4096, 1, 1);
                        
                        processor.onaudioprocess = function(e) {
                            var buffer = e.inputBuffer.getChannelData(0);
                            window.dispatchEvent(new CustomEvent('audioData', { 
                                detail: Array.from(buffer)
                            }));
                        };
                        
                        source.connect(processor);
                        processor.connect(audioContext.destination);
                    })
                    .catch(function(err) {
                        console.error('Error accessing audio:', err);
                    });
                """
                self.driver.execute_script(script)
                
                # Add event listener for audio data
                self.driver.execute_script("""
                    window.addEventListener('audioData', function(e) {
                        window.audioData = e.detail;
                    });
                """)
                
                self.is_capturing_audio = True
                print("Started audio capture")
                
                # Start the audio processing loop
                self._process_audio_loop()
                
        except Exception as e:
            print(f"Error starting audio capture: {e}")

    def stop_audio_capture(self):
        """Stop capturing audio"""
        try:
            if self.is_capturing_audio:
                self.driver.execute_script("""
                    if (window.audioContext) {
                        window.audioContext.close();
                    }
                """)
                self.is_capturing_audio = False
                self.audio_callback = None
                print("Stopped audio capture")
        except Exception as e:
            print(f"Error stopping audio capture: {e}")

    def _process_audio_loop(self):
        """Process captured audio data"""
        if self.is_capturing_audio and self.audio_callback:
            try:
                # Get audio data from JavaScript
                audio_data = self.driver.execute_script("return window.audioData;")
                if audio_data:
                    # Convert to numpy array and send to callback
                    self.audio_callback(audio_data)
                    
                # Schedule next processing
                if self.is_capturing_audio:
                    self.driver.execute_script("""
                        setTimeout(function() {
                            window.dispatchEvent(new Event('processAudio'));
                        }, 100);
                    """)
            except Exception as e:
                print(f"Error in audio processing loop: {e}")



    def take_screenshot(self, filename="images/screenshot.png"):
        """Take a screenshot and overlay coordinate system scaled to 1000x1000."""
        # Take the screenshot
        self.driver.save_screenshot(filename)
        
        try:
            # Open and resize the screenshot to 1000x1000
            image = Image.open(filename)
            draw = ImageDraw.Draw(image)

            try:
                font = ImageFont.truetype("arial.ttf", 15)
            except IOError:
                font = None
            
            # Overlay the mouse position if available
            if self.last_mouse_position:
                # Draw viewport coordinates in red
                viewport_x, viewport_y = self.last_mouse_position
                mouse_size = 10
                draw.ellipse(
                    (viewport_x - mouse_size, viewport_y - mouse_size, 
                     viewport_x + mouse_size, viewport_y + mouse_size),
                    fill='red',
                    outline='black'
                )
                draw.text((viewport_x + 15, viewport_y), 
                         f"Viewport: ({int(viewport_x)}, {int(viewport_y)})", 
                         fill="red", 
                         font=font)
                
                # Draw screenshot coordinates in blue
                screenshot_x, screenshot_y = self.normalize_coordinates(
                    viewport_x, 
                    viewport_y, 
                    from_screenshot=False
                )
                draw.ellipse(
                    (screenshot_x - mouse_size, screenshot_y - mouse_size, 
                     screenshot_x + mouse_size, screenshot_y + mouse_size),
                    fill='blue',
                    outline='black'
                )
                draw.text((screenshot_x + 15, screenshot_y + 25), 
                         f"Screenshot: ({int(screenshot_x)}, {int(screenshot_y)})", 
                         fill="blue", 
                         font=font)

            image = image.resize((self.screenshot_width, self.screenshot_height))
            # Save the modified screenshot
            image.save(filename)
            print(f"Enhanced screenshot saved with viewport and screenshot coordinates at {filename}")
        except Exception as e:
            print(f"Error processing screenshot: {e}")

