"""
The computer environment is the main environment that Bob will interact with.
This needs to include the applications that Bob will use, a functioning mouse and keyboard that Bob will control, and a screen that Bob will see.
"""
from app.env.computer.discord import Discord
from app.env.computer.browser import Browser
from app.env.computer.screen import Screen
from app.env.computer.microphone import Microphone
from app.env.computer.audio import Audio
from app.env.computer.keyboard import Keyboard
from app.env.computer.mouse import Mouse
import logging
import subprocess
import app.env.computer.discord as discord
import threading
import multiprocessing
from typing import Dict, Any, List
import pyautogui
import time


class Computer(multiprocessing.Process):
    def __init__(self):
        super().__init__()
        self.screen = Screen()
        self.audio = Audio()
        self.microphone = Microphone()
        self.keyboard = Keyboard(target=self.screen)
        self.mouse = Mouse(target=self.screen, movement_speed=1.0)

        self.apps = {
            "browser": Browser(audio=self.audio, microphone=self.microphone, screen=self.screen),
            "discord": None  # Will be initialized after browser
        }
        
        # Initialize Discord after browser setup
        self.discord = None
        
        # Start sending audio in a separate thread
        audio_stream_thread = threading.Thread(target=self.microphone.start_sending_audio, daemon=True)
        audio_stream_thread.start()
        
        self.screen_container = None

    def run(self):
        self.startup()

    def startup(self):
        """
        Initializes all components of the computer environment.
        """
        try:
            # Initialize core components
            self.screen.initialize()
            self.audio.initialize() 
            self.microphone.initialize()
            
            # Launch browser in container
            logging.info("Launching browser...")
            if not self.launch_app("browser"):
                raise Exception("Failed to launch browser")
            
            # Wait for browser to be ready
            if not self.apps["browser"].webdriver:
                raise Exception("Browser WebDriver not initialized")
                
            # Ensure screen is receiving frames
            if not self.screen.get_current_frame():
                logging.warning("Screen not receiving frames, checking connection...")
                self.apps["browser"].set_screen(self.screen)
                
            # Initialize and launch Discord after browser is ready
            logging.info("Initializing Discord...")
            self.discord = Discord(browser=self.apps["browser"])
            self.apps["discord"] = self.discord
            
            # Launch Discord
            if not self.launch_app("discord"):
                logging.error("Failed to launch Discord")
            
            # Start input devices
            self.mouse.start()
            self.keyboard.start()
            
            # Initialize frame buffer for eyesight
            self.screen.frame_buffer = []
            
            logging.info("Computer environment started successfully")
            
        except Exception as e:
            logging.error(f"Failed to start computer environment: {e}")
            self.shutdown()
            raise

    def display_ui(self, ui_elements):
        """
        Sends UI elements to the simulated screen for display.
        """
        self.screen.display_elements(ui_elements)

    def launch_app(self, app_name):
        """
        Launches an application in the computer environment.
        """
        app = self.apps.get(app_name)
        if not app:
            logging.error(f"App {app_name} not found.")
            return False
            
        try:
            # Launch the app
            if not app.launch():
                raise Exception(f"Failed to launch {app_name}")
                
            # Route audio/microphone if available
            if hasattr(app, 'audio') and app.audio:
                self.audio.route_output(app_name)
            if hasattr(app, 'microphone') and app.microphone:
                self.microphone.connect_application(app_name)
                
            logging.info(f"Successfully launched {app_name}")
            return True
            
        except Exception as e:
            logging.error(f"Error launching {app_name}: {e}")
            return False

    def close_app(self, app_name):
        """
        Closes an application and cleans up resources.
        """
        app = self.apps.get(app_name)
        if not app:
            logging.error(f"App {app_name} not found.")
            return
            
        try:
            app.close()
            logging.info(f"Successfully closed {app_name}")
        except Exception as e:
            logging.error(f"Error closing {app_name}: {e}")

    def shutdown(self):
        """
        Shuts down all components gracefully.
        """
        self.keyboard.stop()
        self.mouse.stop()
        if self.screen_container:
            self.screen_container.terminate()
            self.screen_container.wait()
            logging.debug("Screen container stopped.")
        # ... existing shutdown code ...

    def get_system_state(self) -> Dict[str, Any]:
        """
        Returns the current system state including running applications,
        resource usage, notifications, and interaction mode.
        """
        return {
            'running_apps': list(self.apps.keys()),
            'resource_usage': self.get_resource_usage(),
            'notifications': self.get_pending_notifications(),
            'interaction_mode': self.get_interaction_mode()
        }

    def get_resource_usage(self) -> Dict[str, float]:
        """
        Returns current resource usage metrics.
        """
        return {
            'cpu': 0.0,  # Placeholder for CPU usage
            'memory': 0.0,  # Placeholder for memory usage
            'disk': 0.0  # Placeholder for disk usage
        }

    def get_pending_notifications(self) -> List[str]:
        """
        Returns list of pending notifications.
        """
        return []

    def get_interaction_mode(self) -> str:
        """
        Returns current interaction mode.
        """
        return 'normal'

    def minimize_all_windows(self):
        """Minimize all windows to show the desktop"""
        # Windows key + D shows the desktop
        pyautogui.hotkey('win', 'd')
        time.sleep(1)  # Wait for windows to minimize
