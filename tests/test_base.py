import unittest
from unittest.mock import patch
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import logging
import os

class BaseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are reused across test methods"""
        # Set up paths
        cls.TEST_DIR = Path(__file__).parent
        cls.FIXTURES_DIR = cls.TEST_DIR / "fixtures"
        cls.SCREENSHOTS_DIR = cls.TEST_DIR / "screenshots"
        cls.CONFIG_DIR = cls.TEST_DIR / "conf"
        cls.LOGS_DIR = cls.TEST_DIR / "logs"
        
        # Create necessary directories
        for directory in [cls.SCREENSHOTS_DIR, cls.FIXTURES_DIR, cls.LOGS_DIR]:
            directory.mkdir(exist_ok=True, parents=True)
        
        # Configure logging
        logging.basicConfig(
            filename=cls.LOGS_DIR / "test.log",
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        
        # Load test configuration
        try:
            with hydra.initialize(config_path=str(cls.CONFIG_DIR.relative_to(Path.cwd()))):
                cls.config = hydra.compose(config_name="test_config")
        except Exception as e:
            logging.error(f"Failed to load config: {e}")
            cls.config = cls._get_default_config()
    
    @staticmethod
    def _get_default_config():
        """Provide default configuration if hydra config fails"""
        return OmegaConf.create({
            "test": {
                "discord": {
                    "server_id": "999382051935506503",
                    "channel_name": "General",
                    "timeout": 30
                },
                "browser": {
                    "default_timeout": 10,
                    "screenshot_dir": "tests/screenshots"
                }
            }
        })
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        super().setUp()
        self.addCleanup(self.cleanup)
        
    def cleanup(self):
        """Clean up after each test"""
        # Clean up screenshot files
        for file in self.SCREENSHOTS_DIR.glob("*.png"):
            try:
                file.unlink()
            except Exception as e:
                logging.warning(f"Failed to delete screenshot {file}: {e}")