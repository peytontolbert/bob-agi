"""
The filesystem of the computer.
"""
import logging
from pathlib import Path
from typing import Optional

class Filesystem:
    def __init__(self):
        self.root = Path("/")
        self.current_dir = self.root
        self.files = {}

    def initialize(self):
        """Initializes virtual filesystem"""
        self.create_directory("/home")
        self.create_directory("/home/agent")
        logging.info("Filesystem initialized")

    def create_directory(self, path: str):
        """Creates new directory"""
        full_path = Path(path)
        self.files[str(full_path)] = {"type": "directory", "contents": {}}
        logging.debug("Created directory: %s", path)

    def write_file(self, path: str, content: str):
        """Writes content to file"""
        full_path = Path(path)
        self.files[str(full_path)] = {"type": "file", "content": content}
        logging.debug("Wrote file: %s", path)

    def read_file(self, path: str) -> Optional[str]:
        """Reads file content"""
        full_path = str(Path(path))
        if full_path in self.files and self.files[full_path]["type"] == "file":
            return self.files[full_path]["content"]
        return None
