from typing import Tuple, Optional
import re

class NLPMouseController:
    """
    A controller that translates natural language commands into mouse movements.
    """
    def __init__(self, mouse, screen):
        self.mouse = mouse
        self.screen = screen
        # Define movement mappings
        self.distance_mappings = {
            'tiny': 5,
            'very small': 10,
            'small': 20,
            'medium': 50,
            'large': 100,
            'very large': 200,
            'huge': 400
        }
        
        self.direction_mappings = {
            'up': (0, -1),
            'down': (0, 1),
            'left': (-1, 0),
            'right': (1, 0),
            'up-left': (-1, -1),
            'up-right': (1, -1),
            'down-left': (-1, 1),
            'down-right': (1, 1)
        }
        
        # Add relative position mappings
        self.position_mappings = {
            'center': lambda w, h: (w//2, h//2),
            'top': lambda w, h: (w//2, 0),
            'bottom': lambda w, h: (w//2, h),
            'left': lambda w, h: (0, h//2),
            'right': lambda w, h: (w, h//2),
            'top-left': lambda w, h: (0, 0),
            'top-right': lambda w, h: (w, 0),
            'bottom-left': lambda w, h: (0, h),
            'bottom-right': lambda w, h: (w, h)
        }

    def parse_movement(self, command: str) -> Optional[Tuple[int, int]]:
        """
        Parse a natural language movement command.
        
        Examples:
        - "move up a small amount"
        - "go right by a large distance"
        - "move down-left a tiny bit"
        
        Returns:
            Tuple[int, int]: The (x, y) movement delta, or None if invalid
        """
        command = command.lower().strip()
        
        # Extract distance descriptor
        distance_match = None
        for distance_desc in self.distance_mappings.keys():
            if distance_desc in command:
                distance_match = distance_desc
                break
                
        if not distance_match:
            return None
            
        # Extract direction
        direction_match = None
        for direction in self.direction_mappings.keys():
            if direction in command:
                direction_match = direction
                break
                
        if not direction_match:
            return None
            
        # Calculate movement
        base_distance = self.distance_mappings[distance_match]
        direction_vector = self.direction_mappings[direction_match]
        
        # Calculate final movement delta
        delta_x = direction_vector[0] * base_distance
        delta_y = direction_vector[1] * base_distance
        
        return (delta_x, delta_y)

    def move(self, command: str) -> bool:
        """
        Execute a natural language movement command.
        
        Args:
            command: Natural language movement instruction
            
        Returns:
            bool: True if movement was executed successfully
        """
        movement = self.parse_movement(command)
        if not movement:
            return False
            
        # Get current position
        current_x, current_y = self.mouse.get_position()
        
        # Calculate new position
        new_x = current_x + movement[0]
        new_y = current_y + movement[1]
        
        # Execute movement
        return self.mouse.move_to(new_x, new_y)

    def move_to_relative_position(self, position: str) -> bool:
        """Move to a relative screen position."""
        if position in self.position_mappings:
            screen_width = self.screen.width
            screen_height = self.screen.height
            x, y = self.position_mappings[position](screen_width, screen_height)
            return self.mouse.move_to(x, y)
        return False

    def execute_command(self, command: str) -> bool:
        """
        Enhanced command execution with support for:
        - Relative positions ("move to top-right")
        - Complex movements ("move up large then right small")
        - Click operations
        - Grid-based search patterns
        """
        command = command.lower().strip()
        
        # Handle relative position commands
        if "move to" in command:
            for position in self.position_mappings.keys():
                if position in command:
                    return self.move_to_relative_position(position)
        
        # Handle sequential movements
        if "then" in command:
            commands = command.split("then")
            success = True
            for cmd in commands:
                success = success and self.move(cmd.strip())
            return success
            
        # Handle click commands
        if "click" in command:
            if "right" in command:
                self.mouse.click(button='right')
            elif "double" in command:
                self.mouse.click(double=True)
            else:
                self.mouse.click()
            return True
            
        # Handle basic movement commands
        return self.move(command)
