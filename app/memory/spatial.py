"""
The SpatialKnowledgeSystem manages spatial relationships, navigation, and environmental understanding.
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
import logging

class SpatialKnowledgeSystem:
    def __init__(self, knowledge_system):
        self.knowledge_system = knowledge_system
        # Computer environment specific spatial maps
        self.interface_maps = {
            'desktop': {},  # Desktop layout
            'applications': {},  # Application window layouts
            'navigation_paths': {}  # Common navigation patterns
        }
        self.ui_locations = {}  # UI element locations
        self.interaction_zones = {}  # Clickable/interactive areas
        
    def add_location(self, location: Dict) -> bool:
        """
        Adds a new UI/interface location.
        
        Args:
            location: Dictionary containing:
                - element_id: UI element identifier
                - coordinates: (x, y) screen coordinates
                - dimensions: (width, height)
                - parent_window: Parent application/window
                - interaction_type: Click, type, drag, etc.
                - landmarks: Nearby UI elements
        """
        try:
            if not self._validate_ui_location(location):
                return False
                
            self.ui_locations[location['element_id']] = location
            self._update_interaction_zones(location)
            
            # Store in vector space for similarity lookups
            vector = self._vectorize_ui_location(location)
            self.knowledge_system.create_entry({
                'vector': vector,
                'metadata': location
            })
            
            return True
            
        except Exception as e:
            logging.error(f"Error adding UI location: {e}")
            return False
    
    def find_path(self, start_element: str, target_element: str) -> List[Dict]:
        """
        Finds a navigation path between UI elements, considering:
        - Window/application switches needed
        - Mouse movements required
        - Intermediate clicks/actions needed
        """
        try:
            if not (start_element in self.ui_locations and 
                   target_element in self.ui_locations):
                return []
                
            start = self.ui_locations[start_element]
            target = self.ui_locations[target_element]
            
            path = []
            
            # Add window/application switch if needed
            if start['parent_window'] != target['parent_window']:
                path.append({
                    'action': 'switch_window',
                    'from': start['parent_window'],
                    'to': target['parent_window']
                })
            
            # Add mouse movement
            path.append({
                'action': 'move_mouse',
                'from': start['coordinates'],
                'to': target['coordinates']
            })
            
            # Add final interaction
            path.append({
                'action': target['interaction_type'],
                'target': target_element,
                'coordinates': target['coordinates']
            })
            
            return path
            
        except Exception as e:
            logging.error(f"Error finding UI navigation path: {e}")
            return []
    
    def get_nearby_locations(self, element_id: str, radius: float) -> List[Dict]:
        """
        Finds UI elements within a given pixel radius of the target element.
        """
        try:
            if element_id not in self.ui_locations:
                return []
                
            current_element = self.ui_locations[element_id]
            nearby = []
            
            for id, element in self.ui_locations.items():
                if id != element_id:
                    # Calculate screen-space distance
                    distance = self._calculate_screen_distance(
                        current_element['coordinates'],
                        element['coordinates']
                    )
                    if distance <= radius:
                        element_info = element.copy()
                        element_info['distance'] = distance
                        nearby.append(element_info)
                        
            return sorted(nearby, key=lambda x: x['distance'])
            
        except Exception as e:
            logging.error(f"Error finding nearby UI elements: {e}")
            return []
    
    def _validate_ui_location(self, location: Dict) -> bool:
        """Validates UI location data."""
        required = ['element_id', 'coordinates', 'dimensions', 
                   'parent_window', 'interaction_type']
        return all(field in location for field in required)
    
    def _calculate_screen_distance(self, coord1: Tuple[int, int], 
                                 coord2: Tuple[int, int]) -> float:
        """Calculates pixel distance between screen coordinates."""
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(coord1, coord2)))
    
    def _update_interaction_zones(self, location: Dict):
        """Updates the map of interactive UI zones."""
        zone = {
            'element_id': location['element_id'],
            'coordinates': location['coordinates'],
            'dimensions': location['dimensions'],
            'interaction_type': location['interaction_type']
        }
        self.interaction_zones[location['element_id']] = zone
