"""
The interface for Bob's thinking and thought processing. Bob is always thinking.
"""

import logging
import threading
import queue
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List
from app.memory.base import BaseMemorySystem
from app.agents.text import TextAgent
from app.memory.knowledge_system import KnowledgeSystem

class Thinking:
    def __init__(self, knowledge_system: KnowledgeSystem):
        self.knowledge_system = knowledge_system
        self.text_agent = TextAgent()
        self.logger = logging.getLogger(__name__)
        self.last_thought = None  # Store the most recent thought

    def process_sensory_input(self, sensory_data: Dict) -> Dict:
        """
        Processes current sensory inputs and generates relevant thoughts.
        
        Args:
            sensory_data: Dictionary containing visual, audio and system state data
        
        Returns:
            Dictionary containing generated thoughts and their metadata
        """
        try:
            # Extract sensory information
            visual_input = sensory_data.get('visual')
            audio_input = sensory_data.get('audio')
            system_state = sensory_data.get('system')

            # Generate thought based on current sensory input
            thought = self._generate_sensory_thought(visual_input, audio_input, system_state)
            
            # Create thought entry with metadata
            thought_entry = {
                'content': thought,
                'timestamp': datetime.now(),
                'type': 'sensory_reaction',
                'priority': self._assess_priority(thought, sensory_data),
                'metadata': {
                    'visual_context': bool(visual_input),
                    'audio_context': bool(audio_input),
                    'system_context': bool(system_state)
                }
            }

            # Update last thought
            self.last_thought = thought_entry
            return thought_entry

        except Exception as e:
            self.logger.error(f"Error processing sensory input: {e}")
            return None

    def current_thoughts(self) -> List[Dict]:
        """
        Returns the current thought if it exists.
        
        Returns:
            List containing the most recent thought entry, or empty list if none exists
        """
        if self.last_thought is None:
            return []
            
        # Return as list for compatibility with existing code
        return [self.last_thought]

    def _generate_sensory_thought(self, visual_input, audio_input, system_state) -> str:
        """
        Generates a thought based on current sensory inputs.
        """
        prompt = f"""
        Based on current sensory inputs:

        Visual Input:
        - Screen content: {visual_input}
        
        Audio Input:
        - Recent speech/sounds: {audio_input}
        
        System State:
        - Current state: {system_state}

        Generate a single thought that:
        1. Interprets the current sensory information
        2. Identifies important changes or patterns
        3. Focuses on relevant interactions or needed responses
        """
        
        return self.text_agent.complete_task(prompt)

    def _assess_priority(self, thought: str, sensory_data: Dict) -> float:
        """
        Assesses the priority of a thought based on sensory input importance.
        """
        # Priority indicators
        visual_indicators = {'error', 'warning', 'alert', 'changed', 'new'}
        audio_indicators = {'speech', 'sound', 'voice', 'said'}
        
        thought_lower = thought.lower()
        
        # Calculate priority scores
        visual_score = sum(indicator in thought_lower for indicator in visual_indicators) * 0.4
        audio_score = sum(indicator in thought_lower for indicator in audio_indicators) * 0.4
        
        # Add base priority for active inputs
        base_priority = (
            0.2 * bool(sensory_data.get('visual')) +
            0.2 * bool(sensory_data.get('audio'))
        )
        
        return min(1.0, visual_score + audio_score + base_priority)

    def add_experience(self, experience: Dict):
        """
        Adds a new experience to the knowledge system.
        """
        self.knowledge_system.add_experience(experience)
