"""
The AutobiographicalMemorySystem manages personal experiences and self-reflection.
"""
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np
import logging

class AutobiographicalMemorySystem:
    def __init__(self, knowledge_system):
        self.knowledge_system = knowledge_system
        self.personal_timeline = []
        self.core_memories = {}  # Key memories that define identity
        
    def store_experience(self, experience: Dict) -> bool:
        """
        Stores a personal experience with emotional and contextual details.
        
        Args:
            experience: Dictionary containing:
                - event: Description of what happened
                - emotions: Emotional response
                - significance: Personal importance (1-10)
                - context: Situational context
                - reflection: Personal thoughts/learnings
        """
        try:
            if not self._validate_experience(experience):
                return False
                
            experience['timestamp'] = datetime.now()
            self.personal_timeline.append(experience)
            
            # Store as core memory if highly significant
            if experience.get('significance', 0) >= 8:
                self.core_memories[experience['event']] = experience
            
            # Create vector and store in knowledge system
            vector = self._vectorize_experience(experience)
            self.knowledge_system.create_entry({
                'vector': vector,
                'metadata': experience
            })
            
            return True
            
        except Exception as e:
            logging.error(f"Error storing experience: {e}")
            return False
    
    def reflect_on_period(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """
        Retrieves and analyzes experiences from a specific time period.
        """
        relevant_experiences = [
            exp for exp in self.personal_timeline 
            if start_date <= exp['timestamp'] <= end_date
        ]
        return self._analyze_experiences(relevant_experiences)
    
    def get_core_memories(self) -> List[Dict]:
        """Returns the most significant memories that shape identity."""
        return list(self.core_memories.values())
    
    def _validate_experience(self, experience: Dict) -> bool:
        required_fields = ['event', 'emotions', 'significance', 'context']
        return all(field in experience for field in required_fields)
    
    def _vectorize_experience(self, experience: Dict) -> np.ndarray:
        # Placeholder for actual embedding logic
        return np.random.rand(384)
    
    def _analyze_experiences(self, experiences: List[Dict]) -> List[Dict]:
        """Analyzes patterns and themes in experiences."""
        # Group by emotion and significance
        analysis = {
            'emotional_patterns': self._analyze_emotions(experiences),
            'significant_events': [e for e in experiences if e.get('significance', 0) >= 7],
            'learnings': self._extract_learnings(experiences)
        }
        return analysis
    
    def _analyze_emotions(self, experiences: List[Dict]) -> Dict:
        """Analyzes emotional patterns in experiences."""
        emotion_counts = {}
        for exp in experiences:
            for emotion in exp.get('emotions', []):
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        return emotion_counts
    
    def _extract_learnings(self, experiences: List[Dict]) -> List[str]:
        """Extracts key learnings from experiences."""
        return [exp.get('reflection', '') for exp in experiences if 'reflection' in exp]

