"""
The TemporalKnowledgeSystem manages time-based knowledge and temporal reasoning.
"""
from typing import Dict, List, Optional
import numpy as np
import logging
from datetime import datetime, timedelta

class TemporalKnowledgeSystem:
    def __init__(self, knowledge_system):
        self.knowledge_system = knowledge_system
        self.temporal_events = []  # Time-ordered events
        self.temporal_patterns = {}  # Recurring patterns
        self.future_predictions = {}  # Predicted future events
        
    def add_temporal_event(self, event: Dict) -> bool:
        """
        Adds a temporal event with timing information.
        
        Args:
            event: Dictionary containing:
                - description: Event description
                - timestamp: When it occurred
                - duration: How long it lasted
                - context: Temporal context
                - importance: Event importance (0-1)
        """
        try:
            if not self._validate_event(event):
                return False
                
            self.temporal_events.append(event)
            self._update_patterns(event)
            
            # Store in vector space
            vector = self._vectorize_event(event)
            self.knowledge_system.create_entry({
                'vector': vector,
                'metadata': event
            })
            
            return True
            
        except Exception as e:
            logging.error(f"Error adding temporal event: {e}")
            return False
    
    def predict_future_events(self, context: Dict, 
                            time_horizon: timedelta) -> List[Dict]:
        """
        Predicts future events based on temporal patterns.
        """
        try:
            predictions = []
            current_time = datetime.now()
            
            # Check patterns for likely future events
            for pattern_name, pattern in self.temporal_patterns.items():
                if self._pattern_matches_context(pattern, context):
                    next_occurrence = self._predict_next_occurrence(
                        pattern, current_time)
                    
                    if next_occurrence and \
                       next_occurrence - current_time <= time_horizon:
                        predictions.append({
                            'event_type': pattern_name,
                            'predicted_time': next_occurrence,
                            'confidence': pattern['confidence'],
                            'based_on': pattern['supporting_events']
                        })
            
            return sorted(predictions, 
                         key=lambda x: x['confidence'], 
                         reverse=True)
                         
        except Exception as e:
            logging.error(f"Error predicting events: {e}")
            return []
    
    def _validate_event(self, event: Dict) -> bool:
        required_fields = ['description', 'timestamp', 'duration', 'context']
        return all(field in event for field in required_fields)
    
    def _vectorize_event(self, event: Dict) -> np.ndarray:
        # Placeholder for actual embedding logic
        return np.random.rand(384)
    
    def _update_patterns(self, new_event: Dict):
        """Updates temporal patterns based on new event."""
        try:
            # Find similar past events
            similar_events = [
                e for e in self.temporal_events
                if self._events_are_similar(e, new_event)
            ]
            
            if similar_events:
                pattern_key = self._generate_pattern_key(new_event)
                
                if pattern_key not in self.temporal_patterns:
                    self.temporal_patterns[pattern_key] = {
                        'events': [],
                        'interval': timedelta(0),
                        'confidence': 0.0,
                        'supporting_events': []
                    }
                
                pattern = self.temporal_patterns[pattern_key]
                pattern['events'].append(new_event)
                pattern['interval'] = self._calculate_average_interval(
                    pattern['events'])
                pattern['confidence'] = self._calculate_pattern_confidence(
                    pattern['events'])
                pattern['supporting_events'] = similar_events[-5:]  # Keep last 5
                
        except Exception as e:
            logging.error(f"Error updating patterns: {e}")
    
    def _events_are_similar(self, event1: Dict, event2: Dict) -> bool:
        """Determines if two events are similar enough to form a pattern."""
        return (
            event1['description'] == event2['description'] and
            self._context_similarity(event1['context'], event2['context']) > 0.8
        )
    
    def _calculate_average_interval(self, events: List[Dict]) -> timedelta:
        """Calculates average time interval between similar events."""
        if len(events) < 2:
            return timedelta(0)
            
        intervals = []
        sorted_events = sorted(events, key=lambda x: x['timestamp'])
        
        for i in range(1, len(sorted_events)):
            interval = sorted_events[i]['timestamp'] - \
                      sorted_events[i-1]['timestamp']
            intervals.append(interval)
            
        return sum(intervals, timedelta(0)) / len(intervals)
    
    def _predict_next_occurrence(self, pattern: Dict, 
                               current_time: datetime) -> Optional[datetime]:
        """Predicts when pattern will next occur."""
        if not pattern['events']:
            return None
            
        last_event = max(pattern['events'], 
                        key=lambda x: x['timestamp'])
        return last_event['timestamp'] + pattern['interval']
