"""
The MetaCognitiveSystem handles self-monitoring, learning strategies, and cognitive control.
"""
from typing import Dict, List, Optional
import numpy as np
import logging
from datetime import datetime

class MetaCognitiveSystem:
    def __init__(self, knowledge_system):
        self.knowledge_system = knowledge_system
        self.performance_history = []
        self.learning_strategies = {}
        self.cognitive_state = {
            'attention_level': 1.0,
            'processing_depth': 1.0,
            'cognitive_load': 0.0
        }
        
    def monitor_performance(self, task: Dict) -> Dict:
        """
        Monitors and evaluates performance on a task.
        
        Args:
            task: Dictionary containing:
                - type: Task type
                - difficulty: Estimated difficulty (0-1)
                - outcome: Success/failure metrics
                - strategies_used: List of strategies applied
        """
        try:
            evaluation = self._evaluate_performance(task)
            self.performance_history.append({
                'task': task,
                'evaluation': evaluation,
                'timestamp': datetime.now(),
                'cognitive_state': self.cognitive_state.copy()
            })
            
            # Update learning strategies based on performance
            self._update_strategies(task, evaluation)
            
            return evaluation
            
        except Exception as e:
            logging.error(f"Error monitoring performance: {e}")
            return {}
    
    def adjust_cognitive_control(self, situation: Dict) -> Dict:
        """
        Adjusts cognitive parameters based on task demands.
        """
        try:
            new_state = self._calculate_optimal_state(situation)
            self.cognitive_state.update(new_state)
            return self.cognitive_state
        except Exception as e:
            logging.error(f"Error adjusting cognitive control: {e}")
            return self.cognitive_state
    
    def add_learning_strategy(self, strategy: Dict) -> bool:
        """
        Adds a new learning strategy to the repertoire.
        """
        try:
            if self._validate_strategy(strategy):
                self.learning_strategies[strategy['name']] = {
                    'description': strategy['description'],
                    'effectiveness': strategy.get('effectiveness', 0.0),
                    'conditions': strategy.get('conditions', [])
                }
                return True
            return False
        except Exception as e:
            logging.error(f"Error adding learning strategy: {e}")
            return False
    
    def get_optimal_strategy(self, task_type: str) -> Optional[Dict]:
        """
        Determines the best learning strategy for a given task.
        """
        try:
            relevant_strategies = [
                (name, data) for name, data in self.learning_strategies.items()
                if task_type in data.get('conditions', [])
            ]
            
            if not relevant_strategies:
                return None
                
            return max(relevant_strategies, key=lambda x: x[1]['effectiveness'])
            
        except Exception as e:
            logging.error(f"Error getting optimal strategy: {e}")
            return None
    
    def _evaluate_performance(self, task: Dict) -> Dict:
        """Evaluates task performance and generates insights."""
        return {
            'success_rate': self._calculate_success_rate(task),
            'efficiency': self._calculate_efficiency(task),
            'areas_for_improvement': self._identify_improvements(task)
        }
    
    def _calculate_success_rate(self, task: Dict) -> float:
        """Calculates task success rate."""
        outcomes = task.get('outcome', {})
        if 'success_metrics' in outcomes:
            return sum(outcomes['success_metrics']) / len(outcomes['success_metrics'])
        return 0.0
    
    def _calculate_efficiency(self, task: Dict) -> float:
        """Calculates task efficiency."""
        time_taken = task.get('time_taken', 1.0)
        success_rate = self._calculate_success_rate(task)
        return success_rate / time_taken if time_taken > 0 else 0.0
    
    def _identify_improvements(self, task: Dict) -> List[str]:
        """Identifies areas for improvement."""
        improvements = []
        if self._calculate_success_rate(task) < 0.8:
            improvements.append("Increase accuracy")
        if self._calculate_efficiency(task) < 0.7:
            improvements.append("Improve speed")
        return improvements
    
    def _calculate_optimal_state(self, situation: Dict) -> Dict:
        """Calculates optimal cognitive parameters for a situation."""
        return {
            'attention_level': min(1.0, situation.get('complexity', 0.5) + 0.2),
            'processing_depth': min(1.0, situation.get('difficulty', 0.5) + 0.3),
            'cognitive_load': situation.get('complexity', 0.5) * 0.8
        }
