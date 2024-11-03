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

    def _validate_strategy(self, strategy: Dict) -> bool:
        """
        Validates that a learning strategy has all required components.
        
        Args:
            strategy: Dictionary containing strategy details with required fields:
                - name: Strategy name
                - description: Strategy description
                - conditions: List of applicable task types
        """
        required_fields = ['name', 'description', 'conditions']
        if not all(field in strategy for field in required_fields):
            logging.warning(f"Strategy missing required fields: {required_fields}")
            return False
            
        if not isinstance(strategy['conditions'], list):
            logging.warning("Strategy conditions must be a list")
            return False
            
        if strategy['name'] in self.learning_strategies:
            logging.warning(f"Strategy {strategy['name']} already exists")
            return False
            
        return True

    def _update_strategies(self, task: Dict, evaluation: Dict) -> None:
        """
        Updates effectiveness ratings of used strategies based on task performance.
        
        Args:
            task: Dictionary containing task details and strategies used
            evaluation: Performance evaluation results
        """
        try:
            strategies_used = task.get('strategies_used', [])
            performance_score = evaluation.get('success_rate', 0.0)
            
            for strategy_name in strategies_used:
                if strategy_name in self.learning_strategies:
                    # Update effectiveness using exponential moving average
                    alpha = 0.3  # Learning rate
                    current_effectiveness = self.learning_strategies[strategy_name]['effectiveness']
                    new_effectiveness = (alpha * performance_score + 
                                      (1 - alpha) * current_effectiveness)
                    
                    self.learning_strategies[strategy_name]['effectiveness'] = new_effectiveness
                    
                    # Log significant changes in effectiveness
                    if abs(new_effectiveness - current_effectiveness) > 0.2:
                        logging.info(f"Significant change in strategy effectiveness: {strategy_name}")
                        
        except Exception as e:
            logging.error(f"Error updating strategies: {e}")

    def get_performance_analytics(self) -> Dict:
        """
        Analyzes performance history to generate insights.
        
        Returns:
            Dictionary containing performance metrics and trends
        """
        try:
            if not self.performance_history:
                return {}
                
            recent_performances = [p['evaluation']['success_rate'] 
                                 for p in self.performance_history[-10:]]
            
            return {
                'average_performance': np.mean(recent_performances),
                'performance_trend': self._calculate_trend(recent_performances),
                'top_strategies': self._get_top_strategies(3),
                'cognitive_load_history': [p['cognitive_state']['cognitive_load'] 
                                         for p in self.performance_history[-10:]]
            }
        except Exception as e:
            logging.error(f"Error generating performance analytics: {e}")
            return {}

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculates trend direction from a series of values."""
        if len(values) < 2:
            return "insufficient_data"
            
        slope = np.polyfit(range(len(values)), values, 1)[0]
        if slope > 0.1:
            return "improving"
        elif slope < -0.1:
            return "declining"
        return "stable"

    def _get_top_strategies(self, n: int) -> List[Dict]:
        """Returns the n most effective learning strategies."""
        sorted_strategies = sorted(
            self.learning_strategies.items(),
            key=lambda x: x[1]['effectiveness'],
            reverse=True
        )
        
        return [
            {
                'name': name,
                'effectiveness': data['effectiveness'],
                'description': data['description']
            }
            for name, data in sorted_strategies[:n]
        ]

    def reset_cognitive_state(self) -> None:
        """Resets cognitive state to default values."""
        self.cognitive_state = {
            'attention_level': 1.0,
            'processing_depth': 1.0,
            'cognitive_load': 0.0
        }
