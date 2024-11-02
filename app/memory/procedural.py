"""
The ProceduralKnowledgeSystem manages knowledge about skills, actions, and procedures.
"""
from typing import Dict, List, Optional
import numpy as np
import logging
from datetime import datetime

class ProceduralKnowledgeSystem:
    def __init__(self, knowledge_system):
        self.knowledge_system = knowledge_system
        self.skills = {}  # Stores learned skills and procedures
        self.action_sequences = {}  # Stores common action patterns
        self.skill_proficiency = {}  # Tracks skill mastery levels
        
    def learn_skill(self, skill: Dict) -> bool:
        """
        Learns a new skill or procedure.
        
        Args:
            skill: Dictionary containing:
                - name: Skill name
                - steps: Ordered list of actions/steps
                - prerequisites: Required prior knowledge
                - difficulty: Skill complexity (0-1)
                - context: When/where skill is applicable
        """
        try:
            if not self._validate_skill(skill):
                return False
                
            skill['learned_at'] = datetime.now()
            skill['proficiency'] = 0.1  # Initial proficiency
            
            self.skills[skill['name']] = skill
            self.skill_proficiency[skill['name']] = {
                'level': 0.1,
                'practice_count': 0,
                'last_used': datetime.now()
            }
            
            # Store in vector space
            vector = self._vectorize_skill(skill)
            self.knowledge_system.create_entry({
                'vector': vector,
                'metadata': skill
            })
            
            return True
            
        except Exception as e:
            logging.error(f"Error learning skill: {e}")
            return False
    
    def practice_skill(self, skill_name: str, performance_metrics: Dict) -> float:
        """
        Updates skill proficiency based on practice performance.
        Returns new proficiency level.
        """
        try:
            if skill_name not in self.skill_proficiency:
                return 0.0
                
            proficiency = self.skill_proficiency[skill_name]
            
            # Update proficiency based on performance
            success_rate = performance_metrics.get('success_rate', 0.0)
            improvement = self._calculate_improvement(
                current_level=proficiency['level'],
                performance=success_rate,
                practice_count=proficiency['practice_count']
            )
            
            # Update skill metrics
            proficiency['level'] = min(1.0, proficiency['level'] + improvement)
            proficiency['practice_count'] += 1
            proficiency['last_used'] = datetime.now()
            
            return proficiency['level']
            
        except Exception as e:
            logging.error(f"Error practicing skill: {e}")
            return 0.0
    
    def get_applicable_skills(self, context: Dict) -> List[Dict]:
        """
        Finds skills applicable to the current context.
        """
        try:
            applicable_skills = []
            for skill_name, skill in self.skills.items():
                if self._context_matches(skill['context'], context):
                    skill_info = skill.copy()
                    skill_info['proficiency'] = self.skill_proficiency[skill_name]['level']
                    applicable_skills.append(skill_info)
            
            return sorted(applicable_skills, 
                         key=lambda x: x['proficiency'], 
                         reverse=True)
                         
        except Exception as e:
            logging.error(f"Error finding applicable skills: {e}")
            return []
    
    def _validate_skill(self, skill: Dict) -> bool:
        required_fields = ['name', 'steps', 'prerequisites', 'difficulty', 'context']
        return all(field in skill for field in required_fields)
    
    def _vectorize_skill(self, skill: Dict) -> np.ndarray:
        # Placeholder for actual embedding logic
        return np.random.rand(384)
    
    def _calculate_improvement(self, current_level: float, 
                             performance: float, 
                             practice_count: int) -> float:
        """Calculates skill improvement based on practice."""
        # Diminishing returns with practice
        base_improvement = performance * 0.1
        practice_factor = 1 / (1 + 0.1 * practice_count)
        return base_improvement * practice_factor
    
    def _context_matches(self, skill_context: Dict, current_context: Dict) -> bool:
        """Determines if a skill is applicable in the current context."""
        return all(
            current_context.get(key) == value
            for key, value in skill_context.items()
        )
