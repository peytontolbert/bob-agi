"""
The BaseMemorySystem class is the foundation for all memory systems in the application.
"""
import logging
from typing import List, Dict, Any
import numpy as np
from .knowledge_system import KnowledgeSystem
from .autobiographical import AutobiographicalMemorySystem
from .causual import CausalKnowledgeSystem
from .conceptual import ConceptualKnowledgeSystem
from .episodic import EpisodicMemorySystem
from .metacognitive import MetaCognitiveSystem
from .procedural import ProceduralKnowledgeSystem
from .semantic import SemanticKnowledgeSystem
from .spatial import SpatialKnowledgeSystem
from .temporal import TemporalKnowledgeSystem
from .internalmonologue import InternalMonologue

class BaseMemorySystem:
    def __init__(self):
        # Initialize the shared knowledge system
        self.knowledge_system = KnowledgeSystem()
        
        # Initialize all memory subsystems with the knowledge system
        self.autobiographical = AutobiographicalMemorySystem(self.knowledge_system)
        self.causal = CausalKnowledgeSystem(self.knowledge_system)
        self.conceptual = ConceptualKnowledgeSystem(self.knowledge_system)
        self.episodic = EpisodicMemorySystem(self.knowledge_system)
        self.metacognitive = MetaCognitiveSystem(self.knowledge_system)
        self.procedural = ProceduralKnowledgeSystem(self.knowledge_system)
        self.semantic = SemanticKnowledgeSystem(self.knowledge_system)
        self.spatial = SpatialKnowledgeSystem(self.knowledge_system)
        self.temporal = TemporalKnowledgeSystem(self.knowledge_system)
        self.internal_monologue = InternalMonologue(self.knowledge_system)
        self.memories: List[Dict[str, Any]] = []
        logging.info("BaseMemorySystem initialized with all subsystems")

    def add_memory(self, memory: Dict[str, Any]) -> bool:
        """
        Adds a memory to appropriate subsystems based on its type.
        """
        try:
            # Add to general memories
            self.memories.append(memory)
            
            # Route to appropriate subsystems
            if 'type' in memory:
                if memory['type'] == 'episodic':
                    self.episodic.store_episode(memory)
                elif memory['type'] == 'semantic':
                    self.semantic.store_concept(memory)
                elif memory['type'] == 'procedural':
                    self.procedural.learn_skill(memory)
                elif memory['type'] == 'spatial':
                    self.spatial.add_location(memory)
                elif memory['type'] == 'temporal':
                    self.temporal.add_temporal_event(memory)
                elif memory['type'] == 'autobiographical':
                    self.autobiographical.store_experience(memory)
                elif memory['type'] == 'internal_monologue':
                    self.internal_monologue.store_inner_thought(memory)
                    
            logging.debug(f"Memory added: {memory}")
            return True
            
        except Exception as e:
            logging.error(f"Error adding memory: {e}")
            return False

    def get_memories(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Retrieves memories based on query parameters.
        """
        try:
            # Create query vector
            query_vector = self._vectorize_query(query)
            
            # Search across all memory systems
            results = self.knowledge_system.search_similar(query_vector)
            
            # Filter and sort results
            filtered_results = self._filter_results(results, query)
            return filtered_results
            
        except Exception as e:
            logging.error(f"Error retrieving memories: {e}")
            return []
    
    def _vectorize_query(self, query: Dict[str, Any]) -> np.ndarray:
        """Creates a vector representation of the query."""
        # Placeholder - replace with actual embedding logic
        return np.random.rand(384)
    
    def _filter_results(self, results: List[Dict], 
                       query: Dict[str, Any]) -> List[Dict]:
        """Filters and ranks search results based on query criteria."""
        filtered = []
        for result in results:
            if self._matches_criteria(result['metadata'], query):
                filtered.append(result)
        return filtered
    
    def _matches_criteria(self, memory: Dict, query: Dict) -> bool:
        """Checks if a memory matches the query criteria."""
        return all(
            key in memory and memory[key] == value
            for key, value in query.items()
        )
