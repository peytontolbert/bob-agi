import logging
import faiss
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime

class KnowledgeSystem:
    def __init__(self):
        self.dimension = 768  # Increased dimension for richer representations
        self.knowledge_base = faiss.IndexFlatL2(self.dimension)
        self.metadata = []
        self.update_timestamps = []
        self.importance_scores = []
        
        # Add computer environment specific tracking
        self.active_application = None
        self.active_window = None
        self.current_view = None
        self.interaction_history = []
        self.system_state = {
            'running_apps': set(),
            'resource_usage': {},
            'notifications': [],
            'interaction_mode': 'idle'
        }

    # Add computer environment specific methods
    def get_active_application(self) -> str:
        """Returns the currently active application."""
        return self.active_application

    def get_active_window(self) -> str:
        """Returns the currently active window."""
        return self.active_window

    def get_current_view(self) -> Dict:
        """Returns the current view state."""
        return self.current_view

    def get_current_action(self) -> Dict:
        """Returns the current user action being performed."""
        if not self.interaction_history:
            return None
        return self.interaction_history[-1]

    def get_interaction_target(self) -> Dict:
        """Returns the current interaction target."""
        if not self.current_view:
            return None
        return self.current_view.get('focused_element')

    def is_mouse_active(self) -> bool:
        """Checks if mouse is currently being used."""
        return self.system_state['interaction_mode'] == 'mouse'

    def is_keyboard_active(self) -> bool:
        """Checks if keyboard is currently being used."""
        return self.system_state['interaction_mode'] == 'keyboard'

    def is_audio_active(self) -> bool:
        """Checks if audio is currently active."""
        return any(app.get('audio_active', False) 
                  for app in self.system_state['running_apps'])

    def get_task_type(self) -> str:
        """Returns the current task type."""
        current_context = self._get_current_context()
        return self._classify_task(current_context)

    def get_task_application(self) -> str:
        """Returns the application associated with current task."""
        return self.active_application

    def get_task_objective(self) -> str:
        """Returns the current task objective."""
        # Implement task objective inference based on context
        context = self._get_current_context()
        return self._infer_objective(context)

    def get_task_progress(self) -> float:
        """Returns the current task progress."""
        # Implement task progress tracking
        return self._calculate_task_progress()

    def get_application_category(self) -> str:
        """Returns the category of the current application."""
        app_categories = {
            'discord': 'communication',
            'browser': 'web',
            'notepad': 'text_editor',
            # Add more categories
        }
        return app_categories.get(self.active_application, 'unknown')

    def get_interaction_mode(self) -> str:
        """Returns the current interaction mode."""
        return self.system_state['interaction_mode']

    def get_running_applications(self) -> List[str]:
        """Returns list of running applications."""
        return list(self.system_state['running_apps'])

    def get_system_state(self) -> Dict:
        """Returns current system state."""
        return self.system_state

    def get_pending_notifications(self) -> List[Dict]:
        """Returns pending notifications."""
        return self.system_state['notifications']

    def get_resource_usage(self) -> Dict:
        """Returns current resource usage."""
        return self.system_state['resource_usage']

    def _get_current_context(self) -> Dict:
        """Builds comprehensive current context."""
        return {
            'application': self.active_application,
            'window': self.active_window,
            'view': self.current_view,
            'interaction_history': self.interaction_history[-5:],  # Last 5 interactions
            'system_state': self.system_state
        }

    def _classify_task(self, context: Dict) -> str:
        """Classifies the current task type based on context."""
        # Implement task classification logic
        if context['application'] == 'discord':
            if context['view'].get('type') == 'voice_channel':
                return 'voice_communication'
            return 'text_communication'
        elif context['application'] == 'browser':
            return 'web_browsing'
        return 'unknown_task'

    def _infer_objective(self, context: Dict) -> str:
        """Infers the current task objective from context."""
        # Implement objective inference logic
        recent_actions = context['interaction_history']
        current_view = context['view']
        
        # Example inference logic
        if context['application'] == 'discord':
            if 'join_voice' in str(recent_actions):
                return 'join_voice_channel'
            elif 'type_message' in str(recent_actions):
                return 'send_message'
        
        return 'unknown_objective'

    def _calculate_task_progress(self) -> float:
        """Calculates current task progress."""
        # Implement progress tracking logic
        # Return value between 0.0 and 1.0
        return 0.0  # Placeholder

    def create_entry(self, entry: Dict[str, Any]) -> bool:
        """
        Enhanced entry creation with additional metadata.
        """
        try:
            if not self._validate_entry(entry):
                raise ValueError("Invalid entry format")
                
            vector = np.array([entry['vector']]).astype('float32')
            self.knowledge_base.add(vector)
            
            # Enhanced metadata
            metadata = {
                **entry['metadata'],
                'creation_time': datetime.now(),
                'last_access': datetime.now(),
                'access_count': 0,
                'importance_score': self._calculate_importance(entry),
                'relationships': self._identify_relationships(entry)
            }
            
            self.metadata.append(metadata)
            self.update_timestamps.append(datetime.now())
            self.importance_scores.append(metadata['importance_score'])
            
            logging.info(f"Enhanced entry created: {metadata}")
            return True
            
        except Exception as e:
            logging.error(f"Error creating entry: {e}")
            return False

    def get_relevant_knowledge(self, context: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Retrieves relevant knowledge based on context and importance.
        """
        try:
            if context:
                # Context-based retrieval
                vector = self._context_to_vector(context)
                results = self.search_similar(vector, k=10)
            else:
                # Importance-based retrieval
                results = self._get_important_knowledge()
            
            # Update access patterns
            self._update_access_patterns([r['metadata'] for r in results])
            
            return results
            
        except Exception as e:
            logging.error(f"Error retrieving knowledge: {e}")
            return []

    def _get_important_knowledge(self, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Retrieves knowledge entries above an importance threshold.
        
        Args:
            threshold: Minimum importance score (0-1)
            
        Returns:
            List of important knowledge entries with metadata
        """
        try:
            important_entries = []
            for idx, score in enumerate(self.importance_scores):
                if score >= threshold:
                    important_entries.append({
                        'metadata': self.metadata[idx],
                        'importance_score': score
                    })
            
            # Sort by importance score
            important_entries.sort(key=lambda x: x['importance_score'], reverse=True)
            return important_entries
            
        except Exception as e:
            logging.error(f"Error retrieving important knowledge: {e}")
            return []

    def _calculate_importance(self, entry: Dict) -> float:
        """
        Calculates importance score based on multiple factors.
        
        Args:
            entry: Knowledge entry with metadata
            
        Returns:
            Importance score between 0.0 and 1.0
        """
        try:
            # Factor 1: Recency
            recency = 0.3
            if 'creation_time' in entry['metadata']:
                age = (datetime.now() - entry['metadata']['creation_time']).total_seconds()
                recency = max(0.0, 1.0 - (age / (24 * 60 * 60)))  # Decay over 24 hours
                
            # Factor 2: Access frequency 
            frequency = 0.2
            if 'access_count' in entry['metadata']:
                frequency = min(1.0, entry['metadata']['access_count'] / 10)  # Cap at 10 accesses
                
            # Factor 3: Relationship density
            relationships = 0.2
            if 'relationships' in entry['metadata']:
                num_relations = len(entry['metadata']['relationships'])
                relationships = min(1.0, num_relations / 5)  # Cap at 5 relationships
                
            # Factor 4: Context relevance
            relevance = self._calculate_context_relevance(entry)
            
            # Combine factors with weights
            importance = (0.3 * recency + 
                         0.2 * frequency +
                         0.2 * relationships +
                         0.3 * relevance)
                         
            return round(min(1.0, importance), 2)
            
        except Exception as e:
            logging.error(f"Error calculating importance: {e}")
            return 0.0

    def _calculate_context_relevance(self, entry: Dict) -> float:
        """
        Calculates how relevant an entry is to current context.
        
        Args:
            entry: Knowledge entry to evaluate
            
        Returns:
            Relevance score between 0.0 and 1.0
        """
        try:
            current_context = self._get_current_context()
            
            # Extract key terms from current context
            context_terms = set()
            for key, value in current_context.items():
                if isinstance(value, dict):
                    context_terms.update(str(v).lower() for v in value.values())
                else:
                    context_terms.add(str(value).lower())
                    
            # Extract terms from entry
            entry_terms = set()
            if 'metadata' in entry:
                for value in entry['metadata'].values():
                    if isinstance(value, (str, int, float)):
                        entry_terms.add(str(value).lower())
                        
            # Calculate overlap
            if not entry_terms or not context_terms:
                return 0.0
                
            intersection = len(entry_terms.intersection(context_terms))
            union = len(entry_terms.union(context_terms))
            
            return round(intersection / union if union > 0 else 0.0, 2)
            
        except Exception as e:
            logging.error(f"Error calculating context relevance: {e}")
            return 0.0

    def _identify_relationships(self, entry: Dict) -> List[Dict]:
        """
        Identifies relationships with existing knowledge.
        
        Args:
            entry: Knowledge entry to find relationships for
            
        Returns:
            List of relationship dictionaries
        """
        try:
            relationships = []
            entry_vector = np.array([entry['vector']]).astype('float32')
            
            # Find similar entries
            distances, indices = self.knowledge_base.search(entry_vector, k=5)
            
            for i, idx in enumerate(indices[0]):
                if idx != -1:  # Valid index
                    relationship = {
                        'related_entry_id': self.metadata[idx].get('id'),
                        'relationship_type': 'similar',
                        'strength': 1.0 - float(distances[0][i]),  # Convert distance to similarity
                        'created_at': datetime.now()
                    }
                    relationships.append(relationship)
                    
            return relationships
            
        except Exception as e:
            logging.error(f"Error identifying relationships: {e}")
            return []

    def search_similar(self, query_vector: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """
        Searches for similar entries using vector similarity.
        """
        try:
            query_vector = np.array([query_vector]).astype('float32')
            distances, indices = self.knowledge_base.search(query_vector, k)
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx != -1:  # Valid index
                    results.append({
                        'metadata': self.metadata[idx],
                        'distance': float(distances[0][i])
                    })
            return results
            
        except Exception as e:
            logging.error(f"Error searching entries: {e}")
            return []

    def _update_access_patterns(self, entries: List[Dict]) -> None:
        """
        Updates access patterns for retrieved entries.
        
        Args:
            entries: List of metadata entries that were accessed
        """
        try:
            for entry in entries:
                if 'id' in entry:  # Find entry in metadata list
                    idx = next((i for i, m in enumerate(self.metadata) 
                              if m.get('id') == entry['id']), None)
                    if idx is not None:
                        # Update access count and timestamp
                        self.metadata[idx]['access_count'] = self.metadata[idx].get('access_count', 0) + 1
                        self.metadata[idx]['last_access'] = datetime.now()
                        
                        # Recalculate importance score
                        self.importance_scores[idx] = self._calculate_importance({
                            'metadata': self.metadata[idx]
                        })
        except Exception as e:
            logging.error(f"Error updating access patterns: {e}")

    def add_experience(self, experience: Dict[str, Any]) -> bool:
        """
        Adds an experience entry to the knowledge system.
        
        Args:
            experience: Dictionary containing:
                - action: The action performed
                - result: Result of the action
                - context: Context in which action was performed
                
        Returns:
            bool: Success status
        """
        try:
            # Convert experience to vector representation
            vector = self._experience_to_vector(experience)
            
            # Create metadata for the experience
            metadata = {
                'type': 'experience',
                'action': experience['action'],
                'result': experience['result'],
                'context': experience['context'],
                'timestamp': datetime.now(),
                'success': experience['result'].get('success', False)
            }
            
            # Create entry with vector and metadata
            entry = {
                'vector': vector,
                'metadata': metadata
            }
            
            # Add to knowledge base
            return self.create_entry(entry)
            
        except Exception as e:
            logging.error(f"Error adding experience: {e}")
            return False

    def _experience_to_vector(self, experience: Dict) -> np.ndarray:
        """
        Converts an experience into a vector representation.
        """
        try:
            # Initialize vector with zeros
            vector = np.zeros(self.dimension, dtype=np.float32)
            
            # Extract key information from experience
            action_type = experience['action'].get('type', '')
            result_success = experience['result'].get('success', False)
            context = experience['context']
            
            # Simple encoding scheme (can be enhanced with proper embeddings)
            # Use different ranges of the vector for different aspects
            vector[0] = 1.0 if result_success else 0.0
            
            # Encode action type
            action_types = ['interaction', 'speech', 'command', 'thought_driven']
            if action_type in action_types:
                idx = action_types.index(action_type)
                vector[1 + idx] = 1.0
                
            # Add more sophisticated encoding as needed
            
            return vector
            
        except Exception as e:
            logging.error(f"Error converting experience to vector: {e}")
            return np.zeros(self.dimension, dtype=np.float32)

    def _validate_entry(self, entry):
        """Validate knowledge entry format"""
        required_fields = ['content', 'timestamp', 'type']
        
        if not isinstance(entry, dict):
            return False
            
        for field in required_fields:
            if field not in entry:
                return False
                
        return True
        
    def add_entry(self, entry):
        """Add new knowledge entry"""
        if self._validate_entry(entry):
            self.entries.append(entry)
            return True
        return False
