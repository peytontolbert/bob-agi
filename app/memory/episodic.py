"""
The EpisodicMemorySystem manages episodic memory, storing and retrieving task-related experiences.
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
import logging
from app.memory.knowledge_system import KnowledgeSystem

class EpisodicMemorySystem:
    def __init__(self, knowledge_system: KnowledgeSystem):
        self.knowledge_system = knowledge_system
        self.episodes: List[Dict] = []
        self.temporal_index: Dict[str, List[int]] = {}  # Maps timeframes to episode indices
        self.emotional_index: Dict[str, List[int]] = {}  # Maps emotional states to episode indices
        self.context_index: Dict[str, List[int]] = {}  # Maps context keys to episode indices
        
    def store_episode(self, episode: Dict) -> bool:
        """
        Stores an episodic memory with temporal context.
        
        Args:
            episode: Dictionary containing:
                - content: The actual memory content
                - context: Environmental/situational context
                - emotions: Associated emotional states
                - importance: Optional importance score (0-1)
                - tags: Optional list of categorical tags
        """
        try:
            if not self._validate_episode(episode):
                logging.error("Invalid episode format")
                return False
                
            # Add temporal metadata
            timestamp = datetime.now()
            episode['stored_at'] = timestamp
            episode['last_accessed'] = timestamp
            episode['access_count'] = 0
            episode['importance'] = episode.get('importance', self._calculate_importance(episode))
            
            # Generate unique ID
            episode['id'] = f"ep_{len(self.episodes)}_{timestamp.strftime('%Y%m%d%H%M%S')}"
            
            # Store in episodes list
            episode_idx = len(self.episodes)
            self.episodes.append(episode)
            
            # Update indices
            self._update_indices(episode, episode_idx)
            
            # Create vector representation and store in knowledge system
            vector = self._vectorize_episode(episode)
            self.knowledge_system.create_entry({
                'vector': vector,
                'metadata': episode
            })
            
            logging.info(f"Stored episode {episode['id']}")
            return True
            
        except Exception as e:
            logging.error(f"Error storing episode: {e}")
            return False
            
    def recall_similar_episodes(self, query_episode: Dict, limit: int = 5) -> List[Dict]:
        """
        Recalls episodes similar to the query episode.
        
        Args:
            query_episode: Episode to find similar matches for
            limit: Maximum number of episodes to return
            
        Returns:
            List of similar episodes with similarity scores
        """
        try:
            query_vector = self._vectorize_episode(query_episode)
            similar_episodes = self.knowledge_system.search_similar(query_vector, k=limit)
            
            # Update access patterns
            for episode in similar_episodes:
                self._update_access_metadata(episode['metadata']['id'])
                
            return similar_episodes
            
        except Exception as e:
            logging.error(f"Error recalling episodes: {e}")
            return []
            
    def recall_by_timeframe(self, start_time: datetime, end_time: datetime) -> List[Dict]:
        """
        Recalls episodes within a specific timeframe.
        """
        try:
            episodes = []
            for episode in self.episodes:
                timestamp = episode['stored_at']
                if start_time <= timestamp <= end_time:
                    episodes.append(episode)
                    self._update_access_metadata(episode['id'])
                    
            return sorted(episodes, key=lambda x: x['importance'], reverse=True)
            
        except Exception as e:
            logging.error(f"Error recalling by timeframe: {e}")
            return []
            
    def recall_by_emotion(self, emotion: str) -> List[Dict]:
        """
        Recalls episodes associated with a specific emotion.
        """
        try:
            episode_indices = self.emotional_index.get(emotion.lower(), [])
            episodes = [self.episodes[idx] for idx in episode_indices]
            
            for episode in episodes:
                self._update_access_metadata(episode['id'])
                
            return sorted(episodes, key=lambda x: x['importance'], reverse=True)
            
        except Exception as e:
            logging.error(f"Error recalling by emotion: {e}")
            return []
            
    def recall_by_context(self, context_key: str, context_value: str) -> List[Dict]:
        """
        Recalls episodes matching a specific context.
        """
        try:
            matching_episodes = []
            context_key = context_key.lower()
            context_value = context_value.lower()
            
            for episode in self.episodes:
                if episode['context'].get(context_key, '').lower() == context_value:
                    matching_episodes.append(episode)
                    self._update_access_metadata(episode['id'])
                    
            return sorted(matching_episodes, key=lambda x: x['importance'], reverse=True)
            
        except Exception as e:
            logging.error(f"Error recalling by context: {e}")
            return []

    def _validate_episode(self, episode: Dict) -> bool:
        """
        Validates episode structure and content.
        """
        required_fields = ['content', 'context', 'emotions']
        
        if not all(field in episode for field in required_fields):
            return False
            
        if not isinstance(episode['content'], str) or not episode['content'].strip():
            return False
            
        if not isinstance(episode['context'], dict):
            return False
            
        if not isinstance(episode['emotions'], (list, dict)):
            return False
            
        return True
        
    def _vectorize_episode(self, episode: Dict) -> np.ndarray:
        """
        Creates a vector representation of an episode.
        
        Args:
            episode: Episode dictionary to vectorize
            
        Returns:
            Vector representation of the episode
        """
        try:
            # Initialize vector with zeros
            vector = np.zeros(384, dtype=np.float32)  # Using same dimension as knowledge system
            
            # Encode basic episode properties
            # Note: In a real implementation, you would use proper embedding models
            
            # Encode emotional state (first 64 dimensions)
            if isinstance(episode['emotions'], dict):
                for i, (emotion, value) in enumerate(episode['emotions'].items()[:8]):
                    vector[i*8] = float(value)
            else:
                for i, emotion in enumerate(episode['emotions'][:8]):
                    vector[i*8] = 1.0
                    
            # Encode context (next 128 dimensions)
            context_str = str(episode['context'])
            context_hash = hash(context_str) % (2**32)
            vector[64:192] = np.frombuffer(context_hash.to_bytes(16, 'big'), dtype=np.float32)
            
            # Encode content (remaining dimensions)
            content_hash = hash(episode['content']) % (2**32)
            vector[192:] = np.frombuffer(content_hash.to_bytes(24, 'big'), dtype=np.float32)
            
            # Normalize vector
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
                
            return vector
            
        except Exception as e:
            logging.error(f"Error vectorizing episode: {e}")
            return np.zeros(384, dtype=np.float32)
            
    def _calculate_importance(self, episode: Dict) -> float:
        """
        Calculates importance score for an episode.
        """
        try:
            importance = 0.0
            
            # Factor 1: Emotional intensity (30%)
            emotional_intensity = 0.0
            if isinstance(episode['emotions'], dict):
                emotional_intensity = sum(episode['emotions'].values()) / len(episode['emotions'])
            else:
                emotional_intensity = len(episode['emotions']) / 5  # Normalize by assuming max 5 emotions
                
            # Factor 2: Context richness (30%)
            context_richness = len(episode['context']) / 10  # Normalize by assuming max 10 context items
            
            # Factor 3: Content length (20%)
            content_length = len(episode['content']) / 1000  # Normalize by assuming max 1000 chars
            
            # Factor 4: Explicit importance if provided (20%)
            explicit_importance = episode.get('importance', 0.5)
            
            # Combine factors
            importance = (0.3 * emotional_intensity +
                        0.3 * min(1.0, context_richness) +
                        0.2 * min(1.0, content_length) +
                        0.2 * explicit_importance)
                        
            return round(min(1.0, importance), 2)
            
        except Exception as e:
            logging.error(f"Error calculating importance: {e}")
            return 0.5
            
    def _update_indices(self, episode: Dict, episode_idx: int) -> None:
        """
        Updates the various indices with the new episode.
        """
        try:
            # Update temporal index
            timeframe = episode['stored_at'].strftime('%Y%m')
            if timeframe not in self.temporal_index:
                self.temporal_index[timeframe] = []
            self.temporal_index[timeframe].append(episode_idx)
            
            # Update emotional index
            if isinstance(episode['emotions'], dict):
                emotions = episode['emotions'].keys()
            else:
                emotions = episode['emotions']
                
            for emotion in emotions:
                emotion = emotion.lower()
                if emotion not in self.emotional_index:
                    self.emotional_index[emotion] = []
                self.emotional_index[emotion].append(episode_idx)
                
            # Update context index
            for key, value in episode['context'].items():
                context_key = f"{key}:{value}".lower()
                if context_key not in self.context_index:
                    self.context_index[context_key] = []
                self.context_index[context_key].append(episode_idx)
                
        except Exception as e:
            logging.error(f"Error updating indices: {e}")
            
    def _update_access_metadata(self, episode_id: str) -> None:
        """
        Updates access metadata for an episode.
        """
        try:
            for episode in self.episodes:
                if episode['id'] == episode_id:
                    episode['last_accessed'] = datetime.now()
                    episode['access_count'] = episode.get('access_count', 0) + 1
                    break
                    
        except Exception as e:
            logging.error(f"Error updating access metadata: {e}")

    def get_episode_stats(self) -> Dict:
        """
        Returns statistics about stored episodes.
        """
        try:
            stats = {
                'total_episodes': len(self.episodes),
                'timeframes': len(self.temporal_index),
                'unique_emotions': len(self.emotional_index),
                'unique_contexts': len(self.context_index),
                'avg_importance': np.mean([ep['importance'] for ep in self.episodes]) if self.episodes else 0,
                'most_common_emotions': self._get_most_common_emotions(5),
                'most_accessed': self._get_most_accessed_episodes(5)
            }
            return stats
            
        except Exception as e:
            logging.error(f"Error generating episode stats: {e}")
            return {}
            
    def _get_most_common_emotions(self, limit: int) -> List[Tuple[str, int]]:
        """Returns the most common emotions across episodes."""
        emotion_counts = {}
        for emotions in self.emotional_index.items():
            emotion_counts[emotions[0]] = len(emotions[1])
        return sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)[:limit]
        
    def _get_most_accessed_episodes(self, limit: int) -> List[Dict]:
        """Returns the most frequently accessed episodes."""
        return sorted(self.episodes, 
                     key=lambda x: x.get('access_count', 0),
                     reverse=True)[:limit]
