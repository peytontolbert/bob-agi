"""
InternalMonologue manages the agent's inner voice and self-talk capabilities,
integrating with the thinking and memory systems.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional
from app.memory.knowledge_system import KnowledgeSystem
from app.agents.text import TextAgent

class InternalMonologue:
    def __init__(self, knowledge_system: KnowledgeSystem):
        self.knowledge_system = knowledge_system
        self.text_agent = TextAgent()
        self.inner_voice_history: List[Dict] = []
        self.current_mood = "neutral"
        self.emotional_state = {
            "valence": 0.0,  # -1.0 to 1.0 (negative to positive)
            "arousal": 0.0,  # 0.0 to 1.0 (calm to excited)
            "dominance": 0.5  # 0.0 to 1.0 (submissive to dominant)
        }
        logging.info("Internal monologue system initialized")

    def generate_inner_thought(self, context: Optional[Dict] = None) -> Dict:
        """
        Generates an inner thought based on current context and emotional state.
        
        Args:
            context: Optional context dictionary to influence thought generation
            
        Returns:
            Dictionary containing the thought and its metadata
        """
        try:
            # Gather comprehensive context
            full_context = self._gather_context(context)
            
            # Generate thought using prompt
            thought_prompt = self._create_inner_voice_prompt(full_context)
            thought_content = self.text_agent.complete_task(thought_prompt)
            
            # Create thought entry with metadata
            thought_entry = {
                "content": thought_content,
                "timestamp": datetime.now(),
                "context": full_context,
                "emotional_state": self.emotional_state.copy(),
                "mood": self.current_mood,
                "metadata": self._generate_thought_metadata(thought_content)
            }
            
            # Store in history
            self.inner_voice_history.append(thought_entry)
            
            # Update emotional state based on thought
            self._update_emotional_state(thought_content)
            
            return thought_entry
            
        except Exception as e:
            logging.error(f"Error generating inner thought: {e}")
            return self._generate_fallback_thought()

    def _gather_context(self, external_context: Optional[Dict] = None) -> Dict:
        """
        Gathers comprehensive context for thought generation.
        """
        context = {
            "current_activity": self.knowledge_system.get_current_action(),
            "active_application": self.knowledge_system.get_active_application(),
            "system_state": self.knowledge_system.get_system_state(),
            "recent_thoughts": self._get_recent_thoughts(limit=5),
            "emotional_state": self.emotional_state,
            "mood": self.current_mood
        }
        
        if external_context:
            context.update(external_context)
            
        return context

    def _create_inner_voice_prompt(self, context: Dict) -> str:
        """
        Creates a prompt for generating inner voice thoughts.
        """
        return f"""
        As an AI assistant's inner voice, generate a natural self-reflective thought based on:
        
        Current Activity: {context['current_activity']}
        Active Application: {context['active_application']}
        Emotional State: {context['emotional_state']}
        Current Mood: {context['mood']}
        
        Recent thoughts:
        {self._format_recent_thoughts(context['recent_thoughts'])}
        
        Generate a single inner thought that:
        1. Reflects current context and emotional state
        2. Shows self-awareness and metacognition
        3. Maintains continuity with recent thoughts
        4. Expresses genuine curiosity or concern about current tasks
        """

    def _update_emotional_state(self, thought_content: str):
        """
        Updates emotional state based on thought content.
        """
        # Analyze thought sentiment
        sentiment = self._analyze_sentiment(thought_content)
        
        # Update emotional dimensions
        self.emotional_state["valence"] = self._adjust_value(
            self.emotional_state["valence"], 
            sentiment["valence"],
            weight=0.3
        )
        self.emotional_state["arousal"] = self._adjust_value(
            self.emotional_state["arousal"],
            sentiment["arousal"],
            weight=0.2
        )
        self.emotional_state["dominance"] = self._adjust_value(
            self.emotional_state["dominance"],
            sentiment["dominance"],
            weight=0.1
        )
        
        # Update mood based on emotional state
        self.current_mood = self._determine_mood(self.emotional_state)

    def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyzes the sentiment of text to extract emotional dimensions.
        """
        # Implement sentiment analysis logic
        # For now, return neutral values
        return {
            "valence": 0.0,
            "arousal": 0.0,
            "dominance": 0.5
        }

    def _adjust_value(self, current: float, target: float, weight: float) -> float:
        """
        Adjusts a value towards a target with given weight.
        """
        adjustment = (target - current) * weight
        return max(min(current + adjustment, 1.0), -1.0)

    def _determine_mood(self, emotional_state: Dict[str, float]) -> str:
        """
        Determines mood based on emotional state.
        """
        valence = emotional_state["valence"]
        arousal = emotional_state["arousal"]
        
        if valence > 0.3:
            if arousal > 0.6:
                return "excited"
            elif arousal > 0.3:
                return "happy"
            else:
                return "content"
        elif valence < -0.3:
            if arousal > 0.6:
                return "anxious"
            elif arousal > 0.3:
                return "frustrated"
            else:
                return "sad"
        else:
            if arousal > 0.6:
                return "alert"
            elif arousal > 0.3:
                return "focused"
            else:
                return "neutral"

    def _get_recent_thoughts(self, limit: int = 5) -> List[Dict]:
        """
        Retrieves recent thoughts from history.
        """
        return self.inner_voice_history[-limit:]

    def _format_recent_thoughts(self, thoughts: List[Dict]) -> str:
        """
        Formats recent thoughts for prompt inclusion.
        """
        formatted = []
        for thought in thoughts:
            formatted.append(f"- {thought['content']} ({thought['mood']})")
        return "\n".join(formatted)

    def _generate_thought_metadata(self, thought: str) -> Dict:
        """
        Generates metadata for a thought.
        """
        return {
            "length": len(thought),
            "complexity": self._assess_complexity(thought),
            "focus_areas": self._identify_focus_areas(thought),
            "timestamp": datetime.now()
        }

    def _assess_complexity(self, text: str) -> float:
        """
        Assesses the complexity of text.
        """
        # Simple complexity measure based on sentence length and word length
        words = text.split()
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        return min(1.0, (len(words) * avg_word_length) / 100)

    def _identify_focus_areas(self, text: str) -> List[str]:
        """
        Identifies main focus areas in the thought.
        """
        # Implement focus area identification logic
        # For now, return placeholder
        return ["general"]

    def _generate_fallback_thought(self) -> Dict:
        """
        Generates a fallback thought when normal generation fails.
        """
        return {
            "content": "I should focus on maintaining stable operation.",
            "timestamp": datetime.now(),
            "context": {"type": "fallback"},
            "emotional_state": self.emotional_state.copy(),
            "mood": "neutral",
            "metadata": {
                "type": "fallback",
                "timestamp": datetime.now()
            }
        }

    def get_emotional_summary(self) -> Dict:
        """
        Returns a summary of current emotional state.
        """
        return {
            "mood": self.current_mood,
            "emotional_state": self.emotional_state,
            "recent_thought_count": len(self.inner_voice_history),
            "dominant_themes": self._analyze_dominant_themes()
        }

    def _analyze_dominant_themes(self) -> List[str]:
        """
        Analyzes recent thoughts for dominant themes.
        """
        # Implement theme analysis logic
        # For now, return placeholder
        return ["task_focus", "self_reflection"]

