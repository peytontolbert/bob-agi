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
        self.memory_system = BaseMemorySystem()
        self.knowledge_system = knowledge_system
        self.text_agent = TextAgent()
        
        # Enhanced thought parameters
        self.THOUGHT_INTERVAL = 1.0  # Increased frequency to 1 second
        self.MAX_THOUGHT_AGE = 60  # Extended to 60 seconds
        self.REFLECTION_INTERVAL = 30  # Reflect every 30 seconds
        
        # Initialize specialized queues
        self.thought_queue = queue.Queue()
        self.active_thoughts = queue.Queue(maxsize=20)  # Increased capacity
        self.reflection_thoughts = queue.Queue(maxsize=10)
        
        # Add last reflection time
        self.last_reflection_time = datetime.now()
        
        # Initialize thinking thread
        self.thinking_thread = None
        self.should_think = False
        
        self.start_thinking()
    
    def start_thinking(self):
        """
        Starts the continuous thinking process in a separate thread.
        """
        if self.thinking_thread is not None and self.thinking_thread.is_alive():
            logging.warning("Thinking process is already running")
            return
            
        self.should_think = True
        self.thinking_thread = threading.Thread(target=self._continuous_thinking, daemon=True)
        self.thinking_thread.start()
        logging.info("Bob has started thinking with all memory systems initialized.")
    
    def stop_thinking(self):
        """
        Safely stops the thinking process.
        """
        self.should_think = False
        if self.thinking_thread:
            self.thinking_thread.join(timeout=5.0)
            logging.info("Thinking process has been stopped")
    
    def _continuous_thinking(self):
        """
        Enhanced continuous thinking process with integrated memory systems.
        """
        while self.should_think:
            try:
                # Gather comprehensive context
                current_context = self._gather_context()
                
                # Get inner thought from internal monologue system
                inner_thought = self.memory_system.internal_monologue.generate_inner_thought(current_context)
                
                # Generate primary thought incorporating internal monologue
                new_thought = self.generate_thoughts(current_context, inner_thought)
                
                # Create rich thought entry
                thought_entry = {
                    'content': new_thought,
                    'timestamp': datetime.now(),
                    'type': 'autonomous_thought',
                    'context': current_context,
                    'inner_thought': inner_thought,
                    'emotional_state': inner_thought['emotional_state'],
                    'metadata': self._generate_thought_metadata(new_thought)
                }
                
                # Process thought through memory systems
                self._process_thought_in_memory_systems(thought_entry)
                
                # Check if reflection is needed
                if self._should_reflect():
                    self._perform_reflection()
                
                time.sleep(self.THOUGHT_INTERVAL)
                
            except Exception as e:
                logging.error(f"Error in thinking process: {e}")
                time.sleep(1.0)
    
    def generate_thoughts(self, context: Dict, inner_thought: Dict) -> str:
        """
        Generates new thoughts based on the current context and internal monologue.
        """
        prompt = self._create_thought_prompt(context, inner_thought)
        thought = self.text_agent.complete_task(prompt)
        return thought
    
    def _create_thought_prompt(self, context: Dict, inner_thought: Dict) -> str:
        """
        Creates an enhanced prompt for thought generation incorporating internal monologue.
        """
        return f"""
        Based on these recent experiences, current context, and internal reflection:
        
        Recent memories: {context.get('recent_memories', [])}
        Active thoughts: {context.get('active_thoughts', [])}
        Current knowledge: {context.get('current_knowledge', [])}
        
        Internal voice: {inner_thought['content']}
        Current mood: {inner_thought['mood']}
        Emotional state: {inner_thought['emotional_state']}
        
        Generate a single, coherent thought that:
        1. Integrates the internal perspective
        2. Reflects current context and priorities
        3. Shows emotional awareness
        4. Maintains cognitive continuity
        """
    
    def _add_to_thought_queue(self, thought):
        """
        Adds a new thought to the thought queue.
        """
        try:
            self.thought_queue.put(thought, block=False)
            logging.debug(f"New thought added: {thought}")
        except queue.Full:
            logging.warning("Thought queue is full, dropping oldest thought")
            try:
                # Remove oldest thought and add new one
                self.thought_queue.get_nowait()
                self.thought_queue.put(thought, block=False)
            except (queue.Empty, queue.Full):
                logging.error("Failed to manage thought queue")
    
    def _update_active_thoughts(self, thought):
        """
        Updates the active thoughts queue, maintaining only recent thoughts.
        """
        try:
            if self.active_thoughts.full():
                self.active_thoughts.get_nowait()  # Remove oldest thought
            self.active_thoughts.put(thought, block=False)
        except queue.Full:
            logging.warning("Failed to update active thoughts queue")
    
    def current_thoughts(self):
        """
        Returns a list of current active thoughts.
        """
        thoughts = []
        now = datetime.now()
        
        # Create a temporary list of all thoughts
        while not self.active_thoughts.empty():
            thought = self.active_thoughts.get()
            if (now - thought['timestamp']).seconds < self.MAX_THOUGHT_AGE:
                thoughts.append(thought)
        
        # Put thoughts back in the queue
        for thought in thoughts:
            try:
                self.active_thoughts.put(thought, block=False)
            except queue.Full:
                break
                
        return thoughts
    
    def add_experience(self, experience):
        """
        Adds a new experience to the autobiographical memory.
        """
        self.memory_system.autobiographical.add_memory(experience)
        logging.info(f"Experience added to autobiographical memory: {experience}")
    
    def recall_memories(self, query):
        """
        Recalls memories based on a query from all memory systems.
        """
        memories = self.memory_system.get_memories(query)
        logging.debug(f"Memories recalled with query '{query}': {memories}")
        return memories
    
    def _process_thought_in_memory_systems(self, thought_entry: Dict[str, Any]):
        """
        Processes a thought through all relevant memory systems.
        """
        # Add to episodic memory
        self.memory_system.episodic.store_episode({
            'type': 'thought',
            'content': thought_entry['content'],
            'timestamp': thought_entry['timestamp']
        })
        
        # Store inner thought in internal monologue memory
        self.memory_system.add_memory({
            'type': 'internal_monologue',
            'content': thought_entry['inner_thought']['content'],
            'emotional_state': thought_entry['emotional_state'],
            'timestamp': thought_entry['timestamp']
        })
        
        # Update semantic memory with concepts
        concepts = self._extract_concepts(thought_entry['content'])
        self.memory_system.semantic.update_concepts(concepts)
        
        # Update causal relationships
        self.memory_system.causal.update_relations(thought_entry)
        
        # Update metacognitive system
        self.memory_system.metacognitive.process_thought(thought_entry)
        
        # Add to regular thought queues
        self._add_to_thought_queue(thought_entry)
        self._update_active_thoughts(thought_entry)
    
    def _should_reflect(self) -> bool:
        """
        Determines if it's time for reflection.
        """
        time_since_reflection = (datetime.now() - self.last_reflection_time).seconds
        return time_since_reflection >= self.REFLECTION_INTERVAL
    
    def _perform_reflection(self):
        """
        Enhanced reflection process incorporating internal monologue.
        """
        recent_thoughts = self.current_thoughts()
        reflection_context = self._gather_context()
        
        # Get emotional summary from internal monologue
        emotional_summary = self.memory_system.internal_monologue.get_emotional_summary()
        
        reflection = self.text_agent.complete_task(
            self._create_reflection_prompt(recent_thoughts, reflection_context, emotional_summary)
        )
        
        reflection_entry = {
            'content': reflection,
            'timestamp': datetime.now(),
            'type': 'reflection',
            'analyzed_thoughts': recent_thoughts,
            'emotional_state': emotional_summary
        }
        
        # Update memory systems with reflection
        self.memory_system.metacognitive.store_reflection(reflection_entry)
        self.memory_system.episodic.store_episode(reflection_entry)
        
        self.last_reflection_time = datetime.now()
    
    def _create_reflection_prompt(self, thoughts: List[Dict], 
                                context: Dict, 
                                emotional_summary: Dict) -> str:
        """
        Creates an enhanced reflection prompt incorporating emotional awareness.
        """
        return f"""
        Analyze recent thoughts and experiences to generate deeper understanding:
        
        Recent thoughts: {thoughts}
        Current context: {context}
        
        Emotional State:
        - Current mood: {emotional_summary['mood']}
        - Emotional dimensions: {emotional_summary['emotional_state']}
        - Dominant themes: {emotional_summary['dominant_themes']}
        
        Consider:
        1. Patterns and relationships between thoughts
        2. Emotional progression and impact
        3. Areas requiring more attention or learning
        4. Consistency with existing knowledge
        5. Potential improvements or adaptations
        
        Generate a comprehensive reflection that synthesizes these elements.
        """
    
    def _generate_thought_metadata(self, thought: str) -> Dict[str, Any]:
        """
        Generates rich metadata for a thought.
        """
        return {
            'concepts': self._extract_concepts(thought),
            'priority': self._assess_thought_priority(thought),
            'category': self._categorize_thought(thought),
            'relations': self._identify_relations(thought)
        }

    # Add helper methods for context gathering
    def _get_current_focus(self) -> str:
        """Gets the current focus of attention."""
        active_thoughts = self.current_thoughts()
        return active_thoughts[0]['content'] if active_thoughts else ""

    def _get_current_location(self) -> str:
        """Gets the current location in the computer environment."""
        # Return the current active window/application
        return {
            'app': self.knowledge_system.get_active_application(),
            'window': self.knowledge_system.get_active_window(),
            'view': self.knowledge_system.get_current_view()
        }

    def _get_current_activity(self) -> str:
        """Gets the current activity being performed in the computer environment."""
        return {
            'type': 'computer_interaction',
            'action': self.knowledge_system.get_current_action(),
            'target': self.knowledge_system.get_interaction_target(),
            'state': {
                'mouse_active': self.knowledge_system.is_mouse_active(),
                'keyboard_active': self.knowledge_system.is_keyboard_active(),
                'audio_active': self.knowledge_system.is_audio_active()
            }
        }

    def _get_current_task(self) -> Dict:
        """Gets the current computer task context."""
        return {
            'type': self.knowledge_system.get_task_type(),
            'application': self.knowledge_system.get_task_application(),
            'objective': self.knowledge_system.get_task_objective(),
            'progress': self.knowledge_system.get_task_progress()
        }

    def _get_current_concept(self) -> str:
        """Gets the current concept being processed."""
        return "current_concept"  # Replace with actual concept tracking

    def _get_current_domain(self) -> str:
        """Gets the current computer interaction domain."""
        return {
            'environment': 'computer',
            'application_category': self.knowledge_system.get_application_category(),
            'interaction_mode': self.knowledge_system.get_interaction_mode()
        }

    def _get_current_situation(self) -> str:
        """Gets the current computer environment situation."""
        return {
            'active_apps': self.knowledge_system.get_running_applications(),
            'system_state': self.knowledge_system.get_system_state(),
            'notifications': self.knowledge_system.get_pending_notifications(),
            'resource_usage': self.knowledge_system.get_resource_usage()
        }

    def _get_current_context(self) -> Dict:
        """Gets the current overall context."""
        return {
            'activity': self._get_current_activity(),
            'location': self._get_current_location(),
            'task': self._get_current_task()
        }

    def _assess_complexity(self) -> float:
        """Assesses current task/situation complexity."""
        return 0.5  # Replace with actual complexity assessment

    def _extract_concepts(self, thought: str) -> List[str]:
        """
        Extracts key concepts from a thought string using NLP techniques.
        
        Args:
            thought: The thought text to analyze
            
        Returns:
            List of identified concepts
        """
        # TODO: Implement more sophisticated NLP concept extraction
        # For now, using simple word-based extraction
        words = thought.lower().split()
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        concepts = [word for word in words if word not in stopwords and len(word) > 3]
        return list(set(concepts))  # Remove duplicates

    def _assess_thought_priority(self, thought: str) -> float:
        """
        Assesses the priority level of a thought from 0.0 to 1.0.
        
        Args:
            thought: The thought text to analyze
            
        Returns:
            Priority score between 0.0 and 1.0
        """
        # Priority factors to consider
        urgency_words = {'urgent', 'immediate', 'critical', 'important', 'priority'}
        emotional_words = {'worry', 'concern', 'excited', 'anxious', 'crucial'}
        
        thought_lower = thought.lower()
        
        # Calculate base priority score
        urgency_score = sum(word in thought_lower for word in urgency_words) * 0.2
        emotional_score = sum(word in thought_lower for word in emotional_words) * 0.15
        
        # Consider current context
        context_relevance = self._assess_context_relevance(thought)
        
        # Combine scores with weights
        priority = min(1.0, urgency_score + emotional_score + (context_relevance * 0.65))
        return round(priority, 2)

    def _categorize_thought(self, thought: str) -> str:
        """
        Categorizes a thought into predefined categories.
        
        Args:
            thought: The thought text to categorize
            
        Returns:
            Category string
        """
        categories = {
            'task_planning': {'plan', 'schedule', 'organize', 'todo', 'task'},
            'problem_solving': {'solve', 'fix', 'improve', 'optimize', 'debug'},
            'observation': {'notice', 'observe', 'see', 'detect', 'identify'},
            'reflection': {'think', 'reflect', 'consider', 'analyze', 'evaluate'},
            'emotion': {'feel', 'happy', 'worried', 'excited', 'concerned'},
            'decision': {'decide', 'choose', 'select', 'determine', 'pick'}
        }
        
        thought_lower = thought.lower()
        
        # Find category with most matching keywords
        max_matches = 0
        best_category = 'general'
        
        for category, keywords in categories.items():
            matches = sum(keyword in thought_lower for keyword in keywords)
            if matches > max_matches:
                max_matches = matches
                best_category = category
                
        return best_category

    def _identify_relations(self, thought: str) -> Dict[str, List[str]]:
        """
        Identifies relationships between concepts in a thought.
        
        Args:
            thought: The thought text to analyze
            
        Returns:
            Dictionary of relationship types and related concepts
        """
        # Extract concepts from the thought
        concepts = self._extract_concepts(thought)
        
        # Get related concepts from semantic memory
        related_concepts = self.memory_system.semantic.get_related_concepts(
            concept_term=concepts[0] if concepts else "",
            max_depth=1
        ) if concepts else []
        
        # Analyze causal relationships - Fix parameter name from source_concept to concept
        causal_relations = self.memory_system.causal.find_causal_relations(
            concept=concepts[0] if concepts else ""
        ) if concepts else []
        
        return {
            'semantic': related_concepts,
            'causal': causal_relations,
            'temporal': self._find_temporal_relations(thought),
            'spatial': self._find_spatial_relations(thought)
        }

    def _assess_context_relevance(self, thought: str) -> float:
        """
        Assesses how relevant a thought is to the current context.
        
        Args:
            thought: The thought to analyze
            
        Returns:
            Relevance score between 0.0 and 1.0
        """
        current_context = self._get_current_context()
        
        # Extract key terms from current context
        context_terms = set()
        for value in current_context.values():
            if isinstance(value, dict):
                context_terms.update(str(v).lower() for v in value.values())
            else:
                context_terms.add(str(value).lower())
        
        # Compare thought terms with context terms
        thought_terms = set(self._extract_concepts(thought))
        
        if not thought_terms or not context_terms:
            return 0.0
            
        # Calculate Jaccard similarity
        intersection = len(thought_terms.intersection(context_terms))
        union = len(thought_terms.union(context_terms))
        
        return round(intersection / union if union > 0 else 0.0, 2)

    def _find_temporal_relations(self, thought: str) -> List[str]:
        """
        Identifies temporal relationships in a thought.
        
        Args:
            thought: The thought text to analyze
            
        Returns:
            List of temporal relationships
        """
        temporal_indicators = {
            'before', 'after', 'during', 'while', 'when',
            'yesterday', 'today', 'tomorrow', 'next', 'previous'
        }
        
        thought_words = thought.lower().split()
        temporal_relations = []
        
        for i, word in enumerate(thought_words):
            if word in temporal_indicators and i < len(thought_words) - 1:
                # Capture the temporal relationship and the following context
                relation = ' '.join(thought_words[i:i+3])  # Include next two words for context
                temporal_relations.append(relation)
                
        return temporal_relations

    def _find_spatial_relations(self, thought: str) -> List[str]:
        """
        Identifies spatial relationships in a thought.
        
        Args:
            thought: The thought text to analyze
            
        Returns:
            List of spatial relationships
        """
        spatial_indicators = {
            'in', 'on', 'at', 'near', 'between',
            'above', 'below', 'inside', 'outside', 'around'
        }
        
        thought_words = thought.lower().split()
        spatial_relations = []
        
        for i, word in enumerate(thought_words):
            if word in spatial_indicators and i < len(thought_words) - 1:
                # Capture the spatial relationship and the following context
                relation = ' '.join(thought_words[i:i+3])  # Include next two words for context
                spatial_relations.append(relation)
                
        return spatial_relations

    def _gather_context(self) -> Dict[str, Any]:
        """
        Gathers comprehensive context from all available sources.
        
        Returns:
            Dict containing current context information
        """
        try:
            # Get recent memories and thoughts
            recent_memories = self.memory_system.get_memories({
                'timestamp': {'$gt': time.time() - 300}  # Last 5 minutes
            })
            
            active_thoughts = self.current_thoughts()
            
            # Get current knowledge context
            current_knowledge = self.knowledge_system.get_relevant_knowledge()
            
            # Get computer environment context
            computer_context = {
                'location': {
                    'application': self.knowledge_system.get_active_application(),
                    'window': self.knowledge_system.get_active_window(),
                    'view': self.knowledge_system.get_current_view()
                },
                'activity': {
                    'type': self.knowledge_system.get_task_type(),
                    'objective': self.knowledge_system.get_task_objective(),
                    'progress': self.knowledge_system.get_task_progress()
                },
                'system_state': {
                    'running_apps': self.knowledge_system.get_running_applications(),
                    'notifications': self.knowledge_system.get_pending_notifications(),
                    'resources': self.knowledge_system.get_resource_usage()
                }
            }

            # Combine all context information
            context = {
                'recent_memories': recent_memories,
                'active_thoughts': active_thoughts,
                'current_knowledge': current_knowledge,
                'computer_context': computer_context,
                'timestamp': time.time()
            }
            
            return context
            
        except Exception as e:
            logging.error(f"Error gathering context: {e}")
            # Return minimal context to allow continued operation
            return {
                'recent_memories': [],
                'active_thoughts': [],
                'current_knowledge': [],
                'computer_context': {},
                'timestamp': time.time()
            }
