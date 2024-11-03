"""
Bob is the main class that will handle Bob's interactions with the world.
"""
import os
from app.actions.hearing import Hearing
from app.actions.eyesight import Eyesight
from app.actions.thinking import Thinking
from app.actions.hands import Hands
from app.actions.voice import Voice
from app.memory.knowledge_system import KnowledgeSystem
from app.agents.text import TextAgent
from app.agents.vision import VisionAgent
from app.agents.speech import SpeechAgent
from collections import deque
from time import time, sleep
import numpy as np
import logging
from app.embeddings.embeddings import UnifiedEmbeddingSpace

class Bob:
    def __init__(self, computer):
        # Initialize core systems
        self.knowledge_system = KnowledgeSystem()
        self.hearing = Hearing(computer.audio)
        self.voice = Voice(computer.microphone)
        self.eyes = Eyesight(computer.screen)
        self.hands = Hands(computer.mouse, computer.keyboard)
        self.thoughts = Thinking(self.knowledge_system)
        
        # Initialize AI agents
        self.text_agent = TextAgent()
        self.vision_agent = VisionAgent()
        self.speech_agent = SpeechAgent()

        # Time windows and processing rates
        self.QUEUE_TIME_WINDOW = 10  # 10 seconds window
        self.VISION_FPS = 5  # 5 frames per second
        self.AUDIO_CHUNK_SIZE = 0.5  # 500ms audio chunks
        
        # Initialize time-based queues
        self.audio_buffer = deque(maxlen=int(self.QUEUE_TIME_WINDOW / self.AUDIO_CHUNK_SIZE))
        self.vision_buffer = deque(maxlen=self.QUEUE_TIME_WINDOW * self.VISION_FPS)
        self.thought_buffer = deque(maxlen=50)
        
        # Action state tracking
        self.current_action = None
        self.action_start_time = None
        self.last_visual_update = 0
        
        # Initialize interaction context
        self.interaction_context = {
            'visual_memory': [],  # Recent visual observations
            'audio_context': [],  # Recent audio context
            'active_elements': [], # Currently visible UI elements
            'last_action': None,  # Last performed action
            'current_goal': None  # Current objective
        }

        # Add thought processing parameters
        self.THOUGHT_INTEGRATION_WINDOW = 30  # Consider thoughts from last 30 seconds
        self.MIN_THOUGHT_CONFIDENCE = 0.5  # Minimum confidence to act on thoughts

        # Add logging configuration
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)
        
        # Initialize state tracking
        self.last_process_time = time()
        self.process_interval = 0.1  # 100ms processing interval

        # Add embedding space initialization
        self.embedding_space = UnifiedEmbeddingSpace()

        # Add context tracking
        self.context_history = deque(maxlen=50)  # Track recent context
        self.last_scene_understanding = None

    def process_environment(self, env):
        """
        Enhanced environmental processing using unified embedding space and context-aware vision.
        """
        current_time = time()
        processed_data = {
            'timestamp': current_time,
            'visual': {},
            'audio': {},
            'context': {},
            'elements': [],
            'thoughts': {},
            'embeddings': {}
        }
        
        # Create unified embeddings for all inputs
        embeddings = self.embedding_space.embed_environmental_input(
            visual_input=env.get('visual', [])[-1] if env.get('visual') else None,
            audio_input=env.get('audio'),
            system_state=env.get('system'),
            text_input=str(self.interaction_context.get('current_goal'))
        )
        processed_data['embeddings'] = embeddings

        # Get current thoughts for context
        current_thoughts = self.thoughts.current_thoughts()
        thought_context = {
            'thought_focus': current_thoughts[0]['content'] if current_thoughts else None,
            'current_goal': self.interaction_context.get('current_goal'),
            'last_action': self.interaction_context.get('last_action')
        }
        
        # Process visual input with context-aware scene understanding
        if env.get('visual'):
            visual_data = self._process_visual_input(
                env['visual'][-1], 
                thought_context
            )
            processed_data['visual'] = visual_data
            
            # Update interaction context
            self.interaction_context['visual_memory'].append(visual_data)
            if len(self.interaction_context['visual_memory']) > 10:
                self.interaction_context['visual_memory'].pop(0)

        # Process audio input
        if env.get('audio'):
            audio_text = self.hearing.get_transcription()
            if audio_text:
                # Process speech commands
                command_response = self.text_agent.complete_task(
                    f"Extract action commands from: {audio_text}"
                )
                
                processed_data['audio'] = {
                    'transcription': audio_text,
                    'commands': command_response
                }

        # Add thought processing
        if current_thoughts:
            thought_analysis = self._process_thoughts(current_thoughts)
            processed_data['thoughts'] = thought_analysis
            
        # Integrate all modalities with embeddings
        processed_data['context'] = self._integrate_modalities(processed_data)
        
        # Update context history
        self.context_history.append(processed_data['context'])
        
        return processed_data

    def _process_visual_input(self, visual_input, thought_context):
        """
        Process visual input using context-aware scene understanding.
        Only detect specific elements when needed for actions.
        """
        try:
            # Get high-level scene understanding using Qwen-VL
            scene_understanding = self.vision_agent.perceive_scene(
                visual_input,
                thought_context
            )
            
            # Store last scene understanding
            self.last_scene_understanding = scene_understanding
            
            visual_data = {
                'scene_understanding': scene_understanding,
                'timestamp': time(),
                'context': thought_context
            }
            
            # Only detect elements if we need precise locations
            if self._needs_element_detection(thought_context):
                detected_elements = self.vision_agent.detect_elements(visual_input)
                visual_data['elements'] = detected_elements
                
            return visual_data
            
        except Exception as e:
            self.logger.error(f"Error processing visual input: {e}")
            return {}

    def _needs_element_detection(self, context):
        """
        Determine if we need precise element detection based on context.
        """
        if not context:
            return False
            
        # Check if current goal or thought focus suggests need for interaction
        interaction_indicators = ['click', 'select', 'find', 'locate', 'interact']
        
        goal = context.get('current_goal', '').lower()
        focus = context.get('thought_focus', '').lower()
        
        return any(indicator in goal or indicator in focus 
                  for indicator in interaction_indicators)

    def _integrate_modalities(self, processed_data):
        """
        Integrate multiple modalities using embeddings and context history.
        """
        # Get unified embedding
        unified_embedding = processed_data['embeddings'].get('unified')
        
        context = {
            'scene_understanding': processed_data['visual'].get('scene_understanding'),
            'audio_input': processed_data['audio'].get('transcription', ''),
            'detected_commands': processed_data['audio'].get('commands', []),
            'recent_actions': self.interaction_context.get('last_action'),
            'current_goal': self.interaction_context.get('current_goal'),
            'current_thoughts': processed_data['thoughts'].get('recent_thoughts', []),
            'thought_focus': processed_data['thoughts'].get('focus'),
            'cognitive_state': processed_data['thoughts'].get('cognitive_state')
        }
        
        # Use unified embedding for knowledge retrieval if available
        if unified_embedding is not None:
            relevant_knowledge = self.knowledge_system.query_by_embedding(
                unified_embedding
            )
            context['relevant_knowledge'] = relevant_knowledge
            
        # Generate integrated understanding
        context['integrated_understanding'] = self._generate_integrated_understanding(context)
        
        return context

    def _generate_integrated_understanding(self, context):
        """
        Generate integrated understanding using the text agent.
        """
        prompt = f"""
        Given the current context:
        Scene: {context['scene_understanding']}
        Audio: {context['audio_input']}
        Goal: {context['current_goal']}
        Recent Action: {context['recent_actions']}
        Current Focus: {context['thought_focus']}
        Cognitive State: {context['cognitive_state']}
        
        Provide a unified understanding of the current situation and any needed actions.
        """
        
        return self.text_agent.complete_task(prompt)

    def _process_thoughts(self, thoughts):
        """
        Processes and analyzes current thoughts for decision making.
        """
        recent_thoughts = [t for t in thoughts 
                         if (time() - t['timestamp'].timestamp()) < self.THOUGHT_INTEGRATION_WINDOW]
        
        if not recent_thoughts:
            return {
                'recent_thoughts': [],
                'focus': None,
                'cognitive_state': 'neutral'
            }

        # Analyze thought patterns and focus
        thought_contents = [t['content'] for t in recent_thoughts]
        thought_emotions = [t.get('emotional_state', {}) for t in recent_thoughts]
        
        # Get current cognitive focus from thoughts
        focus = self.text_agent.complete_task(
            f"Determine the main focus from these thoughts: {thought_contents}"
        )
        
        # Assess cognitive state from emotional patterns
        cognitive_state = self._assess_cognitive_state(thought_emotions)
        
        return {
            'recent_thoughts': recent_thoughts,
            'focus': focus,
            'cognitive_state': cognitive_state,
            'priority_thought': max(recent_thoughts, 
                                  key=lambda x: x.get('metadata', {}).get('priority', 0))
        }

    def decide_action(self, processed_env):
        """
        Enhanced action decision system incorporating thoughts.
        """
        context = processed_env['context']
        thoughts = processed_env['thoughts']
        
        # Check for high-priority thoughts that might override other actions
        if thoughts.get('priority_thought'):
            priority = thoughts['priority_thought'].get('metadata', {}).get('priority', 0)
            if priority > self.MIN_THOUGHT_CONFIDENCE:
                return self._handle_thought_driven_action(thoughts['priority_thought'])
        
        # Continue with existing decision logic...
        if context['detected_commands']:
            return self._handle_command_action(context['detected_commands'])
        
        if context['visual_elements']:
            return self._handle_ui_interaction(context)
        
        return self._generate_verbal_response(context)

    def _handle_command_action(self, commands):
        """
        Processes explicit commands into actions.
        """
        # Convert command to action structure
        action = {
            'type': 'command',
            'command': commands[0],  # Take first command for now
            'timestamp': time()
        }
        return action

    def _handle_ui_interaction(self, context):
        """
        Determines appropriate UI interaction based on visual context.
        """
        elements = context['visual_elements']
        understanding = context['integrated_understanding']
        
        # Find most relevant UI element for current context
        target_element = None
        for element in elements:
            if element['type'] in understanding.lower():
                target_element = element
                break
        
        if target_element:
            return {
                'type': 'interaction',
                'element': target_element,
                'action': 'click',  # or other action based on context
                'timestamp': time()
            }
        
        return None

    def _generate_verbal_response(self, context):
        """
        Generates appropriate verbal response when no physical action is needed.
        """
        prompt = f"""
        Given the current context:
        {context['integrated_understanding']}
        
        Generate an appropriate verbal response.
        """
        
        response = self.text_agent.complete_task(prompt)
        return {
            'type': 'speech',
            'content': response,
            'timestamp': time()
        }

    def _handle_thought_driven_action(self, thought):
        """
        Converts a high-priority thought into an actionable response.
        """
        # Analyze thought content for actionable elements
        action_analysis = self.text_agent.complete_task(
            f"""
            Given this thought: {thought['content']}
            Determine if and how to act on it. Consider:
            1. Is immediate action needed?
            2. What type of action would be most appropriate?
            3. What is the specific target of the action?
            
            Return a structured action recommendation.
            """
        )
        
        # Convert thought into action
        return {
            'type': 'thought_driven',
            'thought': thought['content'],
            'action': action_analysis,
            'priority': thought['metadata']['priority'],
            'timestamp': time()
        }

    def _assess_cognitive_state(self, thought_emotions):
        """
        Analyzes emotional patterns in thoughts to determine cognitive state.
        """
        if not thought_emotions:
            return 'neutral'
            
        # Aggregate emotional indicators
        emotional_states = []
        for emotion in thought_emotions:
            if isinstance(emotion, dict):
                emotional_states.extend(emotion.values())
            elif emotion:
                emotional_states.append(emotion)
        
        # Determine dominant cognitive state
        if emotional_states:
            return max(set(emotional_states), key=emotional_states.count)
        return 'neutral'

    def execute_action(self, action):
        """
        Enhanced action execution with better error handling and feedback.
        """
        if not action:
            return None
            
        try:
            self.action_start_time = time()
            
            if action['type'] == 'interaction':
                # Handle UI interaction
                element = action['element']
                if action.get('action') == 'click':
                    success = self.hands.click_element(element)
                elif action.get('action') == 'type':
                    success = self.hands.type_text(action['text'])
                
                result = {
                    'success': success,
                    'action_type': 'interaction',
                    'target': element,
                    'timestamp': time()
                }
                
            elif action['type'] == 'speech':
                # Handle speech output
                self.voice.start_speaking(action['content'])
                result = {
                    'success': True,
                    'action_type': 'speech',
                    'content': action['content'],
                    'timestamp': time()
                }
                
            elif action['type'] == 'command':
                # Handle system commands
                success = self.execute_system_command(action['command'])
                result = {
                    'success': success,
                    'action_type': 'command',
                    'command': action['command'],
                    'timestamp': time()
                }
            
            self.interaction_context['last_action'] = result
            return result
            
        except Exception as e:
            print(f"Action execution error: {e}")
            return {
                'success': False,
                'error': str(e),
                'action': action,
                'timestamp': time()
            }

    def execute_system_command(self, command):
        """
        Executes system-level commands.
        """
        # Implement system command execution logic
        pass

    def alive(self, computer):
        """
        Main loop that keeps Bob running and processing his environment.
        """
        while True:
            try:
                current_time = time()
                
                # Only process if enough time has elapsed
                if current_time - self.last_process_time >= self.process_interval:
                    # Get environmental input
                    env = self._get_environment(computer)
                    
                    # Process environment and thoughts
                    processed_env = self.process_environment(env)
                    
                    # Decide and execute action
                    action = self.decide_action(processed_env)
                    if action:
                        result = self.execute_action(action)
                        
                        # Add experience to thoughts
                        self.thoughts.add_experience({
                            'action': action,
                            'result': result,
                            'context': processed_env['context']
                        })
                    
                    self.last_process_time = current_time
                
                sleep(0.01)  # Small delay to prevent CPU overload
                
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}", exc_info=True)
                sleep(1.0)  # Longer delay on error

    def _get_environment(self, computer):
        """
        Gets the current state of Bob's environment and generates thoughts based on sensory input.
        """
        try:
            # Get sensory inputs
            visual_input = self.eyes.get_screen_state()
            audio_input = self.hearing.get_audio_input()
            system_state = computer.get_system_state()
            
            # Generate thoughts based on sensory input
            thoughts = self.thoughts.process_sensory_input({
                'visual': visual_input,
                'audio': audio_input,
                'system': system_state
            })

            return {
                'visual': visual_input,
                'audio': audio_input, 
                'system': system_state,
                'thoughts': thoughts
            }
        except Exception as e:
            self.logger.error(f"Error getting environment: {e}")
            return {
                'visual': None,
                'audio': None,
                'system': None,
                'thoughts': None
            }

