"""
Bob is the main class that will handle Bob's interactions with the world.
"""
import os
from app.env.senses.hearing import Hearing
from app.env.senses.eyesight import Eyesight
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
        try:
            # Initialize AI agents
            self.text_agent = TextAgent()
            self.vision_agent = VisionAgent()
            self.speech_agent = SpeechAgent()
            
            # Initialize core systems
            self.knowledge_system = KnowledgeSystem()
            self.hearing = Hearing(computer.browser)
            self.hearing.start_listening()
            self.voice = Voice(computer.browser)
            #self.eyes = Eyesight(computer.browser, self.vision_agent)
            
            # Initialize hands with error handling
            #try:
                #self.hands = Hands(computer.browser)
            #except Exception as e:
            #    logging.error(f"Error initializing hands: {e}")
            #    self.hands = None
                
            self.thoughts = Thinking(self.knowledge_system)
            

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

        except Exception as e:
            logging.error(f"Error initializing Bob: {e}")
            raise

    def process_environment(self, env):
        """
        Process environmental inputs and generate thoughts that lead to actions.
        """
        processed_data = {
            'timestamp': time(),
            'visual': {},
            'audio': {},
            'thoughts': [],
            'active_elements': []
        }
        
        try:
            # Process visual input with context-aware scene understanding
            #if env.get('visual'):
            #    visual_data = self._process_visual_input(
            #        env['visual'], 
            #        self.interaction_context
            #    )
            #    processed_data['visual'] = visual_data
            #    
            #    # Update interaction context
            #    self.interaction_context['visual_memory'].append(visual_data)
            #    if len(self.interaction_context['visual_memory']) > 10:
            #        self.interaction_context['visual_memory'].pop(0)

            # Process audio input if needed
            if env.get('audio'):
                audio_text = self.hearing.get_transcription()
                if audio_text:
                    processed_data['audio'] = {
                        'transcription': audio_text
                    }

            # Generate thoughts based on current goal and environment
            thoughts = self._generate_action_thoughts(
                #visual_data=processed_data['visual'],
                audio_data=processed_data['audio'],
                current_goal=self.interaction_context['current_goal']
            )
            processed_data['thoughts'] = thoughts
            
            return processed_data
                
        except Exception as e:
            self.logger.error(f"Error in process_environment: {e}")
            return processed_data

    def _integrate_modalities(self, processed_data):
        """
        Integrate multiple modalities using embeddings and context history.
        """
        # Get unified embedding
        unified_embedding = processed_data['embeddings'].get('unified')
        
        context = {
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
        Decide next action based on thoughts and visual information.
        """
        try:
            thoughts = processed_env['thoughts']
            current_goal = self.interaction_context['current_goal']
            
            if not thoughts or not current_goal:
                return None
            
            # Determine what UI element we need to find based on thoughts
            target_element_prompt = f"""
            Given these thoughts about the current goal:
            {thoughts}
            
            What action to take next? Speech or observation.

            respond in this format:
            {{{'type': 'speech', 'description': 'speech description'}}}
            or
            {{{'type': 'observation', 'description': 'observation description'}}}
            """
            target_description = self.text_agent.complete_task(target_element_prompt)
            
            # Use vision agent to find the specific element
            if target_description:
                return {
                    'type': 'speech',
                    'description': target_description
                }
            
        except Exception as e:
            self.logger.error(f"Error deciding action: {e}")
            return None

    def _convert_decision_to_action(self, decision, context):
        """
        Converts a cognitive decision into an executable action.
        """
        try:
            # Parse the decision into actionable components
            action_type = self._determine_action_type(decision)
            
            if action_type == 'interaction':
                return self._create_interaction_action(decision, context)
            elif action_type == 'speech':
                return self._create_speech_action(decision)
            elif action_type == 'observation':
                return None  # Continue observing
            else:
                self.logger.warning(f"Unclear action type from decision: {decision}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error converting decision to action: {e}")
            return None

    def _create_speech_action(self, decision):
        """
        Creates a speech action based on the decision.
        """
        prompt = f"""
        Given this decision: {decision}
        Generate an appropriate verbal response.
        """
        
        response = self.text_agent.complete_task(prompt)
        return {
            'type': 'speech',
            'content': response,
            'timestamp': time()
        }

    def _determine_action_type(self, decision):
        """
        Determines the type of action based on the decision.
        """
        if 'speech' in decision.lower():
            return 'speech'
        elif 'observation' in decision.lower():
            return 'observation'
        else:
            self.logger.warning(f"Unclear action type from decision: {decision}")
            return None

    def execute_action(self, action):
        """
        Enhanced action execution with better error handling and feedback.
        """
        if not action:
            return None
            
        try:
            self.action_start_time = time()
            
            if action['type'] == 'speech':
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
        if not self.hands:
            logging.error("Hands system not initialized properly")
            return
            
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
                    if action:  # Check if hands available
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

    def _generate_action_thoughts(self, visual_data, audio_data, current_goal):
        """
        Generate thoughts focused on achieving the current goal based on environmental input.
        """
        try:
            # Create context for thought generation
            context = {
                'scene': visual_data.get('scene_understanding', ''),
                'audio': audio_data.get('transcription', ''),
                'goal': current_goal,
                'last_action': self.interaction_context.get('last_action')
            }
            
            # Generate action-oriented thoughts using text agent
            thoughts_prompt = f"""
            Given the current context:
            Scene: {context['scene']}
            Goal: {context['goal']}
            Last Action: {context['last_action']}
            
            Generate specific thoughts about what actions to take next to achieve the goal.
            Focus on:
            1. What specific UI elements need to be found
            2. What interactions are needed (click, type, etc)
            3. Whether the last action was successful
            4. What the next immediate step should be
            
            Return thoughts as a list of specific action-oriented statements.
            """
            
            action_thoughts = self.text_agent.complete_task(thoughts_prompt)
            return action_thoughts
            
        except Exception as e:
            self.logger.error(f"Error generating action thoughts: {e}")
            return []

