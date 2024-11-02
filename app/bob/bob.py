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
from time import time
import numpy as np

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

    def process_environment(self, env):
        """
        Enhanced environmental processing combining multiple modalities.
        """
        current_time = time()
        processed_data = {
            'timestamp': current_time,
            'visual': {},
            'audio': {},
            'context': {},
            'elements': []
        }
        
        # Process visual input with multiple models
        if env.get('visual'):
            # Get UI element detection from YOLO+SAM
            detected_elements = self.vision_agent.detect_elements(env['visual'][-1])
            
            # Get scene understanding from GPT-4V
            scene_description = self.vision_agent.complete_task(
                "Describe the current screen state and context",
                env['visual'][-1]
            )
            
            # Update visual processing results
            processed_data['visual'] = {
                'elements': detected_elements,
                'scene_description': scene_description,
                'timestamp': current_time
            }
            
            # Update active elements in context
            self.interaction_context['active_elements'] = detected_elements

        # Process audio input
        if env.get('audio'):
            audio_text = self.hearing.get_transcription()
            if audio_text:
                # Process speech commands
                command_response = self.text_agent.complete_task(
                    f"Extract action commands from: {audio_text}"
                )
                
                # Update audio context
                self.interaction_context['audio_context'].append({
                    'text': audio_text,
                    'commands': command_response,
                    'timestamp': current_time
                })
                
                processed_data['audio'] = {
                    'transcription': audio_text,
                    'commands': command_response
                }

        # Combine modalities for context understanding
        processed_data['context'] = self._integrate_modalities(processed_data)
        
        return processed_data

    def _integrate_modalities(self, processed_data):
        """
        Integrates multiple modalities to create a unified context understanding.
        """
        context = {
            'visual_elements': processed_data['visual'].get('elements', []),
            'scene_context': processed_data['visual'].get('scene_description', ''),
            'audio_input': processed_data['audio'].get('transcription', ''),
            'detected_commands': processed_data['audio'].get('commands', []),
            'recent_actions': self.interaction_context.get('last_action'),
            'current_goal': self.interaction_context.get('current_goal')
        }
        
        # Generate integrated understanding
        prompt = f"""
        Given the following context:
        Visual Scene: {context['scene_context']}
        Detected UI Elements: {len(context['visual_elements'])} elements
        Audio Input: {context['audio_input']}
        Current Goal: {context['current_goal']}
        Recent Action: {context['recent_actions']}
        
        Provide a unified understanding of the current situation and recommended action type.
        """
        
        integrated_understanding = self.text_agent.complete_task(prompt)
        context['integrated_understanding'] = integrated_understanding
        
        return context

    def decide_action(self, processed_env):
        """
        Enhanced action decision system combining multiple inputs.
        """
        context = processed_env['context']
        
        # Determine action type based on context
        if context['detected_commands']:
            # Prioritize explicit commands
            return self._handle_command_action(context['detected_commands'])
        
        # Check for UI interaction needs
        if context['visual_elements']:
            return self._handle_ui_interaction(context)
        
        # Default to generating a verbal response
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

