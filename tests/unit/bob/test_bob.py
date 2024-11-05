import pytest
from unittest.mock import Mock, patch, MagicMock
from app.bob.bob import Bob
from app.env.senses.hearing import Hearing
from app.env.senses.eyesight import Eyesight
from app.actions.thinking import Thinking
from app.actions.hands import Hands
from app.actions.voice import Voice
from app.memory.knowledge_system import KnowledgeSystem
from app.agents.text import TextAgent
from app.agents.vision import VisionAgent
from app.agents.speech import SpeechAgent
from app.embeddings.embeddings import UnifiedEmbeddingSpace
import numpy as np

@pytest.fixture
def mock_computer():
    computer = Mock()
    computer.audio = Mock()
    computer.microphone = Mock()
    computer.screen = Mock()
    computer.mouse = Mock()
    computer.keyboard = Mock()
    # Mock get_system_state
    computer.get_system_state = Mock(return_value={'state': 'active'})
    return computer

@pytest.fixture
def mock_dependencies():
    with patch('app.memory.knowledge_system.KnowledgeSystem') as mock_ks, \
         patch('app.actions.hearing.Hearing') as mock_hearing, \
         patch('app.actions.voice.Voice') as mock_voice, \
         patch('app.actions.eyesight.Eyesight') as mock_eyes, \
         patch('app.actions.hands.Hands') as mock_hands, \
         patch('app.actions.thinking.Thinking') as mock_thinking, \
         patch('app.agents.text.TextAgent') as mock_text, \
         patch('app.agents.vision.VisionAgent') as mock_vision, \
         patch('app.agents.speech.SpeechAgent') as mock_speech, \
         patch('app.embeddings.embeddings.UnifiedEmbeddingSpace') as mock_embed:
        
        # Configure mock returns
        mock_hearing.return_value.get_transcription.return_value = "mock transcription"
        mock_hearing.return_value.get_audio_input.return_value = np.array([0.1, 0.2, 0.3])
        
        mock_eyes.return_value.get_screen_state.return_value = [np.zeros((100, 100, 3))]
        
        mock_vision.return_value.perceive_scene.return_value = "mock scene perception"
        mock_vision.return_value.detect_elements.return_value = [{"type": "button", "location": (10, 10)}]
        
        mock_thinking.return_value.current_thoughts.return_value = [
            {"content": "mock thought", "timestamp": Mock(), "metadata": {"priority": 1}}
        ]
        mock_thinking.return_value.process_sensory_input.return_value = [
            {"content": "mock processed thought", "timestamp": Mock()}
        ]
        
        mock_ks.return_value.query_by_embedding.return_value = [
            {"type": "experience", "content": "mock memory"}
        ]
        
        mock_text.return_value.complete_task.return_value = "mock completion"
        
        mock_embed.return_value.embed_environmental_input.return_value = {
            'unified': np.array([0.1, 0.2, 0.3]),
            'visual': np.array([0.4, 0.5, 0.6]),
            'audio': np.array([0.7, 0.8, 0.9])
        }
        
        yield {
            'knowledge_system': mock_ks,
            'hearing': mock_hearing,
            'voice': mock_voice, 
            'eyes': mock_eyes,
            'hands': mock_hands,
            'thinking': mock_thinking,
            'text_agent': mock_text,
            'vision_agent': mock_vision,
            'speech_agent': mock_speech,
            'embedding_space': mock_embed
        }

class TestBob:
    def test_bob_initialization(self, mock_computer, mock_dependencies):
        # Test successful initialization
        bob = Bob(mock_computer)
        assert bob is not None
        assert bob.knowledge_system is not None
        assert bob.hearing is not None
        assert bob.voice is not None
        assert bob.eyes is not None
        assert bob.hands is not None
        assert bob.thoughts is not None
        assert bob.text_agent is not None
        assert bob.vision_agent is not None
        assert bob.speech_agent is not None

    def test_process_environment(self, mock_computer, mock_dependencies):
        bob = Bob(mock_computer)
        
        # Mock environment data
        env = {
            'visual': [np.zeros((100, 100, 3))],  # Mock image array
            'audio': np.array([0.1, 0.2, 0.3]),  # Mock audio array
            'system': {'state': 'active'}
        }

        # Process environment
        result = bob.process_environment(env)

        # Verify result structure
        assert 'timestamp' in result
        assert 'visual' in result
        assert 'audio' in result
        assert 'context' in result
        assert 'elements' in result
        assert 'thoughts' in result
        assert 'embeddings' in result

    def test_decide_action(self, mock_computer, mock_dependencies):
        bob = Bob(mock_computer)

        # Mock processed environment data
        processed_env = {
            'context': {
                'scene_understanding': 'mock scene',
                'integrated_understanding': 'mock understanding'
            },
            'thoughts': {
                'focus': 'mock focus',
                'cognitive_state': 'neutral'
            },
            'audio': {
                'understanding': 'mock audio understanding'
            }
        }

        # Test action decision
        action = bob.decide_action(processed_env)
        assert action is not None

    def test_execute_action(self, mock_computer, mock_dependencies):
        bob = Bob(mock_computer)

        # Test interaction action
        interaction_action = {
            'type': 'interaction',
            'element': {'id': 'button1', 'type': 'button'},
            'action': 'click'
        }
        
        # Configure mock returns
        bob.hands.click_element.return_value = True
        
        result = bob.execute_action(interaction_action)
        assert result['success'] is True
        assert result['action_type'] == 'interaction'

        # Test speech action
        speech_action = {
            'type': 'speech',
            'content': 'Hello world'
        }
        
        result = bob.execute_action(speech_action)
        assert result['success'] is True
        assert result['action_type'] == 'speech'

    def test_error_handling(self, mock_computer, mock_dependencies):
        # Test initialization with failing hands
        mock_dependencies['hands'].side_effect = Exception("Failed to initialize hands")
        
        bob = Bob(mock_computer)
        assert bob.hands is None  # Should handle hands failure gracefully

        # Test process_environment error handling
        bob.embedding_space.embed_environmental_input.side_effect = Exception("Embedding error")
        
        result = bob.process_environment({})
        assert 'timestamp' in result  # Should return basic structure even on error

    def test_get_environment(self, mock_computer, mock_dependencies):
        bob = Bob(mock_computer)
        
        # Test normal operation
        env = bob._get_environment(mock_computer)
        assert 'visual' in env
        assert 'audio' in env
        assert 'system' in env
        assert 'thoughts' in env
        
        # Test error handling
        bob.eyes.get_screen_state.side_effect = Exception("Screen error")
        env = bob._get_environment(mock_computer)
        assert env['visual'] is None  # Should handle errors gracefully
