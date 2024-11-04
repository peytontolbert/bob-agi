import pytest
import numpy as np
import torch
from PIL import Image
from app.embeddings.embeddings import (
    TextEncoder,
    VisionEncoder,
    AudioEncoder,
    UnifiedEmbeddingSpace
)

@pytest.fixture
def text_encoder():
    return TextEncoder()

@pytest.fixture
def vision_encoder():
    return VisionEncoder()

@pytest.fixture
def audio_encoder():
    return AudioEncoder()

@pytest.fixture
def unified_space():
    return UnifiedEmbeddingSpace()

class TestTextEncoder:
    def test_text_encoding(self, text_encoder):
        text = "Hello, this is a test"
        embedding = text_encoder.encode(text)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape[1] == 768  # Default embedding dimension
        assert not np.allclose(embedding, 0)  # Ensure non-zero embedding

class TestVisionEncoder:
    def test_vision_encoding_pil(self, vision_encoder):
        # Create a test image
        image = Image.new('RGB', (224, 224), color='red')
        embedding = vision_encoder.encode(image)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape[1] == 768
        assert not np.allclose(embedding, 0)

    def test_vision_encoding_numpy(self, vision_encoder):
        # Create numpy array image
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        embedding = vision_encoder.encode(image)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape[1] == 768
        assert not np.allclose(embedding, 0)

    def test_vision_encoding_dict(self, vision_encoder):
        # Test with dictionary input
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        image_dict = {'frame': image}
        embedding = vision_encoder.encode(image_dict)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape[1] == 768

class TestAudioEncoder:
    def test_audio_encoding(self, audio_encoder):
        # Create dummy audio data (1 second at 16kHz)
        audio_data = np.random.randn(16000)
        embedding = audio_encoder.encode(audio_data)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape[1] == 768
        assert not np.allclose(embedding, 0)

class TestUnifiedEmbeddingSpace:
    def test_unified_embedding(self, unified_space):
        # Test with multiple modalities
        text = "Test text"
        image = Image.new('RGB', (224, 224), color='blue')
        audio = np.random.randn(16000)
        
        embeddings = unified_space.embed_environmental_input(
            visual_input=image,
            audio_input=audio,
            text_input=text
        )
        
        assert isinstance(embeddings, dict)
        assert 'visual' in embeddings
        assert 'audio' in embeddings
        assert 'text' in embeddings
        assert 'unified' in embeddings
        
        # Check unified embedding
        assert isinstance(embeddings['unified'], np.ndarray)
        assert embeddings['unified'].shape == (768,)

    def test_similarity_computation(self, unified_space):
        # Create two test embeddings
        emb1 = np.random.randn(768)
        emb2 = np.random.randn(768)
        
        similarity = unified_space.compute_similarity(emb1, emb2)
        
        assert isinstance(similarity, float)
        assert -1 <= similarity <= 1  # Cosine similarity range

    def test_knowledge_query(self, unified_space):
        # Create test knowledge base
        query = np.random.randn(768)
        knowledge_base = [np.random.randn(768) for _ in range(5)]
        
        # Add one very similar embedding
        similar_embedding = query + np.random.randn(768) * 0.1
        knowledge_base.append(similar_embedding)
        
        results = unified_space.query_knowledge(
            query, 
            knowledge_base,
            threshold=0.7
        )
        
        assert isinstance(results, list)
        assert all(isinstance(idx, int) for idx in results)

    def test_empty_input(self, unified_space):
        # Test with no inputs
        embeddings = unified_space.embed_environmental_input()
        assert isinstance(embeddings, dict)
        assert len(embeddings) == 0

    def test_system_state_embedding(self, unified_space):
        # Test system state embedding
        system_state = {"status": "active", "memory_usage": 0.75}
        
        embeddings = unified_space.embed_environmental_input(
            system_state=system_state
        )
        
        assert 'system' in embeddings
        assert isinstance(embeddings['system'], np.ndarray)
        assert embeddings['system'].shape[1] == 768
