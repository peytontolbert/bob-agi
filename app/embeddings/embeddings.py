"""
Unified embedding space for Bob's environmental processing.
Handles conversion and integration of multimodal inputs into a common representation.
"""

import numpy as np
from typing import Dict, List, Union, Optional, Any
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, Wav2Vec2Processor, Wav2Vec2ForCTC
from PIL import Image
import logging

class ModalityEncoder:
    """Base encoder for converting different modalities into embeddings"""
    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim
        
    def encode(self, input_data: Any) -> np.ndarray:
        raise NotImplementedError

class TextEncoder(ModalityEncoder):
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
    def encode(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()

class VisionEncoder(ModalityEncoder):
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        
    def encode(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        # Convert image to expected format and get embeddings
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        inputs = self.model.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
        return outputs.numpy()

class AudioEncoder(ModalityEncoder):
    def __init__(self, model_name: str = "facebook/wav2vec2-base-960h"):
        super().__init__()
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        
    def encode(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Encode audio using wav2vec2-bert model.
        
        Args:
            audio_data: numpy array of audio waveform (16kHz sampling rate)
            
        Returns:
            numpy array: Audio embeddings
        """
        try:
            # Preprocess audio
            inputs = self.processor(
                audio_data, 
                sampling_rate=16000,
                return_tensors="pt",
                padding="longest"
            )
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Use mean pooling over time dimension
            embeddings = outputs.last_hidden_state.mean(dim=1)
            
            return embeddings.numpy()
            
        except Exception as e:
            logging.error(f"Error encoding audio: {e}")
            return np.zeros((1, self.embedding_dim))

class UnifiedEmbeddingSpace:
    def __init__(self):
        self.text_encoder = TextEncoder()
        self.vision_encoder = VisionEncoder()
        self.audio_encoder = AudioEncoder()
        self.logger = logging.getLogger(__name__)
        
        # Embedding combination parameters
        self.attention = nn.MultiheadAttention(embed_dim=768, num_heads=8)
        
    def embed_environmental_input(self, 
                                visual_input: Optional[Union[Image.Image, np.ndarray]] = None,
                                audio_input: Optional[np.ndarray] = None,
                                text_input: Optional[str] = None,
                                system_state: Optional[Dict] = None) -> Dict[str, np.ndarray]:
        """
        Convert all environmental inputs into a unified embedding space
        """
        embeddings = {}
        
        try:
            # Process each modality if present
            if visual_input is not None:
                embeddings['visual'] = self.vision_encoder.encode(visual_input)
                
            if audio_input is not None:
                embeddings['audio'] = self.audio_encoder.encode(audio_input)
                
            if text_input is not None:
                embeddings['text'] = self.text_encoder.encode(text_input)
                
            if system_state is not None:
                # Convert system state to text and encode
                system_text = str(system_state)
                embeddings['system'] = self.text_encoder.encode(system_text)
            
            # Combine embeddings using attention mechanism
            unified_embedding = self._combine_embeddings(embeddings)
            embeddings['unified'] = unified_embedding
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Error creating embeddings: {e}")
            return {}
    
    def _combine_embeddings(self, embeddings: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Combine different modality embeddings using attention mechanism
        """
        if not embeddings:
            return np.zeros(768)  # Return zero vector if no embeddings
            
        # Stack all embeddings
        embedding_list = list(embeddings.values())
        stacked = torch.tensor(np.stack(embedding_list))
        
        # Apply self-attention to combine embeddings
        with torch.no_grad():
            attended, _ = self.attention(stacked, stacked, stacked)
            
        # Average the attended embeddings
        unified = attended.mean(dim=0).numpy()
        return unified
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings
        """
        return float(np.dot(embedding1, embedding2) / 
                    (np.linalg.norm(embedding1) * np.linalg.norm(embedding2)))
    
    def query_knowledge(self, query_embedding: np.ndarray, 
                       knowledge_embeddings: List[np.ndarray],
                       threshold: float = 0.7) -> List[int]:
        """
        Find relevant knowledge entries based on embedding similarity
        """
        similarities = [self.compute_similarity(query_embedding, ke) 
                       for ke in knowledge_embeddings]
        return [i for i, sim in enumerate(similarities) if sim >= threshold]

