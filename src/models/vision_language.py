"""
Vision-Language Model wrapper for Image-Text Retrieval.

This module provides a unified interface for loading SigLIP/CLIP and extracting
normalized image and text embeddings.

We use `google/siglip-base-patch16-224` because SigLIP replaces the standard 
softmax contrastive loss with a pairwise sigmoid loss. This leads to better 
zero-shot performance, especially on complex or dense image-text relationships, 
which is an excellent starting point for compositional semantic retrieval.
"""
from __future__ import annotations

from typing import List, Union

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor

import config as cfg


class VisionLanguageModel:
    """Wrapper for Hugging Face Vision-Language models (SigLIP/CLIP)."""

    def __init__(self, model_name: str = cfg.MODEL_NAME, device: str = None):
        """
        Args:
            model_name: Hugging Face model hub identifier.
            device: 'cuda' or 'cpu'. If None, autodetects.
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading VisionLanguageModel: {self.model_name} on {self.device.upper()}...")
        
        # SigLIP architecture is slightly different in HF than standard CLIP
        # Using AutoModel and AutoProcessor handles the architecture transparently
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def get_image_embeddings(self, images: List[Image.Image]) -> torch.Tensor:
        """
        Compute L2-normalized image embeddings.
        
        Args:
            images: List of PIL images.
            
        Returns:
            torch.Tensor of shape (batch_size, embedding_dim)
        """
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        outputs = self.model.get_image_features(**inputs)
        
        # Explicit L2 normalization so Inner Product == Cosine Similarity
        embeddings = F.normalize(outputs, p=2, dim=-1)
        return embeddings

    @torch.no_grad()
    def get_text_embeddings(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Compute L2-normalized text embeddings.
        
        Args:
            texts: Single string or list of strings.
            
        Returns:
            torch.Tensor of shape (batch_size, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]
            
        # SigLIP typically uses padding="max_length" or "longest"
        inputs = self.processor(text=texts, return_tensors="pt", padding="max_length", truncation=True).to(self.device)
        outputs = self.model.get_text_features(**inputs)
        
        # Explicit L2 normalization
        embeddings = F.normalize(outputs, p=2, dim=-1)
        return embeddings
