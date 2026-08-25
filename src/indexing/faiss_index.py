"""
FAISS Index wrapper.

Handles building, saving, loading, and searching the exact Inner Product index.
"""
import os
import json
import numpy as np
import faiss

import config as cfg

class SemanticIndex:
    def __init__(self, index_file=cfg.FAISS_INDEX_FILE, ids_file=cfg.IMAGE_IDS_FILE):
        self.index_file = index_file
        self.ids_file = ids_file
        self.index = None
        self.image_ids = []

    def build_index(self, embeddings_file=cfg.IMAGE_EMBEDDINGS_FILE):
        """Builds a FAISS index from saved embeddings."""
        print(f"Loading embeddings from {embeddings_file}...")
        embeddings = np.load(embeddings_file)
        
        with open(self.ids_file, "r") as f:
            self.image_ids = json.load(f)
            
        assert len(embeddings) == len(self.image_ids), "Mismatched embeddings and IDs"
        
        dim = embeddings.shape[1]
        print(f"Building FAISS IndexFlatIP (dim={dim})...")
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        
        print(f"Saving index to {self.index_file}...")
        faiss.write_index(self.index, self.index_file)

    def load_index(self):
        """Loads FAISS index and ID mapping from disk."""
        if not os.path.exists(self.index_file) or not os.path.exists(self.ids_file):
            raise FileNotFoundError("Index or IDs file missing. Run build_index() first.")
            
        print(f"Loading index from {self.index_file}...")
        self.index = faiss.read_index(self.index_file)
        with open(self.ids_file, "r") as f:
            self.image_ids = json.load(f)

    def search(self, query_embeddings: np.ndarray, top_k: int = 10):
        """
        Search the index.
        Args:
            query_embeddings: np.ndarray of shape (N, D)
            top_k: number of results to return
        Returns:
            distances: np.ndarray of shape (N, top_k)
            indices: np.ndarray of shape (N, top_k)
        """
        if self.index is None:
            self.load_index()
            
        distances, indices = self.index.search(query_embeddings, top_k)
        return distances, indices
        
    def get_image_id(self, idx: int) -> str:
        """Map FAISS numerical index to image_id."""
        return self.image_ids[idx]
