"""
End-to-end text search pipeline.
"""
from typing import List, Dict, Any
import numpy as np

from src.models.vision_language import VisionLanguageModel
from src.indexing.faiss_index import SemanticIndex
from src.data.dataset import Flickr30kDataset

class ImageRetriever:
    """End-to-end retriever orchestrating VLM, Dataset, and FAISS."""
    def __init__(self, vlm: VisionLanguageModel = None, index: SemanticIndex = None, dataset: Flickr30kDataset = None):
        self.vlm = vlm if vlm else VisionLanguageModel()
        self.index = index if index else SemanticIndex()
        self.dataset = dataset if dataset else Flickr30kDataset()
        
        # Ensure index is loaded
        if self.index.index is None:
            self.index.load_index()

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieves top_k images for a given text query.
        
        Returns:
            List of dictionaries with image_id, image_path, and score.
        """
        # 1. Get text embedding
        query_emb = self.vlm.get_text_embeddings(query)
        query_emb_np = query_emb.cpu().numpy().astype(np.float32)
        
        # 2. FAISS Search
        distances, indices = self.index.search(query_emb_np, top_k)
        
        # 3. Format Results
        results = []
        for i in range(top_k):
            idx = indices[0][i]
            score = float(distances[0][i])
            img_id = self.index.get_image_id(idx)
            img_path = self.dataset.get_image_path(img_id)
            
            results.append({
                "image_id": img_id,
                "image_path": img_path,
                "score": score
            })
            
        return results
