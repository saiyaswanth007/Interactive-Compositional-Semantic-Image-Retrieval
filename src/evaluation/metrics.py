"""
Evaluation metrics for Image-Text Retrieval.
"""
import numpy as np
from typing import List, Set

def recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    Recall@K: 1 if ANY relevant image is in the top K retrieved images, else 0.
    Since this is Text->Image, for a specific caption, there is exactly ONE relevant image.
    So Recall@K is 1 if the ground truth image is in the top K.
    """
    top_k_retrieved = retrieved_ids[:k]
    intersection = set(top_k_retrieved).intersection(relevant_ids)
    return 1.0 if len(intersection) > 0 else 0.0

def average_precision(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    """
    Average Precision for a single query.
    """
    hits = 0
    sum_precisions = 0.0
    for i, p in enumerate(retrieved_ids):
        if p in relevant_ids:
            hits += 1
            sum_precisions += hits / (i + 1.0)
    
    if len(relevant_ids) == 0:
        return 0.0
        
    return sum_precisions / len(relevant_ids)

def ndcg_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    nDCG@K for a single query.
    """
    dcg = 0.0
    for i, p in enumerate(retrieved_ids[:k]):
        if p in relevant_ids:
            dcg += 1.0 / np.log2(i + 2) # i is 0-indexed, so i+1 is rank, i+2 is log base
            
    idcg = 0.0
    for i in range(min(len(relevant_ids), k)):
        idcg += 1.0 / np.log2(i + 2)
        
    if idcg == 0.0:
        return 0.0
    return dcg / idcg
