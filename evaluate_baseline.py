#!/usr/bin/env python3
"""
Step 7 — Full evaluation script for the baseline retrieval system.

This script executes large-scale evaluation using batching.
Instead of querying FAISS one text query at a time, it embeds all 
text queries efficiently on the GPU, and then performs a bulk FAISS 
search for maximum throughput.
"""
import os
import sys
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as cfg
from src.data.dataset import Flickr30kDataset
from src.indexing.faiss_index import SemanticIndex
from src.retrieval.search import ImageRetriever
from src.evaluation.metrics import recall_at_k, average_precision, ndcg_at_k

def main():
    print("Loading Dataset...")
    ds = Flickr30kDataset()
    
    print("\nInitializing Retriever (loads Model and FAISS Index)...")
    try:
        retriever = ImageRetriever(dataset=ds)
    except FileNotFoundError:
        print("\n[ERROR] FAISS index or embeddings not found.")
        print("Please run `generate_embeddings.py` and ensure the FAISS index builds.")
        sys.exit(1)
        
    test_ids = ds.get_split("test")
    
    # 1. Build Query Set
    queries = []
    ground_truths = []
    
    print("\nBuilding evaluation query set from Test split...")
    for img_id in test_ids:
        captions = ds.get_captions(img_id)
        for cap in captions:
            queries.append(cap)
            ground_truths.append({img_id})
            
    num_queries = len(queries)
    print(f"Total test queries: {num_queries}")
    
    # 2. Batch Text Embedding Generation
    batch_size = cfg.BATCH_SIZE * 2 # Text is smaller, can use larger batch
    query_embeddings = np.zeros((num_queries, cfg.EMBEDDING_DIM), dtype=np.float32)
    
    vlm = retriever.vlm
    device_type = 'cuda' if 'cuda' in vlm.device else 'cpu'
    
    print("Encoding text queries (AMP enabled)...")
    for start_idx in tqdm(range(0, num_queries, batch_size), desc="Text Encoding"):
        end_idx = min(start_idx + batch_size, num_queries)
        batch_texts = queries[start_idx:end_idx]
        
        # We manually use the model to get the benefit of AMP and explicit batching
        inputs = vlm.processor(text=batch_texts, return_tensors="pt", padding="max_length", truncation=True).to(vlm.device)
        
        with torch.no_grad():
            with torch.autocast(device_type=device_type, enabled=(device_type=='cuda')):
                outputs = vlm.model.get_text_features(**inputs)
            
            outputs = outputs.float()
            embs = torch.nn.functional.normalize(outputs, p=2, dim=-1)
            
        query_embeddings[start_idx:end_idx] = embs.cpu().numpy()
        
    # 3. Bulk FAISS Search
    print("Performing bulk FAISS search for top-10 candidates...")
    distances, indices = retriever.index.index.search(query_embeddings, 10)
    
    # 4. Compute Metrics
    recalls = {1: [], 5: [], 10: []}
    maps = []
    ndcgs = {10: []}
    
    print("Calculating Retrieval Metrics...")
    for q_idx in range(num_queries):
        relevant = ground_truths[q_idx]
        
        # Map FAISS numerical indices back to string image_ids
        retrieved_ids = [retriever.index.get_image_id(idx) for idx in indices[q_idx]]
        
        for k in recalls.keys():
            recalls[k].append(recall_at_k(retrieved_ids, relevant, k))
            
        maps.append(average_precision(retrieved_ids, relevant))
        ndcgs[10].append(ndcg_at_k(retrieved_ids, relevant, 10))
        
    # 5. Output
    print("\n" + "="*50)
    print("Baseline Evaluation Results (Text -> Image)")
    print("="*50)
    print(f"Model       : {cfg.MODEL_NAME}")
    print(f"Dataset     : Flickr30K ({num_queries} queries)")
    print("-" * 50)
    print(f"Recall@1  : {np.mean(recalls[1]):.4f}")
    print(f"Recall@5  : {np.mean(recalls[5]):.4f}")
    print(f"Recall@10 : {np.mean(recalls[10]):.4f}")
    print(f"mAP       : {np.mean(maps):.4f}")
    print(f"nDCG@10   : {np.mean(ndcgs[10]):.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
