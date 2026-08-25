#!/usr/bin/env python3
"""
Step 3 — Offline Image Embedding Generation.

Generates normalized image embeddings for the entire Flickr30k dataset using 
a PyTorch DataLoader and Automatic Mixed Precision (AMP) for maximum throughput.

Output:
    - embeddings/image_embeddings.npy (N x D float32 array)
    - embeddings/image_ids.json (List of length N preserving order)
"""
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as cfg
from src.data.dataset import Flickr30kDataset, Flickr30kImageDataset
from src.models.vision_language import VisionLanguageModel


def custom_collate(batch):
    """Simple collate function to handle PIL images."""
    image_ids = [item['image_id'] for item in batch]
    images = [item['image'] for item in batch]
    indices = [item['index'] for item in batch]
    return {"image_ids": image_ids, "images": images, "indices": indices}


def main():
    print("Initializing dataset...")
    core_ds = Flickr30kDataset()
    image_ds = Flickr30kImageDataset(core_ds, split="all")
    
    # We use multiple workers if running on a standard system.
    # num_workers=4 is a safe default for I/O bound image loading.
    loader = DataLoader(
        image_ds, 
        batch_size=cfg.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4, 
        collate_fn=custom_collate
    )
    
    num_images = len(image_ds)
    image_ids_ordered = core_ds.get_all_image_ids()
    
    print("Loading model...")
    vlm = VisionLanguageModel()
    embedding_dim = cfg.EMBEDDING_DIM
    
    print(f"Allocating {num_images} x {embedding_dim} numpy array...")
    all_embeddings = np.zeros((num_images, embedding_dim), dtype=np.float32)
    
    # Autocast setup for mixed precision
    device_type = 'cuda' if 'cuda' in vlm.device else 'cpu'
    
    print(f"Extracting embeddings using batch size {cfg.BATCH_SIZE} on {vlm.device} (AMP)...")
    
    vlm.model.eval()
    
    for batch in tqdm(loader, desc="Generating Embeddings"):
        images = batch["images"]
        indices = batch["indices"]
        
        # Prepare inputs via the processor
        inputs = vlm.processor(images=images, return_tensors="pt").to(vlm.device)
        
        with torch.no_grad():
            # Use AMP for faster inference if on GPU
            with torch.autocast(device_type=device_type, enabled=(device_type=='cuda')):
                outputs = vlm.model.get_image_features(**inputs)
            
            # Normalization and move to CPU Float32
            # It's important to do normalization in float32 for numerical stability
            outputs = outputs.float()
            embeddings = torch.nn.functional.normalize(outputs, p=2, dim=-1)
            
        # Store in numpy array
        all_embeddings[indices] = embeddings.cpu().numpy()
        
    print(f"\nSaving embeddings to: {cfg.IMAGE_EMBEDDINGS_FILE}")
    np.save(cfg.IMAGE_EMBEDDINGS_FILE, all_embeddings)
    
    print(f"Saving image IDs mapping to: {cfg.IMAGE_IDS_FILE}")
    with open(cfg.IMAGE_IDS_FILE, "w", encoding="utf-8") as f:
        json.dump(image_ids_ordered, f)
        
    print("Offline embedding generation complete.")

if __name__ == "__main__":
    main()
