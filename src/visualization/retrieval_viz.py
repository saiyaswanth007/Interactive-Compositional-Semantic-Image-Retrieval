"""
Visualization utilities for retrieval results.
"""
import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Dict, Any

def visualize_retrieval(query: str, results: List[Dict[str, Any]], top_k: int = 5):
    """
    Displays the query and the top_k retrieved images side by side.
    """
    display_results = results[:top_k]
    num_results = len(display_results)
    
    if num_results == 0:
        print("No results to display.")
        return
        
    fig, axes = plt.subplots(1, num_results, figsize=(4 * num_results, 5))
    if num_results == 1:
        axes = [axes]
        
    fig.suptitle(f'Query: "{query}"', fontsize=16)
    
    for ax, res in zip(axes, display_results):
        img_path = res["image_path"]
        score = res["score"]
        
        try:
            img = Image.open(img_path)
            ax.imshow(img)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            
        ax.axis('off')
        ax.set_title(f"Score: {score:.3f}")
        
    plt.tight_layout()
    plt.show()
