"""
Central configuration for the Interactive Compositional Semantic Image Retrieval project.

All paths, filenames, model settings, and experiment parameters live here.
Every module imports from this file — never hardcode paths elsewhere.
"""
import os

# ---------------------------------------------------------------------------
# 1. Directory layout
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Flickr30k dataset
FLICKR30K_DIR = os.path.join(DATA_DIR, "flickr30k-images-2")
FLICKR30K_CAPTIONS_FILE = os.path.join(FLICKR30K_DIR, "captions.txt")

# Kaggle dataset identifier (for download scripts)
KAGGLE_DATASET = "adityajn105/flickr30k"

# Project output directories
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")
INDEX_DIR = os.path.join(BASE_DIR, "index")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# ---------------------------------------------------------------------------
# 2. Embedding artefact filenames
# ---------------------------------------------------------------------------
IMAGE_EMBEDDINGS_FILE = os.path.join(EMBEDDINGS_DIR, "image_embeddings.npy")
IMAGE_IDS_FILE = os.path.join(EMBEDDINGS_DIR, "image_ids.json")
FAISS_INDEX_FILE = os.path.join(INDEX_DIR, "faiss_index.bin")

# ---------------------------------------------------------------------------
# 3. Model configuration
# ---------------------------------------------------------------------------
# We use SigLIP as the default vision-language encoder.
# google/siglip-base-patch16-224 produces 768-dim embeddings.
MODEL_NAME = "google/siglip-base-patch16-224"
EMBEDDING_DIM = 768

# ---------------------------------------------------------------------------
# 4. Experiment / evaluation settings
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
TEST_SPLIT_RATIO = 0.2          # 20 % of images held out for evaluation
BATCH_SIZE = 64                 # batch size for image embedding generation
TOP_K_VALUES = [1, 5, 10]       # Recall@K values to compute

# ---------------------------------------------------------------------------
# 5. Ensure output directories exist
# ---------------------------------------------------------------------------
for _dir in [DATA_DIR, EMBEDDINGS_DIR, INDEX_DIR, RESULTS_DIR]:
    os.makedirs(_dir, exist_ok=True)
