#!/bin/bash
# ==============================================================================
# Interactive Compositional Semantic Image Retrieval - Baseline Pipeline
# ==============================================================================

# Exit immediately if a command exits with a non-zero status
set -e

echo "============================================================"
echo " Starting Part 1 Baseline: Text -> Image Retrieval System"
echo "============================================================"

echo ""
echo "[Step 1/3] Running Dataset Sanity Checks..."
echo "------------------------------------------------------------"
python3 sanity_check_dataset.py

echo ""
echo "[Step 2/3] Generating Image Embeddings (Offline)..."
echo "------------------------------------------------------------"
echo "Note: This will use AMP and DataLoader if a GPU is available."
python3 generate_embeddings.py

echo ""
echo "[Step 3/3] Evaluating Baseline (Online/Bulk)..."
echo "------------------------------------------------------------"
echo "Note: This builds the FAISS index if it hasn't been built yet."
python3 evaluate_baseline.py

echo ""
echo "============================================================"
echo " Baseline Pipeline Completed Successfully! ✓"
echo "============================================================"
