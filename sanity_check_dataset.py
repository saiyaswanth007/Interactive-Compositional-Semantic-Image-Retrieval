#!/usr/bin/env python3
"""
Step 1 sanity check — verify the Flickr30K dataset is correctly loaded.

Run:
    python sanity_check_dataset.py

Expected output:
    - Dataset summary (image counts, caption counts, split sizes)
    - Example image IDs, paths, and captions
    - Verification that every image has exactly 5 captions
    - Confirmation that train/test splits are disjoint and cover all data
"""
import os
import sys

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.dataset import Flickr30kDataset


def main():
    print("Loading Flickr30K dataset …\n")
    ds = Flickr30kDataset()
    ds.summary()

    # ------------------------------------------------------------------
    # Additional sanity checks
    # ------------------------------------------------------------------
    print("\nRunning sanity checks …\n")

    # 1. Every image has exactly 5 captions
    caps_counts = {img: len(ds.get_captions(img)) for img in ds.get_all_image_ids()}
    unique_counts = set(caps_counts.values())
    print(f"  [CHECK] Unique caption-count values: {unique_counts}")
    assert unique_counts == {5}, f"Expected all images to have 5 captions, got {unique_counts}"
    print("  ✓ Every image has exactly 5 captions.")

    # 2. Image files actually exist
    missing = [img for img in ds.get_all_image_ids() if not os.path.isfile(ds.get_image_path(img))]
    print(f"  [CHECK] Missing image files: {len(missing)}")
    assert len(missing) == 0, f"Missing files: {missing[:5]}"
    print("  ✓ All image files exist on disk.")

    # 3. Train / test are disjoint and cover all images
    train_set = set(ds.get_split("train"))
    test_set = set(ds.get_split("test"))
    all_set = set(ds.get_all_image_ids())

    assert train_set & test_set == set(), "Train and test overlap!"
    assert train_set | test_set == all_set, "Train + test don't cover all images!"
    print(f"  ✓ Train ({len(train_set):,}) + Test ({len(test_set):,}) = "
          f"Total ({len(all_set):,}), disjoint and complete.")

    # 4. Captions are non-empty strings
    empty = 0
    for img in ds.get_all_image_ids():
        for cap in ds.get_captions(img):
            if not cap.strip():
                empty += 1
    print(f"  [CHECK] Empty captions: {empty}")
    assert empty == 0, f"Found {empty} empty captions!"
    print("  ✓ No empty captions.")

    # 5. Quick look at a few test-set examples
    print("\n" + "-" * 60)
    print("  Sample TEST set images:")
    print("-" * 60)
    for img_id in ds.get_split("test")[:3]:
        print(f"\n  {img_id}")
        for i, cap in enumerate(ds.get_captions(img_id)):
            print(f"    [{i+1}] {cap}")

    print("\n" + "=" * 60)
    print("  ALL SANITY CHECKS PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
