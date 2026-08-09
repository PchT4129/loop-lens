# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.


## Project Overview

This is a Visual Place Recognition (VPR) pipeline for loop closure detection in SLAM systems. It uses pretrained CNN features (ResNet18) with cosine similarity to retrieve visually similar places from a database. The project includes a metric learning stage using triplet loss to adapt the descriptor for day-night appearance changes.

## Architecture

### Core Pipeline

The system follows a three-stage pipeline:

1. **Feature Extraction**: Images are passed through a ResNet18 backbone (without final classifier layer) producing 512-D L2-normalized features. This is handled by `ResNet18FeatureExtractor` in `models.py`.

2. **Similarity Retrieval**: Query and database features are matched using dot product (equivalent to cosine similarity for normalized vectors) to find top-K candidates. Implemented in `retrieve_top_k()` in `retrieve.py`.

3. **Metric Learning (Optional)**: The embedding space can be fine-tuned using triplet loss to improve day-night retrieval. The `TripletPlaceDataset` constructs (anchor, positive, negative) triplets where positives are nearby images (tolerance ±N frames) and negatives are distant (gap ≥20 frames). Both anchor and database images must follow the naming pattern `Image###.jpg` for index-based matching.

### Key Modules

- **`models.py`**: `ResNet18FeatureExtractor` — loads pretrained ResNet18, removes classifier, returns L2-normalized 512-D features.
- **`dataset.py`**: `ImageFolderDataset` — generic image loader with recursive directory traversal. `get_default_transform()` applies ResNet normalization (ImageNet mean/std).
- **`extract_features.py`**: Batch processes images and saves {features: Tensor[N, 512], paths: list[str]} to disk. Supports loading a fine-tuned checkpoint.
- **`retrieve.py`**: `retrieve_top_k()` performs similarity matching via matrix multiplication.
- **`evaluate.py`**: Computes Recall@K and Precision@K by parsing image indices from file paths and applying frame tolerance (e.g., ±3 frames).
- **`visualize.py`**: Creates matplotlib visualizations of query + top-K results side-by-side.
- **`triplet_dataset.py`**: `TripletPlaceDataset` implements triplet sampling with configurable positive tolerance and negative gap. Supports training on a split (e.g., Image000-069) via min/max-index filtering.
- **`train_triplet.py`**: Standard training loop using `torch.nn.TripletMarginLoss` and Adam optimizer.

### Data Dependencies

All executables assume the Gardens Point Walking dataset in `data/gardens_point/`:
- Database: `data/gardens_point/database/day_left/Image000.jpg` ... `Image099.jpg`
- Query: `data/gardens_point/query/{day_right,night_right}/Image000.jpg` ... `Image099.jpg`

The image index (extracted from filename) is used as pseudo ground-truth location. Matches are deemed correct if indices are within ±tolerance frames.

## Development Commands

### Setup

```bash
# Create conda environment
conda create -n vpr-loop-closure python=3.11
conda activate vpr-loop-closure

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install dependencies
pip install numpy pillow matplotlib tqdm scikit-learn
```

### Feature Extraction

```bash
# Extract database features (pretrained ResNet18)
python -m src.extract_features \
  --image-dir data/gardens_point/database \
  --output outputs/gardens_database_features.pt

# Extract query features
python -m src.extract_features \
  --image-dir data/gardens_point/query \
  --output outputs/gardens_query_features.pt

# Extract with fine-tuned checkpoint
python -m src.extract_features \
  --image-dir data/gardens_point/query \
  --output outputs/gardens_query_features_triplet.pt \
  --checkpoint outputs/checkpoints/resnet18_triplet.pt
```

### Retrieval and Evaluation

```bash
# Show top-K results for all queries
python -m src.retrieve \
  --database outputs/gardens_database_features.pt \
  --query outputs/gardens_query_features.pt \
  --top-k 5

# Evaluate Recall@K and Precision@K across all query splits
python -m src.evaluate \
  --database outputs/gardens_database_features.pt \
  --query outputs/gardens_query_features.pt \
  --top-k 10 \
  --tolerance 3

# Evaluate held-out sequence (e.g., test split)
python -m src.evaluate \
  --database outputs/gardens_database_features_triplet_split.pt \
  --query outputs/gardens_query_features_triplet_split.pt \
  --top-k 10 \
  --tolerance 3 \
  --split-name night_right \
  --min-index 70 \
  --max-index 99
```

### Visualization

```bash
# Visualize retrieval result for a single query
python -m src.visualize \
  --database outputs/gardens_database_features.pt \
  --query outputs/gardens_query_features.pt \
  --query-index 123 \
  --top-k 5 \
  --output outputs/visualizations/result_123.png
```

### Training with Triplet Loss

```bash
# Train on daytime images using nighttime images as anchors (train split 000-069)
python -m src.train_triplet \
  --anchor-dir data/gardens_point/query/night_right \
  --database-dir data/gardens_point/database/day_left \
  --output outputs/checkpoints/resnet18_triplet_train_000_069.pt \
  --epochs 5 \
  --batch-size 16 \
  --lr 1e-4 \
  --margin 0.2 \
  --min-index 0 \
  --max-index 69
```

## Important Implementation Details

1. **Feature Normalization**: L2 normalization is applied in `ResNet18FeatureExtractor.forward()`. This means dot product between features equals cosine similarity.

2. **Image Index Parsing**: `evaluate.py` and `triplet_dataset.py` extract indices via regex pattern `Image(\d+)\.jpg`. This is critical for ground-truth matching and triplet construction. Filenames must follow this convention.

3. **Triplet Sampling**: The `TripletPlaceDataset` allows training on a temporal subsequence (e.g., Image000-069) while negative candidates can come from the full database. This enables train/test splits for evaluating generalization.

4. **Evaluation Tolerance**: The `--tolerance` parameter in `evaluate.py` allows soft matching (e.g., ±3 frames). This reflects the fact that visually similar places don't need to be at exactly the same location.

5. **Checkpoint Format**: Saved checkpoints use a simple dictionary with `model_state_dict`, `epochs`, `lr`, and `margin` keys. Load via `torch.load(..., map_location="cpu")` and `.load_state_dict()`.

## Known Results and Limitations

- **Pretrained baseline** (day_right: 96% Recall@1, night_right: 51% Recall@1) shows the model handles lateral viewpoint changes well but struggles with day-night appearance shifts.
- **After triplet fine-tuning** (day_right: 99% Recall@1, night_right: 86% Recall@1) performance improves significantly, though held-out test splits show overfitting in top-1 ranking (51% → 57% on held-out night_right).
- The README notes that perfect Recall@5 on held-out data (100%) suggests the descriptor is still useful for candidate retrieval in SLAM pipelines, even if top-1 ranking fails.

## File Organization

- `.gitignore`: Excludes data/, .pt files, checkpoints, and Python cache. Feature files are generated, not committed.
- `outputs/`: Stores extracted features, checkpoints, and visualizations. Feature .pt files and checkpoints are ignored.
- `src/`: All Python modules as executable packages (use `python -m src.module_name`).

