import argparse
from pathlib import Path

import torch


def load_feature_file(path: str | Path, include_meta: bool = False):
    data = torch.load(path, map_location="cpu")
    if "features" not in data or "paths" not in data:
        raise ValueError(f"invalid feature artifact: {path}")
    if data["features"].ndim != 2 or len(data["features"]) != len(data["paths"]):
        raise ValueError(f"feature/path shape mismatch in: {path}")
    if include_meta:
        return data["features"], data["paths"], data.get("meta", {})
    return data["features"], data["paths"]

def retrieve_top_k(
    query_features: torch.Tensor,
    database_features: torch.Tensor,
    k: int = 5,
):
    similarities = query_features @ database_features.T

    top_scores, top_indices = torch.topk(
        similarities,
        k=min(k, database_features.shape[0]),
        dim=1,
    )

    return top_scores, top_indices

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    database_features, database_paths = load_feature_file(args.database)
    query_features, query_paths = load_feature_file(args.query)

    top_scores, top_indices = retrieve_top_k(
        query_features=query_features,
        database_features=database_features,
        k=args.top_k,
    )

    for query_idx, query_path in enumerate(query_paths):
        print("=" * 80)
        print(f"Query: {query_path}")

        for rank in range(top_indices.shape[1]):
            db_idx = top_indices[query_idx, rank].item()
            score = top_scores[query_idx, rank].item()
            db_path = database_paths[db_idx]

            print(f"Top {rank + 1}: score={score:.4f} | {db_path}")


if __name__ == "__main__":
    main()
