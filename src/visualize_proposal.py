"""Render the evidence behind one structured loop-closure proposal."""

import argparse
import json
import os
import tempfile
from pathlib import Path

from src.evaluate import is_correct_match


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance", type=int, default=3)
    args = parser.parse_args()

    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "vpr-matplotlib")
    )
    import matplotlib.pyplot as plt
    from PIL import Image

    proposals = json.loads(args.proposals.read_text(encoding="utf-8"))
    proposal = proposals[args.index]
    accepted = proposal["accepted"]
    correct = is_correct_match(
        proposal["query_path"], proposal["candidate_path"], args.tolerance
    )

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for axis, key, label in (
        (axes[0], "query_path", "Query"),
        (axes[1], "candidate_path", "Top-1 candidate"),
    ):
        axis.imshow(Image.open(proposal[key]).convert("RGB"))
        axis.set_title(f"{label}\n{Path(proposal[key]).name}")
        axis.set_xticks([])
        axis.set_yticks([])

    decision = "ACCEPT" if accepted else "REJECT"
    truth = "correct candidate" if correct else "incorrect candidate"
    colour = "#18864b" if accepted == correct else "#c43c39"
    figure.suptitle(
        f"{decision} · {truth} | sequence={proposal['sequence_score']:.3f} · "
        f"matches={proposal['num_matches']} · inliers={proposal['num_inliers']} · "
        f"ratio={proposal['inlier_ratio']:.3f}",
        color=colour,
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
