"""Create the repository dataset layout from three downloaded traversals."""

import argparse
import re
import shutil
from pathlib import Path

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _natural_key(path: Path) -> tuple[tuple[int, int | str], ...]:
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.lower())
        for part in re.split(r"(\d+)", path.name)
    )


def copy_traversal(source: Path, destination: Path, limit: int | None) -> int:
    images = sorted((
        path for path in source.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ), key=_natural_key)
    if limit is not None:
        images = images[:limit]
    if not images:
        raise ValueError(f"no images found in {source}")

    destination.mkdir(parents=True, exist_ok=True)
    for index, source_path in enumerate(images):
        target = destination / f"Image{index:03d}.jpg"
        if source_path.suffix.lower() in {".jpg", ".jpeg"}:
            shutil.copy2(source_path, target)
        else:
            with Image.open(source_path) as image:
                image.convert("RGB").save(target, quality=95)
    return len(images)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--day-left", type=Path, required=True)
    parser.add_argument("--day-right", type=Path, required=True)
    parser.add_argument("--night-right", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/gardens_point"))
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    traversals = {
        args.output / "database/day_left": args.day_left,
        args.output / "query/day_right": args.day_right,
        args.output / "query/night_right": args.night_right,
    }
    for destination, source in traversals.items():
        count = copy_traversal(source, destination, args.limit)
        print(f"{destination}: {count} images")


if __name__ == "__main__":
    main()
