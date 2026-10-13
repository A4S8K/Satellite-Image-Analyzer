"""Generate labelled-by-construction RGB test imagery, never real satellite data."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def generate_sample(path: Path, width: int = 1920, height: int = 1080) -> None:
    """Create a deterministic synthetic scene for CLI/performance verification."""
    if width < 128 or height < 96:
        raise ValueError("Sample dimensions must be at least 128x96")
    rng = np.random.default_rng(42)
    noise = rng.integers(-10, 11, size=(height, width, 1), dtype=np.int16)
    pixels = np.clip(np.array([38, 105, 45], dtype=np.int16) + noise, 0, 255).astype(
        np.uint8
    )
    image = Image.fromarray(pixels)
    drawing = ImageDraw.Draw(image)
    drawing.rectangle((0, 0, width // 4, height), fill=(24, 77, 143))
    drawing.rectangle((width * 4 // 5, 0, width - 1, height - 1), fill=(182, 184, 183))
    drawing.ellipse(
        (width * 0.38, height * 0.37, width * 0.48, height * 0.55), fill=(255, 135, 12)
    )
    drawing.ellipse(
        (width * 0.40, height * 0.41, width * 0.46, height * 0.51), fill=(255, 225, 10)
    )
    drawing.rectangle(
        (width * 0.59, height * 0.66, width * 0.64, height * 0.72), fill=(235, 60, 8)
    )
    drawing.point((width // 2, height // 4), fill=(255, 120, 10))
    if path.suffix.lower() != ".png":
        raise ValueError("Synthetic sample output must be a PNG file")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Refuse existing paths so generating a fixture cannot replace user imagery.
    with path.open("xb") as handle:
        image.save(handle, format="PNG")


def main() -> None:
    """Generate a local sample with configurable output and dimensions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("data/samples/synthetic_fire.png")
    )
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    args = parser.parse_args()
    try:
        generate_sample(args.output, args.width, args.height)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Generated synthetic RGB test fixture: {args.output}")


if __name__ == "__main__":
    main()
