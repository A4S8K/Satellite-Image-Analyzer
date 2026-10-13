"""Generate small synthetic spectral fixtures; these are not satellite observations."""

import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin


def generate_sample(output: Path) -> None:
    """Create independent synthetic spectral bands with known calibration."""
    paths = [
        output / name
        for name in ("B04.tif", "B08.tif", "synthetic_multiband.tif", "MTD_MSIL2A.xml")
    ]
    if any(path.exists() for path in paths):
        raise FileExistsError(
            "Synthetic output exists; choose a different --output directory"
        )
    output.mkdir(parents=True, exist_ok=True)
    values = np.array([-0.5, 0.1, 0.3, 0.5, 0.8], dtype=np.float32)
    indices = np.minimum(np.arange(256) * 5 // 256, 4)
    red = np.full((160, 256), 0.1, dtype=np.float32)
    nir = np.broadcast_to(
        0.1 * (1 + values[indices]) / (1 - values[indices]), red.shape
    )
    red_dn = np.rint(red * 10000 + 1000).astype(np.uint16)
    nir_dn = np.rint(nir * 10000 + 1000).astype(np.uint16)
    red_dn[:16, :16] = 0
    nir_dn[:16, :16] = 0
    profile = {
        "driver": "GTiff",
        "height": 160,
        "width": 256,
        "dtype": "uint16",
        "crs": "EPSG:32643",
        "transform": from_origin(300000, 4700000, 10, 10),
        "nodata": 0,
    }
    for path, band in zip(paths[:2], (red_dn, nir_dn), strict=True):
        with rasterio.open(path, "w", count=1, **profile) as dataset:
            dataset.write(band, 1)
            dataset.update_tags(SYNTHETIC="true", purpose="software verification only")
    with rasterio.open(paths[2], "w", count=2, **profile) as dataset:
        dataset.write(red_dn, 1)
        dataset.write(nir_dn, 2)
        dataset.set_band_description(1, "B04_RED_SYNTHETIC")
        dataset.set_band_description(2, "B08_NIR_SYNTHETIC")
        dataset.update_tags(SYNTHETIC="true")
    paths[3].write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<Level-2A_User_Product><PROCESSING_LEVEL>Level-2A</PROCESSING_LEVEL>"
        "<PROCESSING_BASELINE>04.00</PROCESSING_BASELINE>"
        "<SPACECRAFT_NAME>Sentinel-2A</SPACECRAFT_NAME>"
        "<BOA_QUANTIFICATION_VALUE>10000</BOA_QUANTIFICATION_VALUE>"
        '<BOA_ADD_OFFSET band_id="3">-1000</BOA_ADD_OFFSET>'
        '<BOA_ADD_OFFSET band_id="7">-1000</BOA_ADD_OFFSET>'
        "<SYNTHETIC>true</SYNTHETIC></Level-2A_User_Product>\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/samples/ndvi"))
    args = parser.parse_args()
    try:
        generate_sample(args.output)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Generated SYNTHETIC spectral test fixtures: {args.output}")


if __name__ == "__main__":
    main()
