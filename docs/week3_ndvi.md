# Week 3: Vegetation Analysis and Real NDVI

## Overview

The `ndvi` command adds measured Red/NIR vegetation analysis to the existing
Satellite Image Analyzer CLI. It uses NumPy and Rasterio, preserves a georeferenced
scientific grid, and exports numerical statistics and optional Matplotlib previews.
It does not create a NIR channel from RGB, add wildfire-risk coefficients, or
modify the Week 2 visual fire detector.

Week 3 is complete: the Windows Python 3.12 regression, CLI, export, formatting,
lint, and type checks passed for version 0.3.0. Verification used synthetic
spectral fixtures, not a real satellite acquisition or ground-truth accuracy study.

## NDVI and mathematical formula

```text
NDVI = (NIR - RED) / (NIR + RED)
```

RED and NIR must be measured spectral channels with compatible radiometric units
and spatial alignment. For example, reflectance RED=0.2 and NIR=0.6 gives NDVI=0.5.
This arithmetic example is not an observation or an accuracy claim.

The engine computes float32 NDVI with vectorized operations. It excludes source
NoData/masked pixels, nonfinite or negative corrected reflectance, and sums less
than or equal to `denominator_epsilon` (default `1e-6`). Invalid NDVI is NaN in the
scientific raster and masked out of statistics. Valid values stay in [-1,1].

## Supported satellite sensors

| `--sensor` | Red | NIR | Native resolution used here | Product calibration |
| --- | --- | --- | --- | --- |
| `sentinel-2` | B04 | B08 | 10 m | Level-2A or Level-1C user-product XML |
| `landsat-8` | B4 | B5 | 30 m | Collection 2 Level-2 MTL text or JSON |
| `landsat-9` | B4 | B5 | 30 m | Collection 2 Level-2 MTL text or JSON |
| `custom` | Explicit Red input | Explicit NIR input | Input reference grid | Explicit coefficients, TIFF calibration, or declared reflectance |

Sentinel B04/B08 belong to the 10 m band group. See
[Copernicus Sentinel-2 mission documentation](https://sentiwiki.copernicus.eu/web/s2-mission).
Landsat 8/9 band assignments and resolutions are documented by
[USGS band designations](https://www.usgs.gov/faqs/what-are-band-designations-landsat-satellites).
Band numbers in a custom stack are not sensor band names: provide the actual
1-based file indexes using `--red-band` and `--nir-band`.

## Installation

Week 3 uses the existing runtime dependencies; no trained-model framework is
required. From the repository root in PowerShell:

```powershell
$env:PYTHONUTF8 = '1'
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe main.py ndvi --help
```

No environment activation or PowerShell execution-policy change is needed.
`sat-analyzer ndvi` and `python -m satellite_analyzer ndvi` expose the same command.

## Input data requirements

Use local GeoTIFFs with a valid CRS, affine transform, and measured spectral bands.
Two modes are supported:

1. Separate single-band rasters: `--red FILE --nir FILE`.
2. One multispectral raster: `--multiband FILE --red-band INDEX --nir-band INDEX`.

The modes cannot be mixed. A normal RGB or false-color display image is not an
acceptable source of inferred NIR. Explicit mapping declares real spectral roles;
it does not transform a visible RGB channel into infrared.

Select bands from the same acquisition and processing product. The application
validates the supplied spatial/radiometric information, but users remain
responsible for matching product metadata to the actual imagery. Keep source
NoData and masks. Do not min/max-normalize, denoise, color-stretch, or resize bands
with the generic RGB preprocessing command before scientific NDVI computation.

### Digital numbers versus reflectance

Conversion is `reflectance = source_value * scale + offset` for each band.
A shared multiplicative scale can cancel in NDVI, but additive offsets do not;
discarding them produces a different index.

The engine selects calibration from one of these declared sources:

- All four explicit `--red-scale`, `--red-offset`, `--nir-scale`, `--nir-offset` values.
- `--metadata`: supported Sentinel user-product XML or Landsat Collection 2 Level-2 MTL.
- Nonidentity TIFF band scale/offset metadata.
- `--input-units reflectance` for already calibrated values, with identity TIFF scaling.
- In `auto` mode only, finite floating input within [0,1] can be inferred as
  reflectance. The report records that inference rather than claiming product-level calibration.

Integer DN without trustworthy calibration fails, even when `--sensor` is supplied.
Partial explicit coefficients, coefficients combined with product metadata, and
declared reflectance with another calibration source fail instead of applying
ambiguous or double scaling. TIFF parameters overridden by explicit/product
calibration are recorded but not applied a second time.

For Sentinel-2, the product XML supplies quantification and per-band offsets:
`reflectance = (DN + ADD_OFFSET) / QUANTIFICATION_VALUE`. Level-2A is surface
reflectance; Level-1C is top-of-atmosphere reflectance. Processing baseline 04.00
and later requires the declared additive offsets; missing offsets are accepted
as zero only with a documented earlier baseline. DN=0 is excluded before scaling.
See [Copernicus product radiometry](https://sentiwiki.copernicus.eu/web/s2-products).

For Landsat Collection 2 Level-2 surface reflectance, MTL values are read from
`LEVEL2_SURFACE_REFLECTANCE_PARAMETERS` for bands 4/5. Level-1 parameters are not
substituted. USGS documents the Collection 2 scale 0.0000275 and offset -0.2;
the implementation still reads the supplied metadata rather than guessing from
the sensor name. See [USGS scaling guidance](https://www.usgs.gov/faqs/how-do-i-use-a-scale-factor-landsat-level-2-science-products).

### Spatial alignment

CRS, dimensions, pixel grid, and transform must match in default strict mode.
Mismatches produce an explicit error. `--align-nir` opts into reprojection of
calibrated NIR to the Red grid. `--resampling nearest` or `bilinear` selects the
method; bilinear is the default. Input metadata, reference grid, and alignment
provenance are saved. Reprojection can change local values and valid coverage.
Scientific outputs remain at the Red reference resolution.

## Example CLI usage

Generate small, explicitly synthetic local data with independent Red/NIR bands,
known Sentinel-style quantification/offsets, and NoData:

```powershell
.\.venv\Scripts\python.exe scripts\generate_ndvi_sample.py
.\.venv\Scripts\python.exe main.py ndvi --red data\samples\ndvi\B04.tif --nir data\samples\ndvi\B08.tif --sensor sentinel-2 --metadata data\samples\ndvi\MTD_MSIL2A.xml --output results\ndvi --visualize --json
```

The generator defaults to `data/samples/ndvi`, creates 160×256 fixtures, marks the
rasters/metadata as synthetic, and refuses to overwrite existing fixtures. Use
`--output` to choose another directory. It is not a real Sentinel acquisition.

The verified fixture produced 40,704 valid pixels, 256 invalid pixels, mean NDVI
approximately 0.241646, and vegetation coverage approximately 60.1415%. Separate
bands and the explicitly mapped stack gave the same counts/statistics.

![NDVI heatmap of explicitly synthetic Red/NIR fixtures](example_ndvi_heatmap.png)

For the generated two-band stack:

```powershell
.\.venv\Scripts\python.exe main.py ndvi --multiband data\samples\ndvi\synthetic_multiband.tif --red-band 1 --nir-band 2 --sensor sentinel-2 --metadata data\samples\ndvi\MTD_MSIL2A.xml --visualize
```

For already calibrated Red/NIR GeoTIFFs:

```powershell
.\.venv\Scripts\python.exe main.py ndvi --red red_reflectance.tif --nir nir_reflectance.tif --input-units reflectance --visualize
```

For real Landsat Collection 2 Level-2 data:

```powershell
.\.venv\Scripts\python.exe main.py ndvi --red data\landsat\SR_B4.tif --nir data\landsat\SR_B5.tif --sensor landsat-9 --metadata data\landsat\PRODUCT_MTL.txt --visualize
```

Additional controls include `--config configs/default.toml`, `--max-pixels`,
`--verbose`, and the explicit calibration/alignment options above. JSON is always
exported; `--json` additionally writes only JSON to stdout. Logs/errors go to stderr.

## Example output

Each run creates a unique directory below the selected output root:

```text
results/ndvi/<red-stem>-ndvi-<UTC timestamp>-<unique suffix>/
├── ndvi_map.tif
├── vegetation_classes.tif
├── vegetation_statistics.json
├── ndvi_heatmap.png          # With --visualize
├── vegetation_classes.png   # With --visualize
└── ndvi_histogram.png        # With --visualize
```

`ndvi_map.tif` is float32 with NaN NoData and an internal validity mask.
`vegetation_classes.tif` is uint8 with invalid class 0, classes 1–5, and a matching
mask. Both preserve CRS, affine transform, and the complete scientific grid.
Class descriptions are included in raster metadata. Only PNG display previews
are limited by `preview_max_dimension`; statistics/histogram populations are not
downsampled.

The heatmap uses a fixed [-1,1] color scale and colorbar; the class map includes
a legend; the histogram includes all valid NDVI samples and their mean. Preview
axes refer to source pixel positions, not an invented projected map layout.

The JSON report includes sensor, formula, spatial grid, input/calibration and
alignment provenance, settings, per-class counts/percentages, mean/min/max/median,
population standard deviation, valid/invalid counts, coverage, timestamps,
measured processing time, limitations, and absolute artifact paths. Numerical
values are calculated from input pixels. Timing covers loading, calibration,
alignment, NDVI, and statistics, excluding artifact export. Previous runs and
source rasters/metadata are preserved.

## Vegetation classification

| Class ID | Name | Default NDVI interval |
| --- | --- | --- |
| 0 | Invalid | No valid NDVI |
| 1 | `non_vegetation` | [-1,0] |
| 2 | `sparse` | (0,0.2) |
| 3 | `low` | [0.2,0.4) |
| 4 | `moderate` | [0.4,0.6) |
| 5 | `dense` | [0.6,1] |

Negative/zero NDVI is grouped as non-vegetation/water/other; it is not a dedicated
water classification. Boundary inclusion is explicit. These initial thresholds
are configurable starting points, not universal ecological standards. Coverage
is the fraction of valid pixels meeting `NDVI >= coverage_threshold` (default 0.2).
Class percentages also use valid pixels only and sum to 100 when any are valid.
All-invalid input has zero valid count and JSON `null` summaries/percentages.

## Configuration

The existing TOML config accepts an independent `[ndvi]` table:

```toml
[ndvi]
sensor = "custom"
input_units = "auto"
alignment = "strict"
resampling = "bilinear"
max_pixels = 150000000
denominator_epsilon = 0.000001
non_vegetation_max = 0.0
sparse_max = 0.2
low_max = 0.4
moderate_max = 0.6
coverage_threshold = 0.2
histogram_bins = 50
preview_max_dimension = 2048
```

Boundary values must increase strictly. CLI options override supplied TOML values.
Optional `metadata_path` in TOML resolves relative to the config file. Built-in
defaults allow use outside the repository without assuming a local config exists.
The per-band pixel budget is a resource guard, not a guarantee that a whole scene
will fit in available memory; use a geospatially valid crop for limited-memory systems.

## Obtaining real satellite data

For Sentinel-2, use [Copernicus Data Space](https://dataspace.copernicus.eu/explore-data)
to select a product and acquisition. Prefer Level-2A surface reflectance for this
workflow. Retain the top-level `MTD_MSIL2A.xml`; select the same granule's B04 and
B08 in its 10 m image folder. Original bands are typically JP2. Convert each to
single-band GeoTIFF with QGIS or GDAL installed separately, preserving raw DN,
CRS, affine transform, and NoData. Do not stretch or resample during conversion.
For Level-1C use its `MTD_MSIL1C.xml` and understand the TOA distinction.

For Landsat, follow [USGS data access](https://www.usgs.gov/landsat-missions/landsat-data-access)
to obtain one Landsat 8/9 Collection 2 Level-2 product. Select its surface
reflectance `SR_B4` and `SR_B5` GeoTIFFs and retain the matching MTL text or JSON.

Do not download a web RGB preview and rename channels as Red/NIR. Keep the
acquisition ID, processing level/baseline, band identity, conversion steps,
license/terms, and metadata with any image used for evaluation. This release
contains no real acquisition and performs no automatic imagery downloads.

## Limitations

- NDVI is a spectral vegetation-condition indicator, not fire detection or a direct moisture measurement.
- Threshold categories vary by ecosystem, season, soil background, sensor, and processing.
- Cloud, shadow, snow, saturation, and other quality masks are not automatically applied.
- Negative corrected reflectance is conservatively excluded; this may remove dark-surface samples.
- Float [0,1] inference is not proof of physical reflectance; explicit provenance is preferable.
- TOA and surface reflectance are distinct and should not be interchanged in comparisons.
- Full-band processing uses memory proportional to raster size; no tiled/out-of-core engine is included.
- Synthetic tests establish implementation behavior; no ground-truth accuracy evaluation is claimed.

## Testing instructions

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m black --check .
.\.venv\Scripts\python.exe -m mypy src
```

Offline fixtures exercise positive/negative/zero NDVI, division protection,
NoData/nonfinite values, calibration sources/offsets, grid validation/alignment,
classification boundaries/statistics, georeferenced exports, strict JSON and CLI
errors. The complete suite also includes Week 1/2 regressions. Only actual test
runs determine the final status; no scientific field-validation result is inferred.

Version 0.3.0 verification: **276 tests passed**, 90% aggregate coverage; Ruff
passed, Black checked 63 files, and mypy checked 41 source files. Installed
`main.py`, module, and `sat-analyzer` entry points and wheel construction passed.
Manual separate/multiband synthetic runs, native GeoTIFF values/masks/grid, and
the three PNG outputs were verified. Tests currently emit Affine multiplication
pending-deprecation warnings and warnings from deliberately invalid fixtures;
these do not indicate failed scientific calculations.

## Day 15–21 implementation stages

| Day | Development task | Implemented component |
| --- | --- | --- |
| 15 | Repository audit and spectral architecture | Separate spectral loader/analyzer contracts |
| 16 | Red/NIR loading and GeoTIFF validation | Band mapping, calibration, CRS/grid validation |
| 17 | NDVI calculation | Vectorized float32 engine and validity mask |
| 18 | Vegetation classes and statistics | Configurable boundaries and valid-pixel summaries |
| 19 | Visualizations and GeoTIFF export | Heatmap, class preview, histogram, scientific rasters |
| 20 | Existing pipeline and CLI integration | Existing application's `ndvi` command |
| 21 | Testing, bug fixes, and documentation | Unit/integration/regression checks and this guide |

These are implementation stages, not claims of seven elapsed calendar days.
Week 3 completion is based on the passed quality gates, not a planned date.

## Definition of done

- [x] NDVI uses supplied Red/NIR spectral bands rather than RGB-derived fake NIR.
- [x] Sentinel-2 B04/B08 metadata-backed input and explicit multiband mapping work.
- [x] NoData, zero/near-zero division, and nonfinite/negative reflectance are handled.
- [x] Configurable vegetation classes and valid-pixel statistics are implemented.
- [x] Heatmap, classification preview, and histogram export work.
- [x] Native-grid GeoTIFF and strict JSON export preserve spatial/provenance information.
- [x] The existing application's CLI integrates NDVI alongside Week 1/2 commands.
- [x] Unit, integration, and regression tests and required quality checks pass.
- [x] README, architecture, input-data guidance, and this NDVI guide are updated.

No genuine Sentinel/Landsat scene or ground-truth labels were available in the
repository for release verification. Acquisition guidance is provided above;
real-scene validation and automatic cloud/quality masking remain next steps.
