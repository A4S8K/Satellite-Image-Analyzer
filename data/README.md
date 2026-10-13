# Image data

Place local visible RGB images in `samples/` to try the fire analyzer. The project
does not download imagery automatically or include a validated fire dataset.

For a local synthetic example:

```powershell
.\.venv\Scripts\python.exe scripts\generate_sample.py
.\.venv\Scripts\python.exe main.py analyze data\samples\synthetic_fire.png --output results
```

The generator creates a clearly labeled synthetic 1920×1080 image and refuses
to overwrite an existing file. This is a software fixture, not a satellite
observation or a validated fire sample.

For your own visible satellite image:

```powershell
.\.venv\Scripts\python.exe main.py analyze data\samples\satellite.jpg --output results
```

Week 2 accepts 8-bit RGB/RGBA or grayscale PNG, JPEG, and suitably interpreted
TIFF imagery. Grayscale lacks color-based fire evidence. High-bit-depth,
multispectral, and ambiguous-band rasters need deliberate visible-image
preparation; analysis rejects them instead of assigning arbitrary bands to RGB.
Use `info` to inspect metadata and `preprocess` for general raster preparation.

Keep large, private, or unlicensed images outside Git. `outputs/` is the Week 1
location for preprocessing outputs; analysis artifacts are written to separate
run directories under `results/` by default.

Tests generate small synthetic images in temporary directories without internet
access. For a shared sample, document source/license, sensor, bands, and any
resizing or color adjustments. A synthetic orange patch is a detector fixture,
not an authentic satellite fire observation or evidence of field accuracy.

## Week 3 spectral data

The `ndvi` command requires measured Red/NIR GeoTIFFs, valid geospatial metadata,
and radiometric calibration. A normal RGB preview cannot supply NIR. Sentinel-2
uses B04/B08; Landsat 8/9 uses B4/B5. Use bands and product metadata from the same
acquisition. Do not apply RGB min/max stretching or generic preprocessing to them.

Generate a reproducible local **synthetic** fixture:

```powershell
.\.venv\Scripts\python.exe scripts\generate_ndvi_sample.py
.\.venv\Scripts\python.exe main.py ndvi --red data\samples\ndvi\B04.tif --nir data\samples\ndvi\B08.tif --sensor sentinel-2 --metadata data\samples\ndvi\MTD_MSIL2A.xml --visualize --json
```

This creates two 160×256 uint16 bands, a two-band stack, and synthetic product XML
with known quantification/additive offsets. Rasters/XML are labeled synthetic.
The generator refuses to overwrite existing files; choose another `--output`
directory if needed. Generated fixtures are ignored by Git and are not real
satellite observations or an accuracy dataset.

Obtain actual imagery through
[Copernicus Data Space](https://dataspace.copernicus.eu/explore-data) or
[USGS Landsat data access](https://www.usgs.gov/landsat-missions/landsat-data-access).
For Sentinel, preserve `MTD_MSIL2A.xml` or `MTD_MSIL1C.xml` while converting the
same granule's B04/B08 JP2 bands to GeoTIFF without changing raw values/grid.
For Landsat, keep Collection 2 Level-2 SR_B4/SR_B5 and the matching MTL file.
Local `sentinel2/` and `landsat/` directories are ignored to keep large products
out of Git. Record source, product/acquisition ID, processing level/baseline,
terms, bands, and conversion steps for every evaluation image.

The [Week 3 guide](../docs/week3_ndvi.md) covers calibration, alignment, valid-pixel
statistics, outputs, limitations, and offline verification.
