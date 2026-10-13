"""Resolve documented Red/NIR calibration without guessing from sensor names.

Calibration is deliberately separate from applying it: the loader must exclude
source nodata (especially Sentinel-2 DN=0) *before* multiplication and addition.
Sentinel-2 L1C is TOA reflectance; L2A and Landsat C2 L2 are surface reflectance.
Their additive offsets matter for NDVI and must never be discarded.
"""

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from xml.etree import ElementTree

import numpy as np
from numpy.typing import NDArray

from satellite_analyzer.core.exceptions import SatelliteAnalyzerError
from satellite_analyzer.core.ndvi_config import NDVISettings

BandRole = Literal["red", "nir"]
_MAX_METADATA_BYTES = 8 * 1024 * 1024
_SENTINEL_BAND_IDS: dict[BandRole, int] = {"red": 3, "nir": 7}
_LANDSAT_BANDS: dict[BandRole, int] = {"red": 4, "nir": 5}


@dataclass(frozen=True, slots=True)
class RadiometricCalibration:
    """An affine conversion plus the evidence used to select its coefficients.

    ``reflectance = source_value * scale + offset``. ``details`` contains only
    JSON-compatible finite values; it records assumptions rather than claiming
    that generic TIFF calibration identifies surface versus TOA reflectance.
    """

    scale: float
    offset: float
    source: str
    reflectance_type: str
    details: dict[str, Any]
    zero_is_nodata: bool = False

    def __post_init__(self) -> None:
        _validate_coefficients(self.scale, self.offset, self.source)
        try:
            json.dumps(self.details, allow_nan=False)
        except (TypeError, ValueError, OverflowError) as error:
            raise SatelliteAnalyzerError(
                "Radiometric provenance must contain finite JSON values"
            ) from error


def resolve_calibration(
    *,
    settings: NDVISettings,
    role: BandRole,
    dtype: np.dtype[Any],
    valid_values: NDArray[Any],
    tiff_scale: float = 1.0,
    tiff_offset: float = 0.0,
) -> RadiometricCalibration:
    """Choose explicit, product metadata, TIFF, or bounded-float calibration.

    ``valid_values`` consists of finite, valid *source* pixels, before any
    radiometric correction. Only reductions are used for bounds, avoiding a
    second image-sized copy. Integer DN has no universal implied scale, even
    when a sensor name is known. Declared reflectance forbids TIFF rescaling.
    """
    if role not in _SENTINEL_BAND_IDS:
        raise SatelliteAnalyzerError(f"Unsupported spectral role: {role!r}")
    _validate_coefficients(tiff_scale, tiff_offset, "TIFF band metadata")
    nonidentity_tiff = tiff_scale != 1.0 or tiff_offset != 0.0
    base_details: dict[str, Any] = {"role": role, "sensor": settings.sensor}

    if settings.input_units == "reflectance":
        if nonidentity_tiff:
            raise SatelliteAnalyzerError(
                "Declared reflectance conflicts with nonidentity TIFF scale/offset; "
                "use auto/dn with calibration, or supply an unscaled reflectance raster"
            )
        return RadiometricCalibration(
            1.0, 0.0, "declared_reflectance", "unspecified", base_details
        )

    explicit_scale = settings.red_scale if role == "red" else settings.nir_scale
    explicit_offset = settings.red_offset if role == "red" else settings.nir_offset
    if explicit_scale is not None and explicit_offset is not None:
        _record_tiff_override(base_details, tiff_scale, tiff_offset)
        return RadiometricCalibration(
            float(explicit_scale),
            float(explicit_offset),
            "explicit_coefficients",
            "unspecified",
            base_details,
            zero_is_nodata=settings.sensor == "sentinel-2",
        )

    if settings.metadata_path is not None:
        calibration = _from_product_metadata(settings.metadata_path, settings, role)
        _record_tiff_override(calibration.details, tiff_scale, tiff_offset)
        return calibration

    if nonidentity_tiff:
        return RadiometricCalibration(
            float(tiff_scale),
            float(tiff_offset),
            "tiff_band_metadata",
            "unspecified",
            base_details,
            zero_is_nodata=settings.sensor == "sentinel-2",
        )

    if settings.input_units == "auto" and np.issubdtype(dtype, np.floating):
        if valid_values.size == 0:
            # No reflectance claim can be inferred from an empty raster. Identity
            # permits the caller to produce the more useful all-invalid error.
            base_details["inference"] = "empty_float_source_no_valid_pixels"
            return RadiometricCalibration(
                1.0, 0.0, "inferred_float_reflectance", "unspecified", base_details
            )
        minimum = float(np.min(valid_values))
        maximum = float(np.max(valid_values))
        if math.isfinite(minimum) and math.isfinite(maximum):
            if 0.0 <= minimum <= maximum <= 1.0:
                base_details.update(
                    inference="finite_float_values_within_0_1",
                    valid_source_min=minimum,
                    valid_source_max=maximum,
                )
                return RadiometricCalibration(
                    1.0,
                    0.0,
                    "inferred_float_reflectance",
                    "unspecified",
                    base_details,
                )
    raise SatelliteAnalyzerError(
        f"Unknown radiometric scale for {role} band ({dtype}); supply explicit "
        "red/nir scale and offset, trusted product metadata, or declare "
        "input_units='reflectance' for already calibrated values"
    )


def _validate_coefficients(scale: float, offset: float, source: str) -> None:
    if not math.isfinite(scale) or scale <= 0 or not math.isfinite(offset):
        raise SatelliteAnalyzerError(
            f"Invalid {source} calibration: scale must be positive and finite; "
            "offset must be finite"
        )


def _record_tiff_override(details: dict[str, Any], scale: float, offset: float) -> None:
    if scale != 1.0 or offset != 0.0:
        details["overridden_tiff_calibration"] = {
            "scale": float(scale),
            "offset": float(offset),
            "applied": False,
        }


def _from_product_metadata(
    path: Path, settings: NDVISettings, role: BandRole
) -> RadiometricCalibration:
    try:
        if not path.is_file():
            raise OSError("metadata path is not a regular file")
        if path.stat().st_size > _MAX_METADATA_BYTES:
            raise ValueError("metadata exceeds the 8 MiB safety limit")
        text = path.read_text(encoding="utf-8-sig").strip()
        if not text:
            raise ValueError("metadata is empty")
        if text.startswith("<"):
            return _sentinel_metadata(text, path, settings, role)
        groups = _landsat_groups(text)
        return _landsat_metadata(groups, path, settings, role)
    except SatelliteAnalyzerError:
        raise
    except (
        OSError,
        UnicodeError,
        ValueError,
        TypeError,
        OverflowError,
        ElementTree.ParseError,
    ) as error:
        raise SatelliteAnalyzerError(
            f"Invalid radiometric metadata '{path}': {error}"
        ) from error


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _xml_value(root: ElementTree.Element, name: str) -> str | None:
    values = [
        (node.text or "").strip()
        for node in root.iter()
        if _local_name(node.tag) == name
    ]
    if len(values) > 1:
        raise ValueError(f"Ambiguous duplicate XML field {name}")
    return values[0] if values else None


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Invalid boolean {label}")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"Missing or invalid numeric {label}") from error
    if not math.isfinite(result):
        raise ValueError(f"Nonfinite {label}")
    return result


def _sentinel_metadata(
    text: str, path: Path, settings: NDVISettings, role: BandRole
) -> RadiometricCalibration:
    # ElementTree never fetches external resources. Reject DTD/entity declarations
    # too, so user-supplied metadata cannot expand internal entities in memory.
    if "<!DOCTYPE" in text.upper() or "<!ENTITY" in text.upper():
        raise ValueError("DTD/entity declarations are not allowed in metadata")
    root = ElementTree.fromstring(text)
    supported_roots = {
        "Level-2A_User_Product": (
            "Level-2A",
            "BOA_QUANTIFICATION_VALUE",
            "BOA_ADD_OFFSET",
        ),
        "Level-1C_User_Product": (
            "Level-1C",
            "QUANTIFICATION_VALUE",
            "RADIO_ADD_OFFSET",
        ),
    }
    level_spec = supported_roots.get(_local_name(root.tag))
    if level_spec is None:
        raise ValueError(
            "Only Sentinel-2 Level-1C/Level-2A user-product XML is supported"
        )
    if settings.sensor not in {"custom", "sentinel-2"}:
        raise ValueError("Sentinel-2 metadata does not match the configured sensor")
    level, quantification_name, offset_name = level_spec
    declared_level = _xml_value(root, "PROCESSING_LEVEL")
    if declared_level is not None and declared_level != level:
        raise ValueError("Sentinel-2 XML root and PROCESSING_LEVEL disagree")
    spacecraft = _xml_value(root, "SPACECRAFT_NAME")
    if spacecraft is not None and spacecraft not in {
        "Sentinel-2A",
        "Sentinel-2B",
        "Sentinel-2C",
        "Sentinel-2D",
    }:
        raise ValueError("Unsupported Sentinel-2 SPACECRAFT_NAME")
    quantification = _number(_xml_value(root, quantification_name), quantification_name)
    if quantification <= 0:
        raise ValueError(f"{quantification_name} must be positive")
    baseline_text = _xml_value(root, "PROCESSING_BASELINE")
    baseline = (
        _number(baseline_text, "PROCESSING_BASELINE")
        if baseline_text is not None
        else None
    )
    if baseline is not None and baseline < 0:
        raise ValueError("PROCESSING_BASELINE must not be negative")
    band_id = _SENTINEL_BAND_IDS[role]
    offset_values = [
        node.text
        for node in root.iter()
        if _local_name(node.tag) == offset_name
        and node.attrib.get("band_id") == str(band_id)
    ]
    if len(offset_values) > 1:
        raise ValueError(f"Duplicate {offset_name} for band_id={band_id}")
    assumed_zero = not offset_values
    if assumed_zero:
        # PB 04.00 introduced radiometric offsets. Historical acquisition dates
        # are insufficient because old scenes can have been reprocessed later.
        if baseline is None or baseline >= 4.0:
            raise ValueError(
                f"Missing {offset_name} for band_id={band_id}; zero offset is "
                "allowed only with a documented pre-04.00 processing baseline"
            )
        additive_dn = 0.0
    else:
        additive_dn = _number(offset_values[0], offset_name)
    details: dict[str, Any] = {
        "metadata_path": str(path),
        "role": role,
        "sensor": "sentinel-2",
        "processing_level": level,
        "processing_baseline": baseline_text,
        "band_id": band_id,
        "quantification_value": quantification,
        "additive_offset_dn": additive_dn,
        "zero_offset_from_pre_04_baseline": assumed_zero,
    }
    return RadiometricCalibration(
        scale=1.0 / quantification,
        offset=additive_dn / quantification,
        source="sentinel_product_metadata",
        reflectance_type="surface" if level == "Level-2A" else "toa",
        details=details,
        zero_is_nodata=True,
    )


def _landsat_groups(text: str) -> dict[str, Any]:
    """Read USGS MTL ODL/JSON while keeping L1 and L2 parameter groups distinct."""
    if text.startswith("{"):
        metadata = json.loads(text, object_pairs_hook=_unique_json_fields)
        if not isinstance(metadata, dict):
            raise ValueError("Landsat MTL must be a JSON object")
        json_root = metadata.get("LANDSAT_METADATA_FILE")
        if not isinstance(json_root, dict):
            raise ValueError("Missing LANDSAT_METADATA_FILE MTL root")
        return json_root

    root: dict[str, Any] = {}
    stack: list[tuple[str, dict[str, Any]]] = []
    finished = False
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if finished:
            raise ValueError("Unexpected text after MTL END")
        if line == "END":
            if stack or not root:
                raise ValueError("Unclosed or missing Landsat MTL groups")
            finished = True
            continue
        matched = re.fullmatch(r"([A-Z][A-Z0-9_]*)\s*=\s*(.+)", line)
        if matched is None:
            raise ValueError(f"Invalid MTL statement on line {line_number}")
        key, value = matched.groups()
        value = value.strip()
        if value.startswith('"') or value.endswith('"'):
            if len(value) < 2 or not (value.startswith('"') and value.endswith('"')):
                raise ValueError(f"Malformed MTL string on line {line_number}")
            value = value[1:-1]
        if key == "GROUP":
            if not stack:
                if value != "LANDSAT_METADATA_FILE" or root:
                    raise ValueError("Expected a single LANDSAT_METADATA_FILE MTL root")
                group = root
            else:
                parent = stack[-1][1]
                if value in parent:
                    raise ValueError(f"Duplicate MTL group {value}")
                group = {}
                parent[value] = group
            stack.append((value, group))
        elif key == "END_GROUP":
            if not stack or stack[-1][0] != value:
                raise ValueError(f"Mismatched MTL END_GROUP {value}")
            stack.pop()
        else:
            if not stack:
                raise ValueError("MTL field outside a group")
            group = stack[-1][1]
            if key in group:
                raise ValueError(f"Duplicate MTL field {key}")
            group[key] = value
    if not finished:
        raise ValueError("Missing Landsat MTL END")
    return root


def _unique_json_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for key, value in pairs:
        if key in fields:
            raise ValueError(f"Duplicate JSON MTL field {key}")
        fields[key] = value
    return fields


def _mtl_group(groups: dict[str, Any], name: str) -> dict[str, Any]:
    group = groups.get(name)
    if not isinstance(group, dict):
        raise ValueError(f"Missing trusted MTL group {name}")
    return group


def _landsat_metadata(
    groups: dict[str, Any], path: Path, settings: NDVISettings, role: BandRole
) -> RadiometricCalibration:
    contents = _mtl_group(groups, "PRODUCT_CONTENTS")
    attributes = _mtl_group(groups, "IMAGE_ATTRIBUTES")
    level = contents.get("PROCESSING_LEVEL")
    if level not in {"L2SP", "L2SR"}:
        raise ValueError(
            "Only Landsat Level-2 surface reflectance L2SP/L2SR is supported"
        )
    collection = _number(contents.get("COLLECTION_NUMBER"), "COLLECTION_NUMBER")
    if collection != 2:
        raise ValueError("Only Landsat Collection 2 metadata is supported")
    spacecraft = attributes.get("SPACECRAFT_ID")
    sensors = {"LANDSAT_8": "landsat-8", "LANDSAT_9": "landsat-9"}
    if not isinstance(spacecraft, str) or spacecraft not in sensors:
        raise ValueError("Only LANDSAT_8/LANDSAT_9 metadata is supported")
    sensor = sensors[spacecraft]
    if settings.sensor not in {"custom", sensor}:
        raise ValueError("Landsat metadata does not match the configured sensor")
    parameters = _mtl_group(groups, "LEVEL2_SURFACE_REFLECTANCE_PARAMETERS")
    band = _LANDSAT_BANDS[role]
    scale = _number(
        parameters.get(f"REFLECTANCE_MULT_BAND_{band}"), "reflectance scale"
    )
    offset = _number(
        parameters.get(f"REFLECTANCE_ADD_BAND_{band}"), "reflectance offset"
    )
    return RadiometricCalibration(
        scale=scale,
        offset=offset,
        source="landsat_mtl_metadata",
        reflectance_type="surface",
        details={
            "metadata_path": str(path),
            "role": role,
            "sensor": sensor,
            "spacecraft_id": spacecraft,
            "processing_level": level,
            "collection_number": 2,
            "band_number": band,
            "coefficient_group": "LEVEL2_SURFACE_REFLECTANCE_PARAMETERS",
        },
        zero_is_nodata=True,
    )
