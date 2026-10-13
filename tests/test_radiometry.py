"""Synthetic radiometric metadata tests: no sensor-based coefficient guesses."""

import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pytest
from numpy.typing import NDArray

from satellite_analyzer.core.exceptions import SatelliteAnalyzerError
from satellite_analyzer.core.ndvi_config import NDVISettings
from satellite_analyzer.io.radiometry import (
    RadiometricCalibration,
    resolve_calibration,
)


def _resolve(
    settings: NDVISettings | None = None,
    *,
    role: Literal["red", "nir"] = "red",
    values: NDArray[Any] | None = None,
    tiff_scale: float = 1.0,
    tiff_offset: float = 0.0,
) -> RadiometricCalibration:
    samples = np.array([1000, 2500], dtype=np.uint16) if values is None else values
    return resolve_calibration(
        settings=settings or NDVISettings(),
        role=role,
        dtype=samples.dtype,
        valid_values=samples,
        tiff_scale=tiff_scale,
        tiff_offset=tiff_offset,
    )


def _sentinel_xml(
    tmp_path: Path,
    *,
    level: str = "Level-2A",
    baseline: str | None = "04.00",
    offsets: tuple[str, str] | None = ("-1000", "-900"),
    quantification: str = "10000",
) -> Path:
    quantification_tag = (
        "BOA_QUANTIFICATION_VALUE" if level == "Level-2A" else "QUANTIFICATION_VALUE"
    )
    offset_tag = "BOA_ADD_OFFSET" if level == "Level-2A" else "RADIO_ADD_OFFSET"
    baseline_field = (
        f"<PROCESSING_BASELINE>{baseline}</PROCESSING_BASELINE>"
        if baseline is not None
        else ""
    )
    offset_fields = (
        f'<{offset_tag} band_id="3">{offsets[0]}</{offset_tag}>'
        f'<{offset_tag} band_id="7">{offsets[1]}</{offset_tag}>'
        if offsets is not None
        else ""
    )
    path = tmp_path / "MTD_MSIL.xml"
    path.write_text(
        f'<n1:{level}_User_Product xmlns:n1="urn:sentinel:synthetic">'
        "<General_Info><Product_Info>"
        f"<PROCESSING_LEVEL>{level}</PROCESSING_LEVEL>{baseline_field}"
        "</Product_Info><Product_Image_Characteristics>"
        f"<{quantification_tag}>{quantification}</{quantification_tag}>"
        f"{offset_fields}</Product_Image_Characteristics></General_Info>"
        f"</n1:{level}_User_Product>",
        encoding="utf-8",
    )
    return path


def _landsat_mtl(
    tmp_path: Path,
    *,
    level: str = "L2SP",
    collection: str = "02",
    spacecraft: str = "LANDSAT_8",
) -> Path:
    path = tmp_path / "scene_MTL.txt"
    path.write_text(
        "GROUP = LANDSAT_METADATA_FILE\n"
        "  GROUP = PRODUCT_CONTENTS\n"
        f'    PROCESSING_LEVEL = "{level}"\n'
        f"    COLLECTION_NUMBER = {collection}\n"
        "  END_GROUP = PRODUCT_CONTENTS\n"
        "  GROUP = IMAGE_ATTRIBUTES\n"
        f'    SPACECRAFT_ID = "{spacecraft}"\n'
        "  END_GROUP = IMAGE_ATTRIBUTES\n"
        "  GROUP = LEVEL1_RADIOMETRIC_RESCALING\n"
        "    REFLECTANCE_MULT_BAND_4 = 0.00002\n"
        "    REFLECTANCE_ADD_BAND_4 = -0.1\n"
        "  END_GROUP = LEVEL1_RADIOMETRIC_RESCALING\n"
        "  GROUP = LEVEL2_SURFACE_REFLECTANCE_PARAMETERS\n"
        "    REFLECTANCE_MULT_BAND_4 = 0.0000275\n"
        "    REFLECTANCE_ADD_BAND_4 = -0.2\n"
        "    REFLECTANCE_MULT_BAND_5 = 0.000028\n"
        "    REFLECTANCE_ADD_BAND_5 = -0.21\n"
        "  END_GROUP = LEVEL2_SURFACE_REFLECTANCE_PARAMETERS\n"
        "END_GROUP = LANDSAT_METADATA_FILE\nEND\n",
        encoding="utf-8",
    )
    return path


def test_explicit_coefficients_override_tiff_without_double_calibration() -> None:
    settings = NDVISettings(
        sensor="sentinel-2",
        input_units="dn",
        red_scale=0.0001,
        red_offset=-0.1,
        nir_scale=0.0002,
        nir_offset=-0.2,
    )
    red = _resolve(settings, tiff_scale=0.00001, tiff_offset=-0.3)
    nir = _resolve(settings, role="nir")
    assert (red.scale, red.offset) == (0.0001, -0.1)
    assert (nir.scale, nir.offset) == (0.0002, -0.2)
    assert red.source == "explicit_coefficients"
    assert red.zero_is_nodata
    assert red.details["overridden_tiff_calibration"] == {
        "scale": 0.00001,
        "offset": -0.3,
        "applied": False,
    }


def test_tiff_calibration_is_used_without_sensor_guess() -> None:
    result = _resolve(tiff_scale=0.00005, tiff_offset=-0.15)
    assert result.source == "tiff_band_metadata"
    assert (result.scale, result.offset) == (0.00005, -0.15)
    assert result.reflectance_type == "unspecified"


@pytest.mark.parametrize("sensor", ["custom", "sentinel-2", "landsat-8", "landsat-9"])
def test_sensor_name_never_implies_integer_scale(sensor: str) -> None:
    settings = NDVISettings.model_validate({"sensor": sensor})
    with pytest.raises(SatelliteAnalyzerError, match="Unknown radiometric scale"):
        _resolve(settings)


@pytest.mark.parametrize("values", [[0.0, 1.0], [0.0], [0.9, 0.3, 0.1]])
def test_auto_bounded_float_is_identity(values: list[float]) -> None:
    samples = np.array(values, dtype=np.float32)
    original = samples.copy()
    result = _resolve(values=samples)
    assert (result.scale, result.offset) == (1.0, 0.0)
    assert result.source == "inferred_float_reflectance"
    assert result.details["valid_source_min"] == pytest.approx(min(values))
    assert result.details["valid_source_max"] == pytest.approx(max(values))
    np.testing.assert_array_equal(samples, original)


@pytest.mark.parametrize("values", [[-0.01, 0.5], [0.5, 1.01], [1000.0], [np.nan]])
def test_auto_float_outside_reflectance_range_requires_calibration(
    values: list[float],
) -> None:
    with pytest.raises(SatelliteAnalyzerError, match="Unknown radiometric scale"):
        _resolve(values=np.array(values, dtype=np.float32))


def test_dn_does_not_infer_float_identity() -> None:
    with pytest.raises(SatelliteAnalyzerError, match="Unknown radiometric scale"):
        _resolve(NDVISettings(input_units="dn"), values=np.array([0.1, 0.2]))


def test_empty_float_preserves_all_invalid_handling() -> None:
    result = _resolve(values=np.array([], dtype=np.float32))
    assert result.details["inference"] == "empty_float_source_no_valid_pixels"
    assert (result.scale, result.offset) == (1.0, 0.0)
    with pytest.raises(SatelliteAnalyzerError, match="Unknown radiometric scale"):
        _resolve(values=np.array([], dtype=np.uint16))


def test_declared_reflectance_identity_can_exceed_one() -> None:
    result = _resolve(
        NDVISettings(sensor="sentinel-2", input_units="reflectance"),
        values=np.array([0.0, 1.5], dtype=np.float32),
    )
    assert result.source == "declared_reflectance"
    assert (result.scale, result.offset) == (1.0, 0.0)
    assert not result.zero_is_nodata  # Valid reflectance zero is not Sentinel DN zero.


@pytest.mark.parametrize("scale,offset", [(0.0001, 0.0), (1.0, -0.1)])
def test_declared_reflectance_rejects_tiff_rescaling(
    scale: float, offset: float
) -> None:
    with pytest.raises(SatelliteAnalyzerError, match="conflicts with nonidentity TIFF"):
        _resolve(
            NDVISettings(input_units="reflectance"),
            tiff_scale=scale,
            tiff_offset=offset,
        )


@pytest.mark.parametrize(
    "scale,offset", [(0.0, 0.0), (-1.0, 0.0), (np.nan, 0.0), (1.0, np.inf)]
)
def test_invalid_tiff_calibration_fails_explicitly(scale: float, offset: float) -> None:
    with pytest.raises(SatelliteAnalyzerError, match="positive and finite"):
        _resolve(tiff_scale=scale, tiff_offset=offset)


def test_calibration_is_frozen_and_json_safe() -> None:
    result = _resolve(NDVISettings(input_units="reflectance"))
    json.dumps(result.details, allow_nan=False)
    with pytest.raises(FrozenInstanceError):
        result.scale = 2.0  # type: ignore[misc]
    with pytest.raises(SatelliteAnalyzerError, match="finite JSON"):
        RadiometricCalibration(1.0, 0.0, "test", "surface", {"bad": np.inf})


@pytest.mark.parametrize("level,type_", [("Level-2A", "surface"), ("Level-1C", "toa")])
def test_sentinel_metadata_offsets_and_band_indices(
    tmp_path: Path, level: str, type_: str
) -> None:
    path = _sentinel_xml(tmp_path, level=level)
    settings = NDVISettings(sensor="sentinel-2", metadata_path=path)
    red = _resolve(settings, tiff_scale=0.5, tiff_offset=2.0)
    nir = _resolve(settings, role="nir")
    assert red.scale == pytest.approx(0.0001)
    assert red.offset == pytest.approx(-0.1)
    assert nir.offset == pytest.approx(-0.09)
    assert red.details["band_id"] == 3
    assert nir.details["band_id"] == 7
    assert red.reflectance_type == type_
    assert red.zero_is_nodata
    assert red.details["overridden_tiff_calibration"]["applied"] is False
    assert 3000 * red.scale + red.offset == pytest.approx(0.2)
    json.dumps(red.details, allow_nan=False)


@pytest.mark.parametrize("level", ["Level-2A", "Level-1C"])
def test_sentinel_pre_four_baseline_documents_zero_offset(
    tmp_path: Path, level: str
) -> None:
    path = _sentinel_xml(tmp_path, level=level, baseline="03.01", offsets=None)
    result = _resolve(NDVISettings(metadata_path=path))
    assert result.offset == 0.0
    assert result.zero_is_nodata
    assert result.details["zero_offset_from_pre_04_baseline"] is True


@pytest.mark.parametrize("baseline", [None, "04.00", "05.11"])
def test_missing_sentinel_offsets_need_known_pre_four_baseline(
    tmp_path: Path, baseline: str | None
) -> None:
    path = _sentinel_xml(tmp_path, baseline=baseline, offsets=None)
    with pytest.raises(SatelliteAnalyzerError, match="documented pre-04.00"):
        _resolve(NDVISettings(metadata_path=path))


@pytest.mark.parametrize("quantification", ["0", "-1", "nan", "inf", "bad"])
def test_sentinel_invalid_quantification_is_rejected(
    tmp_path: Path, quantification: str
) -> None:
    path = _sentinel_xml(tmp_path, quantification=quantification)
    with pytest.raises(SatelliteAnalyzerError):
        _resolve(NDVISettings(metadata_path=path))


@pytest.mark.parametrize(
    "old,new,message",
    [
        ('band_id="7"', 'band_id="8"', "Missing BOA_ADD_OFFSET"),
        ("<PROCESSING_LEVEL>Level-2A", "<PROCESSING_LEVEL>Level-1C", "disagree"),
        ("-1000", "nan", "Nonfinite"),
        ("04.00", "nan", "Nonfinite"),
        ("BOA_QUANTIFICATION_VALUE", "QUANTIFICATION_VALUE", "BOA_QUANTIFICATION"),
    ],
)
def test_sentinel_inconsistent_metadata_fails(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    path = _sentinel_xml(tmp_path)
    path.write_text(
        path.read_text(encoding="utf-8").replace(old, new), encoding="utf-8"
    )
    with pytest.raises(SatelliteAnalyzerError, match=message):
        _resolve(NDVISettings(metadata_path=path), role="nir" if "7" in old else "red")


def test_sentinel_sensor_mismatch_is_rejected(tmp_path: Path) -> None:
    path = _sentinel_xml(tmp_path)
    with pytest.raises(SatelliteAnalyzerError, match="does not match"):
        _resolve(NDVISettings(sensor="landsat-8", metadata_path=path))


@pytest.mark.parametrize("spacecraft", ["LANDSAT_8", "LANDSAT_9"])
@pytest.mark.parametrize("level", ["L2SP", "L2SR"])
def test_landsat_uses_l2_band_four_five_not_toa_parameters(
    tmp_path: Path, spacecraft: str, level: str
) -> None:
    path = _landsat_mtl(tmp_path, spacecraft=spacecraft, level=level)
    settings = NDVISettings(metadata_path=path)
    red = _resolve(settings)
    nir = _resolve(settings, role="nir")
    assert (red.scale, red.offset) == (0.0000275, -0.2)
    assert (nir.scale, nir.offset) == (0.000028, -0.21)
    assert red.details["band_number"] == 4
    assert nir.details["band_number"] == 5
    assert red.reflectance_type == "surface"
    assert red.zero_is_nodata
    assert 20000 * red.scale + red.offset == pytest.approx(0.35)
    json.dumps(red.details, allow_nan=False)


@pytest.mark.parametrize(
    "updates,message",
    [
        ({"level": "L1TP"}, "Level-2"),
        ({"collection": "01"}, "Collection 2"),
        ({"spacecraft": "LANDSAT_7"}, "LANDSAT_8/LANDSAT_9"),
    ],
)
def test_landsat_unsupported_products_are_rejected(
    tmp_path: Path, updates: dict[str, str], message: str
) -> None:
    path = _landsat_mtl(tmp_path, **updates)
    with pytest.raises(SatelliteAnalyzerError, match=message):
        _resolve(NDVISettings(metadata_path=path))


@pytest.mark.parametrize("sensor", ["landsat-9", "sentinel-2"])
def test_landsat_sensor_mismatch_is_rejected(tmp_path: Path, sensor: str) -> None:
    path = _landsat_mtl(tmp_path)
    settings = NDVISettings.model_validate({"sensor": sensor, "metadata_path": path})
    with pytest.raises(SatelliteAnalyzerError, match="does not match"):
        _resolve(settings)


@pytest.mark.parametrize(
    "old,new,message",
    [
        ("0.0000275", "-0.0000275", "positive and finite"),
        ("-0.2", "nan", "Nonfinite"),
        ("LEVEL2_SURFACE_REFLECTANCE_PARAMETERS", "OTHER_PARAMETERS", "trusted MTL"),
        ("END_GROUP = IMAGE_ATTRIBUTES", "END_GROUP = WRONG_GROUP", "Mismatched"),
    ],
)
def test_landsat_inconsistent_metadata_fails(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    path = _landsat_mtl(tmp_path)
    path.write_text(
        path.read_text(encoding="utf-8").replace(old, new), encoding="utf-8"
    )
    with pytest.raises(SatelliteAnalyzerError, match=message):
        _resolve(NDVISettings(metadata_path=path))


def test_landsat_json_mtl_is_supported(tmp_path: Path) -> None:
    path = tmp_path / "scene_MTL.json"
    path.write_text(
        json.dumps(
            {
                "LANDSAT_METADATA_FILE": {
                    "PRODUCT_CONTENTS": {
                        "PROCESSING_LEVEL": "L2SR",
                        "COLLECTION_NUMBER": 2,
                    },
                    "IMAGE_ATTRIBUTES": {"SPACECRAFT_ID": "LANDSAT_9"},
                    "LEVEL2_SURFACE_REFLECTANCE_PARAMETERS": {
                        "REFLECTANCE_MULT_BAND_4": 0.0000275,
                        "REFLECTANCE_ADD_BAND_4": -0.2,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    result = _resolve(NDVISettings(sensor="landsat-9", metadata_path=path))
    assert result.scale == 0.0000275
    assert result.offset == -0.2


@pytest.mark.parametrize(
    "text",
    [
        "",
        "not metadata",
        "<broken>",
        '<!DOCTYPE root [<!ENTITY x "test">]><root>&x;</root>',
        "{}",
        "GROUP = LANDSAT_METADATA_FILE\nEND",
    ],
)
def test_corrupted_metadata_is_a_domain_error(tmp_path: Path, text: str) -> None:
    path = tmp_path / "metadata.txt"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(SatelliteAnalyzerError, match="metadata"):
        _resolve(NDVISettings(metadata_path=path))


def test_missing_metadata_does_not_fall_back_to_tiff(tmp_path: Path) -> None:
    settings = NDVISettings(metadata_path=tmp_path / "missing.xml")
    with pytest.raises(SatelliteAnalyzerError, match="not a regular file"):
        _resolve(settings, tiff_scale=0.0001)
