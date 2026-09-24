""".dat datasets load exactly like their extended-CSV counterparts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hag_regularized_stacking_boosting_meta.io.loaders import (
    DatasetConfig,
    DatasetLoadError,
    load_dataset_bundle,
)


RAW = Path(__file__).resolve().parents[1] / "datasets" / "raw"


def _load(path: Path, **kwargs) -> object:
    return load_dataset_bundle(DatasetConfig(path=str(path), label_mapping={"1": 1, "2": 2}, **kwargs))


@pytest.mark.parametrize("fmt", [None, "auto", "AUTO"])
def test_default_dat_matches_default_csv(fmt: str | None) -> None:
    dat = _load(RAW / "default.dat", format=fmt)
    csv = _load(RAW / "default.csv", format=fmt)

    assert dat.meta == csv.meta == {"m": 10, "n": 13, "c": 2}
    np.testing.assert_array_equal(dat.X, csv.X)
    np.testing.assert_array_equal(dat.y, csv.y)
    np.testing.assert_array_equal(dat.feature_types, csv.feature_types)
    assert dat.quantitative_idx == csv.quantitative_idx
    assert dat.nominal_idx == csv.nominal_idx


def test_tab_padded_crlf_bom_and_decimal_comma(tmp_path: Path) -> None:
    # Layout seen in exported .dat files: leading/trailing tab padding, empty first line,
    # CRLF endings, a BOM, and decimal commas.
    text = (
        "﻿\t\t\t\t\r\n"
        "\t3\t2\t2\t\t\r\n"
        "\t-0,5\t1\t1\t\r\n"
        "\t2,25\t0\t2\t\r\n"
        "\t7\t1\t1\t\r\n"
        "\t1\t0\t\t\r\n"
    )
    path = tmp_path / "tabs.dat"
    path.write_bytes(text.encode("utf-8"))

    ds = _load(path, format="auto")

    assert ds.meta == {"m": 3, "n": 2, "c": 2}
    np.testing.assert_array_equal(ds.X, [[-0.5, 1.0], [2.25, 0.0], [7.0, 1.0]])
    np.testing.assert_array_equal(ds.y, [1, 2, 1])
    assert ds.quantitative_idx == [0]
    assert ds.nominal_idx == [1]


def test_dat_without_header_and_footer_is_all_quantitative(tmp_path: Path) -> None:
    path = tmp_path / "plain.dat"
    path.write_text("1.5 2 1\n3 4 2\n", encoding="utf-8")

    ds = _load(path, has_metadata_header=False, has_feature_type_row=False)

    assert ds.meta == {}
    np.testing.assert_array_equal(ds.X, [[1.5, 2.0], [3.0, 4.0]])
    np.testing.assert_array_equal(ds.y, [1, 2])
    assert ds.feature_types.tolist() == [1, 1]


def test_dat_row_with_missing_value_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "ragged.dat"
    path.write_text("2 2 2\n1 2 1\n3 2\n1 1\n", encoding="utf-8")

    with pytest.raises(DatasetLoadError, match="Data row 2 has 2 values"):
        _load(path)


def test_explicit_csv_format_on_dat_file_explains_fix() -> None:
    with pytest.raises(DatasetLoadError, match="'auto'"):
        _load(RAW / "default.dat", format="csv_extended")
