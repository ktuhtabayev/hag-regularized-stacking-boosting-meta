"""Stage-script plumbing (cli.py) and the scripts themselves, run as a user would."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from hag_regularized_stacking_boosting_meta.cli import (
    hag_run_config_payload,
    latest_run_dir,
    parse_stage_args,
    to_jsonable,
    write_dij_csv,
)
from hag_regularized_stacking_boosting_meta.io.configs import load_default_config
from hag_regularized_stacking_boosting_meta.services.report_builder import (
    _as_number,
    _as_number_column,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "default.yaml"


def test_every_stage_accepts_a_config_path() -> None:
    args = parse_stage_args("demo", argv=["--config", "other.yaml"])
    assert args.config == "other.yaml"
    assert parse_stage_args("demo", argv=[]).config == "configs/default.yaml"


def test_latest_run_is_the_newest_second_then_the_newest_folder(tmp_path: Path) -> None:
    older = tmp_path / "20260101_120000_ffffffff"
    same_second_first = tmp_path / "20260101_120001_ffffffff"
    same_second_last = tmp_path / "20260101_120001_00000000"
    for i, run in enumerate((older, same_second_first, same_second_last)):
        run.mkdir()
        os.utime(run, ns=(10**18 + i * 10**9, 10**18 + i * 10**9))

    # the random suffix must not decide between runs of the same second
    assert latest_run_dir(tmp_path) == same_second_last
    assert latest_run_dir(tmp_path, lambda p: p != same_second_last) == same_second_first
    assert latest_run_dir(tmp_path / "missing") is None
    assert latest_run_dir(tmp_path, lambda p: False) is None


def test_hag_run_config_records_every_hag_parameter() -> None:
    payload = hag_run_config_payload(load_default_config(CONFIG))
    assert payload == {
        "seed": 42,
        "hag": {
            "alpha": 0.3,
            "delta": 0.1,
            "kappa": 15,
            "cr1": 10.0,
            "k1_label": 1,
            "k2_label": 2,
            "majorizing": {"name": "sigmoid", "params": {"k": 1.0, "x0": 0.0}},
        },
    }


def test_dij_csv_without_latent_features_is_a_header(tmp_path: Path) -> None:
    write_dij_csv(tmp_path / "empty.csv", np.zeros((0, 0)))
    assert (tmp_path / "empty.csv").read_text(encoding="utf-8") == "r1\n"

    write_dij_csv(tmp_path / "dij.csv", np.array([[0.5, -1.0], [2.0, 0.25]]))
    lines = (tmp_path / "dij.csv").read_text(encoding="utf-8").splitlines()
    assert lines == ["r1,r2", "0.500000000,-1.000000000", "2.000000000,0.250000000"]


def test_to_jsonable_converts_keys_and_numpy_values() -> None:
    data = {1.0: np.float64(0.5), 2: [np.int64(3)], "a": np.array([1, 2])}
    assert json.dumps(to_jsonable(data)) == '{"1.0": 0.5, "2": [3], "a": [1, 2]}'


@pytest.mark.parametrize(
    "values",
    [
        [1.0, 2.0, 3.0],
        [0.5, 1.25, -2.75],
        [1.0, 2.5, 3.0000000001, -0.0, 7.0],
        [-0.0, 0.0, 4.0],
    ],
)
def test_number_columns_match_the_per_value_conversion(values) -> None:
    expected = np.asarray([_as_number(v) for v in values])
    column = _as_number_column(np.asarray(values))
    assert column.dtype == expected.dtype
    assert column.tobytes() == expected.tobytes()


# ------------------------------------------------------------------
# Scripts end to end (fresh processes, as from the command line)
# ------------------------------------------------------------------

def _config_writing_to(tmp_path: Path) -> Path:
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw["output"] = {"root_dir": str(tmp_path / "out"), "runs_dir": "runs"}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return path


def _run_script(name: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / name), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "cp1252"},  # a legacy Windows code page
    )


def test_stage_scripts_follow_the_config_and_seed(tmp_path: Path) -> None:
    config = str(_config_writing_to(tmp_path))
    runs = tmp_path / "out" / "runs"

    weights = _run_script("run_quantitative_weights.py", "--config", config)
    assert weights.returncode == 0, weights.stderr
    table = json.loads(
        (latest_run_dir(runs / "quantitative_weights") / "criterion1_table.json").read_text(encoding="utf-8")
    )
    assert table["dataset"]["name"] == "default"
    assert table["hag_defaults"]["majorizing_function"] == "sigmoid"

    train = _run_script("run_hag_training.py", "--config", config)
    assert train.returncode == 0, train.stderr
    train_run = latest_run_dir(runs / "train")
    assert json.loads((train_run / "tuplam.json").read_text(encoding="utf-8"))["tuplam"][:2] == [2, 5]

    snews = []
    for _ in range(2):
        proc = _run_script("run_meta_new_object.py", "--config", config, "--seed", "7")
        assert proc.returncode == 0, proc.stderr
        assert f"auto-reused latest train run: {train_run}" in proc.stdout
        snews.append(json.loads((latest_run_dir(runs / "meta_new_object") / "snew.json").read_text(encoding="utf-8")))
    assert snews[0]["snew_init"] == snews[1]["snew_init"]

    # Unicode output (✅) must not crash on a legacy console code page
    margins = _run_script("run_margin_analysis.py", "--config", config)
    assert margins.returncode == 0, margins.stderr
    assert "Margin analysis completed" in margins.stdout
    assert (latest_run_dir(runs / "margin_analysis") / "latent_r1_object_margins.csv").is_file()
