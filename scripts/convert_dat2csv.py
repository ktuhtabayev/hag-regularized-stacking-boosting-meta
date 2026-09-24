from __future__ import annotations

import csv
from pathlib import Path

from hag_regularized_stacking_boosting_meta.io.loaders import read_dat_rows


def main() -> None:
    # input_path = r"datasets\raw\Cancer\Cancer (589, 44, 2).dat"
    # output_path = r"datasets\raw\Cancer\Cancer (589, 44, 2).csv"

    input_path = r"datasets\raw\DDos\DDos (10000, 80, 2).dat"
    output_path = r"datasets\raw\DDos\DDos (10000, 80, 2).csv"

    # Same tokenizer as the dat_matrix loader: spaces or tabs, decimal point or comma.
    # Values are copied as written (full precision); header/feature-type rows are kept,
    # so the CSV loads identically via csv_extended.
    rows = read_dat_rows(Path(input_path))
    width = max(len(r) for r in rows)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for r in rows:
            writer.writerow(r + [""] * (width - len(r)))

    print(f"File converted successfully! Saved as {output_path}")


if __name__ == "__main__":
    main()
