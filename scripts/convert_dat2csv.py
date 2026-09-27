from __future__ import annotations

import argparse
import csv
from pathlib import Path

from hag_regularized_stacking_boosting_meta.io.loaders import read_dat_rows


DEFAULT_INPUT = r"datasets\raw\DDos\DDos (10000, 80, 2).dat"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert a .dat dataset into the equivalent extended CSV.")
    p.add_argument("input_path", nargs="?", default=DEFAULT_INPUT, help=f"DAT file (default: {DEFAULT_INPUT}).")
    p.add_argument("output_path", nargs="?", default=None, help="CSV file (default: input path with .csv).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path) if args.output_path else input_path.with_suffix(".csv")

    # Same tokenizer as the dat_matrix loader: spaces or tabs, decimal point or comma.
    # Values are copied as written (full precision); header/feature-type rows are kept,
    # so the CSV loads identically via csv_extended.
    rows = read_dat_rows(input_path)
    width = max(len(r) for r in rows)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for r in rows:
            writer.writerow(r + [""] * (width - len(r)))

    print(f"File converted successfully! Saved as {output_path}")


if __name__ == "__main__":
    main()
