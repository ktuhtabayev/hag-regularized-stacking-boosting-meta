from __future__ import annotations

import pandas as pd


def main() -> None:
    # input_path = r"datasets\raw\Cancer\Cancer (589, 44, 2).dat"
    # output_path = r"datasets\raw\Cancer\Cancer (589, 44, 2).csv"

    input_path = r"datasets\raw\DDos\DDos (10000, 80, 2).dat"
    output_path = r"datasets\raw\DDos\DDos (10000, 80, 2).csv"

    # df = pd.read_csv(input_path, delimiter="\t", decimal=",")
    df = pd.read_csv(input_path, delimiter="\t", decimal=".")

    df.to_csv(output_path, index=False)

    print(f"File converted successfully! Saved as {output_path}")


if __name__ == "__main__":
    main()