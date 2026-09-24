from hag_regularized_stacking_boosting_meta.io.loaders import (
    DatasetConfig,
    load_dataset_bundle,
    available_formats,
)


def main() -> None:
    print("Available formats:", available_formats())

    # ============================================================
    # DEFAULT PATH (keep this as the main default input)
    # ============================================================
    default_path = "datasets/raw/default.csv"
    # default_path = "datasets/raw/default.dat"

    # ============================================================
    # NON-DEFAULT PATH EXAMPLE (comment out default_path above,
    #    then uncomment this when you want to use it)
    #
    # NOTE: Spaces in the filename are OK in Python strings.
    # ============================================================
    # default_path = (
    #     "datasets/raw/Heart-Disease/Heart Disease (270, 13, 2).csv"
    # )

    cfg = DatasetConfig(
        path=default_path,
        format=None,                   # auto-detect; CSV default behavior
        delimiter=",",
        has_metadata_header=True,       # for your extended CSV
        has_feature_type_row=True,      # last row: 0/1 types
        label_col=-1,                   # label in last column
        label_mapping={"1": 1, "2": 2}, # keep if labels are 1/2; remove if already numeric correct
    )

    ds = load_dataset_bundle(cfg)

    print("Meta:", ds.meta)
    print("X shape:", ds.X.shape)
    print("y shape:", ds.y.shape)
    print("Classes:", sorted(set(ds.y.tolist())))
    print("Feature types:", ds.feature_types.tolist())
    print("Quantitative idx:", ds.quantitative_idx)
    print("Nominal idx:", ds.nominal_idx)

    # optional split (safe even if one side is empty)
    Xq = ds.X[:, ds.quantitative_idx] if ds.quantitative_idx else ds.X[:, :0]
    Xn = ds.X[:, ds.nominal_idx] if ds.nominal_idx else ds.X[:, :0]
    print("Xq shape:", Xq.shape, "| Xn shape:", Xn.shape)


if __name__ == "__main__":
    main()
