from hag_regularized_stacking_boosting_meta.io.configs import load_default_config
from hag_regularized_stacking_boosting_meta.io.loaders import load_dataset_bundle


def main():
    cfg = load_default_config("configs/default.yaml")
    ds = load_dataset_bundle(cfg.dataset)
    print("Loaded:", ds.X.shape, ds.y.shape, ds.meta)
    print("HAG defaults:", cfg.hag)


if __name__ == "__main__":
    main()
