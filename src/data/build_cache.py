"""Member 1, W1.9 — raw parquet -> cached windowed .npz on NVMe scratch.
Deterministic given a seed; writes a JSON sidecar recording the config used."""


def build_cache(root: str, out_dir: str, config: dict) -> None:
    # TODO: wire ThreeWDataset (inventory.py) -> WindowBuilder (windowing.py)
    # -> VariableSelector -> .npz cache, one file per instance, matching
    # make_fake_data.py's contract exactly.
    raise NotImplementedError


if __name__ == "__main__":
    build_cache("data/3W/dataset", "data/cache/", config={})
