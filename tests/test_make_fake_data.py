from pathlib import Path

import numpy as np

from src.data.make_fake_data import make_fake_dataset


def test_make_fake_dataset_matches_cache_contract(tmp_path):
    out = tmp_path / "fake"

    make_fake_dataset(
        out_dir=out,
        n_wells=4,
        n_instances=6,
        n_channels=5,
        window_size=60,
        windows_per_instance=10,
        seed=42,
    )

    files = sorted(out.glob("*.npz"))
    assert len(files) == 6

    required_keys = {
        "X",
        "mask",
        "y",
        "group",
        "inst_id",
        "t_end",
        "is_sim",
        "failure_time",
        "blockage_time",
        "normal_hours",
    }

    for path in files:
        with np.load(path) as z:
            assert required_keys.issubset(z.files)

            n = len(z["y"])

            assert z["X"].shape == (n, 5, 60)
            assert z["mask"].shape == (n, 5, 60)
            assert z["group"].shape == (n,)
            assert z["inst_id"].shape == (n,)
            assert z["t_end"].shape == (n,)
            assert z["is_sim"].shape == (n,)

            assert z["X"].dtype == np.float32
            assert z["mask"].dtype == np.uint8
            assert z["y"].dtype == np.int64
            assert z["group"].dtype == np.int64
            assert z["inst_id"].dtype == np.int64
            assert z["t_end"].dtype == np.float64
            assert z["is_sim"].dtype == np.uint8
def test_fake_event_times_follow_current_semantics(tmp_path):
    out = tmp_path / "fake"

    make_fake_dataset(
        out_dir=out,
        n_wells=4,
        n_instances=20,
        n_channels=5,
        window_size=60,
        windows_per_instance=20,
        seed=42,
    )

    found_event = False

    for path in sorted(out.glob("*.npz")):
        with np.load(path) as z:
            y = z["y"]

            if np.any(y == 1):
                found_event = True

                transient_idx = np.flatnonzero(y == 1)[0]
                established_idx = np.flatnonzero(y == 2)[0]

                assert float(z["failure_time"]) == transient_idx * 60
                assert float(z["blockage_time"]) == established_idx * 60

    assert found_event