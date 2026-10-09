"""
synthetic_labels.py
───────────────────
Generate fake CollectedData_<scorer>.h5 / .csv for an extracted frame set, so
the rest of the pipeline (add_labeled_data -> create_train_dataset ->
train_network -> inference -> report) can be exercised in tests WITHOUT
labeling anything by hand in napari.

The output follows exactly the structure napari-deeplabcut writes (verified
against a real CollectedData_Tamas.h5 / .csv):

    index   : 3-level MultiIndex, unnamed levels
              ("labeled-data", <video_stem>, "<imgNNNN>.png"), sorted by file name
    columns : 3-level MultiIndex named [scorer, bodyparts, coords]
              (<scorer>, <bodypart>, "x" | "y"), bodyparts in config order,
              x then y for each
    values  : float64, pixel coordinates; a bodypart that was not labeled in a
              frame has BOTH x and y NaN (never just one)
    h5      : key "df_with_missing", pandas default (fixed) format, mode="w"
    csv     : plain df.to_csv(), sitting next to the h5
    file    : <frame_set>/labeled-data/<video_stem>/CollectedData_<scorer>.{h5,csv}

Scorer and bodyparts come from the frame set's own config.yaml (i.e. call
init_labeling_config() first, exactly as you would before real labeling).

Only needs numpy / pandas / pyyaml / pytables — no deeplabcut import.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

H5_KEY = "df_with_missing"


def _frame_set_dirs(store_path, folder_id):
    project_dir = Path(store_path).resolve() / "labeled-data" / folder_id
    config_path = project_dir / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"No extraction project at {project_dir} — call "
            f"extract_frames_for_video() for this video/config first."
        )
    nested_root = project_dir / "labeled-data"
    video_dirs = [p for p in nested_root.iterdir() if p.is_dir()] if nested_root.is_dir() else []
    if len(video_dirs) != 1:
        raise RuntimeError(
            f"Expected exactly one labeled-data/<video_stem> folder under "
            f"{project_dir}, found {[p.name for p in video_dirs]}"
        )
    return config_path, video_dirs[0]


def _image_size(frames, cfg, override):
    """(width, height): explicit override > first extracted PNG > the crop
    entry in config.yaml's video_sets (same '0, W, 0, H' DLC stores)."""
    if override is not None:
        return int(override[0]), int(override[1])
    try:
        from PIL import Image

        with Image.open(frames[0]) as im:
            return im.size
    except Exception:
        pass
    for entry in (cfg.get("video_sets") or {}).values():
        crop = (entry or {}).get("crop")
        if crop:
            x1, x2, y1, y2 = (float(v) for v in str(crop).split(","))
            return int(x2 - x1), int(y2 - y1)
    raise RuntimeError(
        "Could not determine image size from the PNGs or config.yaml — "
        "pass image_size=(width, height)."
    )


def _labeled_count_range(n_labeled, n_bodyparts):
    if n_labeled is None:
        return n_bodyparts, n_bodyparts
    lo, hi = (n_labeled, n_labeled) if isinstance(n_labeled, int) else n_labeled
    lo, hi = max(1, int(lo)), min(n_bodyparts, int(hi))
    if lo > hi:
        raise ValueError(f"n_labeled={n_labeled!r} is impossible with {n_bodyparts} bodyparts.")
    return lo, hi


def generate_synthetic_labels(
    store_path,
    folder_id,
    seed=0,
    n_labeled="auto",
    image_size=None,
    overwrite=False,
):
    """Write synthetic CollectedData_<scorer>.h5 + .csv for one frame set.

    store_path / folder_id: same as every other label_store function (e.g.
        prj.store, "HDMI-A__default"). Every *.png in the frame set gets a row.

    seed: makes the output reproducible.

    n_labeled: how many bodyparts are labeled per frame.
        "auto" (default) -> random 30–70 % of the bodyparts per frame, like the
                            real data (3–7 of 10 there), so NaN handling is
                            exercised;
        int / (lo, hi)   -> exact count / random count in that range;
        None             -> every bodypart in every frame (no NaNs).

    image_size: (width, height); detected automatically when omitted.

    overwrite: if CollectedData_<scorer>.h5 already exists, raise
        FileExistsError unless True (same convention as the rest of the store).

    Returns the written h5 path.
    """
    config_path, frames_dir = _frame_set_dirs(store_path, folder_id)
    cfg = yaml.safe_load(config_path.read_text()) or {}

    scorer = cfg.get("scorer")
    bodyparts = list(cfg.get("bodyparts") or [])
    if not scorer or not bodyparts:
        raise ValueError(
            f"{config_path} has no scorer/bodyparts — call "
            f"init_labeling_config('{folder_id}', ...) first."
        )

    frames = sorted(frames_dir.glob("*.png"))
    if not frames:
        raise FileNotFoundError(f"No *.png frames in {frames_dir}")

    h5_path = frames_dir / f"CollectedData_{scorer}.h5"
    csv_path = frames_dir / f"CollectedData_{scorer}.csv"
    if h5_path.exists() and not overwrite:
        raise FileExistsError(f"{h5_path} already exists. Pass overwrite=True to replace it.")

    width, height = _image_size(frames, cfg, image_size)
    rng = np.random.default_rng(seed)
    n_frames, n_bp = len(frames), len(bodyparts)

    # Fixed "anatomy" (bodypart offsets from a body centre) + per-frame centre
    # movement + small jitter -> consistent, plausible-looking labels.
    spread = 0.08 * min(width, height)
    offsets = rng.normal(0.0, spread, size=(n_bp, 2))
    centres = rng.uniform([0.25 * width, 0.25 * height], [0.75 * width, 0.75 * height], size=(n_frames, 2))
    jitter = rng.normal(0.0, 0.01 * min(width, height), size=(n_frames, n_bp, 2))
    xy = centres[:, None, :] + offsets[None, :, :] + jitter
    xy[..., 0] = np.clip(xy[..., 0], 0, width)
    xy[..., 1] = np.clip(xy[..., 1], 0, height)

    # NaN out whole bodyparts (x AND y together)
    if n_labeled == "auto":
        lo, hi = _labeled_count_range((round(0.3 * n_bp), round(0.7 * n_bp)), n_bp)
    else:
        lo, hi = _labeled_count_range(n_labeled, n_bp)
    for i in range(n_frames):
        k = int(rng.integers(lo, hi + 1))
        unlabeled = rng.permutation(n_bp)[k:]
        xy[i, unlabeled, :] = np.nan

    columns = pd.MultiIndex.from_product(
        [[scorer], bodyparts, ["x", "y"]], names=["scorer", "bodyparts", "coords"]
    )
    index = pd.MultiIndex.from_tuples([("labeled-data", frames_dir.name, f.name) for f in frames])
    df = pd.DataFrame(xy.reshape(n_frames, n_bp * 2), index=index, columns=columns, dtype="float64")

    df.to_hdf(h5_path, key=H5_KEY, mode="w")
    df.to_csv(csv_path)
    print(f"🧪 Synthetic labels ({n_frames} frames, {n_bp} bodyparts) → {h5_path}")
    return h5_path
