"""
label_store.py
───────────────
A persistent "labeled-data store" that extract_frames / label_frames (napari)
can operate on directly, WITHOUT going through deeplabcut.create_new_project().

create_new_project() is just: mkdir a few folders, build a config dict from
auxiliaryfunctions.create_config_template(), override some keys, write it.
No hidden registration happens anywhere else. So we do that ourselves, once,
pointed at whatever folder fits our layout — no forced timestamped project
name, no mandatory first video.

Store layout on disk:
    <store_path>/
        config.yaml         ← same shape DLC itself would generate
        labeled-data/
            <video_stem>/   ← PNG frames + CollectedData_<scorer>.h5, per video

Videos are NEVER copied or symlinked into the store — video_sets just points
at wherever the raw video already lives (same behavior your existing
3a_register_and_extract.py already relies on via add_new_videos(copy_videos=False)).

Extraction and labeling are deliberately decoupled. deeplabcut.extract_frames()
(automatic mode) only ever reads video_sets / numframes2pick / start / stop
from config.yaml — it never touches scorer, bodyparts, or skeleton. Those only
start to matter for label_frames()/napari, which writes
CollectedData_<scorer>.h5 keyed by bodyparts. So init_store() sets up nothing
more than what extraction needs; call add_labeling_schema() separately,
whenever you actually know your bodyparts/skeleton and are ready to label.

Usage
─────
    from label_store import init_store, add_video, extract_frames_for_video

    config_path = init_store(
        store_path="/data/DLC_PINK_CORNERED_labeling/label_store",
        numframes2pick=50,
    )

    add_video(config_path, "/raw_store/HDMI-A/data/event_042/cam1_HDMI-A.mp4")
    extract_frames_for_video(config_path, "/raw_store/HDMI-A/data/event_042/cam1_HDMI-A.mp4")

    # Later, once you're ready to label (not needed for extraction above):
    add_labeling_schema(
        config_path,
        scorer="Egor",
        bodyparts=[
            "right_mitten_wrist", "right_mitten_tip",
            "left_mitten_wrist", "left_mitten_tip",
            "1_corner_table", "2_corner_table", "3_corner_table", "4_corner_table",
        ],
        skeleton=[
            ["right_mitten_wrist", "right_mitten_tip"],
            ["left_mitten_wrist", "left_mitten_tip"],
            ["1_corner_table", "2_corner_table"],
            ["2_corner_table", "3_corner_table"],
            ["3_corner_table", "4_corner_table"],
            ["4_corner_table", "1_corner_table"],
        ],
    )

    # Labeling: your existing 3c_label_video.py already takes a config path +
    # video stem — it should work UNCHANGED against this store's config.yaml,
    # since DLC can't tell a store config apart from a project config.
"""

import matplotlib
matplotlib.use("Agg")  # must come before importing deeplabcut, matches your existing scripts

from datetime import date as _date
from pathlib import Path

import deeplabcut
from deeplabcut.utils import auxiliaryfunctions


def init_store(
    store_path,
    engine="pytorch",
    numframes2pick=20,
    start=0,
    stop=1,
    **overrides,
):
    """Bootstrap a label store for frame extraction: writes config.yaml +
    labeled-data/, without deeplabcut.create_new_project(). Idempotent — if
    config.yaml already exists, returns its path untouched.

    Deliberately does NOT ask for task/scorer/bodyparts/skeleton —
    deeplabcut.extract_frames() (automatic mode) only reads video_sets,
    numframes2pick, start, and stop from the config, so none of that
    labeling-schema stuff needs to exist yet. Call add_labeling_schema()
    later, once you're ready to label, to fill it in.

    Any DLC config key can still be force-set via **overrides, e.g. pcutoff=0.1.
    """
    store_path = Path(store_path).resolve()
    config_path = store_path / "config.yaml"

    if config_path.exists():
        print(f"⏭  Store already initialized: {config_path}")
        return config_path

    (store_path / "labeled-data").mkdir(parents=True, exist_ok=True)

    cfg, _ = auxiliaryfunctions.create_config_template()

    cfg["project_path"] = str(store_path)
    cfg["video_sets"] = {}
    cfg["engine"] = engine
    cfg["numframes2pick"] = numframes2pick
    cfg["start"] = start
    cfg["stop"] = stop
    cfg["date"] = _date.today().strftime("%b%d")

    for key, value in overrides.items():
        cfg[key] = value

    auxiliaryfunctions.write_config(str(config_path), cfg)
    print(f"✅ Store initialized (extraction-only): {config_path}")
    return config_path


def add_labeling_schema(store_config, scorer, bodyparts, skeleton, task=None, **overrides):
    """Fill in what label_frames()/napari actually needs: scorer, bodyparts,
    skeleton, and optionally task. Not required for extract_frames() — only
    call this once you know your bodyparts and are about to start labeling.

    Safe to call any time before labeling starts. If you change bodyparts
    after CollectedData_<scorer>.h5 files already exist for some videos,
    those files will be out of sync with the new schema — only redefine the
    schema before real labeling begins, or you'll need to relabel/rename
    those columns by hand afterward.
    """
    store_config = str(store_config)
    cfg = auxiliaryfunctions.read_config(store_config)

    cfg["scorer"] = scorer
    cfg["bodyparts"] = list(bodyparts)
    cfg["skeleton"] = [list(pair) for pair in skeleton]
    cfg["multianimalproject"] = False
    if task is not None:
        cfg["Task"] = task

    for key, value in overrides.items():
        cfg[key] = value

    auxiliaryfunctions.write_config(store_config, cfg)
    print(f"✅ Labeling schema set: {store_config}")


def _video_crop_entry(video_path):
    """Read width/height straight from the video and build the same
    'crop' entry DLC itself would store in video_sets, without going
    through add_new_videos (which insists on placing a symlink/copy under
    <project_path>/videos, and on Windows that requires Developer Mode or
    an elevated shell just to create the symlink)."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video to read its dimensions: {video_path}")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()

    if width <= 0 or height <= 0:
        raise IOError(f"Got invalid dimensions ({width}x{height}) for video: {video_path}")

    return {"crop": f"0, {width}, 0, {height}"}


def add_video(store_config, video_path):
    """Register a raw video into the store's video_sets, in place —
    no copy, no symlink. video_path can point anywhere, including inside
    your raw camera-position/data/event folder structure.

    Writes directly to config.yaml instead of calling
    deeplabcut.add_new_videos(), because that function always tries to
    place a symlink or a copy under <project_path>/videos/ regardless of
    copy_videos, which breaks on Windows without Developer Mode/admin and
    also requires a videos/ folder that this store deliberately never
    creates."""
    store_config = str(store_config)
    video_path = str(Path(video_path).resolve())

    if not Path(video_path).is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cfg = auxiliaryfunctions.read_config(store_config)
    video_sets = cfg.get("video_sets") or {}
    if video_path in video_sets:
        print(f"⏭  Already registered: {video_path}")
        return

    video_sets[video_path] = _video_crop_entry(video_path)
    cfg["video_sets"] = video_sets
    auxiliaryfunctions.write_config(store_config, cfg)
    print(f"✅ Registered: {video_path}")


def extract_frames_for_video(store_config, video_path, algo="kmeans", mode="automatic", userfeedback=False):
    """Extract frames for exactly one video. Temporarily narrows video_sets to
    just this video (same trick as your 3a_register_and_extract.py) so kmeans
    frame selection runs per-video instead of pooling across the whole store.

    Not safe to run concurrently with another process touching the same
    store's config.yaml — same single-writer assumption your existing script
    already relies on.
    """
    store_config = str(store_config)
    video_path = str(Path(video_path).resolve())

    full_cfg = auxiliaryfunctions.read_config(store_config)
    video_sets = full_cfg.get("video_sets") or {}
    if video_path not in video_sets:
        raise ValueError(f"{video_path} is not registered — call add_video() first")

    stem = Path(video_path).stem
    labeled_dir = Path(full_cfg["project_path"]) / "labeled-data" / stem
    if labeled_dir.is_dir() and any(p.suffix == ".png" for p in labeled_dir.iterdir()):
        print(f"⏭  Frames already extracted for {stem}")
        return

    narrowed_cfg = dict(full_cfg)
    narrowed_cfg["video_sets"] = {video_path: video_sets[video_path]}
    auxiliaryfunctions.write_config(store_config, narrowed_cfg)

    try:
        deeplabcut.extract_frames(store_config, mode=mode, algo=algo, userfeedback=userfeedback)
    finally:
        # restore full video_sets regardless of success/failure above
        auxiliaryfunctions.write_config(store_config, full_cfg)

    print(f"🎞  Extracted frames: {stem}")