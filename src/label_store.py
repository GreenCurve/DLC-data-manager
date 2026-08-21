"""
label_store.py
───────────────
A persistent "labeled-data store" that extract_frames / label_frames (napari)
can operate on directly, WITHOUT going through deeplabcut.create_new_project().

Design (v2)
───────────
1) Extraction and labeling are fully decoupled, PER FRAME SET.
   Every (video, extraction_config) pair gets its own tiny DLC project at

       <store_path>/labeled-data/<video_stem>__<config_name>/
           config.yaml                          ← extraction-only, written by
                                                    extract_frames_for_video()
           labeled-data/
               <video_stem>/
                   *.png
                   CollectedData_<scorer>.h5     ← written once you label

   That nested layout is exactly what DLC's own label_frames()/napari expects
   (project_path/labeled-data/<video_stem>/...), so each frame set can carry
   its OWN bodyparts/scorer/skeleton via init_labeling_config(), added later,
   without touching any other frame set's schema. This is what solves the
   "bodyparts differ per video" problem — there is no single store-wide
   config.yaml that everything has to agree on.

2) Extraction parameters are a first-class, versioned object: ExtractionConfig.
   It holds exactly what deeplabcut.extract_frames() needs (algo, mode,
   userfeedback, numframes2pick, start, stop, engine). Presets are saved as
   YAML under <store_path>/extraction_configs/<name>.yaml and referenced by
   name wherever you extract. Keep as many presets as you want
   (default.yaml, dense.yaml, uniform_pass.yaml, ...).

3) The store supports duplicates on purpose. You explicitly pick a video AND
   a named extraction config every time you extract; the resulting frame
   folder is named "<video_stem>__<config_name>" so re-extracting the same
   video with a different config never collides with or overwrites an
   existing frame set. manifest.yaml at the store root is the index: for
   every frame folder it records which raw video and which exact extraction
   config (full snapshot, not just the name) produced it, plus frame count
   and timestamp — so "what frames came from which raw data, extracted how"
   is always answerable without touching DLC or walking folders by hand.

Videos are NEVER copied or symlinked into the store — video_sets just points
at wherever the raw video already lives.

Usage
─────
    from label_store import (
        init_store, extract_frames_for_video, init_labeling_config,
        ExtractionConfig, save_extraction_config, list_extractions,
    )

    store_path = init_store("/data/frames_store")   # seeds default.yaml preset

    # Extract using the auto-created "default" preset:
    extract_frames_for_video(store_path, "/raw_videos/HDMI-A.mp4")
    # -> labeled-data/HDMI-A__default/labeled-data/HDMI-A/*.png

    # Define + use a different preset -> lands in its OWN folder:
    save_extraction_config(store_path, ExtractionConfig(name="dense", numframes2pick=150))
    extract_frames_for_video(store_path, "/raw_videos/HDMI-A.mp4", config_name="dense")
    # -> labeled-data/HDMI-A__dense/labeled-data/HDMI-A/*.png  (independent of the above)

    # See what's been extracted:
    list_extractions(store_path)

    # Once ready to label a SPECIFIC frame set (not needed for extraction):
    init_labeling_config(
        store_path, "HDMI-A__default",
        scorer="Egor",
        bodyparts=["right_mitten_wrist", "right_mitten_tip", ...],
        skeleton=[["right_mitten_wrist", "right_mitten_tip"], ...],
    )
    # Your existing 3c_label_video.py points napari at
    # store_path/labeled-data/HDMI-A__default/config.yaml — unchanged usage,
    # DLC can't tell this apart from a normal project config.
"""

import matplotlib
matplotlib.use("Agg")  # must come before importing deeplabcut, matches your existing scripts

import yaml
from dataclasses import dataclass, asdict, field
from datetime import date as _date, datetime
from pathlib import Path

import deeplabcut
from deeplabcut.utils import auxiliaryfunctions


# ────────────────────────────────────────────────────────────────────────
# Extraction config: versioned, named presets
# ────────────────────────────────────────────────────────────────────────

@dataclass
class ExtractionConfig:
    """Exactly the arguments deeplabcut.extract_frames() (automatic mode)
    consumes — nothing about bodyparts/scorer/skeleton lives here."""
    name: str = "default"
    algo: str = "kmeans"
    mode: str = "automatic"
    userfeedback: bool = False
    numframes2pick: int = 20
    start: float = 0.0
    stop: float = 1.0
    engine: str = "pytorch"

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


def _extraction_configs_dir(store_path):
    return Path(store_path).resolve() / "extraction_configs"


def save_extraction_config(store_path, cfg: ExtractionConfig, overwrite=False):
    """Persist an ExtractionConfig preset as extraction_configs/<name>.yaml."""
    cfg_dir = _extraction_configs_dir(store_path)
    cfg_dir.mkdir(parents=True, exist_ok=True)
    path = cfg_dir / f"{cfg.name}.yaml"
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Extraction config '{cfg.name}' already exists at {path}. "
            f"Pass overwrite=True to replace it."
        )
    with open(path, "w") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)
    print(f"✅ Extraction config saved: {path}")
    return path


def load_extraction_config(store_path, name="default") -> ExtractionConfig:
    path = _extraction_configs_dir(store_path) / f"{name}.yaml"
    if not path.exists():
        available = list_extraction_configs(store_path)
        raise FileNotFoundError(
            f"No extraction config named '{name}' in {path.parent}. "
            f"Available: {available or '(none)'}"
        )
    with open(path) as f:
        d = yaml.safe_load(f) or {}
    d.setdefault("name", name)
    return ExtractionConfig.from_dict(d)


def list_extraction_configs(store_path):
    cfg_dir = _extraction_configs_dir(store_path)
    if not cfg_dir.is_dir():
        return []
    return sorted(p.stem for p in cfg_dir.glob("*.yaml"))


# ────────────────────────────────────────────────────────────────────────
# Manifest: which raw video + which extraction config produced which frames
# ────────────────────────────────────────────────────────────────────────

def _manifest_path(store_path):
    return Path(store_path).resolve() / "manifest.yaml"


def _load_manifest(store_path):
    path = _manifest_path(store_path)
    if not path.exists():
        return {"extractions": {}}
    with open(path) as f:
        return yaml.safe_load(f) or {"extractions": {}}


def _save_manifest(store_path, manifest):
    with open(_manifest_path(store_path), "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)


def list_extractions(store_path):
    """All recorded frame sets: folder_id -> {video_path, config_name, ...}."""
    return _load_manifest(store_path).get("extractions", {})


def get_extraction(store_path, folder_id):
    record = list_extractions(store_path).get(folder_id)
    if record is None:
        raise KeyError(
            f"No extraction recorded for '{folder_id}'. "
            f"Known: {list(list_extractions(store_path))}"
        )
    return record


def _record_extraction(store_path, folder_id, video_path, video_stem, ex_cfg, frames_dir, project_config_path):
    manifest = _load_manifest(store_path)
    extractions = manifest.setdefault("extractions", {})
    frame_count = sum(1 for p in frames_dir.iterdir() if p.suffix == ".png")
    extractions[folder_id] = {
        "video_path": video_path,
        "video_stem": video_stem,
        "config_name": ex_cfg.name,
        "config": ex_cfg.to_dict(),
        "frame_count": frame_count,
        "project_config": str(project_config_path),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    _save_manifest(store_path, manifest)


# ────────────────────────────────────────────────────────────────────────
# Store bootstrap
# ────────────────────────────────────────────────────────────────────────

def init_store(store_path, **default_extraction_overrides):
    """Bootstrap a store: labeled-data/, extraction_configs/ (with a
    'default' preset), and an empty manifest.yaml. Idempotent.

    There is deliberately NO single store-wide config.yaml anymore — each
    frame set gets its own project config (see extract_frames_for_video /
    init_labeling_config), and extraction presets live in
    extraction_configs/. Any ExtractionConfig field can be overridden for
    the auto-created default preset via **default_extraction_overrides,
    e.g. init_store(path, numframes2pick=50).
    """
    store_path = Path(store_path).resolve()
    (store_path / "labeled-data").mkdir(parents=True, exist_ok=True)
    (store_path / "extraction_configs").mkdir(parents=True, exist_ok=True)

    if not _manifest_path(store_path).exists():
        _save_manifest(store_path, {"extractions": {}})

    default_path = _extraction_configs_dir(store_path) / "default.yaml"
    if not default_path.exists():
        save_extraction_config(store_path, ExtractionConfig(name="default", **default_extraction_overrides))
        print(f"✅ Store initialized: {store_path}")
    else:
        print(f"⏭  Store already initialized: {store_path}")

    return store_path


# ────────────────────────────────────────────────────────────────────────
# Extraction
# ────────────────────────────────────────────────────────────────────────

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


def folder_id_for(video_path, config_name):
    return f"{Path(video_path).stem}__{config_name}"


def extract_frames_for_video(store_path, video_path, config_name="default", overwrite=False):
    """Extract frames for one (video, extraction_config) pair into its own
    nested mini-project:

        labeled-data/<video_stem>__<config_name>/
            config.yaml
            labeled-data/<video_stem>/*.png

    Explicitly names both the raw video and the extraction preset — this is
    what lets the same video be extracted multiple times, differently,
    without one run clobbering another. Records the result in manifest.yaml.

    Idempotent: if this exact (video, config) pair was already extracted,
    skips DLC and just returns the existing project config path (pass
    overwrite=True to force re-extraction).
    """
    store_path = Path(store_path).resolve()
    video_path = str(Path(video_path).resolve())
    if not Path(video_path).is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    ex_cfg = load_extraction_config(store_path, config_name)
    video_stem = Path(video_path).stem
    folder_id = folder_id_for(video_path, ex_cfg.name)
    project_dir = store_path / "labeled-data" / folder_id
    frames_dir = project_dir / "labeled-data" / video_stem
    project_config_path = project_dir / "config.yaml"

    already_extracted = frames_dir.is_dir() and any(p.suffix == ".png" for p in frames_dir.iterdir())
    if already_extracted and not overwrite:
        print(f"⏭  Already extracted: {folder_id}")
        _record_extraction(store_path, folder_id, video_path, video_stem, ex_cfg, frames_dir, project_config_path)
        return project_config_path

    crop_entry = _video_crop_entry(video_path)
    project_dir.mkdir(parents=True, exist_ok=True)

    cfg, _ = auxiliaryfunctions.create_config_template()
    cfg["project_path"] = str(project_dir)
    cfg["video_sets"] = {video_path: crop_entry}
    cfg["engine"] = ex_cfg.engine
    cfg["numframes2pick"] = ex_cfg.numframes2pick
    cfg["start"] = ex_cfg.start
    cfg["stop"] = ex_cfg.stop
    cfg["date"] = _date.today().strftime("%b%d")
    auxiliaryfunctions.write_config(str(project_config_path), cfg)

    deeplabcut.extract_frames(
        str(project_config_path),
        mode=ex_cfg.mode,
        algo=ex_cfg.algo,
        userfeedback=ex_cfg.userfeedback,
    )

    if not frames_dir.is_dir():
        raise RuntimeError(f"DLC did not produce the expected frames dir: {frames_dir}")

    _record_extraction(store_path, folder_id, video_path, video_stem, ex_cfg, frames_dir, project_config_path)
    frame_count = sum(1 for p in frames_dir.iterdir() if p.suffix == ".png")
    print(f"🎞  Extracted {frame_count} frames → labeled-data/{folder_id}/labeled-data/{video_stem}")
    return project_config_path


# ────────────────────────────────────────────────────────────────────────
# Labeling schema — set PER frame set, once you're ready to label it
# ────────────────────────────────────────────────────────────────────────

def init_labeling_config(store_path, folder_id, scorer, bodyparts, skeleton, task=None, **overrides):
    """Fill in what label_frames()/napari needs — scorer, bodyparts,
    skeleton — for ONE frame set (folder_id, e.g. "HDMI-A__default").

    Edits that frame set's own config.yaml in place; every other frame set
    in the store keeps whatever schema (or lack of one) it already has.
    This is what makes bodyparts safely differ across videos/extractions.

    Safe to call any time before labeling starts for this frame set. If you
    change bodyparts after CollectedData_<scorer>.h5 already exists for it,
    that file will be out of sync with the new schema.
    """
    store_path = Path(store_path).resolve()
    project_config_path = store_path / "labeled-data" / folder_id / "config.yaml"
    if not project_config_path.exists():
        raise FileNotFoundError(
            f"No extraction project at {project_config_path.parent} — "
            f"call extract_frames_for_video() for this video/config first."
        )

    cfg = auxiliaryfunctions.read_config(str(project_config_path))
    cfg["scorer"] = scorer
    cfg["bodyparts"] = list(bodyparts)
    cfg["skeleton"] = [list(pair) for pair in skeleton]
    cfg["multianimalproject"] = False
    if task is not None:
        cfg["Task"] = task

    for key, value in overrides.items():
        cfg[key] = value

    auxiliaryfunctions.write_config(str(project_config_path), cfg)
    print(f"✅ Labeling schema set: {project_config_path}")
    return project_config_path