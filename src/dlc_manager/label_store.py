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

3) The store supports duplicates on purpose — extract_frames_for_video() is
   NOT idempotent. Every call creates a brand-new frame folder, even if you
   pass the exact same video and config_name twice in a row. By default the
   folder is named "<video_stem>__<config_name>", auto-incrementing to
   "..._2", "..._3", ... whenever that name is already taken; pass an
   explicit folder_name= to control it yourself instead. manifest.yaml at
   the store root is the index: for every frame folder it records which raw
   video and which exact extraction config (full snapshot, not just the
   name) produced it, plus frame count and timestamp — so "what frames came
   from which raw data, extracted how" is always answerable without
   touching DLC or walking folders by hand.

4) extract_frames_for_video() also accepts a FOLDER instead of a single
   video file. When given a folder, it recurses through it, finds every
   video file inside (any extension in VIDEO_EXTENSIONS, at any depth), and
   calls itself on each one individually — each video still gets its own
   auto-generated frame-set folder and its own manifest entry, exactly as
   if you'd called extract_frames_for_video() on it directly. Because each
   video needs its own folder_id, an explicit folder_name can't be combined
   with a folder input.

Videos are NEVER copied or symlinked into the store — video_sets just points
at wherever the raw video already lives.

Usage
─────
    from dlc_manager import (
        init_store, extract_frames_for_video, init_labeling_config,
        label_frames_for, ExtractionConfig, save_extraction_config, list_extractions,
    )

    store_path = init_store("/data/frames_store")   # seeds default.yaml preset

    # Extract using the auto-created "default" preset:
    extract_frames_for_video(store_path, "/raw_videos/HDMI-A.mp4")
    # -> labeled-data/HDMI-A__default/labeled-data/HDMI-A/*.png

    # Same video, same preset, called again -> NOT skipped, gets its own folder:
    extract_frames_for_video(store_path, "/raw_videos/HDMI-A.mp4")
    # -> labeled-data/HDMI-A__default__2/labeled-data/HDMI-A/*.png

    # Define + use a different preset -> its own folder too:
    save_extraction_config(store_path, ExtractionConfig(name="dense", numframes2pick=150))
    extract_frames_for_video(store_path, "/raw_videos/HDMI-A.mp4", config_name="dense")
    # -> labeled-data/HDMI-A__dense/labeled-data/HDMI-A/*.png

    # Or name the folder yourself:
    extract_frames_for_video(store_path, "/raw_videos/HDMI-A.mp4", config_name="dense", folder_name="HDMI-A_run3")
    # -> labeled-data/HDMI-A_run3/labeled-data/HDMI-A/*.png

    # Point at a whole folder of raw videos instead of a single file -> each
    # video inside (recursively) gets extracted on its own:
    extract_frames_for_video(store_path, "/raw_videos/session_07/")
    # -> labeled-data/HDMI-A__default/labeled-data/HDMI-A/*.png
    # -> labeled-data/HDMI-B__default/labeled-data/HDMI-B/*.png
    # -> ...

    # See what's been extracted:
    list_extractions(store_path)

    # Once ready to label a SPECIFIC frame set (not needed for extraction):
    init_labeling_config(
        store_path, "HDMI-A__default",
        scorer="Egor",
        bodyparts=["right_mitten_wrist", "right_mitten_tip", ...],
        skeleton=[["right_mitten_wrist", "right_mitten_tip"], ...],
    )
    # Launch napari pre-loaded with that frame set — no manual drag-and-drop:
    label_frames_for(store_path, "HDMI-A__default")
"""

import inspect
import shutil
from datetime import date as _date
from pathlib import Path

import deeplabcut
from deeplabcut.utils import auxiliaryfunctions

from . import extraction_config as extraction_config_module
from . import manifest as manifest_module


# Recognized video file extensions when extract_frames_for_video() is given
# a folder instead of a single video (matched case-insensitively).
VIDEO_EXTENSIONS = {
    ".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv",
    ".m4v", ".mpg", ".mpeg", ".webm", ".asf",
}


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

    if not manifest_module._manifest_path(store_path).exists():
        manifest_module._save_manifest(store_path, {"extractions": {}})

    default_path = extraction_config_module._extraction_configs_dir(store_path) / "default.yaml"
    if not default_path.exists():
        extraction_config_module.save_extraction_config(
            store_path, extraction_config_module.ExtractionConfig(name="default", **default_extraction_overrides)
        )
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


def _next_available_folder_id(store_path, base_id):
    """base_id if free, otherwise base_id__2, base_id__3, ... — first name
    under labeled-data/ that doesn't exist yet."""
    labeled_data_dir = Path(store_path).resolve() / "labeled-data"
    candidate = base_id
    n = 2
    while (labeled_data_dir / candidate).exists():
        candidate = f"{base_id}__{n}"
        n += 1
    return candidate


def _find_videos_in_folder(folder_path):
    """All files under folder_path (recursively) whose extension matches
    VIDEO_EXTENSIONS, sorted for deterministic ordering."""
    return sorted(
        p for p in folder_path.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )


def extract_frames_for_video(store_path, video_path, config_name="default", folder_name=None, overwrite=False):
    """Extract frames for one (video, extraction_config) call into its own
    nested mini-project:

        labeled-data/<folder_id>/
            config.yaml
            labeled-data/<video_stem>/*.png

    Every call creates a NEW folder instance — this is intentionally not
    idempotent. Even calling it twice with the identical video and the
    identical config_name produces two separate frame sets, because you may
    want repeated kmeans/uniform passes over the same source to sample
    different frames.

    video_path may also be a FOLDER. In that case this recurses through the
    folder (including subfolders), finds every file whose extension is in
    VIDEO_EXTENSIONS, and calls extract_frames_for_video() on each one in
    turn with the same config_name/overwrite. folder_name can't be combined
    with a folder input (each discovered video needs its own auto-generated
    folder_id) — pass folder_name only when video_path is a single file.
    Returns a dict of {video_path: config.yaml path} instead of a single
    path in this case.

    folder_name:
        - Omitted (default): folder_id auto-increments off
          "<video_stem>__<config_name>" — first call gets that exact name,
          later calls get "..._2", "..._3", etc. (first free name under
          labeled-data/).
        - Given: used as the exact folder_id. If it already exists, raises
          FileExistsError unless overwrite=True (which deletes and
          replaces that folder's contents).

    Records the result in manifest.yaml under its (possibly auto-generated)
    folder_id. Returns the new project's config.yaml path.
    """
    store_path = Path(store_path).resolve()
    input_path = Path(video_path).resolve()

    if input_path.is_dir():
        if folder_name is not None:
            raise ValueError(
                "folder_name can't be used when video_path points to a folder — "
                "each video found inside needs its own auto-generated folder_id. "
                "Omit folder_name (or call extract_frames_for_video() per file "
                "if you need explicit names)."
            )
        video_files = _find_videos_in_folder(input_path)
        if not video_files:
            raise FileNotFoundError(
                f"No video files (looked for {sorted(VIDEO_EXTENSIONS)}) found "
                f"under folder: {input_path}"
            )
        results = {}
        for vf in video_files:
            print(f"📁 {vf.relative_to(input_path)}")
            results[str(vf)] = extract_frames_for_video(
                store_path, vf, config_name=config_name, overwrite=overwrite
            )
        return results

    video_path = str(input_path)
    if not Path(video_path).is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    ex_cfg = extraction_config_module.load_extraction_config(store_path, config_name)
    video_stem = Path(video_path).stem
    base_id = folder_id_for(video_path, ex_cfg.name)

    if folder_name is not None:
        folder_id = folder_name
        project_dir = store_path / "labeled-data" / folder_id
        if project_dir.exists():
            if not overwrite:
                raise FileExistsError(
                    f"labeled-data/{folder_id} already exists. Pass overwrite=True "
                    f"to replace it, pick a different folder_name, or omit "
                    f"folder_name to auto-generate a fresh one."
                )
            shutil.rmtree(project_dir)
    else:
        # Always a brand-new folder — never reuses an existing one, so
        # duplicate (video, config) calls can never collide.
        folder_id = _next_available_folder_id(store_path, base_id)
        project_dir = store_path / "labeled-data" / folder_id

    frames_dir = project_dir / "labeled-data" / video_stem
    project_config_path = project_dir / "config.yaml"

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

    manifest_module._record_extraction(store_path, folder_id, video_path, video_stem, ex_cfg, frames_dir, project_config_path)
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


# ────────────────────────────────────────────────────────────────────────
# Launching napari — no manual "find the folder, drag it in" needed
# ────────────────────────────────────────────────────────────────────────

def label_frames_for(store_path, folder_id, multiple=False):
    """Open napari, pre-loaded with this frame set's images + config —
    equivalent to manually running `napari` and dragging in
    labeled-data/<folder_id>/labeled-data/<video_stem>/ plus
    labeled-data/<folder_id>/config.yaml yourself.

    Thin wrapper around deeplabcut.label_frames(), which is what actually
    activates the napari-deeplabcut plugin and opens both. Blocks (via
    napari.run()) until you close the napari window; whatever you save
    (Ctrl+S) lands in the usual CollectedData_<scorer>.h5/.csv inside that
    frame set's folder.

    multiple=True enables multi-individual labeling — leave False for a
    single-animal project.
    """
    store_path = Path(store_path).resolve()
    project_config_path = store_path / "labeled-data" / folder_id / "config.yaml"
    if not project_config_path.exists():
        raise FileNotFoundError(
            f"No extraction project at {project_config_path.parent} — "
            f"call extract_frames_for_video() for this video/config first."
        )

    cfg = auxiliaryfunctions.read_config(str(project_config_path))
    if not cfg.get("bodyparts"):
        raise ValueError(
            f"{project_config_path} has no bodyparts set yet — "
            f"call init_labeling_config(store_path, '{folder_id}', ...) first."
        )

    # deeplabcut.label_frames()'s signature varies across DLC versions — some
    # accept a `multiple` kwarg for multi-individual labeling, some don't
    # (multianimalproject in config.yaml is enough on newer versions). Only
    # pass it through if this installed version actually supports it, so
    # calling with the default multiple=False never breaks on any version.
    accepted_params = inspect.signature(deeplabcut.label_frames).parameters
    kwargs = {}
    if "multiple" in accepted_params:
        kwargs["multiple"] = multiple
    elif multiple:
        raise TypeError(
            "Your installed deeplabcut.label_frames() doesn't accept a "
            "'multiple' argument. If you need multi-individual labeling, set "
            "multianimalproject: true directly in this frame set's config.yaml "
            "instead, or upgrade deeplabcut."
        )

    deeplabcut.label_frames(str(project_config_path), **kwargs)

    # deeplabcut.label_frames() opens the napari viewer but, on some DLC
    # versions, returns immediately WITHOUT blocking until you close it. If
    # nothing keeps the process alive, the script ends right after this call,
    # tearing down the Qt app (and its background threads, e.g. the update
    # "StatusChecker") mid-session — that's the "opens for a second then
    # crashes with QThread: Destroyed while thread is still running" failure.
    # napari.run() enters napari's own event loop and simply returns once you
    # close the window, so labeling actually gets to happen first.
    import napari
    napari.run()
