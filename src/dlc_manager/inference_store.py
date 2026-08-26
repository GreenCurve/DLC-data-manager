"""
inference_store.py
───────────────────
A store of inference runs: each run binds one or more raw videos to an
already-trained network project (a network_store.py NetworkProject) and
holds whatever that pairing produces — prediction files from
deeplabcut.analyze_videos(), and, optionally, a rendered video with
predictions overlaid from deeplabcut.create_labeled_video().

Store layout on disk:
    <inference_runs>/
        manifest.yaml                       ← every run recorded: which
                                               network project + shuffle,
                                               which videos, when analyzed
        <run_id>/                           ← one folder per run, used as
            <video>.h5, .pickle, .csv         DLC's destfolder — everything
            <video>_labeled.mp4               analyze_videos()/
                                               create_labeled_video() write
                                               lands here, never mixed in
                                               with the raw video's own
                                               folder or another run's output

Design
──────
1) A run is created empty (create_run()) — it just records which videos are
   bound to which trained network project/shuffle. Nothing is analyzed yet.
   This mirrors network_store.create_project() being separate from
   add_labeled_data()/train_network(): binding inputs and actually running
   DLC are two different steps, so a run can be created and inspected (or
   reused for both analyze_videos() and create_labeled_video()) without
   forcing everything to happen in one call.

2) analyze_videos() wraps deeplabcut.analyze_videos(), pinned to
   Engine.PYTORCH like the rest of this package, with destfolder always
   this run's own folder (so prediction files never land next to the raw
   video or get mixed across runs). It writes prediction DATA files
   (.h5/.csv/.pickle) — not a video.

3) create_labeled_video() is a separate step on purpose: DLC's
   analyze_videos() never produces a viewable video by itself, only the
   underlying prediction files. Rendering an actual video with the
   predicted keypoints drawn on it is a second, more expensive pass
   (deeplabcut.create_labeled_video()) that reads the prediction file
   analyze_videos() already wrote — so it requires analyze_videos() to have
   run first for the same run/videos, and is kept as its own call instead
   of being folded into analyze_videos() automatically.

4) videos accepted by create_run() (and, if overridden, by
   analyze_videos()/create_labeled_video()) follow the exact same
   conventions as label_store.extract_frames_for_video(): a single video
   file, a folder (recursed for anything in label_store.VIDEO_EXTENSIONS),
   or a list mixing either. Videos are never copied — analyze_videos() /
   create_labeled_video() read them from wherever they already live.

Usage
─────
    from dlc_manager import create_run, analyze_videos, create_labeled_video

    run_dir = create_run(
        project.inference_runs,
        network_config=net.config_path,      # or the NetworkProject handle itself
        videos=r"/raw_videos/HDMI-A.mp4",
        shuffle=1,
    )

    analyze_videos(run_dir)
    create_labeled_video(run_dir)

    # or, via the InferenceRun handle (what DataProject.create_inference_run() returns):
    run = project.create_inference_run(network=net, videos=r"/raw_videos/session_07/")
    run.analyze_videos()
    run.create_labeled_video()
    run.list_predictions()
    run.list_labeled_videos()
"""

from dataclasses import dataclass
from datetime import date as _date, datetime
from pathlib import Path

import yaml

import deeplabcut
from deeplabcut.compat import Engine

from . import label_store as label_store_module


# ────────────────────────────────────────────────────────────────────────
# manifest.yaml — every run recorded: network project, videos, shuffle,
# and whatever analyze_videos()/create_labeled_video() have done so far
# ────────────────────────────────────────────────────────────────────────

def _manifest_path(store_path):
    return Path(store_path).resolve() / "manifest.yaml"


def _load_manifest(store_path):
    path = _manifest_path(store_path)
    if not path.exists():
        return {"runs": {}}
    with open(path) as f:
        return yaml.safe_load(f) or {"runs": {}}


def _save_manifest(store_path, manifest):
    with open(_manifest_path(store_path), "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)


def list_runs(store_path):
    """All recorded inference runs: run_id -> {network_project,
    network_config, videos, shuffle, created_at, analyzed_at?, ...}."""
    return _load_manifest(store_path).get("runs", {})


def get_run(store_path, run_id):
    record = list_runs(store_path).get(run_id)
    if record is None:
        raise KeyError(
            f"No inference run named '{run_id}' in {Path(store_path).resolve()}. "
            f"Known: {list(list_runs(store_path))}"
        )
    return record


def _update_run(store_path, run_id, updates):
    manifest = _load_manifest(store_path)
    runs = manifest.setdefault("runs", {})
    if run_id not in runs:
        raise KeyError(f"No inference run named '{run_id}' in {store_path}")
    runs[run_id].update(updates)
    _save_manifest(store_path, manifest)


# ────────────────────────────────────────────────────────────────────────
# Resolving the videos= argument — same conventions as
# label_store.extract_frames_for_video(): single file, folder (recursed),
# or a list mixing either
# ────────────────────────────────────────────────────────────────────────

def _resolve_videos(videos):
    if isinstance(videos, (list, tuple, set)):
        resolved = []
        for v in videos:
            resolved.extend(_resolve_videos(v))
        seen, deduped = set(), []
        for p in resolved:
            if p not in seen:
                seen.add(p)
                deduped.append(p)
        return deduped

    path = Path(videos).resolve()
    if path.is_dir():
        found = label_store_module._find_videos_in_folder(path)
        if not found:
            raise FileNotFoundError(
                f"No video files (looked for {sorted(label_store_module.VIDEO_EXTENSIONS)}) "
                f"found under folder: {path}"
            )
        return [str(p) for p in found]

    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {path}")
    return [str(path)]


# ────────────────────────────────────────────────────────────────────────
# Run bootstrap — bind video(s) to a trained network project, nothing
# analyzed yet
# ────────────────────────────────────────────────────────────────────────

def _next_available_run_dir(store_path, base_name):
    store_path = Path(store_path)
    candidate = base_name
    n = 2
    while (store_path / candidate).exists():
        candidate = f"{base_name}_{n}"
        n += 1
    return candidate


def create_run(store_path, network_config, videos, name=None, shuffle=1, **overrides):
    """Create a new inference run: bind one or more videos to an already-
    trained network project. Nothing is analyzed yet — call
    analyze_videos(run_dir) (or InferenceRun.analyze_videos()) next.

    network_config: a network project's config.yaml path, or the
        NetworkProject handle itself (its .config_path is used).
    videos: a single video file, a folder (recursed for anything in
        label_store.VIDEO_EXTENSIONS), or a list mixing either.
    name: run folder name under store_path.
        - Omitted: auto-generated as "<network-project-name>-shuffle<N>-
          <date>", auto-incrementing ("_2", "_3", ...) if that name is
          already taken.
        - Given: used verbatim. Raises FileExistsError if it already
          exists (pick a different name, or omit it to auto-generate).
    shuffle: which trained shuffle this run targets — stored on the run so
        later analyze_videos()/create_labeled_video() calls don't need to
        repeat it, though either call can still override it.
    **overrides: extra kwargs merged into every analyze_videos() call made
        against this run by default (e.g. save_as_csv=True) — anything
        passed directly to analyze_videos() at call time still wins.

    Returns the new run's folder (Path) — pass this to analyze_videos() /
    create_labeled_video(), or wrap it in InferenceRun(run_dir=...).
    """
    store_path = Path(store_path).resolve()
    store_path.mkdir(parents=True, exist_ok=True)

    config_path = Path(getattr(network_config, "config_path", network_config)).resolve()
    if not config_path.exists():
        raise FileNotFoundError(
            f"Network project config not found: {config_path} — pass a "
            f"network project's config.yaml path, or the NetworkProject "
            f"handle itself (e.g. from create_network_project() / "
            f"get_network_project())."
        )

    video_list = _resolve_videos(videos)
    network_name = config_path.parent.name

    date_str = _date.today().strftime("%b%d")
    base_name = name or f"{network_name}-shuffle{shuffle}-{date_str}"

    if name is not None:
        run_dir = store_path / name
        if run_dir.exists():
            raise FileExistsError(
                f"{run_dir} already exists. Pass a different name, or omit "
                f"name to auto-generate a fresh one."
            )
    else:
        run_dir = store_path / _next_available_run_dir(store_path, base_name)

    run_dir.mkdir(parents=True)

    manifest = _load_manifest(store_path)
    manifest.setdefault("runs", {})[run_dir.name] = {
        "network_project": network_name,
        "network_config": str(config_path),
        "videos": video_list,
        "shuffle": shuffle,
        "overrides": dict(overrides),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    _save_manifest(store_path, manifest)

    print(
        f"✅ Inference run created: {run_dir} "
        f"({len(video_list)} video(s), network={network_name}, shuffle={shuffle})"
    )
    return run_dir


# ────────────────────────────────────────────────────────────────────────
# Analysis — thin wrappers, pinned to the pytorch engine
# ────────────────────────────────────────────────────────────────────────

def analyze_videos(run_dir, shuffle=None, videos=None, **kwargs):
    """Wraps deeplabcut.analyze_videos(), pinned to Engine.PYTORCH.

    Writes prediction files (.h5/.csv/.pickle) into run_dir itself
    (destfolder) — never next to the raw video, never mixed with another
    run's output.

    shuffle: overrides the shuffle this run was created with, if given.
    videos: overrides/subsets which video(s) to analyze (same conventions
        as create_run()'s videos= — single file, folder, or list). Defaults
        to every video this run was created with.
    **kwargs: merged over this run's own **overrides (from create_run()),
        then passed straight through to deeplabcut.analyze_videos() —
        anything passed here wins over a same-named create_run() override.

    Returns whatever deeplabcut.analyze_videos() returns (the DLC scorer
    name string) and records analyzed_at + the video list actually
    analyzed in manifest.yaml.
    """
    run_dir = Path(run_dir).resolve()
    store_path, run_id = run_dir.parent, run_dir.name
    record = get_run(store_path, run_id)

    shuffle = record["shuffle"] if shuffle is None else shuffle
    video_list = _resolve_videos(videos) if videos is not None else list(record["videos"])

    call_kwargs = dict(record.get("overrides") or {})
    call_kwargs.update(kwargs)
    call_kwargs.setdefault("engine", Engine.PYTORCH)

    result = deeplabcut.analyze_videos(
        record["network_config"], video_list,
        shuffle=shuffle, destfolder=str(run_dir), **call_kwargs,
    )

    _update_run(store_path, run_id, {
        "shuffle": shuffle,
        "analyzed_videos": video_list,
        "analyzed_at": datetime.now().isoformat(timespec="seconds"),
        "scorer": result,
    })
    print(f"✅ Analyzed {len(video_list)} video(s) → {run_dir}")
    return result


def create_labeled_video(run_dir, shuffle=None, videos=None, **kwargs):
    """Wraps deeplabcut.create_labeled_video() to render an actual video
    with predicted keypoints drawn on it. This is a separate step from
    analyze_videos() on purpose: analyze_videos() only ever writes
    prediction DATA files (.h5/.csv/.pickle), never a video — rendering one
    is a second, heavier pass that reads the prediction file
    analyze_videos() already wrote for the same video/shuffle/destfolder.

    Requires analyze_videos(run_dir) to have been called first (for these
    videos, at this shuffle) — raises RuntimeError otherwise.

    shuffle / videos / **kwargs: same conventions as analyze_videos().
    videos defaults to whatever was last analyzed for this run (falling
    back to the run's full bound video list if that's somehow unset).

    Output lands in run_dir itself (destfolder), as "<video_stem>_labeled.mp4".
    Records labeled_video_at in manifest.yaml.
    """
    run_dir = Path(run_dir).resolve()
    store_path, run_id = run_dir.parent, run_dir.name
    record = get_run(store_path, run_id)

    if not record.get("analyzed_at"):
        raise RuntimeError(
            f"Inference run '{run_id}' hasn't been analyzed yet — call "
            f"analyze_videos(run_dir) first."
        )

    shuffle = record["shuffle"] if shuffle is None else shuffle
    video_list = (
        _resolve_videos(videos) if videos is not None
        else list(record.get("analyzed_videos") or record["videos"])
    )

    deeplabcut.create_labeled_video(
        record["network_config"], video_list,
        shuffle=shuffle, destfolder=str(run_dir), **kwargs,
    )

    _update_run(store_path, run_id, {
        "labeled_video_at": datetime.now().isoformat(timespec="seconds"),
    })
    print(f"🎬 Labeled video(s) created → {run_dir}")


# ────────────────────────────────────────────────────────────────────────
# Inspecting a run's output
# ────────────────────────────────────────────────────────────────────────

def list_predictions(run_dir):
    """Prediction files (*.h5) currently sitting in this run's folder."""
    return sorted(Path(run_dir).resolve().glob("*.h5"))


def list_labeled_videos(run_dir):
    """Rendered videos (*_labeled.mp4) currently sitting in this run's
    folder."""
    return sorted(Path(run_dir).resolve().glob("*_labeled.mp4"))


# ────────────────────────────────────────────────────────────────────────
# Store bootstrap
# ────────────────────────────────────────────────────────────────────────

def init_store(store_path):
    """Bootstrap an inference store: just the folder + an empty
    manifest.yaml. Idempotent. Individual runs are created on demand via
    create_run()."""
    store_path = Path(store_path).resolve()
    store_path.mkdir(parents=True, exist_ok=True)
    if not _manifest_path(store_path).exists():
        _save_manifest(store_path, {"runs": {}})
        print(f"✅ Inference store initialized: {store_path}")
    else:
        print(f"⏭  Inference store already initialized: {store_path}")
    return store_path


# ────────────────────────────────────────────────────────────────────────
# InferenceRun — convenience handle, same role as network_store.py's
# NetworkProject: every method is a thin forward to the free functions
# above with run_dir already filled in
# ────────────────────────────────────────────────────────────────────────

@dataclass
class InferenceRun:
    run_dir: Path

    @property
    def store_path(self):
        return self.run_dir.parent

    @property
    def run_id(self):
        return self.run_dir.name

    @property
    def record(self):
        """Fresh snapshot of this run's manifest entry."""
        return get_run(self.store_path, self.run_id)

    def analyze_videos(self, shuffle=None, videos=None, **kwargs):
        return analyze_videos(self.run_dir, shuffle=shuffle, videos=videos, **kwargs)

    def create_labeled_video(self, shuffle=None, videos=None, **kwargs):
        return create_labeled_video(self.run_dir, shuffle=shuffle, videos=videos, **kwargs)

    def list_predictions(self):
        return list_predictions(self.run_dir)

    def list_labeled_videos(self):
        return list_labeled_videos(self.run_dir)

    def __repr__(self):
        return f"InferenceRun(run_id={self.run_id!r}, run_dir={str(self.run_dir)!r})"
