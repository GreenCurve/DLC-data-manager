"""
network_store.py
─────────────────
Consumer side of the pipeline: a store of independent DeepLabCut 2D
projects. Each project is a normal, trainable DLC project — just built
from copied labeled-data instead of by pointing DLC at real videos.

Store layout on disk:
    <network_store>/
        <project_id>/                       ← a normal DLC project
            config.yaml
            labeled-data/
                <folder_id>/                ← copied wholesale from a
                    *.png                      label_store frame set,
                    CollectedData_<scorer>.h5  e.g. "HDMI-A__default"
            training-datasets/
            dlc-models-pytorch/
            videos/                         ← intentionally always empty
            sources.yaml                    ← which label_store folder(s)
                                               were copied in, and when

Design
──────
1) create_project() never calls deeplabcut.create_new_project() — that
   function insists on at least one real video path, which we don't have
   (a network project here is trained purely from copied frames). Instead
   the project scaffold + config.yaml are built by hand, the same way
   label_store.init_store() sidesteps deeplabcut.create_new_project().
   video_sets is left as {} PERMANENTLY, not just at creation time.

   This isn't just a shortcut — deeplabcut.generate_training_dataset's
   merge_annotateddatasets() has a documented fallback: when it can't match
   labeled-data folders against video_sets, it scans every folder directly
   under labeled-data/ for a CollectedData_<scorer>.h5 instead. Empty
   video_sets reliably takes that path.

2) add_labeled_data() copies (never links/symlinks) a label_store frame-set
   folder — labeled-data/<folder_id> from a label_store, e.g.
   ".../frames_store/labeled-data/HDMI-A__default" — wholesale into the
   project, named after its folder_id (not the bare video stem), so two
   different extractions of the same source video never collide inside one
   network project.

   On the FIRST call for a fresh project (still has no bodyparts), the
   incoming folder's scorer/bodyparts/skeleton are adopted onto the
   project's own config.yaml. Every later call validates the incoming
   folder's bodyparts match exactly — DLC needs one consistent bodypart
   schema across an entire project, so a mismatch is caught here at copy
   time instead of failing confusingly at training-dataset time.

3) create_train_dataset() / train_network() are thin wrappers around
   deeplabcut.create_training_dataset() / deeplabcut.train_network(),
   pinned to the pytorch engine.

Usage
─────
    from dlc_manager import (
        create_project, add_labeled_data, create_train_dataset, train_network,
    )

    project_config = create_project(project.network_store, name="mitten_tracker")

    add_labeled_data(project_config, r".../frames_store/labeled-data/HDMI-A__default")
    add_labeled_data(project_config, r".../frames_store/labeled-data/HDMI-A__dense")

    create_train_dataset(project_config)
    train_network(project_config, epochs=600)
"""

import copy
import re
import shutil
import pandas as pd
import yaml
from datetime import date as _date, datetime
from pathlib import Path

import deeplabcut
from deeplabcut.compat import Engine
from deeplabcut.utils import auxiliaryfunctions
from pydantic import ValidationError


# ────────────────────────────────────────────────────────────────────────
# sources.yaml — which label_store folder(s) were copied into this project
# ────────────────────────────────────────────────────────────────────────

def _sources_path(project_dir):
    return Path(project_dir) / "sources.yaml"


def _load_sources(project_dir):
    path = _sources_path(project_dir)
    if not path.exists():
        return {"copied": []}
    with open(path) as f:
        return yaml.safe_load(f) or {"copied": []}


def _save_sources(project_dir, sources):
    with open(_sources_path(project_dir), "w") as f:
        yaml.safe_dump(sources, f, sort_keys=False)


def list_labeled_data(project_config):
    """What's been copied into this project so far: [{folder_id, video_stem,
    source_folder, frame_count, copied_at}, ...]."""
    project_dir = Path(project_config).resolve().parent
    return _load_sources(project_dir).get("copied", [])


# ────────────────────────────────────────────────────────────────────────
# Project bootstrap — no video required
# ────────────────────────────────────────────────────────────────────────

def _next_available_project_dir(store_path, base_name):
    store_path = Path(store_path)
    candidate = base_name
    n = 2
    while (store_path / candidate).exists():
        candidate = f"{base_name}_{n}"
        n += 1
    return candidate


def create_project(store_path, name=None, scorer="Egor", engine="pytorch", **overrides):
    """Create a new, empty DeepLabCut 2D project under store_path — no
    video path needed. Does NOT call deeplabcut.create_new_project();
    builds the same scaffold by hand instead (see module docstring).

    name: project folder name.
        - Omitted: auto-generated as "network-<scorer>-<date>", auto-
          incrementing ("_2", "_3", ...) if that name is already taken.
        - Given: used verbatim. Raises FileExistsError if it already
          exists (pick a different name, or omit it to auto-generate).

    Any DLC config key can be overridden via **overrides, e.g. pcutoff=0.9.
    Returns the new project's config.yaml path.
    """
    store_path = Path(store_path).resolve()
    store_path.mkdir(parents=True, exist_ok=True)

    date_str = _date.today().strftime("%b%d")
    base_name = name or f"network-{scorer}-{date_str}"

    if name is not None:
        project_dir = store_path / name
        if project_dir.exists():
            raise FileExistsError(
                f"{project_dir} already exists. Pass a different name, or "
                f"omit name to auto-generate a fresh one."
            )
    else:
        project_dir = store_path / _next_available_project_dir(store_path, base_name)

    for sub in ("labeled-data", "training-datasets", "dlc-models-pytorch", "videos"):
        (project_dir / sub).mkdir(parents=True, exist_ok=True)

    cfg, _ = auxiliaryfunctions.create_config_template()
    cfg["Task"] = name or base_name
    cfg["scorer"] = scorer
    cfg["date"] = date_str
    cfg["project_path"] = str(project_dir)
    cfg["video_sets"] = {}      # stays empty permanently — see module docstring
    cfg["bodyparts"] = []       # adopted from the first add_labeled_data() call
    cfg["skeleton"] = []
    cfg["engine"] = engine
    cfg["multianimalproject"] = False

    for key, value in overrides.items():
        cfg[key] = value

    config_path = project_dir / "config.yaml"
    auxiliaryfunctions.write_config(str(config_path), cfg)
    _save_sources(project_dir, {"copied": []})

    print(f"✅ Network project created: {config_path}")
    return config_path


# ────────────────────────────────────────────────────────────────────────
# Copying labeled data in
# ────────────────────────────────────────────────────────────────────────

def _relocate_annotation_paths(dest_dir, old_folder_name, new_folder_name, scorer):
    """napari-deeplabcut bakes each labeled image's path into
    CollectedData_<scorer>.h5's row index as
    ("labeled-data", <folder name at labeling time>, <filename>). Since we
    copy the frames into a differently-named folder here (folder_id instead
    of bare video_stem, to avoid collisions), that embedded folder name has
    to be updated to match — otherwise DLC looks for images under the old
    name and fails with a FileNotFoundError at create_training_dataset time.
    """
    h5_path = dest_dir / f"CollectedData_{scorer}.h5"
    if not h5_path.exists():
        # Nothing labeled yet for this frame set — nothing to fix.
        return

    with pd.HDFStore(h5_path, mode="r") as store:
        keys = store.keys()
    if len(keys) != 1:
        raise RuntimeError(f"Expected exactly one table in {h5_path}, found {keys}")
    key = keys[0]

    df = pd.read_hdf(h5_path, key=key)
    if not isinstance(df.index, pd.MultiIndex) or df.index.nlevels < 3:
        raise RuntimeError(f"Unexpected annotation index structure in {h5_path}: {df.index}")

    # Standard 3-level index: (top, video_folder, filename) — rename only
    # the middle level, wherever it matches the folder's old name.
    df.index = pd.MultiIndex.from_tuples(
        [
            (t[0], new_folder_name if t[-2] == old_folder_name else t[-2], t[-1])
            if len(t) == 3
            else t
            for t in df.index
        ],
        names=df.index.names,
    )
    df.to_hdf(h5_path, key=key, mode="w")

    # The .csv sidecar (if any) is a human-readable export DLC doesn't read
    # for training — drop it rather than leave it silently stale.
    csv_path = dest_dir / f"CollectedData_{scorer}.csv"
    if csv_path.exists():
        csv_path.unlink()


def add_labeled_data(project_config, source_folder, overwrite=False):
    """Copy a label_store frame-set folder wholesale into this project.

    source_folder: the frame-set folder itself, e.g.
        <label_store>/labeled-data/HDMI-A__default
    i.e. the folder containing that frame set's own config.yaml and its
    nested labeled-data/<video_stem>/*.png + CollectedData_<scorer>.h5.

    Copies into project's labeled-data/<folder_id> (folder_id = source
    folder's own name, e.g. "HDMI-A__default") — not the bare video stem —
    so two different extractions of the same video never collide here.

    First call on a fresh project (no bodyparts yet) adopts the incoming
    scorer/bodyparts/skeleton as this project's schema. Later calls require
    an exact bodypart match; raises ValueError otherwise.
    """
    project_config = Path(project_config).resolve()
    project_dir = project_config.parent
    source_folder = Path(source_folder).resolve()
    source_config = source_folder / "config.yaml"

    if not source_config.exists():
        raise FileNotFoundError(
            f"{source_config} not found — source_folder should be a label_store "
            f"frame-set folder (labeled-data/<folder_id>), e.g. "
            f".../frames_store/labeled-data/HDMI-A__default"
        )

    source_cfg = auxiliaryfunctions.read_config(str(source_config))
    source_bodyparts = source_cfg.get("bodyparts") or []
    if not source_bodyparts:
        raise ValueError(
            f"{source_config} has no bodyparts set yet — label that frame set "
            f"(init_labeling_config) before adding it to a network project."
        )

    proj_cfg = auxiliaryfunctions.read_config(str(project_config))
    if not proj_cfg.get("bodyparts"):
        proj_cfg["scorer"] = source_cfg["scorer"]
        proj_cfg["bodyparts"] = list(source_bodyparts)
        proj_cfg["skeleton"] = [list(pair) for pair in source_cfg.get("skeleton", [])]
        proj_cfg["multianimalproject"] = source_cfg.get("multianimalproject", False)
        auxiliaryfunctions.write_config(str(project_config), proj_cfg)
        print(f"✅ Project schema adopted from {source_folder.name}: {len(source_bodyparts)} bodyparts")
    elif list(proj_cfg["bodyparts"]) != list(source_bodyparts):
        raise ValueError(
            f"Bodypart mismatch: project expects {proj_cfg['bodyparts']}, "
            f"but {source_folder.name} has {source_bodyparts}. Every frame set "
            f"in one network project must share the same bodyparts/order."
        )

    # find the actual leaf folder holding the frames: <source_folder>/labeled-data/<video_stem>
    nested_root = source_folder / "labeled-data"
    nested_dirs = [p for p in nested_root.iterdir() if p.is_dir()] if nested_root.is_dir() else []
    if len(nested_dirs) != 1:
        raise RuntimeError(
            f"Expected exactly one video folder under {nested_root}, found {len(nested_dirs)}."
        )
    frames_src = nested_dirs[0]
    video_stem = frames_src.name

    folder_id = source_folder.name
    dest = project_dir / "labeled-data" / folder_id
    if dest.exists():
        if not overwrite:
            raise FileExistsError(
                f"{dest} already exists in this project. Pass overwrite=True to "
                f"replace it — this frame set was likely already added."
            )
        shutil.rmtree(dest)

    shutil.copytree(frames_src, dest)
    _relocate_annotation_paths(dest, old_folder_name=video_stem, new_folder_name=folder_id, scorer=proj_cfg["scorer"])

    sources = _load_sources(project_dir)
    sources.setdefault("copied", []).append({
        "folder_id": folder_id,
        "video_stem": video_stem,
        "source_folder": str(source_folder),
        "frame_count": sum(1 for p in dest.iterdir() if p.suffix == ".png"),
        "copied_at": datetime.now().isoformat(timespec="seconds"),
    })
    _save_sources(project_dir, sources)

    print(f"✅ Copied {folder_id} → {dest}")
    return dest


# ────────────────────────────────────────────────────────────────────────
# Training — thin wrappers, pinned to the pytorch engine
# ────────────────────────────────────────────────────────────────────────

def create_train_dataset(project_config, net_type="resnet_50", **kwargs):
    """Wraps deeplabcut.create_training_dataset(), pinned to Engine.PYTORCH
    to match everywhere else in this project."""
    return deeplabcut.create_training_dataset(
        str(project_config), net_type=net_type, engine=Engine.PYTORCH, **kwargs
    )


def _with_wandb_logger(pytorch_cfg_updates, project_config, shuffle, wandb_project, wandb_run_name, wandb_tags, wandb_image_log_interval):
    """Merge a `logger: {type: WandbLogger, ...}` block into
    pytorch_cfg_updates, so callers configure W&B via a few plain kwargs
    instead of hand-building the nested pytorch_config.yaml dict.

    Does nothing if wandb_project is None (the default) — training then
    behaves exactly as before, logging only to the model folder.

    run_name default: "<network-project-name>-shuffle<N>" so runs are
    identifiable in the W&B UI without extra bookkeeping.
    """
    if wandb_project is None:
        return pytorch_cfg_updates or {}

    project_config = Path(project_config)
    logger_cfg = {
        "type": "WandbLogger",
        "project_name": wandb_project,
        "run_name": wandb_run_name or f"{project_config.parent.name}-shuffle{shuffle}",
    }
    if wandb_tags:
        logger_cfg["tags"] = list(wandb_tags)
    if wandb_image_log_interval is not None:
        logger_cfg["image_log_interval"] = wandb_image_log_interval

    updates = dict(pytorch_cfg_updates or {})
    # Merge rather than overwrite, in case a caller already passed other
    # top-level pytorch_config.yaml keys via pytorch_cfg_updates.
    updates["logger"] = {**updates.get("logger", {}), **logger_cfg}
    return updates


_wandb_logger_build_patched = False


def _patch_wandb_logger_build():
    """Best-effort fix for the actual bug: some installed DLC versions
    (confirmed: 3.0.1) have a WandbLogger config *schema* that includes a
    field (e.g. wandb_kwargs) the real WandbLogger *class* constructor
    doesn't accept — so DLC's own LOGGER.build() raises TypeError deep
    inside training, well after config validation already passed, right as
    it's about to call wandb.init(). No pytorch_cfg_updates trimming can
    prevent this: the field comes from the schema's own defaults, not from
    anything we send.

    Rather than disabling W&B logging to work around that, wrap DLC's
    LOGGER.build() so that if it hits exactly this TypeError, it parses the
    bad keyword straight out of DLC's own error message, drops just that
    key, and retries — so the real WandbLogger still gets constructed and
    logging actually happens. Idempotent (safe to call every train_network()
    call) and strictly best-effort: if DLC's internals don't look like what
    we expect here (different registry API in a different DLC version),
    this quietly does nothing and train_network()'s existing TypeError
    fallback (disable W&B, keep training) remains the safety net.
    """
    global _wandb_logger_build_patched
    if _wandb_logger_build_patched:
        return
    try:
        from deeplabcut.pose_estimation_pytorch.apis import training as _dlc_training
    except ImportError:
        return
    logger_registry = getattr(_dlc_training, "LOGGER", None)
    if logger_registry is None or not hasattr(logger_registry, "build"):
        return

    original_build = logger_registry.build

    def patched_build(cfg, *args, **kwargs):
        cfg = dict(cfg)
        for _ in range(5):  # generous cap in case more than one field is bad
            try:
                return original_build(cfg, *args, **kwargs)
            except TypeError as e:
                m = re.search(r"unexpected keyword argument '([^']+)'", str(e))
                if not m or m.group(1) not in cfg:
                    raise
                bad_key = m.group(1)
                del cfg[bad_key]
                print(
                    f"⚠️  Dropping unsupported logger field '{bad_key}' — your "
                    f"installed DLC's WandbLogger constructor doesn't accept it "
                    f"even though its own config schema includes it. Retrying "
                    f"so W&B logging still goes through."
                )
        return original_build(cfg, *args, **kwargs)

    logger_registry.build = patched_build
    _wandb_logger_build_patched = True


def _strip_extra_forbidden_fields(pytorch_cfg_updates, validation_error):
    """DLC validates pytorch_cfg_updates against a pydantic schema that's
    tied to your installed DLC version — e.g. some versions' WandbLogger
    schema doesn't accept 'tags' even though the online docs say any
    wandb.init() kwarg is allowed. Rather than hard-fail on that mismatch,
    pull the offending field name(s) straight out of the ValidationError
    (pydantic reports them as "extra_forbidden") and drop just those keys.

    pydantic's error loc for a rejected field on a discriminated-union
    model (like `logger: WandbLogger`) includes the union arm's type name
    as a synthetic path segment — e.g. loc=("logger", "WandbLogger",
    "tags") — which isn't an actual key in the plain dict we built (there's
    no updates["logger"]["WandbLogger"]). So the walk below skips any loc
    segment that doesn't match a real dict key instead of giving up, and
    deletes the leaf field from wherever it actually landed.

    Returns (trimmed_dict, [dropped field paths]), or (None, []) if the
    error contained nothing we know how to strip (caller should re-raise).
    """
    updates = copy.deepcopy(pytorch_cfg_updates)
    dropped = []
    for err in validation_error.errors():
        if err.get("type") != "extra_forbidden":
            continue
        loc = err.get("loc", ())
        if not loc:
            continue
        leaf = loc[-1]
        node = updates
        for part in loc[:-1]:
            if isinstance(node, dict) and part in node and isinstance(node[part], dict):
                node = node[part]
            # else: this loc segment doesn't correspond to a real key
            # (e.g. a discriminated-union arm name) — stay at current node.
        if isinstance(node, dict) and leaf in node:
            del node[leaf]
            dropped.append(".".join(str(p) for p in loc))
    if not dropped:
        return None, []
    return updates, dropped


def train_network(
    project_config,
    shuffle=1,
    epochs=600,
    save_epochs=25,
    display_iters=500,
    batch_size=24,
    pytorch_cfg_updates=None,
    wandb_project=None,
    wandb_run_name=None,
    wandb_tags=None,
    wandb_image_log_interval=None,
    **kwargs,
):
    """Wraps deeplabcut.train_network(), pinned to Engine.PYTORCH.

    W&B logging (optional): pass wandb_project to have this run logged to
    Weights & Biases. Requires `pip install "deeplabcut[wandb]"` and having
    run `wandb login` once beforehand — see the package README for setup.

        net.train_network(epochs=600, wandb_project="mitten-tracker")

    wandb_run_name defaults to "<network-project-name>-shuffle<N>".
    wandb_tags: optional list of strings, e.g. ["resnet50", "split=0"].
    wandb_image_log_interval: optional int — if set, periodically logs a
    sample train/test image with predicted heatmaps to W&B.

    Leave wandb_project unset (the default) for no change in behavior —
    training logs only to the model folder, same as before this option
    existed. Any of these can also be set by hand via pytorch_cfg_updates
    (see the DLC pytorch_config.yaml `logger` reference) if you need more
    control than these kwargs expose.

    If your installed DLC's config schema rejects one of these fields (its
    pydantic model is stricter than the online docs — this happens, e.g.
    some versions reject the WandbLogger `tags` field), that field is
    dropped and training is retried automatically, with a warning printed.

    Separately, some DLC versions (confirmed: 3.0.1) have an internal bug
    where the WandbLogger *class* doesn't accept every field its own config
    *schema* claims to support (a `wandb_kwargs` TypeError at logger
    construction time, well after config validation passes). This can't be
    fixed by trimming our config — the field comes from the schema's own
    defaults — so instead DLC's logger-construction call is patched (see
    _patch_wandb_logger_build) to drop just that field and retry, so W&B
    logging still actually happens. If that patch doesn't apply cleanly to
    your installed DLC version, W&B logging is disabled for this run
    instead of crashing training, with a warning explaining why.

    Either way, training itself is never blocked by a broken W&B config.
    """
    pytorch_cfg_updates = _with_wandb_logger(
        pytorch_cfg_updates, project_config, shuffle,
        wandb_project, wandb_run_name, wandb_tags, wandb_image_log_interval,
    )
    if pytorch_cfg_updates.get("logger"):
        _patch_wandb_logger_build()
    train_kwargs = dict(
        shuffle=shuffle,
        epochs=epochs,
        save_epochs=save_epochs,
        display_iters=display_iters,
        batch_size=batch_size,
        engine=Engine.PYTORCH,
        **kwargs,
    )

    updates = pytorch_cfg_updates
    max_remediations = 3  # generous headroom for stacking fixes (e.g. tags + wandb_kwargs)
    for _ in range(max_remediations):
        try:
            return deeplabcut.train_network(
                str(project_config), pytorch_cfg_updates=updates, **train_kwargs
            )
        except ValidationError as e:
            trimmed, dropped = _strip_extra_forbidden_fields(updates, e)
            if trimmed is None:
                raise
            print(
                f"⚠️  Your installed DLC version's config schema doesn't support: "
                f"{', '.join(dropped)} — retrying without them."
            )
            updates = trimmed
        except TypeError as e:
            if not updates.get("logger") or "wandb" not in str(e).lower():
                raise
            print(
                f"⚠️  Your installed DeepLabCut has an internal bug constructing its W&B "
                f"logger ({e}) — disabling W&B logging for this run and continuing "
                f'without it. Run `pip install -U "deeplabcut[wandb]"` to fix this properly.'
            )
            updates = {k: v for k, v in updates.items() if k != "logger"}

    # Remediation attempts exhausted — one last try, letting any error propagate normally.
    return deeplabcut.train_network(str(project_config), pytorch_cfg_updates=updates, **train_kwargs)
    
def evaluate_network(project_config, shuffle=1, per_keypoint_evaluation=True, plotting=False, **kwargs):
    """Wraps deeplabcut.evaluate_network(), pinned to Engine.PYTORCH."""
    return deeplabcut.evaluate_network(
        str(project_config), Shuffles=[shuffle], engine=Engine.PYTORCH,
        per_keypoint_evaluation=per_keypoint_evaluation, plotting=plotting, **kwargs,
    )
