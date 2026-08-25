"""
project.py
──────────
Replaces the old Setup.py.

Setup.py used to hardcode BASE_DIR as "wherever this file's src/ folder's
parent is" and assume a `data/` folder sat right next to `src/`. That's fine
for a single throwaway script, but breaks completely once this becomes an
installable package: a package has no idea where any particular user's data
lives, and it shouldn't assume src/ and data/ are siblings at all.

Instead, a consumer of this package picks any directory to hold their data
project — local disk, external drive, wherever — and calls
init_data_project() on it. DataProject then exposes every label_store /
network_store / extraction_config / manifest operation as a method on
itself, so a usage script just does:

    import dlc_manager as dlm
    prj = dlm.init_data_project(r"/home/ccldlc/Desktop/DLC_project/")
    prj.extract_frames_for_video(r"/raw_videos/HDMI-A.mp4")
    prj.init_labeling_config("HDMI-A__default", scorer="Egor", bodyparts=[...], skeleton=[...])
    net = prj.create_network_project(name="mitten_tracker")
    net.add_labeled_data(prj.store / "labeled-data" / "HDMI-A__default")

The underlying module-level functions in label_store.py / network_store.py /
extraction_config.py / manifest.py still work exactly as before and are
still exported from the package — DataProject is a convenience layer on
top, not a replacement.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml

from . import label_store as label_store_module
from . import network_store as network_store_module
from . import extraction_config as extraction_config_module
from . import manifest as manifest_module


PROJECT_CONFIG_FILENAME = "project_config.yaml"


# ────────────────────────────────────────────────────────────────────────
# project_config.yaml — high-level metadata living at the project root
# (separate from frames_store/manifest.yaml and any network project's own
# config.yaml — this is the ONE file that's about the DataProject itself).
# ────────────────────────────────────────────────────────────────────────

def _project_config_path(root):
    return Path(root) / PROJECT_CONFIG_FILENAME


def _load_project_config(root):
    path = _project_config_path(root)
    if not path.exists():
        return None
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _save_project_config(root, cfg):
    with open(_project_config_path(root), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


@dataclass
class NetworkProject:
    """A handle onto one independent DLC project living under a
    DataProject's network_store/. Nothing here is stateful beyond the
    config_path — every method is a thin forward to network_store.py with
    that path already filled in, so you don't have to keep passing
    project_config around by hand once you've got the handle.
    """
    config_path: Path

    @property
    def name(self):
        return self.config_path.parent.name

    def add_labeled_data(self, source_folder, overwrite=False):
        return network_store_module.add_labeled_data(
            self.config_path, source_folder, overwrite=overwrite
        )

    def create_train_dataset(self, net_type="resnet_50", **kwargs):
        return network_store_module.create_train_dataset(
            self.config_path, net_type=net_type, **kwargs
        )

    def train_network(self, **kwargs):
        return network_store_module.train_network(self.config_path, **kwargs)

    def list_labeled_data(self):
        return network_store_module.list_labeled_data(self.config_path)

    def __repr__(self):
        return f"NetworkProject(name={self.name!r}, config_path={str(self.config_path)!r})"


@dataclass
class DataProject:
    """Resolved paths for one data project's on-disk layout, plus every
    label_store / network_store / extraction_config / manifest operation
    exposed as a method (bound to this project's store/network_store) so
    you never have to pass store_path around by hand:

        <root>/
            project_config.yaml   high-level metadata for this DataProject
                                   itself (created_at, free-form metadata).
            raw_videos/            your own raw video files — created as a
                                   place to drop them, but never written to
                                   or managed by this package.
            frames_store/          label_store territory: extracted +
                                   labeled frame sets.
            network_store/         network_store territory: independent DLC
                                   projects trained from copied labeled-data
                                   — each one accessed via a NetworkProject
                                   handle (see create_network_project /
                                   get_network_project).
    """
    root: Path
    raw_videos: Path
    store: Path
    network_store: Path

    # ---------------------------------------------------------------
    # project-level metadata (project_config.yaml at the root)
    # ---------------------------------------------------------------

    @property
    def metadata(self):
        """Read-only snapshot of this project's free-form metadata dict."""
        cfg = _load_project_config(self.root) or {}
        return dict(cfg.get("metadata", {}))

    def get_metadata(self, key, default=None):
        return self.metadata.get(key, default)

    def set_metadata(self, key, value):
        """Persist one metadata key onto project_config.yaml."""
        cfg = _load_project_config(self.root) or {}
        cfg.setdefault("metadata", {})[key] = value
        _save_project_config(self.root, cfg)

    # ---------------------------------------------------------------
    # frames_store: extraction configs + manifest (extraction_config.py, manifest.py)
    # ---------------------------------------------------------------

    def save_extraction_config(self, cfg, overwrite=False):
        return extraction_config_module.save_extraction_config(self.store, cfg, overwrite=overwrite)

    def load_extraction_config(self, name="default"):
        return extraction_config_module.load_extraction_config(self.store, name=name)

    def list_extraction_configs(self):
        return extraction_config_module.list_extraction_configs(self.store)

    def list_extractions(self):
        return manifest_module.list_extractions(self.store)

    def get_extraction(self, folder_id):
        return manifest_module.get_extraction(self.store, folder_id)

    # ---------------------------------------------------------------
    # frames_store: extraction + labeling (label_store.py)
    # ---------------------------------------------------------------

    def extract_frames_for_video(self, video_path, config_name="default", folder_name=None, overwrite=False):
        return label_store_module.extract_frames_for_video(
            self.store, video_path, config_name=config_name, folder_name=folder_name, overwrite=overwrite
        )

    def init_labeling_config(self, folder_id, scorer, bodyparts, skeleton, task=None, **overrides):
        return label_store_module.init_labeling_config(
            self.store, folder_id, scorer, bodyparts, skeleton, task=task, **overrides
        )

    def label_frames_for(self, folder_id, multiple=False):
        return label_store_module.label_frames_for(self.store, folder_id, multiple=multiple)

    def folder_id_for(self, video_path, config_name):
        return label_store_module.folder_id_for(video_path, config_name)

    def frame_set_path(self, folder_id):
        """Path to a frame set's own folder under frames_store/labeled-data/
        <folder_id> — e.g. to hand to a NetworkProject.add_labeled_data()."""
        return self.store / "labeled-data" / folder_id

    # ---------------------------------------------------------------
    # network_store: independent DLC projects (network_store.py)
    # ---------------------------------------------------------------

    def create_network_project(self, name=None, scorer="Egor", engine="pytorch", **overrides) -> NetworkProject:
        config_path = network_store_module.create_project(
            self.network_store, name=name, scorer=scorer, engine=engine, **overrides
        )
        return NetworkProject(config_path=config_path)

    def get_network_project(self, name) -> NetworkProject:
        """Reload a handle onto a network project created earlier (in this
        or an earlier session) by name."""
        config_path = self.network_store / name / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(
                f"No network project named '{name}' under {self.network_store}. "
                f"Known: {self.list_network_projects()}"
            )
        return NetworkProject(config_path=config_path)

    def list_network_projects(self):
        """Names of every network project under network_store/ so far."""
        if not self.network_store.is_dir():
            return []
        return sorted(
            p.name for p in self.network_store.iterdir() if (p / "config.yaml").exists()
        )


def init_data_project(root_dir, **metadata) -> DataProject:
    """Bootstrap (or just resolve, if it already exists) a data project's
    top-level folder layout under root_dir.

    Creates raw_videos/, frames_store/, and network_store/ if they don't
    already exist, and bootstraps frames_store/ (labeled-data/,
    extraction_configs/ with a 'default' preset, manifest.yaml) right away
    — no separate init_store() call needed. network_store/ project(s) are
    still created on demand, via prj.create_network_project().

    Also creates (or reads) project_config.yaml at the root, which holds
    this DataProject's own high-level metadata — a free-form dict, set via
    **metadata here or later via prj.set_metadata(key, value). Not required
    for anything else in this package to work; it's just a place to keep
    project-wide notes (e.g. species="cat", camera_rig="HDMI-A/B") without
    having to know the folder layout.

    Idempotent — safe to call on every run of a usage script. Any
    **metadata passed on a later call is merged into (not replacing)
    whatever's already saved.
    """
    root = Path(root_dir).resolve()
    raw_videos = root / "raw_videos"
    store = root / "frames_store"
    network_store = root / "network_store"

    for d in (raw_videos, store, network_store):
        d.mkdir(parents=True, exist_ok=True)

    label_store_module.init_store(store)

    cfg = _load_project_config(root)
    if cfg is None:
        cfg = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "metadata": dict(metadata),
        }
        _save_project_config(root, cfg)
    elif metadata:
        cfg.setdefault("metadata", {}).update(metadata)
        _save_project_config(root, cfg)

    return DataProject(
        root=root,
        raw_videos=raw_videos,
        store=store,
        network_store=network_store,
    )