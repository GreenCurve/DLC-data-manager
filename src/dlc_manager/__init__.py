"""
dlc_manager
───────────
Installable package for managing a DeepLabCut-based pipeline:

    project.py            data-project bootstrap (raw_videos/, frames_store/,
                           network_store/ layout) — replaces the old, hardcoded
                           Setup.py.
    extraction_config.py  versioned ExtractionConfig presets.
    manifest.py           manifest.yaml: which video + config produced which
                           frame set.
    label_store.py        extraction + labeling, decoupled per frame set.
    network_store.py      independent DLC "network" projects trained from
                           copied labeled-data.

This package contains no project-specific values (no bodyparts, no video
names, no fixed folder layout beyond raw_videos/frames_store/network_store).
Anything specific to one actual project (e.g. the mitten tracker) belongs in
a usage script that imports this package — see the examples shipped
alongside your data project, not in here.

Every label_store/network_store/extraction_config/manifest operation below
is also available as a method on DataProject (and, for network projects,
on the NetworkProject handle returned by create_network_project /
get_network_project), so the common case is just:

    import dlc_manager as dlm
    prj = dlm.init_data_project(r"/home/ccldlc/Desktop/DLC_project/")
    prj.extract_frames_for_video(r"/raw_videos/HDMI-A.mp4")
    net = prj.create_network_project(name="mitten_tracker")
    net.add_labeled_data(prj.frame_set_path("HDMI-A__default"))

The free functions imported below still work standalone (store_path /
project_config passed explicitly) for anyone who prefers that, or needs to
operate on a store/project without a full DataProject around it.
"""

from .project import DataProject, NetworkProject, init_data_project

from .extraction_config import (
    ExtractionConfig,
    save_extraction_config,
    load_extraction_config,
    list_extraction_configs,
)

from .manifest import (
    list_extractions,
    get_extraction,
)

from .label_store import (
    init_store,
    extract_frames_for_video,
    init_labeling_config,
    label_frames_for,
    import_legacy_project,
    folder_id_for,
    VIDEO_EXTENSIONS,
)

from .network_store import (
    create_project,
    add_labeled_data,
    create_train_dataset,
    train_network,
    list_labeled_data,
)

__all__ = [
    "DataProject",
    "NetworkProject",
    "init_data_project",
    "ExtractionConfig",
    "save_extraction_config",
    "load_extraction_config",
    "list_extraction_configs",
    "list_extractions",
    "get_extraction",
    "init_store",
    "extract_frames_for_video",
    "init_labeling_config",
    "label_frames_for",
    "import_legacy_project",
    "folder_id_for",
    "VIDEO_EXTENSIONS",
    "create_project",
    "add_labeled_data",
    "create_train_dataset",
    "train_network",
    "list_labeled_data",
]