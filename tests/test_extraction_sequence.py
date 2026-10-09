"""
tests/test_extraction_sequence.py
──────────────────────────────────
Validates the extraction sequence through dlc_manager's project-centered
API — DataProject, returned by init_data_project() — rather than the
module-level label_store.py/extraction_config.py/manifest.py functions
directly:

    prj = init_data_project(root)                                   # bootstraps
                                                                      # frames_store/
                                                                      # (labeled-data/,
                                                                      # extraction_configs/
                                                                      # default.yaml,
                                                                      # manifest.yaml)
                                                                      # + raw_videos/,
                                                                      # network_store/,
                                                                      # project_config.yaml
    prj.save_extraction_config(ExtractionConfig(name="default", ...), overwrite=True)
    prj.extract_frames_for_video(VIDEO, config_name="default")        # not idempotent
    prj.extract_frames_for_video(VIDEO, config_name="default")        # -> new folder
    prj.save_extraction_config(ExtractionConfig(name="dense", ...))
    prj.extract_frames_for_video(VIDEO, config_name="dense")
    prj.list_extractions()

init_data_project() already bootstraps frames_store/ (labeled-data/,
extraction_configs/ with a "default" preset, manifest.yaml) as part of
building the root layout — there's no separate init_store() call needed
the way the old, non-project-centered flow required.

prj.extract_frames_for_video() is intentionally NOT idempotent: every call
creates a brand-new labeled-data/<folder_id> instance. Default calls
auto-increment ("HDMI-A__default", "HDMI-A__default__2", ...); an explicit
folder_name= pins the name and requires overwrite=True to replace it.

prj.extract_frames_for_video() also accepts a FOLDER instead of a single
video file: it recurses through the folder, finds every file matching
VIDEO_EXTENSIONS, and extracts each one individually (own auto-generated
folder_id per video), returning a dict of {video_path: config.yaml path}.
folder_name= is rejected outright when given a folder, since it can't apply
to more than one discovered video.

This runs REAL deeplabcut.extract_frames() (kmeans + uniform) against
VIDEO — it is a slow integration test, not a mock-based unit test.
Throwaway data projects are used (under pytest's tmp_path) so your real
data project on disk is never touched; only the raw video (read-only) is
reused from VIDEO (and copied into scratch folders for the folder-input
tests, since extraction needs distinct video files/stems).

The pure filesystem-naming helpers (_next_available_folder_id,
_find_videos_in_folder) aren't exposed on DataProject — they're internal
to label_store.py — so those few tests still import label_store directly.

The second half of the file covers NetworkProject (network_store.py), reached
through prj.create_network_project() / prj.get_network_project():

    net = prj.create_network_project(name="mitten_tracker")      # no video needed
    net.add_labeled_data(prj.frame_set_path("HDMI-A__default"))   # copies the frame set in
    net.list_labeled_data()

Those tests never label anything by hand: frame sets get their
CollectedData_<scorer>.h5/.csv from generate_synthetic_labels()
(dlc_manager/synthetic_labels.py), which writes the same structure napari
does. Creation / lookup / repeated-creation tests only need an empty throwaway
DataProject (no DLC at all); the add_labeled_data tests share one module-scoped
project with a handful of cheap frame sets (uniform extraction, few frames)
and give every test its own fresh network project, so they never interfere
with each other. No test ever mutates the shared frame sets — anything that
needs to change a source works on a copy under tmp_path.

Run from repo root with your DLC env active:
    pytest tests/test_extraction_sequence.py -v -s
"""

import shutil
import sys
from datetime import date
from pathlib import Path

# tests/ -> repo root -> src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import yaml
import pytest

from dlc_manager import init_data_project, ExtractionConfig
from dlc_manager import add_labeled_data as add_labeled_data_fn   # free-function twin of NetworkProject.add_labeled_data
from synthetic_labels import generate_synthetic_labels
from dlc_manager.label_store import (
    _next_available_folder_id,
    _find_videos_in_folder,
    VIDEO_EXTENSIONS,
)

# There's no Setup.py / BASE_DIR anymore — project.py's init_data_project()
# replaced it, and a data project's root is just whatever directory you
# point it at. This test only needs the repo root to find the real test
# video on disk.
REPO_ROOT = Path(__file__).resolve().parent.parent
VIDEO = REPO_ROOT / "data" / "raw_videos" / "SCENE Object 1 HDMI-A_Jul09_12-20-07_synced.mp4"
if not VIDEO.is_file():
    pytest.skip(
        f"Real test video not found at {VIDEO} — this is a slow integration "
        f"suite that runs actual deeplabcut.extract_frames() against it. "
        f"Place a video there to run these tests.",
        allow_module_level=True,
    )

# Derived from the real file on disk rather than hardcoded — the video
# placed at data/raw_videos/ may keep its original camera filename (e.g.
# "SCENE Object 1 HDMI-A_Jul09_12-20-07_synced.mp4") instead of being
# renamed to HDMI-A.mp4, and folder IDs are always built from that stem.
VIDEO_STEM = VIDEO.stem
DEFAULT_ID = f"{VIDEO_STEM}__default"
DEFAULT_ID_2 = f"{VIDEO_STEM}__default__2"
DENSE_ID = f"{VIDEO_STEM}__dense"
CUSTOM_ID = "custom_run"

# Mirrors the shape of a usage-script call against dlc_manager — kept in one
# place so the test fails loudly if that call and real usage ever drift.
SCORER = "Egor"
BODYPARTS = [
    "right_mitten_wrist", "right_mitten_tip",
    "left_mitten_wrist", "left_mitten_tip",
    "1_corner_table", "2_corner_table", "3_corner_table", "4_corner_table",
]
SKELETON = [
    ["right_mitten_wrist", "right_mitten_tip"],
    ["left_mitten_wrist", "left_mitten_tip"],
    ["1_corner_table", "2_corner_table"],
    ["2_corner_table", "3_corner_table"],
    ["3_corner_table", "4_corner_table"],
    ["4_corner_table", "1_corner_table"],
]


def _frames_dir(prj, folder_id, video_stem=VIDEO_STEM):
    return prj.frame_set_path(folder_id) / "labeled-data" / video_stem


def _project_config(prj, folder_id):
    return prj.frame_set_path(folder_id) / "config.yaml"


# ────────────────────────────────────────────────────────────────────────
# Pure filesystem-naming logic — no DLC involved, cheap. Internal to
# label_store.py (not exposed on DataProject), so imported directly.
# ────────────────────────────────────────────────────────────────────────

def test_next_available_folder_id_uses_base_when_free(tmp_path):
    assert _next_available_folder_id(tmp_path, "bar") == "bar"


def test_next_available_folder_id_skips_existing(tmp_path):
    (tmp_path / "labeled-data" / "foo").mkdir(parents=True)
    (tmp_path / "labeled-data" / "foo__2").mkdir(parents=True)
    assert _next_available_folder_id(tmp_path, "foo") == "foo__3"


def test_find_videos_in_folder_is_recursive_and_extension_filtered(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.mp4").write_bytes(b"")
    (tmp_path / "sub" / "b.MOV").write_bytes(b"")           # case-insensitive
    (tmp_path / "notes.txt").write_bytes(b"")                # ignored
    (tmp_path / "sub" / "c.mp4.bak").write_bytes(b"")        # ignored

    found = _find_videos_in_folder(tmp_path)
    names = sorted(p.name for p in found)
    assert names == ["a.mp4", "b.MOV"]


def test_video_extensions_are_lowercase():
    # extension matching lowercases the suffix, so entries must already be
    # lowercase or case-insensitive matching silently breaks.
    assert all(ext == ext.lower() for ext in VIDEO_EXTENSIONS)


# ────────────────────────────────────────────────────────────────────────
# Main sequence: init_data_project -> default -> default (again) -> dense
# Run once for the module (real DLC calls are slow); tests below only
# assert against the result.
# ────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def executed_project(tmp_path_factory):
    prj = init_data_project(tmp_path_factory.mktemp("data_project"))
    # init_data_project() seeds a "default" preset at its own defaults —
    # override it (project-centered: through the project, not the file)
    # to match this suite's expected frame counts.
    prj.save_extraction_config(ExtractionConfig(name="default", numframes2pick=50), overwrite=True)

    prj.extract_frames_for_video(VIDEO, config_name="default")
    # Same video, same config, called again -> must NOT reuse the first folder.
    prj.extract_frames_for_video(VIDEO, config_name="default")

    prj.save_extraction_config(ExtractionConfig(name="dense", numframes2pick=150, algo="uniform"))
    prj.extract_frames_for_video(VIDEO, config_name="dense")

    return prj


# ── init_data_project() ──

def test_project_scaffold_created(executed_project):
    prj = executed_project
    assert prj.raw_videos.is_dir()
    assert prj.network_store.is_dir()
    assert (prj.root / "project_config.yaml").is_file()

    assert prj.store.is_dir()
    assert (prj.store / "labeled-data").is_dir()
    assert (prj.store / "extraction_configs" / "default.yaml").is_file()
    assert (prj.store / "manifest.yaml").is_file()


def test_init_data_project_is_idempotent(executed_project, capsys):
    # Re-running init_data_project() on the same root must not disturb the
    # extraction config this suite already saved onto it.
    init_data_project(executed_project.root)
    captured = capsys.readouterr()
    assert "already initialized" in captured.out

    cfg = executed_project.load_extraction_config("default")
    assert cfg.numframes2pick == 50  # untouched by the second call


# ── first extract_frames_for_video("default") call ──

def test_default_extraction_layout_and_count(executed_project):
    project_config = _project_config(executed_project, DEFAULT_ID)
    frames_dir = _frames_dir(executed_project, DEFAULT_ID)

    assert project_config.is_file()
    assert frames_dir.is_dir()

    pngs = list(frames_dir.glob("*.png"))
    assert len(pngs) == 50, f"expected 50 frames in {frames_dir}, got {len(pngs)}"


def test_default_project_config_points_at_source_video(executed_project):
    cfg = yaml.safe_load(_project_config(executed_project, DEFAULT_ID).read_text())
    assert Path(list(cfg["video_sets"].keys())[0]) == Path(VIDEO).resolve()
    assert cfg["numframes2pick"] == 50


# ── second call, same video + same config -> auto-incremented folder ──

def test_repeated_default_call_creates_a_new_folder_not_a_reuse(executed_project):
    frames_dir_1 = _frames_dir(executed_project, DEFAULT_ID)
    frames_dir_2 = _frames_dir(executed_project, DEFAULT_ID_2)

    assert frames_dir_1.is_dir()
    assert frames_dir_2.is_dir()
    assert frames_dir_1 != frames_dir_2

    pngs_2 = list(frames_dir_2.glob("*.png"))
    assert len(pngs_2) == 50


def test_first_default_folder_untouched_by_the_second_call(executed_project):
    pngs = list(_frames_dir(executed_project, DEFAULT_ID).glob("*.png"))
    assert len(pngs) == 50


# ── dense preset ──

def test_dense_preset_saved(executed_project):
    cfg = executed_project.load_extraction_config("dense")
    assert cfg.numframes2pick == 150
    assert cfg.algo == "uniform"
    assert "dense" in executed_project.list_extraction_configs()


def test_dense_extraction_lands_in_its_own_folder(executed_project):
    frames_dir = _frames_dir(executed_project, DENSE_ID)
    assert frames_dir.is_dir()

    pngs = list(frames_dir.glob("*.png"))
    assert len(pngs) == 150, f"expected 150 frames in {frames_dir}, got {len(pngs)}"


# ── manifest ──

def test_manifest_has_all_three_entries(executed_project):
    manifest = executed_project.list_extractions()
    assert set(manifest.keys()) == {DEFAULT_ID, DEFAULT_ID_2, DENSE_ID}


@pytest.mark.parametrize(
    "folder_id, config_name, algo, frame_count",
    [
        (DEFAULT_ID, "default", "kmeans", 50),
        (DEFAULT_ID_2, "default", "kmeans", 50),
        (DENSE_ID, "dense", "uniform", 150),
    ],
)
def test_manifest_record_fields(executed_project, folder_id, config_name, algo, frame_count):
    record = executed_project.get_extraction(folder_id)
    assert record["config_name"] == config_name
    assert record["config"]["algo"] == algo
    assert record["frame_count"] == frame_count
    assert Path(record["video_path"]) == Path(VIDEO).resolve()
    assert record["video_stem"] == VIDEO_STEM
    assert Path(record["project_config"]) == _project_config(executed_project, folder_id)


# ── init_labeling_config() ──

@pytest.fixture(scope="module")
def labeled_project(executed_project):
    """Runs the real labeling-schema call once, against HDMI-A__default, so
    tests below only assert against the result."""
    executed_project.init_labeling_config(
        DEFAULT_ID,
        scorer=SCORER,
        bodyparts=BODYPARTS,
        skeleton=SKELETON,
    )
    return executed_project


def test_labeling_config_sets_scorer_and_bodyparts(labeled_project):
    cfg = yaml.safe_load(_project_config(labeled_project, DEFAULT_ID).read_text())
    assert cfg["scorer"] == SCORER
    # order matters — DLC keys CollectedData_<scorer>.h5 columns by this order
    assert cfg["bodyparts"] == BODYPARTS


def test_labeling_config_sets_skeleton_as_list_of_pairs(labeled_project):
    cfg = yaml.safe_load(_project_config(labeled_project, DEFAULT_ID).read_text())
    assert cfg["skeleton"] == SKELETON
    assert all(isinstance(pair, list) and len(pair) == 2 for pair in cfg["skeleton"])
    # every bodypart referenced in the skeleton must actually be a defined bodypart
    referenced = {name for pair in cfg["skeleton"] for name in pair}
    assert referenced <= set(cfg["bodyparts"])


def test_labeling_config_forces_single_animal(labeled_project):
    cfg = yaml.safe_load(_project_config(labeled_project, DEFAULT_ID).read_text())
    assert cfg["multianimalproject"] is False


def test_labeling_config_preserves_extraction_fields(labeled_project):
    """Adding scorer/bodyparts/skeleton must not clobber what extraction
    already wrote into this same config.yaml."""
    record = labeled_project.get_extraction(DEFAULT_ID)
    cfg = yaml.safe_load(_project_config(labeled_project, DEFAULT_ID).read_text())

    assert cfg["project_path"] == str(labeled_project.frame_set_path(DEFAULT_ID))
    assert Path(list(cfg["video_sets"].keys())[0]) == Path(VIDEO).resolve()
    assert cfg["numframes2pick"] == 50
    assert cfg["engine"] == record["config"]["engine"]

    # frames on disk are untouched by editing the config
    pngs = list(_frames_dir(labeled_project, DEFAULT_ID).glob("*.png"))
    assert len(pngs) == 50


def test_labeling_config_does_not_touch_other_folders(labeled_project):
    default_2_cfg = yaml.safe_load(_project_config(labeled_project, DEFAULT_ID_2).read_text())
    dense_cfg = yaml.safe_load(_project_config(labeled_project, DENSE_ID).read_text())

    for cfg in (default_2_cfg, dense_cfg):
        assert not cfg.get("bodyparts")
        assert not cfg.get("skeleton")
        assert cfg.get("scorer") in (None, "")


def test_labeling_config_rejects_unknown_folder(executed_project):
    with pytest.raises(FileNotFoundError):
        executed_project.init_labeling_config(
            "does-not-exist__default",
            scorer="Egor", bodyparts=["nose"], skeleton=[],
        )


# ────────────────────────────────────────────────────────────────────────
# folder_name= behavior — isolated project, minimal extra DLC calls
# ────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def folder_name_project(tmp_path_factory):
    prj = init_data_project(tmp_path_factory.mktemp("folder_name_project"))
    prj.save_extraction_config(ExtractionConfig(name="default", numframes2pick=20), overwrite=True)
    prj.extract_frames_for_video(VIDEO, config_name="default", folder_name=CUSTOM_ID)
    return prj


def test_explicit_folder_name_is_used_verbatim(folder_name_project):
    project_config = _project_config(folder_name_project, CUSTOM_ID)
    frames_dir = _frames_dir(folder_name_project, CUSTOM_ID)

    assert project_config.is_file()
    assert frames_dir.is_dir()

    record = folder_name_project.get_extraction(CUSTOM_ID)
    assert record["frame_count"] == len(list(frames_dir.glob("*.png")))
    assert record["frame_count"] > 0


def test_explicit_folder_name_collision_without_overwrite_raises(folder_name_project):
    # No overwrite -> must raise BEFORE touching DLC (folder stays as-is).
    before = {p.name: p.stat().st_mtime_ns for p in _frames_dir(folder_name_project, CUSTOM_ID).glob("*.png")}

    with pytest.raises(FileExistsError):
        folder_name_project.extract_frames_for_video(VIDEO, config_name="default", folder_name=CUSTOM_ID)

    after = {p.name: p.stat().st_mtime_ns for p in _frames_dir(folder_name_project, CUSTOM_ID).glob("*.png")}
    assert before == after  # untouched by the failed call


def test_explicit_folder_name_collision_with_overwrite_replaces(folder_name_project):
    folder_name_project.extract_frames_for_video(
        VIDEO, config_name="default", folder_name=CUSTOM_ID, overwrite=True
    )

    frames_dir = _frames_dir(folder_name_project, CUSTOM_ID)
    assert frames_dir.is_dir()
    pngs = list(frames_dir.glob("*.png"))
    assert len(pngs) == 20  # this project's default preset (numframes2pick=20)

    record = folder_name_project.get_extraction(CUSTOM_ID)
    assert record["frame_count"] == 20


# ────────────────────────────────────────────────────────────────────────
# Folder input — prj.extract_frames_for_video(<folder>, ...)
#
# extraction still needs real, distinct video files (cv2 has to open them
# and DLC has to run kmeans/uniform on each), so VIDEO is copied under
# different stems/locations rather than mocked. Two real videos is enough
# to prove recursion + per-video isolation without tripling the runtime of
# the "default" sequence above.
# ────────────────────────────────────────────────────────────────────────

TOP_VIDEO_STEM = "cam-top"
NESTED_VIDEO_STEM = "cam-nested"
TOP_ID = f"{TOP_VIDEO_STEM}__default"
NESTED_ID = f"{NESTED_VIDEO_STEM}__default"


@pytest.fixture(scope="module")
def raw_videos_folder(tmp_path_factory):
    """A folder with one video at the top level, one nested a level down,
    and a couple of non-video files that must be ignored."""
    root = tmp_path_factory.mktemp("raw_videos")
    src = Path(VIDEO)

    shutil.copy(src, root / f"{TOP_VIDEO_STEM}{src.suffix}")
    (root / "notes.txt").write_text("not a video, must be ignored")

    nested = root / "session_02"
    nested.mkdir()
    shutil.copy(src, nested / f"{NESTED_VIDEO_STEM}{src.suffix}")
    (nested / "README.md").write_text("also not a video")

    return root


@pytest.fixture(scope="module")
def folder_extraction_project(tmp_path_factory, raw_videos_folder):
    prj = init_data_project(tmp_path_factory.mktemp("folder_input_project"))
    prj.save_extraction_config(ExtractionConfig(name="default", numframes2pick=20), overwrite=True)
    results = prj.extract_frames_for_video(raw_videos_folder, config_name="default")
    return prj, results


def test_folder_input_extracts_every_video_recursively(folder_extraction_project):
    prj, _results = folder_extraction_project

    for folder_id, stem in [(TOP_ID, TOP_VIDEO_STEM), (NESTED_ID, NESTED_VIDEO_STEM)]:
        project_config = _project_config(prj, folder_id)
        frames_dir = _frames_dir(prj, folder_id, video_stem=stem)

        assert project_config.is_file(), f"missing config for {folder_id}"
        assert frames_dir.is_dir(), f"missing frames dir for {folder_id}"

        pngs = list(frames_dir.glob("*.png"))
        assert len(pngs) == 20, f"expected 20 frames for {folder_id}, got {len(pngs)}"


def test_folder_input_ignores_non_video_files(folder_extraction_project):
    prj, _results = folder_extraction_project
    manifest = prj.list_extractions()
    # exactly the two real videos — notes.txt/README.md never became entries
    assert set(manifest.keys()) == {TOP_ID, NESTED_ID}


def test_folder_input_returns_dict_keyed_by_resolved_video_path(folder_extraction_project, raw_videos_folder):
    prj, results = folder_extraction_project

    assert isinstance(results, dict)
    assert len(results) == 2

    top_video = (raw_videos_folder / f"{TOP_VIDEO_STEM}{Path(VIDEO).suffix}").resolve()
    nested_video = (raw_videos_folder / "session_02" / f"{NESTED_VIDEO_STEM}{Path(VIDEO).suffix}").resolve()

    assert set(results.keys()) == {str(top_video), str(nested_video)}
    assert results[str(top_video)] == _project_config(prj, TOP_ID)
    assert results[str(nested_video)] == _project_config(prj, NESTED_ID)


def test_folder_input_records_correct_source_video_per_entry(folder_extraction_project, raw_videos_folder):
    prj, _results = folder_extraction_project

    top_record = prj.get_extraction(TOP_ID)
    nested_record = prj.get_extraction(NESTED_ID)

    top_video = (raw_videos_folder / f"{TOP_VIDEO_STEM}{Path(VIDEO).suffix}").resolve()
    nested_video = (raw_videos_folder / "session_02" / f"{NESTED_VIDEO_STEM}{Path(VIDEO).suffix}").resolve()

    assert Path(top_record["video_path"]) == top_video
    assert top_record["video_stem"] == TOP_VIDEO_STEM
    assert Path(nested_record["video_path"]) == nested_video
    assert nested_record["video_stem"] == NESTED_VIDEO_STEM


def test_folder_input_rejects_explicit_folder_name(tmp_path_factory, raw_videos_folder):
    prj = init_data_project(tmp_path_factory.mktemp("folder_input_reject_project"))
    prj.save_extraction_config(ExtractionConfig(name="default", numframes2pick=20), overwrite=True)
    # Must raise before any DLC call — folder_name can't map to >1 video.
    with pytest.raises(ValueError):
        prj.extract_frames_for_video(raw_videos_folder, config_name="default", folder_name="whatever")
    assert prj.list_extractions() == {}


def test_folder_input_raises_when_no_videos_found(tmp_path_factory):
    prj = init_data_project(tmp_path_factory.mktemp("folder_input_empty_project"))
    empty_folder = tmp_path_factory.mktemp("empty_raw_videos")
    (empty_folder / "readme.txt").write_text("no videos in here")

    with pytest.raises(FileNotFoundError):
        prj.extract_frames_for_video(empty_folder, config_name="default")


def test_folder_input_matches_extensions_case_insensitively(tmp_path_factory, raw_videos_folder):
    """A second folder containing only an uppercase-extension copy of the
    source video must still be picked up."""
    prj = init_data_project(tmp_path_factory.mktemp("folder_input_uppercase_project"))
    prj.save_extraction_config(ExtractionConfig(name="default", numframes2pick=20), overwrite=True)
    root = tmp_path_factory.mktemp("raw_videos_uppercase")
    src = Path(VIDEO)
    uppercase_video = root / f"cam-upper{src.suffix.upper()}"
    shutil.copy(src, uppercase_video)

    results = prj.extract_frames_for_video(root, config_name="default")

    assert len(results) == 1
    folder_id = "cam-upper__default"
    assert _project_config(prj, folder_id).is_file()
    pngs = list(_frames_dir(prj, folder_id, video_stem="cam-upper").glob("*.png"))
    assert len(pngs) == 20

# ════════════════════════════════════════════════════════════════════════
# NetworkProject — network_store.py via prj.create_network_project() & co.
# ════════════════════════════════════════════════════════════════════════
#
# Shared helpers / constants for everything below.

NET_SCORER = "NetOwner"      # scorer a network project is created with — deliberately
                             # different from every labeler, so "first add adopts the
                             # labeler's scorer" is observable
LABELER_A = "Tamas"
LABELER_B = "Anna"
NET_FRAMES = 8               # frames per extraction in the shared network_env project

SET_A = "set_a"                      # labeled by LABELER_A, partially labeled (NaNs)
SET_B = "set_b"                      # labeled by LABELER_B, fully labeled, same schema as A
SET_SUBSET = "set_subset"            # labeled, but with fewer bodyparts than A
SET_REORDERED = "set_reordered"      # labeled, same bodyparts as A in a different order
SET_NO_LABELS = "set_no_labels"      # schema set (init_labeling_config) but no CollectedData file
SET_NO_SCHEMA = "set_no_schema"      # extracted only — no scorer/bodyparts at all
ALL_NET_SETS = [SET_A, SET_B, SET_SUBSET, SET_REORDERED, SET_NO_LABELS, SET_NO_SCHEMA]


def _cfg(path):
    return yaml.safe_load(Path(path).read_text())


def _net_dir(net):
    return net.config_path.parent


def _dest(net, folder_id):
    """Where add_labeled_data() puts a frame set inside a network project."""
    return _net_dir(net) / "labeled-data" / folder_id


def _src_frames(prj, folder_id):
    """The frame set's own labeled-data/<video_stem> folder in frames_store."""
    return prj.frame_set_path(folder_id) / "labeled-data" / VIDEO_STEM


def _png_names(folder):
    return sorted(p.name for p in Path(folder).glob("*.png"))


def _copy_frame_set(prj, folder_id, tmp_path):
    """Throwaway copy of a frame set laid out like a frames_store, so a test
    can mutate it (regenerate labels, add a stray folder, ...) without
    touching the shared one. Returns (store_root, frame_set_folder)."""
    store_root = tmp_path / "store_copy"
    copy = store_root / "labeled-data" / folder_id
    shutil.copytree(prj.frame_set_path(folder_id), copy)
    return store_root, copy


# ── fixtures ──

@pytest.fixture
def fresh_prj(tmp_path):
    """Empty throwaway DataProject — enough for creation/lookup tests, which
    never touch DLC."""
    return init_data_project(tmp_path / "fresh_project")


@pytest.fixture(scope="module")
def network_env(tmp_path_factory):
    """One DataProject holding every frame set the add_labeled_data tests
    need. Extraction is cheap on purpose (uniform, NET_FRAMES frames); labels
    come from generate_synthetic_labels() instead of napari."""
    prj = init_data_project(tmp_path_factory.mktemp("network_env"))
    prj.save_extraction_config(
        ExtractionConfig(name="default", algo="uniform", numframes2pick=NET_FRAMES), overwrite=True
    )
    for folder_id in ALL_NET_SETS:
        prj.extract_frames_for_video(VIDEO, config_name="default", folder_name=folder_id)

    prj.init_labeling_config(SET_A, scorer=LABELER_A, bodyparts=BODYPARTS, skeleton=SKELETON)
    generate_synthetic_labels(prj.store, SET_A, seed=1)

    prj.init_labeling_config(SET_B, scorer=LABELER_B, bodyparts=BODYPARTS, skeleton=SKELETON)
    generate_synthetic_labels(prj.store, SET_B, seed=2, n_labeled=None)   # every bodypart, no NaNs

    prj.init_labeling_config(SET_SUBSET, scorer=LABELER_A, bodyparts=BODYPARTS[:-1], skeleton=[])
    generate_synthetic_labels(prj.store, SET_SUBSET, seed=3)

    prj.init_labeling_config(SET_REORDERED, scorer=LABELER_A, bodyparts=list(reversed(BODYPARTS)), skeleton=SKELETON)
    generate_synthetic_labels(prj.store, SET_REORDERED, seed=4)

    prj.init_labeling_config(SET_NO_LABELS, scorer=LABELER_A, bodyparts=BODYPARTS, skeleton=SKELETON)
    # (deliberately no generate_synthetic_labels for SET_NO_LABELS / SET_NO_SCHEMA)
    return prj


@pytest.fixture
def net(network_env):
    """A brand-new, empty network project for ONE test."""
    return network_env.create_network_project(scorer=NET_SCORER)


# ────────────────────────────────────────────────────────────────────────
# generate_synthetic_labels() — the fixtures above rely on it, so check
# its output really has DLC's CollectedData layout
# ────────────────────────────────────────────────────────────────────────

def test_synthetic_labels_follow_dlc_collecteddata_layout(network_env):
    src = _src_frames(network_env, SET_A)
    h5 = src / f"CollectedData_{LABELER_A}.h5"
    assert h5.is_file()
    assert (src / f"CollectedData_{LABELER_A}.csv").is_file()

    df = pd.read_hdf(h5)
    assert list(df.columns.names) == ["scorer", "bodyparts", "coords"]
    assert set(df.columns.get_level_values("scorer")) == {LABELER_A}
    # bodyparts in config order, x then y for each
    assert list(dict.fromkeys(df.columns.get_level_values("bodyparts"))) == BODYPARTS
    assert list(df.columns.get_level_values("coords")) == ["x", "y"] * len(BODYPARTS)

    assert df.index.nlevels == 3
    assert set(df.index.get_level_values(0)) == {"labeled-data"}
    assert set(df.index.get_level_values(1)) == {VIDEO_STEM}
    assert list(df.index.get_level_values(2)) == _png_names(src)      # one row per frame, sorted
    assert (df.dtypes == "float64").all()


def test_synthetic_labels_leave_whole_bodyparts_unlabeled(network_env):
    df = pd.read_hdf(_src_frames(network_env, SET_A) / f"CollectedData_{LABELER_A}.h5")
    n_bp = len(BODYPARTS)

    nan = df.isna().to_numpy().reshape(len(df), n_bp, 2)
    assert (nan[..., 0] == nan[..., 1]).all(), "a bodypart must be NaN in both x and y or neither"
    labeled_per_frame = (~nan[..., 0]).sum(axis=1)
    assert labeled_per_frame.min() >= 1
    assert labeled_per_frame.max() < n_bp           # default mode really exercises missing labels
    assert df.isna().any().any()

    full = pd.read_hdf(_src_frames(network_env, SET_B) / f"CollectedData_{LABELER_B}.h5")
    assert not full.isna().any().any()               # n_labeled=None -> everything labeled


def test_synthetic_labels_refuse_to_overwrite_without_flag(network_env):
    # Raises before writing, so it's safe on the shared frame set.
    with pytest.raises(FileExistsError):
        generate_synthetic_labels(network_env.store, SET_A)


def test_synthetic_labels_require_a_labeling_schema(network_env):
    with pytest.raises(ValueError):
        generate_synthetic_labels(network_env.store, SET_NO_SCHEMA)
    assert not list(_src_frames(network_env, SET_NO_SCHEMA).glob("CollectedData_*"))


# ────────────────────────────────────────────────────────────────────────
# NetworkProject: creation
# ────────────────────────────────────────────────────────────────────────

def test_network_project_creation_scaffold(fresh_prj):
    net = fresh_prj.create_network_project(name="mitten_tracker", scorer="Egor")
    project_dir = fresh_prj.network_store / "mitten_tracker"

    assert net.name == "mitten_tracker"
    assert net.config_path == project_dir / "config.yaml"
    assert net.config_path.is_file()
    for sub in ("labeled-data", "training-datasets", "dlc-models-pytorch", "videos"):
        assert (project_dir / sub).is_dir(), f"missing {sub}/"
    assert list((project_dir / "videos").iterdir()) == []          # intentionally always empty
    assert (project_dir / "sources.yaml").is_file()

    cfg = _cfg(net.config_path)
    assert cfg["Task"] == "mitten_tracker"
    assert cfg["scorer"] == "Egor"
    assert cfg["engine"] == "pytorch"
    assert cfg["project_path"] == str(project_dir)
    assert cfg["multianimalproject"] is False
    # no schema until the first add_labeled_data(); video_sets stays empty for good
    assert not cfg["video_sets"]
    assert not cfg["bodyparts"]
    assert not cfg["skeleton"]


def test_network_project_creation_applies_config_overrides(fresh_prj):
    net = fresh_prj.create_network_project(name="tuned", pcutoff=0.9, batch_size=4)
    cfg = _cfg(net.config_path)
    assert cfg["pcutoff"] == 0.9
    assert cfg["batch_size"] == 4


def test_new_network_project_has_no_labeled_data(fresh_prj):
    net = fresh_prj.create_network_project(name="empty")
    assert net.list_labeled_data() == []
    assert list((_net_dir(net) / "labeled-data").iterdir()) == []


def test_auto_named_network_project_uses_scorer_and_date(fresh_prj):
    net = fresh_prj.create_network_project(scorer="Alice")
    assert net.name == f"network-Alice-{date.today().strftime('%b%d')}"
    assert net.config_path.is_file()


# ── repeated creation: same shape as test_repeated_default_call_creates_a_new_folder_not_a_reuse ──

def test_repeated_auto_named_creation_creates_a_new_project_not_a_reuse(fresh_prj):
    first = fresh_prj.create_network_project()
    mtime_before = first.config_path.stat().st_mtime_ns

    second = fresh_prj.create_network_project()
    third = fresh_prj.create_network_project()

    assert second.name == f"{first.name}_2"
    assert third.name == f"{first.name}_3"
    assert len({first.config_path, second.config_path, third.config_path}) == 3
    for n in (first, second, third):
        assert n.config_path.is_file()
        assert (_net_dir(n) / "sources.yaml").is_file()

    # the first project was not rewritten by the later calls
    assert first.config_path.stat().st_mtime_ns == mtime_before
    assert fresh_prj.list_network_projects() == sorted([first.name, second.name, third.name])


def test_auto_naming_skips_a_name_taken_by_an_explicit_project(fresh_prj):
    auto_base = f"network-Egor-{date.today().strftime('%b%d')}"
    explicit = fresh_prj.create_network_project(name=auto_base)
    auto = fresh_prj.create_network_project()          # default scorer is "Egor"

    assert explicit.name == auto_base                  # explicit name used verbatim
    assert auto.name == f"{auto_base}_2"


def test_explicit_network_project_name_collision_raises_and_leaves_existing_untouched(fresh_prj):
    first = fresh_prj.create_network_project(name="tracker", pcutoff=0.5)
    before = first.config_path.read_text()

    with pytest.raises(FileExistsError):
        fresh_prj.create_network_project(name="tracker", pcutoff=0.9)

    assert first.config_path.read_text() == before     # untouched by the failed call
    assert fresh_prj.list_network_projects() == ["tracker"]


# ────────────────────────────────────────────────────────────────────────
# NetworkProject: reading back (get_network_project / list_network_projects)
# ────────────────────────────────────────────────────────────────────────

def test_get_network_project_returns_an_equivalent_handle(fresh_prj):
    created = fresh_prj.create_network_project(name="reload_me")
    reloaded = fresh_prj.get_network_project("reload_me")

    assert reloaded == created
    assert reloaded.name == "reload_me"
    assert reloaded.config_path == created.config_path
    assert "reload_me" in repr(reloaded)


def test_get_network_project_unknown_name_raises(fresh_prj):
    fresh_prj.create_network_project(name="exists")
    with pytest.raises(FileNotFoundError, match="does_not_exist"):
        fresh_prj.get_network_project("does_not_exist")


def test_list_network_projects_is_sorted_and_ignores_non_projects(fresh_prj):
    assert fresh_prj.list_network_projects() == []

    fresh_prj.create_network_project(name="b_proj")
    fresh_prj.create_network_project(name="a_proj")
    (fresh_prj.network_store / "stray_folder").mkdir()                 # no config.yaml -> not a project
    (fresh_prj.network_store / "notes.txt").write_text("not a project")

    assert fresh_prj.list_network_projects() == ["a_proj", "b_proj"]


# ────────────────────────────────────────────────────────────────────────
# NetworkProject.add_labeled_data — first add, copy semantics
# ────────────────────────────────────────────────────────────────────────

def test_first_add_adopts_the_frame_sets_schema_and_copies_frames(network_env, net):
    assert _cfg(net.config_path)["scorer"] == NET_SCORER     # before: the scorer it was created with

    dest = net.add_labeled_data(network_env.frame_set_path(SET_A))

    assert dest == _dest(net, SET_A)
    cfg = _cfg(net.config_path)
    assert cfg["scorer"] == LABELER_A            # adopted from the frame set, not NET_SCORER
    assert cfg["bodyparts"] == BODYPARTS
    assert cfg["skeleton"] == SKELETON
    assert cfg["multianimalproject"] is False
    assert not cfg["video_sets"]                 # stays empty after adding data

    # frames land directly in labeled-data/<folder_id>/ (no extra <video_stem> level)
    src_pngs = _png_names(_src_frames(network_env, SET_A))
    assert len(src_pngs) > 0
    assert _png_names(dest) == src_pngs
    assert (dest / f"CollectedData_{LABELER_A}.h5").is_file()
    assert not list(dest.glob("*.csv"))          # stale csv sidecars are dropped


def test_add_records_provenance_in_list_labeled_data(network_env, net):
    net.add_labeled_data(network_env.frame_set_path(SET_A))

    entries = net.list_labeled_data()
    assert len(entries) == 1
    entry = entries[0]
    assert entry["folder_id"] == SET_A
    assert entry["video_stem"] == VIDEO_STEM
    assert Path(entry["source_folder"]) == network_env.frame_set_path(SET_A).resolve()
    assert entry["source_scorer"] == LABELER_A
    assert entry["frame_count"] == len(_png_names(_src_frames(network_env, SET_A)))
    assert entry["copied_at"]


def test_add_rewrites_annotation_index_to_the_new_folder_and_keeps_values(network_env, net):
    dest = net.add_labeled_data(network_env.frame_set_path(SET_A))

    src_df = pd.read_hdf(_src_frames(network_env, SET_A) / f"CollectedData_{LABELER_A}.h5")
    dst_df = pd.read_hdf(dest / f"CollectedData_{LABELER_A}.h5")

    assert src_df.isna().any().any()                         # synthetic data really has gaps to carry over
    # index level 1 (the baked-in folder name) now points at the network project's folder_id ...
    assert set(dst_df.index.get_level_values(1)) == {SET_A}
    assert set(src_df.index.get_level_values(1)) == {VIDEO_STEM}   # ... while the source is untouched
    assert set(dst_df.index.get_level_values(0)) == {"labeled-data"}
    assert list(dst_df.index.get_level_values(2)) == list(src_df.index.get_level_values(2))
    # every annotated row refers to a PNG that actually exists in the copy
    for name in dst_df.index.get_level_values(2):
        assert (dest / name).is_file()
    # labels themselves are byte-for-byte the same, NaNs included
    assert dst_df.columns.equals(src_df.columns)
    np.testing.assert_array_equal(dst_df.to_numpy(), src_df.to_numpy())


def test_add_normalizes_scorer_to_the_project_scorer(network_env, net):
    net.add_labeled_data(network_env.frame_set_path(SET_A))        # project scorer becomes LABELER_A
    dest_b = net.add_labeled_data(network_env.frame_set_path(SET_B))   # labeled by LABELER_B

    assert (dest_b / f"CollectedData_{LABELER_A}.h5").is_file()
    assert not (dest_b / f"CollectedData_{LABELER_B}.h5").exists()
    assert not list(dest_b.glob("*.csv"))

    df = pd.read_hdf(dest_b / f"CollectedData_{LABELER_A}.h5")
    assert set(df.columns.get_level_values("scorer")) == {LABELER_A}
    assert set(df.index.get_level_values(1)) == {SET_B}
    assert not df.isna().any().any()                               # fully labeled set stays fully labeled

    by_id = {e["folder_id"]: e for e in net.list_labeled_data()}
    assert by_id[SET_B]["source_scorer"] == LABELER_B              # labeler provenance kept

    # the source frame set still has its ORIGINAL file names / scorer / index
    src = _src_frames(network_env, SET_B)
    assert (src / f"CollectedData_{LABELER_B}.h5").is_file()
    assert (src / f"CollectedData_{LABELER_B}.csv").is_file()
    assert not list(src.glob(f"CollectedData_{LABELER_A}.*"))
    src_df = pd.read_hdf(src / f"CollectedData_{LABELER_B}.h5")
    assert set(src_df.columns.get_level_values("scorer")) == {LABELER_B}
    assert set(src_df.index.get_level_values(1)) == {VIDEO_STEM}


def test_added_data_is_an_independent_copy_not_a_link(network_env, net):
    other = network_env.create_network_project(scorer=NET_SCORER)
    dest_1 = net.add_labeled_data(network_env.frame_set_path(SET_A))
    dest_2 = other.add_labeled_data(network_env.frame_set_path(SET_A))

    victim = _png_names(dest_1)[0]
    assert not dest_1.is_symlink() and not (dest_1 / victim).is_symlink()
    (dest_1 / victim).unlink()

    assert (dest_2 / victim).is_file()                               # other network project unaffected
    assert (_src_frames(network_env, SET_A) / victim).is_file()      # so is the frame set itself


# ────────────────────────────────────────────────────────────────────────
# NetworkProject.add_labeled_data — several frame sets, different call styles
# ────────────────────────────────────────────────────────────────────────

def test_add_several_frame_sets_into_one_project(network_env, net):
    net.add_labeled_data(network_env.frame_set_path(SET_A))          # Path
    net.add_labeled_data(str(network_env.frame_set_path(SET_B)))     # plain str

    assert [e["folder_id"] for e in net.list_labeled_data()] == [SET_A, SET_B]
    assert sorted(p.name for p in (_net_dir(net) / "labeled-data").iterdir()) == [SET_A, SET_B]
    cfg = _cfg(net.config_path)
    assert cfg["bodyparts"] == BODYPARTS                              # schema unchanged by the 2nd add
    assert cfg["scorer"] == LABELER_A

    # both frame sets come from the SAME video, yet nothing collides: one annotation
    # table per folder_id, and together their row keys are unique (what DLC's merge needs)
    frames = [
        pd.read_hdf(_dest(net, s) / f"CollectedData_{LABELER_A}.h5") for s in (SET_A, SET_B)
    ]
    combined = pd.concat(frames)
    assert combined.index.is_unique
    assert len(combined) == sum(len(_png_names(_dest(net, s))) for s in (SET_A, SET_B))
    assert list(combined.columns) == list(frames[0].columns)


def test_reloaded_handle_sees_the_added_data(network_env, net):
    net.add_labeled_data(network_env.frame_set_path(SET_A))
    net.add_labeled_data(network_env.frame_set_path(SET_B))

    reloaded = network_env.get_network_project(net.name)
    assert reloaded.list_labeled_data() == net.list_labeled_data()
    assert [e["folder_id"] for e in reloaded.list_labeled_data()] == [SET_A, SET_B]


def test_module_level_add_labeled_data_matches_the_handle_method(network_env, net):
    other = network_env.create_network_project(scorer=NET_SCORER)

    add_labeled_data_fn(net.config_path, str(network_env.frame_set_path(SET_A)))
    other.add_labeled_data(network_env.frame_set_path(SET_A))

    assert _cfg(net.config_path)["bodyparts"] == _cfg(other.config_path)["bodyparts"]
    assert _cfg(net.config_path)["scorer"] == _cfg(other.config_path)["scorer"]
    assert sorted(p.name for p in _dest(net, SET_A).iterdir()) == sorted(p.name for p in _dest(other, SET_A).iterdir())
    np.testing.assert_array_equal(
        pd.read_hdf(_dest(net, SET_A) / f"CollectedData_{LABELER_A}.h5").to_numpy(),
        pd.read_hdf(_dest(other, SET_A) / f"CollectedData_{LABELER_A}.h5").to_numpy(),
    )


def test_add_frame_set_with_schema_but_no_labels_copies_frames_only(network_env, net):
    dest = net.add_labeled_data(network_env.frame_set_path(SET_NO_LABELS))

    assert _png_names(dest) == _png_names(_src_frames(network_env, SET_NO_LABELS))
    assert not list(dest.glob("CollectedData_*"))                    # nothing labeled yet -> no annotation file
    assert _cfg(net.config_path)["bodyparts"] == BODYPARTS           # schema still adopted
    assert [e["folder_id"] for e in net.list_labeled_data()] == [SET_NO_LABELS]
    assert net.list_labeled_data()[0]["frame_count"] == len(_png_names(dest))


def test_frame_set_training_usage_reflects_add_labeled_data(network_env, net):
    other = network_env.create_network_project(scorer=NET_SCORER)
    assert net.name not in network_env.frame_set_training_usage(SET_A)

    net.add_labeled_data(network_env.frame_set_path(SET_A))

    used_by = network_env.frame_set_training_usage(SET_A)
    assert net.name in used_by
    assert other.name not in used_by                                  # never got SET_A
    assert net.name not in network_env.frame_set_training_usage(SET_B)


# ────────────────────────────────────────────────────────────────────────
# NetworkProject.add_labeled_data — duplicates / overwrite
# ────────────────────────────────────────────────────────────────────────

def test_adding_the_same_frame_set_twice_without_overwrite_raises(network_env, net):
    dest = net.add_labeled_data(network_env.frame_set_path(SET_A))
    files_before = {p.name: p.stat().st_mtime_ns for p in dest.iterdir()}
    entries_before = net.list_labeled_data()

    with pytest.raises(FileExistsError):
        net.add_labeled_data(network_env.frame_set_path(SET_A))

    assert {p.name: p.stat().st_mtime_ns for p in dest.iterdir()} == files_before   # untouched
    assert net.list_labeled_data() == entries_before                                # no duplicate record


def test_add_with_overwrite_replaces_the_copy_and_does_not_duplicate_the_record(network_env, net, tmp_path):
    # Work on a throwaway copy of SET_A (same folder name -> same folder_id) so the
    # shared frame set keeps its original labels.
    store_root, source = _copy_frame_set(network_env, SET_A, tmp_path)
    dest = net.add_labeled_data(source)
    h5 = dest / f"CollectedData_{LABELER_A}.h5"
    old_df = pd.read_hdf(h5)
    (dest / "stale.txt").write_text("left over from the first add")

    # relabel the source with different synthetic data, then re-add
    generate_synthetic_labels(store_root, SET_A, seed=99, overwrite=True)
    new_src_df = pd.read_hdf(source / "labeled-data" / VIDEO_STEM / f"CollectedData_{LABELER_A}.h5")
    assert not new_src_df.equals(old_df.set_axis(new_src_df.index, axis=0))   # different seed -> different labels

    with pytest.raises(FileExistsError):
        net.add_labeled_data(source)
    net.add_labeled_data(source, overwrite=True)

    new_df = pd.read_hdf(h5)
    np.testing.assert_array_equal(new_df.to_numpy(), new_src_df.to_numpy())
    assert not new_df.equals(old_df)
    assert not (dest / "stale.txt").exists()                          # folder was replaced, not merged into

    entries = net.list_labeled_data()
    assert [e["folder_id"] for e in entries] == [SET_A]               # replaced, not appended
    assert entries[0]["frame_count"] == len(_png_names(dest))


# ────────────────────────────────────────────────────────────────────────
# NetworkProject.add_labeled_data — rejected inputs
# ────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("bad_set", [SET_SUBSET, SET_REORDERED])
def test_add_with_different_bodyparts_or_order_is_rejected(network_env, net, bad_set):
    net.add_labeled_data(network_env.frame_set_path(SET_A))
    entries_before = net.list_labeled_data()

    with pytest.raises(ValueError, match="Bodypart mismatch"):
        net.add_labeled_data(network_env.frame_set_path(bad_set))

    assert not _dest(net, bad_set).exists()                           # nothing copied
    assert net.list_labeled_data() == entries_before
    assert _cfg(net.config_path)["bodyparts"] == BODYPARTS            # schema unchanged


def test_first_add_defines_the_schema_every_later_add_must_match(network_env, net):
    net.add_labeled_data(network_env.frame_set_path(SET_SUBSET))
    cfg = _cfg(net.config_path)
    assert cfg["bodyparts"] == BODYPARTS[:-1]
    assert not cfg["skeleton"]

    with pytest.raises(ValueError, match="Bodypart mismatch"):
        net.add_labeled_data(network_env.frame_set_path(SET_A))       # full list != the 9 adopted

    assert [e["folder_id"] for e in net.list_labeled_data()] == [SET_SUBSET]


def test_add_frame_set_without_bodyparts_is_rejected(network_env, net):
    with pytest.raises(ValueError, match="no bodyparts"):
        net.add_labeled_data(network_env.frame_set_path(SET_NO_SCHEMA))

    assert not _dest(net, SET_NO_SCHEMA).exists()
    assert not _cfg(net.config_path)["bodyparts"]                     # project did not adopt an empty schema
    assert net.list_labeled_data() == []


def test_add_folder_that_is_not_a_frame_set_is_rejected(net, tmp_path):
    not_a_frame_set = tmp_path / "not_a_frame_set"
    not_a_frame_set.mkdir()

    with pytest.raises(FileNotFoundError):
        net.add_labeled_data(not_a_frame_set)
    assert net.list_labeled_data() == []


def test_add_frame_set_with_several_video_folders_is_rejected(network_env, net, tmp_path):
    _store_root, source = _copy_frame_set(network_env, SET_A, tmp_path)
    (source / "labeled-data" / "another_video").mkdir()               # now ambiguous: which one holds the frames?

    with pytest.raises(RuntimeError):
        net.add_labeled_data(source)

    assert not _dest(net, SET_A).exists()
    assert net.list_labeled_data() == []
