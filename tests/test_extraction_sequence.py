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

Run from repo root with your DLC env active:
    pytest tests/test_extraction_sequence.py -v -s
"""

import shutil
import sys
from pathlib import Path

# tests/ -> repo root -> src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml
import pytest

from dlc_manager import init_data_project, ExtractionConfig
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