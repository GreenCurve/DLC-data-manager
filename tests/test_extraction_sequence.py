"""
tests/test_extraction_sequence.py
──────────────────────────────────
Validates the sequence from src/main.py against the current label_store.py:

    init_store(...)
    extract_frames_for_video(..., config_name="default")   # not idempotent
    save_extraction_config(..., ExtractionConfig(name="dense", ...))
    extract_frames_for_video(..., config_name="dense")
    list_extractions(...)

extract_frames_for_video() is intentionally NOT idempotent anymore: every
call creates a brand-new labeled-data/<folder_id> instance. Default calls
auto-increment ("HDMI-A__default", "HDMI-A__default__2", ...); an explicit
folder_name= pins the name and requires overwrite=True to replace it.

extract_frames_for_video() also accepts a FOLDER instead of a single video
file: it recurses through the folder, finds every file matching
VIDEO_EXTENSIONS, and extracts each one individually (own auto-generated
folder_id per video), returning a dict of {video_path: config.yaml path}.
folder_name= is rejected outright when given a folder, since it can't apply
to more than one discovered video.

This runs REAL deeplabcut.extract_frames() (kmeans + uniform) against
Setup.VIDEO — it is a slow integration test, not a mock-based unit test.
Two throwaway stores are used (under pytest's tmp_path) so your real
frames_store on disk is never touched; only the raw video (read-only) is
reused from Setup.VIDEO (and copied into scratch folders for the
folder-input tests, since extraction needs distinct video files/stems).

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

import Setup
from label_store import (
    init_store,
    extract_frames_for_video,
    init_labeling_config,
    _next_available_folder_id,
    _find_videos_in_folder,
    VIDEO_EXTENSIONS,
)
from Extraction_config_Module import ExtractionConfig, save_extraction_config
from Manifest_Module import list_extractions, get_extraction


VIDEO_STEM = "HDMI-A"
DEFAULT_ID = f"{VIDEO_STEM}__default"
DEFAULT_ID_2 = f"{VIDEO_STEM}__default__2"
DENSE_ID = f"{VIDEO_STEM}__dense"
CUSTOM_ID = "custom_run"

# Mirrors the exact call now live in src/main.py — kept in one place so the
# test fails loudly if that call and this test ever drift apart.
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


def _frames_dir(store_path, folder_id, video_stem=VIDEO_STEM):
    return store_path / "labeled-data" / folder_id / "labeled-data" / video_stem


def _project_config(store_path, folder_id):
    return store_path / "labeled-data" / folder_id / "config.yaml"


# ────────────────────────────────────────────────────────────────────────
# Pure filesystem-naming logic — no DLC involved, cheap
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
# Main sequence: init_store -> default -> default (again) -> dense
# Run once for the module (real DLC calls are slow); tests below only
# assert against the result.
# ────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def executed_store(tmp_path_factory):
    store_path = init_store(tmp_path_factory.mktemp("frames_store"), numframes2pick=50)

    extract_frames_for_video(store_path, Setup.VIDEO, config_name="default")
    # Same video, same config, called again -> must NOT reuse the first folder.
    extract_frames_for_video(store_path, Setup.VIDEO, config_name="default")

    save_extraction_config(store_path, ExtractionConfig(name="dense", numframes2pick=150, algo="uniform"))
    extract_frames_for_video(store_path, Setup.VIDEO, config_name="dense")

    return store_path


# ── init_store() ──

def test_store_scaffold_created(executed_store):
    assert (executed_store / "labeled-data").is_dir()
    assert (executed_store / "extraction_configs" / "default.yaml").is_file()
    assert (executed_store / "manifest.yaml").is_file()


def test_init_store_is_idempotent(executed_store, capsys):
    init_store(executed_store, numframes2pick=999)
    captured = capsys.readouterr()
    assert "already initialized" in captured.out

    cfg = yaml.safe_load((executed_store / "extraction_configs" / "default.yaml").read_text())
    assert cfg["numframes2pick"] == 50  # untouched by the second call


# ── first extract_frames_for_video("default") call ──

def test_default_extraction_layout_and_count(executed_store):
    project_config = _project_config(executed_store, DEFAULT_ID)
    frames_dir = _frames_dir(executed_store, DEFAULT_ID)

    assert project_config.is_file()
    assert frames_dir.is_dir()

    pngs = list(frames_dir.glob("*.png"))
    assert len(pngs) == 50, f"expected 50 frames in {frames_dir}, got {len(pngs)}"


def test_default_project_config_points_at_source_video(executed_store):
    cfg = yaml.safe_load(_project_config(executed_store, DEFAULT_ID).read_text())
    assert Path(list(cfg["video_sets"].keys())[0]) == Path(Setup.VIDEO).resolve()
    assert cfg["numframes2pick"] == 50


# ── second call, same video + same config -> auto-incremented folder ──

def test_repeated_default_call_creates_a_new_folder_not_a_reuse(executed_store):
    frames_dir_1 = _frames_dir(executed_store, DEFAULT_ID)
    frames_dir_2 = _frames_dir(executed_store, DEFAULT_ID_2)

    assert frames_dir_1.is_dir()
    assert frames_dir_2.is_dir()
    assert frames_dir_1 != frames_dir_2

    pngs_2 = list(frames_dir_2.glob("*.png"))
    assert len(pngs_2) == 50


def test_first_default_folder_untouched_by_the_second_call(executed_store):
    pngs = list(_frames_dir(executed_store, DEFAULT_ID).glob("*.png"))
    assert len(pngs) == 50


# ── dense preset ──

def test_dense_preset_saved(executed_store):
    cfg = yaml.safe_load((executed_store / "extraction_configs" / "dense.yaml").read_text())
    assert cfg["numframes2pick"] == 150
    assert cfg["algo"] == "uniform"


def test_dense_extraction_lands_in_its_own_folder(executed_store):
    frames_dir = _frames_dir(executed_store, DENSE_ID)
    assert frames_dir.is_dir()

    pngs = list(frames_dir.glob("*.png"))
    assert len(pngs) == 150, f"expected 150 frames in {frames_dir}, got {len(pngs)}"


# ── manifest ──

def test_manifest_has_all_three_entries(executed_store):
    manifest = list_extractions(executed_store)
    assert set(manifest.keys()) == {DEFAULT_ID, DEFAULT_ID_2, DENSE_ID}


@pytest.mark.parametrize(
    "folder_id, config_name, algo, frame_count",
    [
        (DEFAULT_ID, "default", "kmeans", 50),
        (DEFAULT_ID_2, "default", "kmeans", 50),
        (DENSE_ID, "dense", "uniform", 150),
    ],
)
def test_manifest_record_fields(executed_store, folder_id, config_name, algo, frame_count):
    record = get_extraction(executed_store, folder_id)
    assert record["config_name"] == config_name
    assert record["config"]["algo"] == algo
    assert record["frame_count"] == frame_count
    assert Path(record["video_path"]) == Path(Setup.VIDEO).resolve()
    assert record["video_stem"] == VIDEO_STEM
    assert Path(record["project_config"]) == _project_config(executed_store, folder_id)


# ── init_labeling_config() — the exact call now live in src/main.py ──

@pytest.fixture(scope="module")
def labeled_store(executed_store):
    """Runs the real labeling-schema call from main.py once, against
    HDMI-A__default, so tests below only assert against the result."""
    init_labeling_config(
        executed_store, DEFAULT_ID,
        scorer=SCORER,
        bodyparts=BODYPARTS,
        skeleton=SKELETON,
    )
    return executed_store


def test_labeling_config_sets_scorer_and_bodyparts(labeled_store):
    cfg = yaml.safe_load(_project_config(labeled_store, DEFAULT_ID).read_text())
    assert cfg["scorer"] == SCORER
    # order matters — DLC keys CollectedData_<scorer>.h5 columns by this order
    assert cfg["bodyparts"] == BODYPARTS


def test_labeling_config_sets_skeleton_as_list_of_pairs(labeled_store):
    cfg = yaml.safe_load(_project_config(labeled_store, DEFAULT_ID).read_text())
    assert cfg["skeleton"] == SKELETON
    assert all(isinstance(pair, list) and len(pair) == 2 for pair in cfg["skeleton"])
    # every bodypart referenced in the skeleton must actually be a defined bodypart
    referenced = {name for pair in cfg["skeleton"] for name in pair}
    assert referenced <= set(cfg["bodyparts"])


def test_labeling_config_forces_single_animal(labeled_store):
    cfg = yaml.safe_load(_project_config(labeled_store, DEFAULT_ID).read_text())
    assert cfg["multianimalproject"] is False


def test_labeling_config_preserves_extraction_fields(labeled_store):
    """Adding scorer/bodyparts/skeleton must not clobber what extraction
    already wrote into this same config.yaml."""
    record = get_extraction(labeled_store, DEFAULT_ID)
    cfg = yaml.safe_load(_project_config(labeled_store, DEFAULT_ID).read_text())

    assert cfg["project_path"] == str(labeled_store / "labeled-data" / DEFAULT_ID)
    assert Path(list(cfg["video_sets"].keys())[0]) == Path(Setup.VIDEO).resolve()
    assert cfg["numframes2pick"] == 50
    assert cfg["engine"] == record["config"]["engine"]

    # frames on disk are untouched by editing the config
    pngs = list(_frames_dir(labeled_store, DEFAULT_ID).glob("*.png"))
    assert len(pngs) == 50


def test_labeling_config_does_not_touch_other_folders(labeled_store):
    default_2_cfg = yaml.safe_load(_project_config(labeled_store, DEFAULT_ID_2).read_text())
    dense_cfg = yaml.safe_load(_project_config(labeled_store, DENSE_ID).read_text())

    for cfg in (default_2_cfg, dense_cfg):
        assert not cfg.get("bodyparts")
        assert not cfg.get("skeleton")
        assert cfg.get("scorer") in (None, "")


def test_labeling_config_rejects_unknown_folder(executed_store):
    with pytest.raises(FileNotFoundError):
        init_labeling_config(
            executed_store, "does-not-exist__default",
            scorer="Egor", bodyparts=["nose"], skeleton=[],
        )


# ────────────────────────────────────────────────────────────────────────
# folder_name= behavior — isolated store, minimal extra DLC calls
# ────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def folder_name_store(tmp_path_factory):
    store_path = init_store(tmp_path_factory.mktemp("folder_name_store"), numframes2pick=20)
    extract_frames_for_video(store_path, Setup.VIDEO, config_name="default", folder_name=CUSTOM_ID)
    return store_path


def test_explicit_folder_name_is_used_verbatim(folder_name_store):
    project_config = _project_config(folder_name_store, CUSTOM_ID)
    frames_dir = _frames_dir(folder_name_store, CUSTOM_ID)

    assert project_config.is_file()
    assert frames_dir.is_dir()

    record = get_extraction(folder_name_store, CUSTOM_ID)
    assert record["frame_count"] == len(list(frames_dir.glob("*.png")))
    assert record["frame_count"] > 0


def test_explicit_folder_name_collision_without_overwrite_raises(folder_name_store):
    # No overwrite -> must raise BEFORE touching DLC (folder stays as-is).
    before = {p.name: p.stat().st_mtime_ns for p in _frames_dir(folder_name_store, CUSTOM_ID).glob("*.png")}

    with pytest.raises(FileExistsError):
        extract_frames_for_video(folder_name_store, Setup.VIDEO, config_name="default", folder_name=CUSTOM_ID)

    after = {p.name: p.stat().st_mtime_ns for p in _frames_dir(folder_name_store, CUSTOM_ID).glob("*.png")}
    assert before == after  # untouched by the failed call


def test_explicit_folder_name_collision_with_overwrite_replaces(folder_name_store):
    extract_frames_for_video(
        folder_name_store, Setup.VIDEO, config_name="default", folder_name=CUSTOM_ID, overwrite=True
    )

    frames_dir = _frames_dir(folder_name_store, CUSTOM_ID)
    assert frames_dir.is_dir()
    pngs = list(frames_dir.glob("*.png"))
    assert len(pngs) == 20  # this store's default preset (numframes2pick=20)

    record = get_extraction(folder_name_store, CUSTOM_ID)
    assert record["frame_count"] == 20


# ────────────────────────────────────────────────────────────────────────
# Folder input — extract_frames_for_video(store_path, <folder>, ...)
#
# extraction still needs real, distinct video files (cv2 has to open them
# and DLC has to run kmeans/uniform on each), so Setup.VIDEO is copied under
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
    src = Path(Setup.VIDEO)

    shutil.copy(src, root / f"{TOP_VIDEO_STEM}{src.suffix}")
    (root / "notes.txt").write_text("not a video, must be ignored")

    nested = root / "session_02"
    nested.mkdir()
    shutil.copy(src, nested / f"{NESTED_VIDEO_STEM}{src.suffix}")
    (nested / "README.md").write_text("also not a video")

    return root


@pytest.fixture(scope="module")
def folder_extraction_store(tmp_path_factory, raw_videos_folder):
    store_path = init_store(tmp_path_factory.mktemp("folder_input_store"), numframes2pick=20)
    results = extract_frames_for_video(store_path, raw_videos_folder, config_name="default")
    return store_path, results


def test_folder_input_extracts_every_video_recursively(folder_extraction_store):
    store_path, _results = folder_extraction_store

    for folder_id, stem in [(TOP_ID, TOP_VIDEO_STEM), (NESTED_ID, NESTED_VIDEO_STEM)]:
        project_config = _project_config(store_path, folder_id)
        frames_dir = _frames_dir(store_path, folder_id, video_stem=stem)

        assert project_config.is_file(), f"missing config for {folder_id}"
        assert frames_dir.is_dir(), f"missing frames dir for {folder_id}"

        pngs = list(frames_dir.glob("*.png"))
        assert len(pngs) == 20, f"expected 20 frames for {folder_id}, got {len(pngs)}"


def test_folder_input_ignores_non_video_files(folder_extraction_store):
    store_path, _results = folder_extraction_store
    manifest = list_extractions(store_path)
    # exactly the two real videos — notes.txt/README.md never became entries
    assert set(manifest.keys()) == {TOP_ID, NESTED_ID}


def test_folder_input_returns_dict_keyed_by_resolved_video_path(folder_extraction_store, raw_videos_folder):
    store_path, results = folder_extraction_store

    assert isinstance(results, dict)
    assert len(results) == 2

    top_video = (raw_videos_folder / f"{TOP_VIDEO_STEM}{Path(Setup.VIDEO).suffix}").resolve()
    nested_video = (raw_videos_folder / "session_02" / f"{NESTED_VIDEO_STEM}{Path(Setup.VIDEO).suffix}").resolve()

    assert set(results.keys()) == {str(top_video), str(nested_video)}
    assert results[str(top_video)] == _project_config(store_path, TOP_ID)
    assert results[str(nested_video)] == _project_config(store_path, NESTED_ID)


def test_folder_input_records_correct_source_video_per_entry(folder_extraction_store, raw_videos_folder):
    store_path, _results = folder_extraction_store

    top_record = get_extraction(store_path, TOP_ID)
    nested_record = get_extraction(store_path, NESTED_ID)

    top_video = (raw_videos_folder / f"{TOP_VIDEO_STEM}{Path(Setup.VIDEO).suffix}").resolve()
    nested_video = (raw_videos_folder / "session_02" / f"{NESTED_VIDEO_STEM}{Path(Setup.VIDEO).suffix}").resolve()

    assert Path(top_record["video_path"]) == top_video
    assert top_record["video_stem"] == TOP_VIDEO_STEM
    assert Path(nested_record["video_path"]) == nested_video
    assert nested_record["video_stem"] == NESTED_VIDEO_STEM


def test_folder_input_rejects_explicit_folder_name(tmp_path_factory, raw_videos_folder):
    store_path = init_store(tmp_path_factory.mktemp("folder_input_reject_store"), numframes2pick=20)
    # Must raise before any DLC call — folder_name can't map to >1 video.
    with pytest.raises(ValueError):
        extract_frames_for_video(store_path, raw_videos_folder, config_name="default", folder_name="whatever")
    assert list_extractions(store_path) == {}


def test_folder_input_raises_when_no_videos_found(tmp_path_factory):
    store_path = init_store(tmp_path_factory.mktemp("folder_input_empty_store"))
    empty_folder = tmp_path_factory.mktemp("empty_raw_videos")
    (empty_folder / "readme.txt").write_text("no videos in here")

    with pytest.raises(FileNotFoundError):
        extract_frames_for_video(store_path, empty_folder, config_name="default")


def test_folder_input_matches_extensions_case_insensitively(tmp_path_factory, raw_videos_folder):
    """A second folder containing only an uppercase-extension copy of the
    source video must still be picked up."""
    store_path = init_store(tmp_path_factory.mktemp("folder_input_uppercase_store"), numframes2pick=20)
    root = tmp_path_factory.mktemp("raw_videos_uppercase")
    src = Path(Setup.VIDEO)
    uppercase_video = root / f"cam-upper{src.suffix.upper()}"
    shutil.copy(src, uppercase_video)

    results = extract_frames_for_video(store_path, root, config_name="default")

    assert len(results) == 1
    folder_id = "cam-upper__default"
    assert _project_config(store_path, folder_id).is_file()
    pngs = list(_frames_dir(store_path, folder_id, video_stem="cam-upper").glob("*.png"))
    assert len(pngs) == 20