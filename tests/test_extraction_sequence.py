"""
tests/test_extraction_sequence.py
──────────────────────────────────
Validates the exact sequence from src/main.py:

    init_store(...)
    extract_frames_for_video(..., config_name="default")
    save_extraction_config(..., ExtractionConfig(name="dense", ...))
    extract_frames_for_video(..., config_name="dense")
    list_extractions(...)

This runs REAL deeplabcut.extract_frames() (kmeans + uniform) against
Setup.VIDEO — it is a slow integration test, not a mock-based unit test.
It uses a throwaway store under pytest's tmp_path so your real
frames_store on disk is never touched; only the raw video (read-only) is
reused from Setup.VIDEO.

Run from repo root with your DLC env active:
    pytest test_extraction_sequence.py -v -s
"""

import sys
from pathlib import Path

# tests/ -> repo root -> src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml
import pytest

import Setup
import Extraction_config_Module
from label_store import (
    init_store,
    extract_frames_for_video,
    init_labeling_config,
    list_extractions,
    get_extraction,
)


EXPECTED_DEFAULT_FRAMES = 50
EXPECTED_DENSE_FRAMES = 150
VIDEO_STEM = "HDMI-A"
DEFAULT_ID = f"{VIDEO_STEM}__default"
DENSE_ID = f"{VIDEO_STEM}__dense"


# ────────────────────────────────────────────────────────────────────────
# Run the sequence exactly once for the whole module (real DLC calls are
# slow) — individual tests below only assert against the result.
# ────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def executed_store(tmp_path_factory):
    store_path = init_store(tmp_path_factory.mktemp("frames_store"), numframes2pick=50)

    extract_frames_for_video(store_path, Setup.VIDEO, config_name="default")

    Extraction_config_Module.save_extraction_config(store_path, Extraction_config_Module.ExtractionConfig(name="dense", numframes2pick=150, algo="uniform"))
    extract_frames_for_video(store_path, Setup.VIDEO, config_name="dense")

    return store_path


def _frames_dir(store_path, folder_id):
    return store_path / "labeled-data" / folder_id / "labeled-data" / VIDEO_STEM


def _project_config(store_path, folder_id):
    return store_path / "labeled-data" / folder_id / "config.yaml"


# ────────────────────────────────────────────────────────────────────────
# init_store()
# ────────────────────────────────────────────────────────────────────────

def test_store_scaffold_created(executed_store):
    assert (executed_store / "labeled-data").is_dir()
    assert (executed_store / "extraction_configs" / "default.yaml").is_file()
    assert (executed_store / "manifest.yaml").is_file()


def test_init_store_is_idempotent(executed_store, capsys):
    # calling again must not reset presets/manifest
    init_store(executed_store, numframes2pick=999)
    captured = capsys.readouterr()
    assert "already initialized" in captured.out

    cfg = yaml.safe_load((executed_store / "extraction_configs" / "default.yaml").read_text())
    assert cfg["numframes2pick"] == 50  # untouched by the second call


# ────────────────────────────────────────────────────────────────────────
# extract_frames_for_video("default")
# ────────────────────────────────────────────────────────────────────────

def test_default_extraction_layout_and_count(executed_store):
    project_config = _project_config(executed_store, DEFAULT_ID)
    frames_dir = _frames_dir(executed_store, DEFAULT_ID)

    assert project_config.is_file()
    assert frames_dir.is_dir()

    pngs = list(frames_dir.glob("*.png"))
    assert len(pngs) == EXPECTED_DEFAULT_FRAMES, (
        f"expected {EXPECTED_DEFAULT_FRAMES} frames in {frames_dir}, got {len(pngs)}"
    )


def test_default_project_config_points_at_source_video(executed_store):
    cfg = yaml.safe_load(_project_config(executed_store, DEFAULT_ID).read_text())
    assert Path(list(cfg["video_sets"].keys())[0]) == Path(Setup.VIDEO).resolve()
    assert cfg["numframes2pick"] == 50


# ────────────────────────────────────────────────────────────────────────
# save_extraction_config("dense") + extract_frames_for_video("dense")
# ────────────────────────────────────────────────────────────────────────

def test_dense_preset_saved(executed_store):
    cfg = yaml.safe_load((executed_store / "extraction_configs" / "dense.yaml").read_text())
    assert cfg["numframes2pick"] == 150
    assert cfg["algo"] == "uniform"


def test_dense_extraction_lands_in_separate_folder(executed_store):
    frames_dir = _frames_dir(executed_store, DENSE_ID)
    assert frames_dir.is_dir()

    pngs = list(frames_dir.glob("*.png"))
    assert len(pngs) == EXPECTED_DENSE_FRAMES, (
        f"expected {EXPECTED_DENSE_FRAMES} frames in {frames_dir}, got {len(pngs)}"
    )


def test_duplicate_extraction_did_not_touch_the_original(executed_store):
    default_pngs = list(_frames_dir(executed_store, DEFAULT_ID).glob("*.png"))
    assert len(default_pngs) == EXPECTED_DEFAULT_FRAMES


# ────────────────────────────────────────────────────────────────────────
# list_extractions() / get_extraction() — the manifest
# ────────────────────────────────────────────────────────────────────────

def test_manifest_has_both_entries(executed_store):
    manifest = list_extractions(executed_store)
    assert set(manifest.keys()) == {DEFAULT_ID, DENSE_ID}


@pytest.mark.parametrize(
    "folder_id, config_name, algo, frame_count",
    [
        (DEFAULT_ID, "default", "kmeans", EXPECTED_DEFAULT_FRAMES),
        (DENSE_ID, "dense", "uniform", EXPECTED_DENSE_FRAMES),
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


# ────────────────────────────────────────────────────────────────────────
# Idempotent re-run (not in the pasted sequence, but implied by "skip if
# already extracted" behavior — worth locking down)
# ────────────────────────────────────────────────────────────────────────

def test_rerunning_extraction_skips_and_leaves_frames_untouched(executed_store, capsys):
    frames_dir = _frames_dir(executed_store, DEFAULT_ID)
    before = {p.name: p.stat().st_mtime_ns for p in frames_dir.glob("*.png")}

    extract_frames_for_video(executed_store, Setup.VIDEO, config_name="default")

    captured = capsys.readouterr()
    assert "Already extracted" in captured.out

    after = {p.name: p.stat().st_mtime_ns for p in frames_dir.glob("*.png")}
    assert before == after


# ────────────────────────────────────────────────────────────────────────
# init_labeling_config() — schema stays independent per frame set
# ────────────────────────────────────────────────────────────────────────

def test_labeling_schema_is_independent_per_folder(executed_store):
    init_labeling_config(
        executed_store, DEFAULT_ID,
        scorer="Egor",
        bodyparts=["nose", "tail"],
        skeleton=[["nose", "tail"]],
    )

    default_cfg = yaml.safe_load(_project_config(executed_store, DEFAULT_ID).read_text())
    dense_cfg = yaml.safe_load(_project_config(executed_store, DENSE_ID).read_text())

    assert default_cfg["scorer"] == "Egor"
    assert default_cfg["bodyparts"] == ["nose", "tail"]
    # the dense frame set must be completely unaffected
    assert not dense_cfg.get("bodyparts")


def test_labeling_config_rejects_unknown_folder(executed_store):
    with pytest.raises(FileNotFoundError):
        init_labeling_config(
            executed_store, "does-not-exist__default",
            scorer="Egor", bodyparts=["nose"], skeleton=[],
        )
