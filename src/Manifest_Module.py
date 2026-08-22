import yaml
from dataclasses import dataclass, asdict, field
from datetime import date as _date, datetime
from pathlib import Path



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