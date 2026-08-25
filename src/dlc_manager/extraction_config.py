import yaml
from dataclasses import dataclass, asdict
from pathlib import Path


# ────────────────────────────────────────────────────────────────────────
# Extraction config: versioned, named presets
# ────────────────────────────────────────────────────────────────────────

@dataclass
class ExtractionConfig:
    """Exactly the arguments deeplabcut.extract_frames() (automatic mode)
    consumes — nothing about bodyparts/scorer/skeleton lives here."""
    name: str = "default"
    algo: str = "kmeans"
    mode: str = "automatic"
    userfeedback: bool = False
    numframes2pick: int = 20
    start: float = 0.0
    stop: float = 1.0
    engine: str = "pytorch"

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


def _extraction_configs_dir(store_path):
    return Path(store_path).resolve() / "extraction_configs"


def save_extraction_config(store_path, cfg: ExtractionConfig, overwrite=False):
    """Persist an ExtractionConfig preset as extraction_configs/<name>.yaml."""
    cfg_dir = _extraction_configs_dir(store_path)
    cfg_dir.mkdir(parents=True, exist_ok=True)
    path = cfg_dir / f"{cfg.name}.yaml"
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Extraction config '{cfg.name}' already exists at {path}. "
            f"Pass overwrite=True to replace it."
        )
    with open(path, "w") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)
    print(f"✅ Extraction config saved: {path}")
    return path


def load_extraction_config(store_path, name="default") -> ExtractionConfig:
    path = _extraction_configs_dir(store_path) / f"{name}.yaml"
    if not path.exists():
        available = list_extraction_configs(store_path)
        raise FileNotFoundError(
            f"No extraction config named '{name}' in {path.parent}. "
            f"Available: {available or '(none)'}"
        )
    with open(path) as f:
        d = yaml.safe_load(f) or {}
    d.setdefault("name", name)
    return ExtractionConfig.from_dict(d)


def list_extraction_configs(store_path):
    cfg_dir = _extraction_configs_dir(store_path)
    if not cfg_dir.is_dir():
        return []
    return sorted(p.stem for p in cfg_dir.glob("*.yaml"))
