# DLC data manager

## Quickstart — using the package

**1. Install DeepLabCut** (GPU/platform-specific, do this first):
```
pip install 'deeplabcut[gui]'   # local machine — needed for labeling
pip install deeplabcut          # Colab / headless — training & inference only
```

**2. Install dlc_manager:**
```
pip install --pre git+https://github.com/GreenCurve/DLC-data-manager.git
```
(`--pre` is required — pins a pydantic pre-release.)

**3. (Optional) W&B logging:**
```
wandb login
```
In Colab, use a Secrets-panel API key instead — see Colab notes below.

---

## Quickstart — developing / testing the package

**1. Install DeepLabCut** (same as above):
```
pip install 'deeplabcut[gui]'
```

**2. Clone the repo and install editable, with dev deps:**
```
git clone https://github.com/GreenCurve/DLC-data-manager.git
cd DLC-data-manager
pip install --pre -e ".[dev]"
```
(`-e` so edits under `src/dlc_manager` take effect immediately without reinstalling; `--pre` for the pydantic pin as above; `[dev]` pulls in pytest.)

**3. Run the tests**, which call real `deeplabcut.extract_frames()` against actual video:
```
pytest tests/test_extraction_sequence.py -v -s
```
This needs a real video at `data/raw_videos/<name>.mp4`; without one, these tests skip themselves automatically.

If one person is doing both roles: quickstart 2 is a superset of quickstart 1 (same DeepLabCut install, then a different package-install step), so just use quickstart 2.

---

## Updating a production install after a dev change

Because production installs the package straight from git (`pip install --pre git+...`) rather than from a versioned index like PyPI, `pip install --upgrade` alone is **not reliable** here: pip decides whether a reinstall is needed by comparing the package version (`0.1.0` in `pyproject.toml`), and if that version string hasn't changed since the last install, pip may silently keep the old code even though the commit on `main` is newer.

**Safest one-liner, always gets the latest `main`:**
```
pip install --pre --force-reinstall --no-deps git+https://github.com/GreenCurve/DLC-data-manager.git
```
- `--force-reinstall` skips pip's "is this version already satisfied?" check and reinstalls regardless.
- `--no-deps` avoids re-resolving/reinstalling `pyyaml`/`pandas`/`pydantic`/`wandb` unnecessarily — drop it if a dependency version has also changed.

**Verify what actually got installed:**
```
pip freeze | grep dlc_manager
```
For a git install this prints the exact commit hash it resolved to (e.g. `dlc_manager @ git+https://github.com/GreenCurve/DLC-data-manager.git@<sha>`), so you can confirm it matches the commit you expect to be on `main`.

**More reproducible alternative — pin to a tag or commit instead of `main`:**
```
pip install --pre git+https://github.com/GreenCurve/DLC-data-manager.git@<tag-or-commit-sha>
```
This is worth doing once a change is confirmed working in dev: tag the commit (`git tag v0.2.0 && git push --tags`), bump `version` in `pyproject.toml` to match, and point prod installs at that tag. Then a plain `pip install --pre --upgrade git+...@v0.2.0` behaves predictably, and anyone can tell from `pip freeze` exactly which release prod is running — rather than everyone being on a floating, unlabeled `main`.

---

## API reference

There are two ways to drive this package — pick whichever fits:

```python
import dlc_manager as dlm

# 1) DataProject (recommended) — store paths are resolved for you
prj = dlm.init_data_project(r"/home/ccldlc/Desktop/DLC_project/")
prj.extract_frames_for_video(r"/raw_videos/HDMI-A.mp4")
net = prj.create_network_project(name="mitten_tracker")
net.add_labeled_data(prj.frame_set_path("HDMI-A__default"))
net.create_train_dataset()
net.train_network(epochs=600)

# 2) Standalone functions — pass store_path/project_config yourself
store = dlm.init_store("/data/frames_store")
dlm.extract_frames_for_video(store, r"/raw_videos/HDMI-A.mp4")
```

The tables below document the `DataProject` / `NetworkProject` / `InferenceRun` methods (the recommended surface). Every one of them is a thin wrapper around a standalone module-level function of the same behavior — see [Standalone functions](#standalone-functions) for those, useful if you're operating on a store without a full `DataProject` around it.

### Handles

| Constructor | Description |
| --- | --- |
| **dlm.init_data_project**(*root_dir*, [**metadata]) | Bootstraps (or just resolves, if it already exists) a data project's folder layout — `raw_videos/`, `frames_store/`, `network_store/`, `inference_runs/`, plus `project_config.yaml` — under `root_dir`. Idempotent, safe to call on every run of a usage script. `**metadata` is merged into `project_config.yaml`'s free-form metadata dict. Returns a **`DataProject`**. |
| **prj.create_network_project**(...) | See [Network projects](#network-projects) below. Returns a **`NetworkProject`** handle. |
| **prj.create_inference_run**(...) | See [Inference runs](#inference-runs) below. Returns an **`InferenceRun`** handle. |

### Project metadata

| Method | Description |
| --- | --- |
| **prj.metadata** | Property. Read-only snapshot of this project's free-form metadata dict (from `project_config.yaml`). |
| **prj.get_metadata**(*key*, [*default*]) | Reads one metadata key, e.g. `species`, `camera_rig`. |
| **prj.set_metadata**(*key*, *value*) | Persists one metadata key onto `project_config.yaml`, merging with whatever's already saved. |

### Extraction configs

`ExtractionConfig` is a versioned, named preset holding exactly what `deeplabcut.extract_frames()` needs — nothing about bodyparts/scorer/skeleton lives here (those are per frame-set, see [Frame extraction & labeling](#frame-extraction--labeling)).

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| **name** | *str* | `"default"` | Preset name — determines the saved filename, `extraction_configs/<name>.yaml`. |
| **algo** | *str* | `"kmeans"` | Frame-selection algorithm passed to `deeplabcut.extract_frames()`. |
| **mode** | *str* | `"automatic"` | Extraction mode passed to `deeplabcut.extract_frames()`. |
| **userfeedback** | *bool* | `False` | Whether DLC should prompt for confirmation before extracting. |
| **numframes2pick** | *int* | `20` | Number of frames to extract per video. |
| **start** / **stop** | *float* | `0.0` / `1.0` | Fraction of the video (0–1) to sample frames from. |
| **engine** | *str* | `"pytorch"` | DLC engine to record on the extraction project's config. |

| Method | Description |
| --- | --- |
| **prj.save_extraction_config**(*cfg*, [*overwrite=False*]) | Persists an `ExtractionConfig` as `extraction_configs/<cfg.name>.yaml`. Raises `FileExistsError` unless `overwrite=True`. |
| **prj.load_extraction_config**([*name="default"*]) | Loads a saved preset by name. Raises `FileNotFoundError` (listing what's available) if it doesn't exist. |
| **prj.list_extraction_configs**() | Names of every saved preset. |

### Frame extraction & labeling

Every `(video, extraction_config)` pair gets its own nested mini DLC project under `frames_store/labeled-data/<folder_id>/`, so bodyparts/scorer/skeleton can safely differ per frame set. Extraction is **not idempotent** — every call creates a brand-new frame folder.

| Method | Description |
| --- | --- |
| **prj.extract_frames_for_video**(*video_path*, [*config_name="default"*], [*folder_name*], [*overwrite=False*]) | Extracts frames for one video using the named `ExtractionConfig` preset. `video_path` may also be a folder — every video file found under it (recursively) is extracted individually, and a `{video_path: config.yaml path}` dict is returned instead of a single path. Without `folder_name`, the folder id auto-increments off `<video_stem>__<config_name>` so repeat calls never collide. Records the result in `manifest.yaml`. Returns the new project's `config.yaml` path. |
| **prj.init_labeling_config**(*folder_id*, *scorer*, *bodyparts*, *skeleton*, [*task*], [**overrides]) | Sets the labeling schema — scorer, bodyparts, skeleton — for one frame set, in place. Every other frame set keeps its own schema untouched, which is what lets bodyparts differ across videos. |
| **prj.label_frames_for**(*folder_id*, [*multiple=False*]) | Launches napari pre-loaded with this frame set's images + config (via `deeplabcut.label_frames()`), blocking until the window is closed. `multiple=True` enables multi-individual labeling. Requires `init_labeling_config()` to have been called first. |
| **prj.import_legacy_project**(*dlc_project_root*, [*config_name="imported"*]) | Imports every labeled frame set from an old, flat (pre-store) DLC project into `frames_store/`, one new frame-set folder per video, matched against this project's own `raw_videos/` by filename stem. Returns `{video_path: config.yaml path}`. |
| **prj.folder_id_for**(*video_path*, *config_name*) | Utility: the folder id a given video/config pair would produce (`<video_stem>__<config_name>`), before any auto-incrementing. |
| **prj.frame_set_path**(*folder_id*) | Path to a frame set's own folder, `frames_store/labeled-data/<folder_id>` — handy for passing to `NetworkProject.add_labeled_data()`. |

### Frame-set inspection & deletion

| Method | Description |
| --- | --- |
| **prj.list_extractions**() | Everything recorded in `manifest.yaml`: `folder_id -> {video_path, config_name, config, frame_count, updated_at, ...}`. |
| **prj.get_extraction**(*folder_id*) | The manifest record for one frame set. Raises `KeyError` (listing what's known) if it doesn't exist. |
| **prj.frame_set_training_usage**(*folder_id*) | Names of every network project this frame set has already been copied into via `add_labeled_data()`. Empty list if never used for training. |
| **prj.delete_frame_set**(*folder_id*, [*force=False*]) | Deletes a frame set's folder **and** its manifest entry together. Refuses if the frame set is already labeled, or has been used to train a network project, unless `force=True` — network-project copies are independent and untouched either way. |

### Network projects

A network project is a normal, trainable DLC project built purely from labeled-data copied out of one or more frame sets — no real video path required.

| Method | Description |
| --- | --- |
| **prj.create_network_project**([*name*], [*scorer="Egor"*], [*engine="pytorch"*], [**overrides]) | Creates a new, empty DLC project under `network_store/` (without calling `deeplabcut.create_new_project()`). `name` auto-generates as `network-<scorer>-<date>` if omitted; raises `FileExistsError` if given and already taken. `**overrides` can set any DLC config key. Returns a **`NetworkProject`**. |
| **prj.get_network_project**(*name*) | Reloads a `NetworkProject` handle created in this or an earlier session, by name. |
| **prj.list_network_projects**() | Names of every network project under `network_store/`. |
| **net.add_labeled_data**(*source_folder*, [*overwrite=False*]) | Copies a `frames_store` frame-set folder (e.g. `prj.frame_set_path("HDMI-A__default")`) wholesale into this project, named after its `folder_id`. The first call on a fresh project adopts the incoming scorer/bodyparts/skeleton as the project's schema; later calls require an exact bodypart match. |
| **net.create_train_dataset**([*net_type="resnet_50"*], [**kwargs]) | Wraps `deeplabcut.create_training_dataset()`, pinned to the pytorch engine. |
| **net.train_network**([*wandb_project*], [*wandb_run_name*], [*wandb_tags*], [*wandb_image_log_interval*], [**kwargs]) | Wraps `deeplabcut.train_network()`, pinned to the pytorch engine (`epochs`, `shuffle`, `batch_size`, `pytorch_cfg_updates`, etc. all pass through via `**kwargs`). Passing `wandb_project` logs the run to Weights & Biases (`pip install "deeplabcut[wandb]"` + `wandb login` first); config fields your installed DLC version doesn't support are dropped automatically with a warning rather than failing the run. |
| **net.evaluate_network**([*shuffle=1*], [*per_keypoint_evaluation=True*], [**kwargs]) | Wraps `deeplabcut.evaluate_network()`, pinned to the pytorch engine. |
| **net.list_labeled_data**() | What's been copied into this project so far: `[{folder_id, video_stem, source_folder, frame_count, copied_at}, ...]`. |
| **net.name** | Property. The project's folder name. |

### Inference runs

An inference run binds one or more raw videos to an already-trained network project and holds whatever that pairing produces — prediction files, and optionally a rendered video with predictions overlaid.

| Method | Description |
| --- | --- |
| **prj.create_inference_run**(*network*, *videos*, [*name*], [*shuffle=1*], [**overrides]) | Binds video(s) to a trained network project as a new run — nothing is analyzed yet. `network` is a `NetworkProject` handle or a `config.yaml` path; `videos` is a single file, a folder (recursed), or a list mixing either. `**overrides` are merged into every `analyze_videos()` call made against this run by default. Returns an **`InferenceRun`**. |
| **prj.get_inference_run**(*run_id*) | Reloads an `InferenceRun` handle created in this or an earlier session, by its folder name. |
| **prj.list_inference_runs**() | `run_id -> manifest record` for every inference run so far. |
| **run.analyze_videos**([*shuffle*], [*videos*], [**kwargs]) | Wraps `deeplabcut.analyze_videos()`, pinned to the pytorch engine, writing prediction files (`.h5`/`.csv`/`.pickle`) into the run's own folder. Defaults to every video the run was created with; records `analyzed_at` in the run's manifest entry. |
| **run.create_labeled_video**([*shuffle*], [*videos*], [**kwargs]) | Wraps `deeplabcut.create_labeled_video()` to render a video with predicted keypoints drawn on it, as `<video_stem>_labeled.mp4`. Requires `analyze_videos()` to have run first for these videos/shuffle — raises `RuntimeError` otherwise. |
| **run.list_predictions**() | Prediction files (`*.h5`) currently in this run's folder. |
| **run.list_labeled_videos**() | Rendered videos (`*_labeled.mp4`) currently in this run's folder. |
| **run.record** | Property. Fresh snapshot of this run's manifest entry (`network_project`, `videos`, `shuffle`, `analyzed_at`, ...). |
| **run.run_id** / **run.store_path** | Properties. The run's folder name, and its parent `inference_runs/` path. |

### Reports / audit

`generate_report()` reads a project's `raw_videos/`, `frames_store/`, `network_store/`, and `inference_runs/` straight off disk (no `deeplabcut` import required) and writes a single self-contained HTML page: raw-video distribution across `raw_videos/` subfolders, extraction/labeling coverage per video, bodypart/skeleton config groups (with near-duplicate detection), which frame sets have been used for training, which raw videos have been run through inference, and duplicate raw videos / labeled data / frames.

| Method | Description |
| --- | --- |
| **prj.generate_report**([*output*], [*similarity_threshold=0.7*], [*check_frame_duplicates=False*]) | Builds and writes the report. `output` defaults to `<root>/dlm_report.html`. `similarity_threshold` is the Jaccard threshold (on bodyparts) above which two not-identical config groups are flagged as "similar". `check_frame_duplicates=True` also hashes every PNG frame to find byte-identical duplicates (off by default — can be slow on large stores). Returns the report's output `Path`. |

Also runnable straight from the command line, without a `DataProject` at all:
```
python -m dlc_manager.report /path/to/project_root
python -m dlc_manager.report /path/to/project_root -o report.html --check-frame-duplicates
```

### Standalone functions

Everything above also works without a `DataProject`, by importing the module-level functions directly and passing the relevant store path / config path yourself — useful for scripting against a store in isolation. Each behaves exactly as its `DataProject`/handle counterpart described above.

| Module | Functions |
| --- | --- |
| `label_store.py` | `init_store(store_path, **default_extraction_overrides)` · `extract_frames_for_video(store_path, video_path, config_name="default", folder_name=None, overwrite=False)` · `init_labeling_config(store_path, folder_id, scorer, bodyparts, skeleton, task=None, **overrides)` · `label_frames_for(store_path, folder_id, multiple=False)` · `import_legacy_project(store_path, dlc_project_root, raw_videos_root, config_name="imported")` · `folder_id_for(video_path, config_name)` · `delete_frame_set(store_path, folder_id, force=False)` · `VIDEO_EXTENSIONS` (constant set of recognized video file extensions) |
| `network_store.py` | `create_project(store_path, name=None, scorer="Egor", engine="pytorch", **overrides)` · `add_labeled_data(project_config, source_folder, overwrite=False)` · `create_train_dataset(project_config, net_type="resnet_50", **kwargs)` · `train_network(project_config, shuffle=1, epochs=600, save_epochs=25, display_iters=500, batch_size=24, pytorch_cfg_updates=None, wandb_project=None, wandb_run_name=None, wandb_tags=None, wandb_image_log_interval=None, **kwargs)` · `evaluate_network(project_config, shuffle=1, per_keypoint_evaluation=True, plotting=False, **kwargs)` · `list_labeled_data(project_config)` |
| `inference_store.py` | `init_store(store_path)` · `create_run(store_path, network_config, videos, name=None, shuffle=1, **overrides)` · `analyze_videos(run_dir, shuffle=None, videos=None, **kwargs)` · `create_labeled_video(run_dir, shuffle=None, videos=None, **kwargs)` · `list_runs(store_path)` · `get_run(store_path, run_id)` · `list_predictions(run_dir)` · `list_labeled_videos(run_dir)` |
| `extraction_config.py` | `ExtractionConfig` (dataclass) · `save_extraction_config(store_path, cfg, overwrite=False)` · `load_extraction_config(store_path, name="default")` · `list_extraction_configs(store_path)` |
| `manifest.py` | `list_extractions(store_path)` · `get_extraction(store_path, folder_id)` |
| `report.py` | `generate_report(root, output=None, similarity_threshold=0.7, check_frame_duplicates=False)` · `build_report(root, similarity_threshold, check_frame_duplicates)` (returns the raw data dict, no HTML) · `render_html(data)` (data dict → HTML string) |