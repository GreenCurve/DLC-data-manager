"""
report.py
─────────
Audit/visualization for a DataProject (raw_videos/ + frames_store/, plus
network_store/ and inference_runs/ if present): produces a single self-
contained HTML report answering:

    1. How raw videos are distributed among subfolders of raw_videos/
    2. How many frame sets (extractions) each raw video has, and how many
       videos-with-data each raw_videos subfolder has
    3. Which frame sets share the same/similar bodyparts+skeleton config
    4. Which frame sets have been used to train a network project, and
       which raw videos have been run through an inference run
    5. Duplicate raw videos (byte-identical) and duplicate labeled data
       (same keypoint content, byte-identical h5, or byte-identical PNGs)

Design
──────
1) Deliberately reads the store's on-disk YAML/HDF5 files directly instead
   of going through label_store.py/network_store.py/inference_store.py (or
   deeplabcut itself) — so building a report is fast, needs nothing heavier
   than pyyaml + matplotlib (+ optionally pandas/tables for content-level
   duplicate detection), and works even against a project on a machine
   with no deeplabcut install. The trade-off is that a couple of small
   things (VIDEO_EXTENSIONS, folder-layout conventions) are duplicated here
   rather than imported from label_store.py, on purpose.

2) build_report() gathers everything into one plain dict; render_html()
   turns that dict into the HTML/matplotlib report. generate_report() is
   the one-call convenience that does both and writes the file — that's
   the function exposed as DataProject.generate_report() and the
   recommended way to call this module. build_report()/render_html() stay
   available separately for anyone who wants the raw data (e.g. to feed a
   notebook) without paying for HTML rendering.

Usage
─────
    from dlc_manager import generate_report

    generate_report(r"/home/ccldlc/Desktop/DLC_project/")
    # -> writes dlm_report.html at the project root, returns its path

    generate_report(project_root, output="audit.html", check_frame_duplicates=True)

    # or, via the DataProject handle (same store paths already filled in):
    prj.generate_report(check_frame_duplicates=True)

Also runnable from the command line:
    python -m dlc_manager.report /path/to/project_root
    python -m dlc_manager.report /path/to/project_root -o report.html
    python -m dlc_manager.report /path/to/project_root --check-frame-duplicates
    python -m dlc_manager.report /path/to/project_root --similarity-threshold 0.6

project_root must contain raw_videos/ and frames_store/ (the layout
init_data_project() creates).
"""

import argparse
import base64
import hashlib
import io
import sys
from collections import defaultdict
from pathlib import Path

import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import pandas as pd
    HAVE_PANDAS = True
except ImportError:
    HAVE_PANDAS = False


VIDEO_EXTENSIONS = {
    ".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv",
    ".m4v", ".mpg", ".mpeg", ".webm", ".asf",
}
HASH_CHUNK = 1024 * 1024


# ────────────────────────────────────────────────────────────────────────
# helpers
# ────────────────────────────────────────────────────────────────────────

def sha256_file(path, chunk=HASH_CHUNK):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def html_escape(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


# ────────────────────────────────────────────────────────────────────────
# 1) raw_videos scan
# ────────────────────────────────────────────────────────────────────────

def scan_raw_videos(raw_videos_root):
    """Returns list of dicts: {path, stem, top_subfolder, rel_dir}."""
    raw_videos_root = Path(raw_videos_root).resolve()
    videos = []
    for p in sorted(raw_videos_root.rglob("*")):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
            rel = p.relative_to(raw_videos_root)
            top = rel.parts[0] if len(rel.parts) > 1 else "(top-level)"
            videos.append({
                "path": p,
                "stem": p.stem,
                "top_subfolder": top,
                "rel_dir": str(rel.parent) if len(rel.parts) > 1 else "",
            })
    return videos


# ────────────────────────────────────────────────────────────────────────
# 2) manifest + frame-set configs
# ────────────────────────────────────────────────────────────────────────

def load_manifest(store_path):
    path = Path(store_path) / "manifest.yaml"
    if not path.exists():
        return {}
    with open(path) as f:
        return (yaml.safe_load(f) or {}).get("extractions", {})


def load_frame_set_configs(store_path):
    """folder_id -> parsed config.yaml dict (or None if unreadable)."""
    labeled_data = Path(store_path) / "labeled-data"
    configs = {}
    if not labeled_data.is_dir():
        return configs
    for folder in sorted(p for p in labeled_data.iterdir() if p.is_dir()):
        cfg_path = folder / "config.yaml"
        if not cfg_path.exists():
            continue
        try:
            with open(cfg_path) as f:
                configs[folder.name] = yaml.safe_load(f) or {}
        except Exception as e:
            configs[folder.name] = {"__error__": str(e)}
    return configs


def find_labeled_h5(store_path, folder_id, video_stem, scorer):
    """Path to CollectedData_<scorer>.h5 for this frame set, if it exists."""
    candidate_dir = Path(store_path) / "labeled-data" / folder_id / "labeled-data" / video_stem
    if not scorer:
        # scorer unknown -- look for any CollectedData_*.h5
        if candidate_dir.is_dir():
            hits = list(candidate_dir.glob("CollectedData_*.h5"))
            return hits[0] if hits else None
        return None
    h5_path = candidate_dir / f"CollectedData_{scorer}.h5"
    return h5_path if h5_path.exists() else None


def load_network_projects(network_store_path):
    """project_name -> list of copied entries (from that project's own
    sources.yaml — see network_store.add_labeled_data()), for every
    network project under network_store/. {} if network_store/ doesn't
    exist or has no projects yet."""
    network_store_path = Path(network_store_path)
    result = {}
    if not network_store_path.is_dir():
        return result
    for pdir in sorted(p for p in network_store_path.iterdir() if p.is_dir()):
        if not (pdir / "config.yaml").exists():
            continue
        sources_path = pdir / "sources.yaml"
        copied = []
        if sources_path.exists():
            with open(sources_path) as f:
                copied = (yaml.safe_load(f) or {}).get("copied", [])
        result[pdir.name] = copied
    return result


def load_inference_runs(inference_runs_path):
    """run_id -> manifest record, straight from inference_runs/manifest.yaml
    (see inference_store.py). {} if inference_runs/ doesn't exist yet."""
    path = Path(inference_runs_path) / "manifest.yaml"
    if not path.exists():
        return {}
    with open(path) as f:
        return (yaml.safe_load(f) or {}).get("runs", {})


def resolve_video_match(recorded_path, raw_videos):
    """Match a manifest 'video_path' string to one of the scanned raw
    videos. Tries resolved-path equality first (works if paths are still
    valid on this machine), then falls back to filename-stem match (works
    even if the project was moved / recorded on another machine)."""
    try:
        resolved = Path(recorded_path).resolve()
    except Exception:
        resolved = None
    if resolved is not None:
        for v in raw_videos:
            if v["path"] == resolved:
                return v
    stem = Path(recorded_path).stem
    stem_matches = [v for v in raw_videos if v["stem"] == stem]
    if len(stem_matches) == 1:
        return stem_matches[0]
    return None  # no match, or ambiguous


# ────────────────────────────────────────────────────────────────────────
# 3) config grouping (bodyparts + skeleton)
# ────────────────────────────────────────────────────────────────────────

def bodyparts_key(bodyparts):
    return tuple(sorted(bodyparts or []))


def skeleton_key(skeleton):
    return tuple(sorted(tuple(sorted(map(str, pair))) for pair in (skeleton or [])))


def group_configs(frame_configs):
    """Exact groups: (bodyparts_key, skeleton_key) -> [folder_id, ...]."""
    groups = defaultdict(list)
    for folder_id, cfg in frame_configs.items():
        if not cfg or "__error__" in cfg:
            continue
        key = (bodyparts_key(cfg.get("bodyparts")), skeleton_key(cfg.get("skeleton")))
        groups[key].append(folder_id)
    return groups


def jaccard(set_a, set_b):
    a, b = set(set_a), set(set_b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def cluster_similar_groups(groups, threshold):
    """Union-find over exact-config groups whose bodyparts sets are
    >= threshold Jaccard-similar (but not already identical). Returns list
    of clusters, each a list of group keys, restricted to clusters with
    >1 member (i.e. skips groups that don't loosely match anything else)."""
    keys = list(groups.keys())
    parent = {k: k for k in keys}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            bp_i = set(keys[i][0])
            bp_j = set(keys[j][0])
            if keys[i] == keys[j]:
                continue
            if jaccard(bp_i, bp_j) >= threshold:
                union(keys[i], keys[j])

    clusters = defaultdict(list)
    for k in keys:
        clusters[find(k)].append(k)
    return [c for c in clusters.values() if len(c) > 1]


# ────────────────────────────────────────────────────────────────────────
# 4) duplicate detection
# ────────────────────────────────────────────────────────────────────────

def find_duplicate_videos(raw_videos):
    by_hash = defaultdict(list)
    for v in raw_videos:
        try:
            h = sha256_file(v["path"])
        except Exception as e:
            h = f"__error__:{e}"
        by_hash[h].append(v["path"])
    return {h: paths for h, paths in by_hash.items() if len(paths) > 1 and not h.startswith("__error__")}


def h5_content_signature(h5_path):
    """Content-level signature of a CollectedData_*.h5, robust to scorer
    name: drops the 'scorer' level of the column MultiIndex and hashes the
    remaining (bodyparts, coords) columns + values + row index. Falls back
    to a raw byte hash if pandas/pytables isn't available."""
    if not HAVE_PANDAS:
        return ("bytes", sha256_file(h5_path))
    try:
        df = pd.read_hdf(h5_path)
        cols = df.columns
        if hasattr(cols, "droplevel") and "scorer" in (cols.names or []):
            df = df.copy()
            df.columns = cols.droplevel("scorer")
        payload = (
            tuple(str(c) for c in df.columns.to_flat_index()),
            tuple(str(i) for i in df.index),
            df.to_numpy().round(3).tobytes(),
        )
        h = hashlib.sha256()
        for part in payload:
            h.update(repr(part).encode("utf-8") if not isinstance(part, bytes) else part)
        return ("content", h.hexdigest())
    except Exception:
        return ("bytes", sha256_file(h5_path))


def find_duplicate_labeled_data(store_path, frame_configs, manifest):
    """folder_ids grouped by identical labeled-data content."""
    by_sig = defaultdict(list)
    modes = set()
    for folder_id, cfg in frame_configs.items():
        if not cfg or "__error__" in cfg:
            continue
        video_sets = cfg.get("video_sets") or {}
        video_stem = None
        if video_sets:
            video_stem = Path(next(iter(video_sets))).stem
        if not video_stem and folder_id in manifest:
            video_stem = manifest[folder_id].get("video_stem")
        if not video_stem:
            continue
        h5_path = find_labeled_h5(store_path, folder_id, video_stem, cfg.get("scorer"))
        if not h5_path:
            continue
        mode, sig = h5_content_signature(h5_path)
        modes.add(mode)
        by_sig[sig].append(folder_id)
    dupes = {sig: ids for sig, ids in by_sig.items() if len(ids) > 1}
    return dupes, modes


def find_duplicate_frames(store_path):
    """Byte-identical PNG frames anywhere in labeled-data/ (opt-in: can be
    slow on large stores)."""
    labeled_data = Path(store_path) / "labeled-data"
    by_hash = defaultdict(list)
    for png in labeled_data.rglob("*.png"):
        try:
            h = sha256_file(png)
        except Exception:
            continue
        by_hash[h].append(png)
    return {h: paths for h, paths in by_hash.items() if len(paths) > 1}


# ────────────────────────────────────────────────────────────────────────
# report assembly
# ────────────────────────────────────────────────────────────────────────

def build_report(root, similarity_threshold, check_frame_duplicates):
    root = Path(root).resolve()
    raw_videos_root = root / "raw_videos"
    store_path = root / "frames_store"
    network_store_path = root / "network_store"
    inference_runs_path = root / "inference_runs"

    if not raw_videos_root.is_dir():
        raise FileNotFoundError(
            f"No raw_videos/ folder under {root} — is this a DataProject root? "
            f"(see init_data_project())"
        )
    if not store_path.is_dir():
        raise FileNotFoundError(
            f"No frames_store/ folder under {root} — is this a DataProject root? "
            f"(see init_data_project())"
        )

    raw_videos = scan_raw_videos(raw_videos_root)
    manifest = load_manifest(store_path)
    frame_configs = load_frame_set_configs(store_path)
    network_projects = load_network_projects(network_store_path)
    inference_runs = load_inference_runs(inference_runs_path)

    # ---- training usage: folder_id -> [network_project names] ----
    training_usage = defaultdict(list)
    for proj_name, copied in network_projects.items():
        for entry in copied:
            fid = entry.get("folder_id")
            if fid:
                training_usage[fid].append(proj_name)

    # ---- inference usage: raw video path -> [{run_id, network_project}, ...] ----
    inference_usage = defaultdict(list)
    unmatched_inference_entries = []
    for run_id, record in inference_runs.items():
        net_proj = record.get("network_project")
        for vp in record.get("videos", []):
            match = resolve_video_match(vp, raw_videos)
            if match is None:
                unmatched_inference_entries.append((run_id, vp))
                continue
            inference_usage[match["path"]].append({
                "run_id": run_id,
                "network_project": net_proj,
                "analyzed": bool(record.get("analyzed_at")),
                "labeled_video": bool(record.get("labeled_video_at")),
            })

    # ---- Q1: distribution per subfolder ----
    per_subfolder = defaultdict(list)
    for v in raw_videos:
        per_subfolder[v["top_subfolder"]].append(v)
    subfolder_names = sorted(per_subfolder)
    subfolder_counts = [len(per_subfolder[s]) for s in subfolder_names]

    # ---- Q2: frame sets per raw video, videos-with-data per subfolder ----
    extractions_per_video = defaultdict(list)  # video path (resolved) -> [folder_id,...]
    unmatched_manifest_entries = []
    for folder_id, record in manifest.items():
        match = resolve_video_match(record.get("video_path", ""), raw_videos)
        if match is None:
            unmatched_manifest_entries.append((folder_id, record.get("video_path")))
            continue
        extractions_per_video[match["path"]].append(folder_id)

    def is_labeled(folder_id):
        cfg = frame_configs.get(folder_id) or {}
        if "__error__" in cfg:
            return False
        record = manifest.get(folder_id, {})
        video_stem = record.get("video_stem")
        h5 = find_labeled_h5(store_path, folder_id, video_stem, cfg.get("scorer"))
        return h5 is not None

    video_rows = []
    for v in raw_videos:
        folder_ids = extractions_per_video.get(v["path"], [])
        labeled_ids = [fid for fid in folder_ids if is_labeled(fid)]
        trained_ids = [fid for fid in folder_ids if fid in training_usage]
        runs_for_video = inference_usage.get(v["path"], [])
        video_rows.append({
            "video": v["path"].name,
            "subfolder": v["top_subfolder"],
            "n_extractions": len(folder_ids),
            "n_labeled": len(labeled_ids),
            "folder_ids": folder_ids,
            "n_trained": len(trained_ids),
            "trained_ids": trained_ids,
            "n_inference_runs": len(runs_for_video),
            "inference_runs": runs_for_video,
        })

    subfolder_stats = defaultdict(lambda: {"total": 0, "with_extraction": 0, "with_labels": 0})
    for row in video_rows:
        s = subfolder_stats[row["subfolder"]]
        s["total"] += 1
        if row["n_extractions"] > 0:
            s["with_extraction"] += 1
        if row["n_labeled"] > 0:
            s["with_labels"] += 1

    # ---- Q3: config groups ----
    groups = group_configs(frame_configs)
    group_list = sorted(groups.items(), key=lambda kv: -len(kv[1]))
    similar_clusters = cluster_similar_groups(groups, similarity_threshold)

    # ---- Q4: duplicates ----
    dup_videos = find_duplicate_videos(raw_videos)
    dup_labeled, h5_modes = find_duplicate_labeled_data(store_path, frame_configs, manifest)
    dup_frames = find_duplicate_frames(store_path) if check_frame_duplicates else None

    return dict(
        root=root, raw_videos=raw_videos, manifest=manifest, frame_configs=frame_configs,
        subfolder_names=subfolder_names, subfolder_counts=subfolder_counts,
        video_rows=video_rows, subfolder_stats=subfolder_stats,
        unmatched_manifest_entries=unmatched_manifest_entries,
        group_list=group_list, groups=groups, similar_clusters=similar_clusters,
        similarity_threshold=similarity_threshold,
        dup_videos=dup_videos, dup_labeled=dup_labeled, h5_modes=h5_modes,
        dup_frames=dup_frames,
        network_projects=network_projects, inference_runs=inference_runs,
        training_usage=training_usage, inference_usage=inference_usage,
        unmatched_inference_entries=unmatched_inference_entries,
    )


def render_html(data):
    R = data
    parts = []
    parts.append(f"""<!doctype html><html><head><meta charset="utf-8">
<title>DLC project audit — {html_escape(R['root'].name)}</title>
<style>
body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#1a1a1a;background:#fafafa}}
h1{{font-size:1.5rem}} h2{{margin-top:2.5rem;border-bottom:2px solid #ddd;padding-bottom:.3rem}}
table{{border-collapse:collapse;width:100%;margin:1rem 0;font-size:.9rem}}
th,td{{border:1px solid #ddd;padding:.4rem .6rem;text-align:left}}
th{{background:#f0f0f0}} tr:nth-child(even){{background:#f7f7f7}}
.badge{{display:inline-block;background:#eee;border-radius:4px;padding:.1rem .5rem;font-size:.8rem;margin:.1rem}}
.warn{{color:#a15c00}} .ok{{color:#1a7a1a}} .bad{{color:#b00020}}
img.chart{{max-width:100%;display:block;margin:1rem 0}}
code{{background:#eee;padding:.1rem .3rem;border-radius:3px}}
.small{{color:#666;font-size:.85rem}}
</style></head><body>
<h1>DLC project audit — <code>{html_escape(R['root'])}</code></h1>
<p class="small">raw_videos: {len(R['raw_videos'])} video files &nbsp;|&nbsp;
frame sets in manifest: {len(R['manifest'])} &nbsp;|&nbsp;
config groups: {len(R['group_list'])} &nbsp;|&nbsp;
network projects: {len(R['network_projects'])} &nbsp;|&nbsp;
inference runs: {len(R['inference_runs'])}</p>
""")

    # ---- Section 1 ----
    fig, ax = plt.subplots(figsize=(7, max(2, 0.4 * len(R['subfolder_names']))))
    ax.barh(R['subfolder_names'], R['subfolder_counts'], color="#4C72B0")
    ax.set_xlabel("raw video count")
    ax.set_title("Raw videos per subfolder")
    fig.tight_layout()
    chart1 = fig_to_b64(fig)

    parts.append("<h2>1. Raw video distribution across subfolders</h2>")
    parts.append(f'<img class="chart" src="data:image/png;base64,{chart1}">')
    parts.append("<table><tr><th>Subfolder</th><th>Video count</th></tr>")
    for name, count in zip(R['subfolder_names'], R['subfolder_counts']):
        parts.append(f"<tr><td>{html_escape(name)}</td><td>{count}</td></tr>")
    parts.append("</table>")

    # ---- Section 2 ----
    parts.append("<h2>2. Labeled data per raw video / per subfolder</h2>")
    sf_names = sorted(R['subfolder_stats'])
    totals = [R['subfolder_stats'][s]['total'] for s in sf_names]
    ext_only = [R['subfolder_stats'][s]['with_extraction'] - R['subfolder_stats'][s]['with_labels'] for s in sf_names]
    labeled = [R['subfolder_stats'][s]['with_labels'] for s in sf_names]
    no_data = [t - e - l for t, e, l in zip(totals, ext_only, labeled)]

    fig, ax = plt.subplots(figsize=(7, max(2, 0.4 * len(sf_names))))
    ax.barh(sf_names, labeled, label="has labels", color="#55A868")
    ax.barh(sf_names, ext_only, left=labeled, label="extracted, not labeled", color="#DD8452")
    ax.barh(sf_names, no_data, left=[a + b for a, b in zip(labeled, ext_only)], label="no data", color="#C44E52")
    ax.set_xlabel("video count")
    ax.set_title("Videos with data, per subfolder")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    chart2 = fig_to_b64(fig)
    parts.append(f'<img class="chart" src="data:image/png;base64,{chart2}">')

    parts.append("<table><tr><th>Subfolder</th><th>Total videos</th><th>With labels</th>"
                  "<th>Extracted only</th><th>No data</th></tr>")
    for s, t, l, e, n in zip(sf_names, totals, labeled, ext_only, no_data):
        parts.append(f"<tr><td>{html_escape(s)}</td><td>{t}</td><td class='ok'>{l}</td>"
                      f"<td class='warn'>{e}</td><td class='bad'>{n}</td></tr>")
    parts.append("</table>")

    parts.append("<h3>Per-video breakdown</h3>")
    parts.append("<table><tr><th>Video</th><th>Subfolder</th><th># frame sets</th>"
                  "<th># labeled frame sets</th><th>Frame set IDs</th>"
                  "<th># used in training</th><th># inference runs</th></tr>")
    for row in sorted(R['video_rows'], key=lambda r: (-r['n_extractions'], r['video'])):
        ids = ", ".join(f"<span class='badge'>{html_escape(i)}</span>" for i in row['folder_ids']) or "<span class='small'>—</span>"
        trained_cls = "ok" if row['n_trained'] else "small"
        infer_cls = "ok" if row['n_inference_runs'] else "small"
        parts.append(f"<tr><td>{html_escape(row['video'])}</td><td>{html_escape(row['subfolder'])}</td>"
                      f"<td>{row['n_extractions']}</td><td>{row['n_labeled']}</td><td>{ids}</td>"
                      f"<td class='{trained_cls}'>{row['n_trained']}</td>"
                      f"<td class='{infer_cls}'>{row['n_inference_runs']}</td></tr>")
    parts.append("</table>")

    if R['unmatched_manifest_entries']:
        parts.append(f"<p class='warn'>⚠ {len(R['unmatched_manifest_entries'])} manifest entries could not be "
                      f"matched to a video currently in raw_videos/ (moved/renamed/deleted source video):</p><ul>")
        for fid, vp in R['unmatched_manifest_entries']:
            parts.append(f"<li><code>{html_escape(fid)}</code> → recorded path <code>{html_escape(vp)}</code></li>")
        parts.append("</ul>")

    # ---- Section 3 ----
    parts.append("<h2>3. Config groups (bodyparts + skeleton)</h2>")
    fig, ax = plt.subplots(figsize=(7, max(2, 0.35 * min(len(R['group_list']), 20))))
    top_groups = R['group_list'][:20]
    labels = [f"group {i+1} ({len(bp)} bp)" for i, ((bp, _), _) in enumerate(top_groups)]
    sizes = [len(ids) for _, ids in top_groups]
    ax.barh(labels[::-1], sizes[::-1], color="#8172B2")
    ax.set_xlabel("# frame sets in group")
    ax.set_title("Config group sizes (top 20)")
    fig.tight_layout()
    chart3 = fig_to_b64(fig)
    parts.append(f'<img class="chart" src="data:image/png;base64,{chart3}">')

    parts.append("<table><tr><th>Group</th><th># frame sets</th><th># bodyparts</th>"
                  "<th>Bodyparts</th><th>Frame set IDs</th></tr>")
    for i, ((bp, sk), ids) in enumerate(R['group_list']):
        bp_str = ", ".join(bp) if bp else "<span class='small'>(none set)</span>"
        id_badges = " ".join(f"<span class='badge'>{html_escape(x)}</span>" for x in ids)
        parts.append(f"<tr><td>{i+1}</td><td>{len(ids)}</td><td>{len(bp)}</td>"
                      f"<td class='small'>{html_escape(bp_str)}</td><td>{id_badges}</td></tr>")
    parts.append("</table>")

    parts.append(f"<h3>Similar (not identical) groups — Jaccard ≥ {R['similarity_threshold']} on bodyparts</h3>")
    if not R['similar_clusters']:
        parts.append("<p class='small'>No near-matching groups found above the threshold.</p>")
    else:
        for cluster in R['similar_clusters']:
            parts.append("<div style='margin-bottom:1rem;padding:.5rem;border:1px solid #ddd;border-radius:6px'>")
            parts.append("<ul>")
            for key in cluster:
                bp, sk = key
                ids = R['groups'][key]
                bp_str = ", ".join(bp) if bp else "(none set)"
                id_badges = " ".join(f"<span class='badge'>{html_escape(x)}</span>" for x in ids)
                parts.append(f"<li><span class='small'>[{len(bp)} bodyparts] {html_escape(bp_str)}</span><br>{id_badges}</li>")
            parts.append("</ul></div>")

    # ---- Section 4: training usage ----
    parts.append("<h2>4. Training usage — frame sets copied into a network project</h2>")
    all_folder_ids = sorted(R['frame_configs'])
    if not R['network_projects']:
        parts.append("<p class='small'>No network_store/ projects found — nothing has been used for training yet.</p>")
    else:
        used_ids = [fid for fid in all_folder_ids if fid in R['training_usage']]
        unused_ids = [fid for fid in all_folder_ids if fid not in R['training_usage']]
        parts.append(f"<p class='small'>{len(used_ids)} of {len(all_folder_ids)} frame sets have been used to "
                      f"train at least one network project.</p>")
        parts.append("<table><tr><th>Frame set ID</th><th>Used in network project(s)</th></tr>")
        for fid in used_ids:
            projs = " ".join(f"<span class='badge'>{html_escape(p)}</span>" for p in R['training_usage'][fid])
            parts.append(f"<tr><td>{html_escape(fid)}</td><td>{projs}</td></tr>")
        parts.append("</table>")
        if unused_ids:
            parts.append("<h3>Not yet used for training</h3>")
            id_badges = " ".join(f"<span class='badge'>{html_escape(x)}</span>" for x in unused_ids)
            parts.append(f"<p class='small'>{id_badges}</p>")

    # ---- Section 5: inference usage ----
    parts.append("<h2>5. Inference usage — raw videos run through an inference run</h2>")
    if not R['inference_runs']:
        parts.append("<p class='small'>No inference_runs/ found — nothing has been run through inference yet.</p>")
    else:
        videos_with_runs = [row for row in R['video_rows'] if row['n_inference_runs'] > 0]
        parts.append(f"<p class='small'>{len(videos_with_runs)} of {len(R['video_rows'])} raw videos have at least "
                      f"one inference run.</p>")
        parts.append("<table><tr><th>Video</th><th>Subfolder</th><th>Inference run(s)</th>"
                      "<th>Network project(s)</th><th>Analyzed</th><th>Labeled video</th></tr>")
        for row in sorted(videos_with_runs, key=lambda r: (-r['n_inference_runs'], r['video'])):
            run_badges = " ".join(f"<span class='badge'>{html_escape(r['run_id'])}</span>" for r in row['inference_runs'])
            net_badges = " ".join(sorted({html_escape(r['network_project'] or '?') for r in row['inference_runs']}))
            analyzed = "✅" if any(r['analyzed'] for r in row['inference_runs']) else ""
            labeled_vid = "🎬" if any(r['labeled_video'] for r in row['inference_runs']) else ""
            parts.append(f"<tr><td>{html_escape(row['video'])}</td><td>{html_escape(row['subfolder'])}</td>"
                          f"<td>{run_badges}</td><td>{net_badges}</td>"
                          f"<td class='ok'>{analyzed}</td><td class='ok'>{labeled_vid}</td></tr>")
        parts.append("</table>")

        if R['unmatched_inference_entries']:
            parts.append(f"<p class='warn'>⚠ {len(R['unmatched_inference_entries'])} inference-run video entries "
                          f"could not be matched to a video currently in raw_videos/ (moved/renamed/deleted "
                          f"source video):</p><ul>")
            for run_id, vp in R['unmatched_inference_entries']:
                parts.append(f"<li><code>{html_escape(run_id)}</code> → recorded path <code>{html_escape(vp)}</code></li>")
            parts.append("</ul>")

    # ---- Section 6 ----
    parts.append("<h2>6. Duplicate detection</h2>")
    parts.append("<h3>Duplicate raw video files (byte-identical)</h3>")
    if not R['dup_videos']:
        parts.append("<p class='ok small'>No byte-identical duplicate raw videos found.</p>")
    else:
        parts.append("<table><tr><th>SHA-256</th><th>Files</th></tr>")
        for h, paths in R['dup_videos'].items():
            file_list = "<br>".join(html_escape(str(p)) for p in paths)
            parts.append(f"<tr><td class='small'>{h[:12]}…</td><td>{file_list}</td></tr>")
        parts.append("</table>")

    mode_note = ", ".join(sorted(R['h5_modes'])) or "n/a"
    parts.append(f"<h3>Duplicate labeled data <span class='small'>(comparison mode: {mode_note})</span></h3>")
    if not R['dup_labeled']:
        parts.append("<p class='ok small'>No duplicate labeled-data content found.</p>")
    else:
        parts.append("<table><tr><th>Signature</th><th>Frame set IDs</th></tr>")
        for sig, ids in R['dup_labeled'].items():
            id_badges = " ".join(f"<span class='badge'>{html_escape(x)}</span>" for x in ids)
            parts.append(f"<tr><td class='small'>{str(sig)[:12]}…</td><td>{id_badges}</td></tr>")
        parts.append("</table>")

    if R['dup_frames'] is not None:
        parts.append("<h3>Duplicate PNG frames (byte-identical)</h3>")
        if not R['dup_frames']:
            parts.append("<p class='ok small'>No byte-identical duplicate frames found.</p>")
        else:
            parts.append(f"<p class='warn'>{len(R['dup_frames'])} groups of identical frames found.</p>")
            parts.append("<table><tr><th>SHA-256</th><th>Files</th></tr>")
            for h, paths in list(R['dup_frames'].items())[:200]:
                file_list = "<br>".join(html_escape(str(p)) for p in paths)
                parts.append(f"<tr><td class='small'>{h[:12]}…</td><td class='small'>{file_list}</td></tr>")
            parts.append("</table>")
    else:
        parts.append("<p class='small'>Frame-level (PNG) duplicate check skipped — pass "
                      "<code>--check-frame-duplicates</code> to enable (can be slow on large stores).</p>")

    parts.append("</body></html>")
    return "\n".join(parts)


# ────────────────────────────────────────────────────────────────────────
# generate_report() — the one-call, recommended entry point (build +
# render + write), and what DataProject.generate_report() forwards to
# ────────────────────────────────────────────────────────────────────────

def generate_report(root, output=None, similarity_threshold=0.7, check_frame_duplicates=False):
    """Build the audit report for one DataProject root and write it out as
    a single self-contained HTML file.

    root: a DataProject root — must contain raw_videos/ and frames_store/
        (the layout init_data_project() creates). Raises FileNotFoundError
        if either is missing.
    output: where to write the report.
        - Omitted (default): "<root>/dlm_report.html".
        - Given: written there exactly (relative paths are resolved
          against the current working directory, not root).
    similarity_threshold: Jaccard threshold (on bodyparts) above which two
        NOT-identical config groups are flagged as "similar" in section 3
        of the report.
    check_frame_duplicates: if True, also hashes every PNG frame under
        frames_store/ to find byte-identical duplicate frames (section 6).
        Off by default — can be slow on large stores.

    Returns the report's output Path.
    """
    root = Path(root).resolve()
    data = build_report(root, similarity_threshold, check_frame_duplicates)
    html = render_html(data)

    output_path = Path(output).resolve() if output is not None else root / "dlm_report.html"
    output_path.write_text(html, encoding="utf-8")
    print(f"✅ Report written to {output_path}")
    if not HAVE_PANDAS:
        print("⚠  pandas/pytables not available — labeled-data duplicate check fell back to raw byte hashing "
              "(won't catch relabels under a different scorer name). `pip install pandas tables` for content-level comparison.")
    return output_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("project_root", help="Path to the DataProject root (contains raw_videos/, frames_store/)")
    ap.add_argument("-o", "--output", default="dlm_report.html", help="Output HTML path")
    ap.add_argument("--similarity-threshold", type=float, default=0.7,
                     help="Jaccard threshold on bodyparts for 'similar' (not identical) config groups (default 0.7)")
    ap.add_argument("--check-frame-duplicates", action="store_true",
                     help="Also hash every PNG frame to find byte-identical duplicate frames (slower)")
    args = ap.parse_args()

    try:
        generate_report(
            args.project_root, output=args.output,
            similarity_threshold=args.similarity_threshold,
            check_frame_duplicates=args.check_frame_duplicates,
        )
    except (FileNotFoundError, ValueError) as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
