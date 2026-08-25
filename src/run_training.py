"""
run_training.py
────────────────
Usage script (not part of the dlc_manager package itself — this lives
alongside your data project, same idea as the old create_training_dataset.py
/ train_network.py pair).

1. Resolves the data project (raw_videos/, frames_store/, network_store/).
2. Creates (or reuses) a network project.
3. Adds each imported label_store frame set to it (the folder_ids below
   come straight from your import_legacy_project() run log).
4. Builds the training dataset (resnet_50, 1 shuffle — same as before).
5. Trains (same epochs/save_epochs/display_iters/batch_size/scheduler as
   before), optionally logged to Weights & Biases.
"""

import dlc_manager as dlm

# ── Edit these two ──────────────────────────────────────────────────────
PROJECT_ROOT = r'/home/ccldlc/Desktop/DLC_project/'     # the DataProject root
NETWORK_PROJECT_NAME = "DLC_good_old_black(missing some data)"         # name for this network project
# ─────────────────────────────────────────────────────────────────────────

# Folder IDs from the import_legacy_project() run — the two "no source
# video found" skips are left out on purpose, everything else that was
# actually imported is included.
IMPORTED_FOLDER_IDS = [
    "SCENE Object 1 HDMI-A_Jun23_11-42-33_synced__imported__2",
    "SCENE Object 1 HDMI-A_May12_12-26-24_synced_10__imported",
    "SCENE Object 1 HDMI-A_May12_12-26-24_synced_11__imported",
    "SCENE Object 1 HDMI-A_May12_12-26-24_synced_12__imported",
    "SCENE Object 1 HDMI-A_May12_12-26-24_synced_15__imported",
    "SCENE Object 1 HDMI-A_May12_12-26-24_synced_16__imported",
    "SCENE Object 1 HDMI-A_May12_12-26-24_synced_17__imported",
    "SCENE Object 1 HDMI-A_May12_12-26-24_synced_19__imported",
    "SCENE Object 1 HDMI-A_May12_12-26-24_synced_2__imported",
    "SCENE Object 1 HDMI-A_May12_12-26-24_synced_5__imported",
    "SCENE Object 1 HDMI-A_May12_12-26-24_synced_7__imported",
    "SCENE Object 1 SDI_Jun23_11-42-33_synced__imported",
]


def main():
    prj = dlm.init_data_project(PROJECT_ROOT)

    # Reuse the network project if it already exists (e.g. re-running this
    # script after adding more imported frame sets); create it otherwise.
    try:
        net = prj.get_network_project(NETWORK_PROJECT_NAME)
    except FileNotFoundError:
        net = prj.create_network_project(name=NETWORK_PROJECT_NAME)

    for folder_id in IMPORTED_FOLDER_IDS:
        net.add_labeled_data(prj.frame_set_path(folder_id), overwrite=True)

    net.create_train_dataset(net_type="resnet_50", num_shuffles=1)

    net.train_network(
        shuffle=1,
        epochs=600,
        save_epochs=25,
        display_iters=500,
        batch_size=24,
        pytorch_cfg_updates={
            "runner.scheduler.params.milestones": [350, 480],
        },
        # Comment these three out to skip W&B logging entirely.
        wandb_project="mitten-tracker",
        wandb_run_name=None,   # defaults to "<network-project-name>-shuffle1"
        wandb_tags=["resnet50", "shuffle1"],
    )


if __name__ == "__main__":
    main()
