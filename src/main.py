from label_store import (
    init_store,
    extract_frames_for_video,
    init_labeling_config,
    ExtractionConfig,
    save_extraction_config,
    list_extractions,
)

STORE = r"C:\Users\Egor\Documents\GitHub\DLC data managenet\data\frames_store"
VIDEO = r"C:\Users\Egor\Documents\GitHub\DLC data managenet\data\raw_videos\HDMI-A.mp4"

store_path = init_store(STORE, numframes2pick=50)  # seeds extraction_configs/default.yaml

# Extract with the default preset -> labeled-data/HDMI-A__default/...
extract_frames_for_video(store_path, VIDEO, config_name="default")

# --- Duplicate extraction, same video, different preset -----------------
# Because both video AND config are named explicitly, this lands in its own
# folder (HDMI-A__dense) instead of overwriting the run above.
#
save_extraction_config(store_path, ExtractionConfig(name="dense", numframes2pick=150, algo="uniform"))
extract_frames_for_video(store_path, VIDEO, config_name="dense")

print(list_extractions(store_path))

# --- Once you're ready to label a SPECIFIC frame set ---------------------
# (not needed just to extract frames above; bodyparts can differ per folder)
#
# init_labeling_config(
#     store_path, "HDMI-A__default",
#     scorer="Egor",
#     bodyparts=[
#         "right_mitten_wrist", "right_mitten_tip",
#         "left_mitten_wrist", "left_mitten_tip",
#         "1_corner_table", "2_corner_table", "3_corner_table", "4_corner_table",
#     ],
#     skeleton=[
#         ["right_mitten_wrist", "right_mitten_tip"],
#         ["left_mitten_wrist", "left_mitten_tip"],
#         ["1_corner_table", "2_corner_table"],
#         ["2_corner_table", "3_corner_table"],
#         ["3_corner_table", "4_corner_table"],
#         ["4_corner_table", "1_corner_table"],
#     ],
# )
#
# Your existing 3c_label_video.py should work unchanged, pointed at:
#   store_path/labeled-data/HDMI-A__default/config.yaml