from label_store import init_store, add_video, extract_frames_for_video

config_path = init_store(
    store_path=r"C:\Users\Egor\Documents\GitHub\DLC data managenet\data\frames_store",
    numframes2pick=50,
)

add_video(config_path, r"C:\Users\Egor\Documents\GitHub\DLC data managenet\data\raw_videos\HDMI-A.mp4")
extract_frames_for_video(config_path, r"C:\Users\Egor\Documents\GitHub\DLC data managenet\data\raw_videos\HDMI-A.mp4")

# Once you're ready to label (not needed just to extract frames above):
#
# from label_store import add_labeling_schema
#
# add_labeling_schema(
#     config_path,
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