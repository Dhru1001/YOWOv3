# # import os
# # import json
# # import glob
# # from tqdm import tqdm

# # VIDEOS = "/fsxnew/dhrumil.shah/A2Seek_data/A2Seek/The_Splited/A2Seek"
# # LABELS = "/fsxnew/dhrumil.shah/A2Seek_data/A2Seek/Labels/All_Frame_Labels"
# # OUT = "/fsxnew/dhrumil.shah/A2Seek_data/processed"

# # RGB = os.path.join(OUT, "rgb-images")
# # LBL = os.path.join(OUT, "labels")

# # os.makedirs(RGB, exist_ok=True)
# # os.makedirs(LBL, exist_ok=True)


# # # -------------------------------------------------
# # # COPY / LINK FRAMES
# # # -------------------------------------------------

# # print("Preparing rgb-images...")

# # for split in ["train", "test"]:

# #     split_dir = os.path.join(VIDEOS, split)

# #     for vid in tqdm(os.listdir(split_dir)):

# #         src = os.path.join(split_dir, vid)

# #         if not os.path.isdir(src):
# #             continue

# #         vid_id = vid.replace("@", "_@_")

# #         dst = os.path.join(RGB, vid_id)

# #         os.makedirs(dst, exist_ok=True)

# #         for f in os.listdir(src):

# #             if not f.endswith(".jpg"):
# #                 continue

# #             s = os.path.join(src, f)
# #             d = os.path.join(dst, f)

# #             if not os.path.exists(d):
# #                 os.symlink(s, d)


# # # -------------------------------------------------
# # # BUILD LABELS
# # # -------------------------------------------------

# # train_list = []
# # test_list = []

# # print("Building labels...")

# # for jf in tqdm(glob.glob(LABELS + "/*.json")):

# #     with open(jf) as f:
# #         data = json.load(f)

# #     vgroup = data["video_group"]
# #     vname = data["video_name"]

# #     vid_id = f"{vgroup}_@_{vname}"

# #     rgb_dir = os.path.join(RGB, vid_id)

# #     if not os.path.exists(rgb_dir):
# #         continue

# #     # find split
# #     if os.path.exists(os.path.join(VIDEOS, "train", f"{vgroup}@{vname}")):
# #         split = "train"
# #     else:
# #         split = "test"

# #     W = data["attribute"]["width"]
# #     H = data["attribute"]["height"]

# #     out_dir = os.path.join(LBL, vid_id)
# #     os.makedirs(out_dir, exist_ok=True)

# #     for fid, finfo in data["info"].items():

# #         objs = finfo.get("anomaly_objects", [])

# #         if not objs:
# #             continue

# #         bboxes = []
# #         labels = []

# #         for obj in objs:

# #             cls = int(obj.get("anomaly_type", "A20")[1:]) - 1

# #             for box in obj.get("bbox", []):

# #                 x1 = box[0] / W
# #                 y1 = box[1] / H
# #                 x2 = box[2] / W
# #                 y2 = box[3] / H

# #                 if x2 <= x1 or y2 <= y1:
# #                     continue

# #                 bboxes.append([x1, y1, x2, y2])
# #                 labels.append(cls)

# #         if not bboxes:
# #             continue

# #         fid = int(fid)

# #         out = os.path.join(out_dir, f"{fid:05d}.json")

# #         with open(out, "w") as f:
# #             json.dump({"bboxes": bboxes, "labels": labels}, f)

# #         entry = f"labels/{vid_id}/{fid:05d}.json"

# #         if split == "train":
# #             train_list.append(entry)
# #         else:
# #             test_list.append(entry)


# # # -------------------------------------------------
# # # SAVE LISTS
# # # -------------------------------------------------

# # with open(os.path.join(OUT, "trainlist.txt"), "w") as f:
# #     f.write("\n".join(train_list))

# # with open(os.path.join(OUT, "testlist.txt"), "w") as f:
# #     f.write("\n".join(test_list))

# # print("Train:", len(train_list))
# # print("Test :", len(test_list))



# import os
# import json
# import glob
# from tqdm import tqdm

# VIDEOS = "/fsxnew/dhrumil.shah/A2Seek_data/A2Seek/The_Focused/Focused_Image/"
# LABELS = "/fsxnew/dhrumil.shah/A2Seek_data/A2Seek/Labels/All_Frame_Labels"
# OUT = "/fsxnew/dhrumil.shah/A2Seek_data/processed"

# RGB = os.path.join(OUT, "rgb-images")
# LBL = os.path.join(OUT, "labels")

# os.makedirs(RGB, exist_ok=True)
# os.makedirs(LBL, exist_ok=True)


# # -------------------------------------------------
# # COPY / LINK FRAMES
# # -------------------------------------------------

# print("Preparing rgb-images...")

# # for split in ["train", "test"]:

# #     split_dir = os.path.join(VIDEOS, split)

# #     for vid in tqdm(os.listdir(split_dir)):

# #         src = os.path.join(split_dir, vid)

# #         if not os.path.isdir(src):
# #             continue

# #         vid_id = vid.replace("@", "_@_")

# #         dst = os.path.join(RGB, vid_id)

# #         os.makedirs(dst, exist_ok=True)

# #         for f in os.listdir(src):

# #             if not f.endswith(".jpg"):
# #                 continue

# #             s = os.path.join(src, f)
# #             d = os.path.join(dst, f)

# #             if not os.path.exists(d):
# #                 os.symlink(s, d)

# for vid_id in tqdm(os.listdir(VIDEOS)):
#     src_video_dir = os.path.join(VIDEOS, vid_id, "resized")
    
#     if not os.path.isdir(src_video_dir):
#         continue

#     dst_video_dir = os.path.join(RGB, vid_id)
#     os.makedirs(dst_video_dir, exist_ok=True)

#     # Link all frames (00000.jpg, 00016.jpg, etc.)
#     for f in os.listdir(src_video_dir):
#         if f.endswith(".jpg"):
#             s = os.path.join(src_video_dir, f)
#             d = os.path.join(dst_video_dir, f)
#             if not os.path.exists(d):
#                 os.symlink(s, d)
# # -------------------------------------------------
# # BUILD LABELS
# # -------------------------------------------------

# train_list = []
# test_list = []

# print("Building labels...")

# for jf in tqdm(glob.glob(LABELS + "/*.json")):

#     with open(jf) as f:
#         data = json.load(f)

#     vgroup = data["video_group"]
#     vname = data["video_name"]

#     vid_id = f"{vgroup}_@_{vname}"

#     rgb_dir = os.path.join(RGB, vid_id)

#     if not os.path.exists(rgb_dir):
#         continue

#     # detect split
#     if os.path.exists(os.path.join(VIDEOS, "train", f"{vgroup}@{vname}")):
#         split = "train"
#     else:
#         split = "test"

#     W = data["attribute"]["width"]
#     H = data["attribute"]["height"]

#     out_dir = os.path.join(LBL, vid_id)
#     os.makedirs(out_dir, exist_ok=True)

#     # --- FIX: get actual existing frames ---
#     existing_frames = {
#         int(f.split(".")[0])
#         for f in os.listdir(rgb_dir)
#         if f.endswith(".jpg")
#     }

#     for fid, finfo in data["info"].items():

#         fid = int(fid)

#         # --- FIX: skip frames that don't exist ---
#         if fid not in existing_frames:
#             continue

#         objs = finfo.get("anomaly_objects", [])

#         if not objs:
#             continue

#         bboxes = []
#         labels = []

#         for obj in objs:

#             cls = int(obj.get("anomaly_type", "A20")[1:]) - 1

#             for box in obj.get("bbox", []):

#                 x1 = box[0] / W
#                 y1 = box[1] / H
#                 x2 = box[2] / W
#                 y2 = box[3] / H

#                 if x2 <= x1 or y2 <= y1:
#                     continue

#                 bboxes.append([x1, y1, x2, y2])
#                 labels.append(cls)

#         if not bboxes:
#             continue

#         out = os.path.join(out_dir, f"{fid:05d}.json")

#         with open(out, "w") as f:
#             json.dump({"bboxes": bboxes, "labels": labels}, f)

#         entry = f"labels/{vid_id}/{fid:05d}.json"

#         if split == "train":
#             train_list.append(entry)
#         else:
#             test_list.append(entry)


# # -------------------------------------------------
# # SAVE LISTS
# # -------------------------------------------------

# with open(os.path.join(OUT, "trainlist.txt"), "w") as f:
#     f.write("\n".join(train_list))

# with open(os.path.join(OUT, "testlist.txt"), "w") as f:
#     f.write("\n".join(test_list))

# print("Train:", len(train_list))
# print("Test :", len(test_list))




import os
import json
import glob
from tqdm import tqdm

# --- PATH CONFIGURATION ---
# Where the focused images actually are
VIDEOS_ROOT = "/fsxnew/dhrumil.shah/A2Seek_data/A2Seek/The_Focused/Focused_Image"
# Where the original train/test folder structure lives (to detect the split)
SPLIT_SOURCE = "/fsxnew/dhrumil.shah/A2Seek_data/A2Seek/The_Splited/A2Seek"
# Where the raw JSON labels are
LABELS_ROOT = "/fsxnew/dhrumil.shah/A2Seek_data/A2Seek/Labels/All_Frame_Labels"
# Where the processed output should go
OUT = "/fsxnew/dhrumil.shah/A2Seek_data/processed"

RGB = os.path.join(OUT, "rgb-images")
LBL = os.path.join(OUT, "labels")

os.makedirs(RGB, exist_ok=True)
os.makedirs(LBL, exist_ok=True)

# -------------------------------------------------
# 1. PREPARE RGB IMAGES (Linking from Focused_Image)
# -------------------------------------------------
print("Linking RGB images from Focused_Image...")

# Focused_Image contains folders named 'video_group_XXX_@_video_X'
for vid_id in tqdm(os.listdir(VIDEOS_ROOT)):
    src_video_dir = os.path.join(VIDEOS_ROOT, vid_id, "resized")
    
    if not os.path.isdir(src_video_dir):
        continue

    # We keep the vid_id name (group_@_video) for the processed folder
    dst_video_dir = os.path.join(RGB, vid_id)
    os.makedirs(dst_video_dir, exist_ok=True)

    for f in os.listdir(src_video_dir):
        if f.endswith(".jpg"):
            s = os.path.join(src_video_dir, f)
            d = os.path.join(dst_video_dir, f)
            if not os.path.exists(d):
                os.symlink(s, d)

# -------------------------------------------------
# 2. BUILD LABELS & SPLIT LISTS
# -------------------------------------------------
train_list = []
test_list = []

print("Building labels and detecting splits...")

for jf in tqdm(glob.glob(LABELS_ROOT + "/*.json")):
    with open(jf) as f:
        data = json.load(f)

    vgroup = data["video_group"]
    vname = data["video_name"]

    # vid_id matches the folder name in Focused_Image (group_@_video)
    # folder_name matches the folder name in The_Splited (group@video)
    vid_id = f"{vgroup}_@_{vname}"
    folder_name = f"{vgroup}@{vname}"

    rgb_dir = os.path.join(RGB, vid_id)

    # Only process if we actually have the images for this video
    if not os.path.exists(rgb_dir):
        continue

    # CORRECT SPLIT DETECTION
    # We check if the video folder exists in the original 'train' directory
    if os.path.exists(os.path.join(SPLIT_SOURCE, "train", folder_name)):
        split = "train"
    elif os.path.exists(os.path.join(SPLIT_SOURCE, "test", folder_name)):
        split = "test"
    else:
        # Fallback if split cannot be determined
        continue

    W = data["attribute"]["width"]
    H = data["attribute"]["height"]

    out_dir = os.path.join(LBL, vid_id)
    os.makedirs(out_dir, exist_ok=True)

    # Cache existing frames to avoid repeated disk hits
    existing_frames = {
        int(f.split(".")[0])
        for f in os.listdir(rgb_dir)
        if f.endswith(".jpg")
    }

    for fid, finfo in data["info"].items():
        fid_int = int(fid)

        # Skip if the image frame doesn't exist in our focused set
        if fid_int not in existing_frames:
            continue

        objs = finfo.get("anomaly_objects", [])
        if not objs:
            continue

        bboxes = []
        labels = []

        for obj in objs:
            # Format: "A01" -> index 0
            cls = int(obj.get("anomaly_type", "A20")[1:]) - 1

            for box in obj.get("bbox", []):
                # Normalize coordinates
                x1, y1, x2, y2 = box[0]/W, box[1]/H, box[2]/W, box[3]/H

                # Valid box check
                if x2 <= x1 or y2 <= y1:
                    continue

                bboxes.append([x1, y1, x2, y2])
                labels.append(cls)

        if not bboxes:
            continue

        # Save individual frame annotation
        out_json_path = os.path.join(out_dir, f"{fid_int:05d}.json")
        with open(out_json_path, "w") as f:
            json.dump({"bboxes": bboxes, "labels": labels}, f)

        # Record entry for trainlist/testlist
        entry = f"labels/{vid_id}/{fid_int:05d}.json"
        if split == "train":
            train_list.append(entry)
        else:
            test_list.append(entry)

# -------------------------------------------------
# 3. SAVE FINAL LISTS
# -------------------------------------------------
with open(os.path.join(OUT, "trainlist.txt"), "w") as f:
    f.write("\n".join(train_list))

with open(os.path.join(OUT, "testlist.txt"), "w") as f:
    f.write("\n".join(test_list))

print(f"--- Processing Complete ---")
print(f"Train samples: {len(train_list)}")
print(f"Test samples : {len(test_list)}")