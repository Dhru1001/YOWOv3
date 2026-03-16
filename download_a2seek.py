# from huggingface_hub import snapshot_download

# snapshot_download(
#     repo_id="Hayneyday/A2Seek",
#     repo_type="dataset",
#     local_dir="/fsxnew/dhrumil.shah/A2Seek_data/A2Seek",
#     allow_patterns=[
#         "The_Focused/*",
#         "The_Splited/*",
#         "Labels/All_Frame_Labels.rar"
#     ]
# )

import os
from huggingface_hub import snapshot_download

# Define the local destination
OUT_DIR = "/fsxnew/dhrumil.shah/YOWOv3/"
os.makedirs(OUT_DIR, exist_ok=True)

# The repository ID
REPO = "manh6054/YOWOv3"

# Using wildcards to ensure we catch them even if there's a slight path variation
required_patterns = [
    "**/v8_s.pth",
    "**/kinetics_shufflenetv2_2.0x_RGB_16_best.pth"
]

print(f"Downloading required weights to {OUT_DIR}...")

try:
    snapshot_download(
        repo_id=REPO,
        local_dir=OUT_DIR,
        allow_patterns=required_patterns,
        repo_type="model" # Explicitly setting repo_type
    )
    print("\nDownload finished. Checking folder structure...")
    
    # Quick check to see if files exist
    for root, dirs, files in os.walk(OUT_DIR):
        for file in files:
            if file.endswith(".pth"):
                print(f"Found: {os.path.join(root, file)}")
                
except Exception as e:
    print(f"An error occurred: {e}")