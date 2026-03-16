# import torch
# import cv2
# import numpy as np
# from cus_datasets.build_dataset import build_dataset
# from utils.build_config import build_config

# def debug_visual_samples(config_path, num_samples=5):
#     config = build_config(config_path)
#     # Force phase to train to check normalization/augmentation
#     dataset = build_dataset(config, phase='train')
    
#     # We want the original image too for drawing
#     for i in range(num_samples):
#         # Pick a random index
#         idx = np.random.randint(0, len(dataset))
        
#         # Get data from your actual loader
#         clip, boxes, labels = dataset[idx]
        
#         # clip is [C, T, H, W], we want the keyframe (last frame in T)
#         # Convert tensor back to numpy for CV2 [H, W, C]
#         keyframe = clip[:, -1, :, :].permute(1, 2, 0).numpy()
#         keyframe = (keyframe * 255).astype(np.uint8)
#         keyframe = cv2.cvtColor(keyframe, cv2.COLOR_RGB2BGR)
        
#         h, w, _ = keyframe.shape
        
#         print(f"Sample {i}: Index {idx}")
#         print(f"Boxes shape: {boxes.shape}, Labels shape: {labels.shape}")

#         for box, lbl in zip(boxes, labels):
#             # Convert normalized back to pixel values
#             x1, y1, x2, y2 = box
#             x1, x2 = int(x1 * w), int(x2 * w)
#             y1, y2 = int(y1 * h), int(y2 * h)
            
#             # Get class ID from one-hot
#             class_id = np.argmax(lbl)
            
#             # Draw
#             cv2.rectangle(keyframe, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(keyframe, f"ID:{class_id}", (x1, y1-5), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#         # Save to disk
#         out_name = f"debug_sample_{i}.jpg"
#         cv2.imwrite(out_name, keyframe)
#         print(f"Saved: {out_name}")

# if __name__ == "__main__":
#     debug_visual_samples("config/cf2/a2seek_config.yaml")


import torch
import cv2
import numpy as np
import os
from PIL import Image
from cus_datasets.build_dataset import build_dataset
from utils.build_config import build_config

def end_to_end_debug(config_path, num_samples=3):
    config = build_config(config_path)
    # Load dataset
    dataset = build_dataset(config, phase='train')
    
    # ImageNet stats for denormalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    print(f"\n{'='*60}")
    print(f"STARTING END-TO-END DEBUGGING")
    print(f"{'='*60}\n")

    for i in range(num_samples):
        idx = np.random.randint(0, len(dataset))
        
        # 1. Get Metadata manually from the dataset object to find the raw file
        ann_rel_path = dataset.lines[idx]
        parts = ann_rel_path.split('/')
        video_name = parts[-2]
        frame_str = parts[-1].split('.')[0]
        
        raw_img_path = os.path.join(dataset.data_path, video_name, f"{frame_str}.jpg")
        raw_pil = Image.open(raw_img_path).convert('RGB')
        raw_np = np.array(raw_pil)
        
        # 2. Get Processed Data from the Loader
        clip, boxes, labels = dataset[idx]
        
        # 3. Diagnostics for Printing
        print(f"--- SAMPLE {i+1} [Index: {idx}] ---")
        print(f"File Path: {raw_img_path}")
        print(f"RAW: Res: {raw_np.shape[1]}x{raw_np.shape[0]} | Channels: {raw_np.shape[2]} | Format: RGB (PIL)")
        
        # Clip is [C, T, H, W]
        processed_h, processed_w = clip.shape[2], clip.shape[3]
        print(f"PROCESSED: Res: {processed_w}x{processed_h} | Stride: {config['sampling_rate']} | Tensor: {clip.shape}")
        
        # 4. Process the Raw Image for Saving (Green Boxes)
        raw_bgr = cv2.cvtColor(raw_np, cv2.COLOR_RGB2BGR)
        for box in boxes:
            x1, y1, x2, y2 = box.numpy()
            cv2.rectangle(raw_bgr, (int(x1*raw_np.shape[1]), int(y1*raw_np.shape[0])), 
                          (int(x2*raw_np.shape[1]), int(y2*raw_np.shape[0])), (0, 255, 0), 3)

        # 5. Process the Model Input (Denormalize + Red Boxes)
        # Take keyframe, denormalize, convert to BGR
        model_eye = clip[:, -1, :, :] * std + mean
        model_eye = (model_eye.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        model_eye_bgr = cv2.cvtColor(model_eye, cv2.COLOR_RGB2BGR)
        
        for box in boxes:
            x1, y1, x2, y2 = box.numpy()
            cv2.rectangle(model_eye_bgr, (int(x1*processed_w), int(y1*processed_h)), 
                          (int(x2*processed_w), int(y2*processed_h)), (0, 0, 255), 2)

        # 6. Save results
        cv2.imwrite(f"debug_{i}_ORIGINAL.jpg", raw_bgr)
        cv2.imwrite(f"debug_{i}_MODEL_INPUT.jpg", model_eye_bgr)
        
        print(f"Labels Found: {labels.shape[0]} | One-Hot Sum: {labels.sum().item()}")
        print(f"Status: Files saved as debug_{i}_ORIGINAL.jpg and debug_{i}_MODEL_INPUT.jpg\n")

if __name__ == "__main__":
    end_to_end_debug("config/cf2/a2seek_config.yaml")