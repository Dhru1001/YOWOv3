import torch
import torchvision.transforms.functional as FT
import numpy as np
from PIL import Image

def normalize(clip):
    """Stack images into a 4D tensor and normalize."""
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    out = []
    for img in clip:
        img = FT.to_tensor(img)  # Converts to [0, 1] and [C, H, W]
        img = FT.normalize(img, mean=mean, std=std)
        out.append(img)
    
    # Final shape: [C, T, H, W]
    clip_tensor = torch.stack(out, dim=0).permute(1, 0, 2, 3)
    return clip_tensor

class A2SeekHighResTransform:
    """
    H100 Optimized Transform:
    Keeps Aspect Ratio via Letterboxing (Padding)
    Prevents 'Needle' boxes and INF loss.
    """
    def __init__(self, img_size=640):
        self.img_size = img_size if isinstance(img_size, int) else img_size[0]

    def __call__(self, clip, targets):
        # 1. Calculate Resize Scale keeping Aspect Ratio
        w_orig, h_orig = clip[0].size
        ratio = min(self.img_size / w_orig, self.img_size / h_orig)
        nw, nh = int(w_orig * ratio), int(h_orig * ratio)
        
        # 2. Calculate Padding offsets to center the image
        dw, dh = (self.img_size - nw) // 2, (self.img_size - nh) // 2

        # 3. Resize and Pad Clip
        new_clip = []
        for img in clip:
            img_resized = img.resize((nw, nh), Image.BILINEAR)
            # Create gray canvas (128, 128, 128) - better for training than pure black
            padded_img = Image.new("RGB", (self.img_size, self.img_size), (128, 128, 128))
            padded_img.paste(img_resized, (dw, dh))
            new_clip.append(padded_img)

        # 4. Correct Bounding Boxes
        # targets: [N, 4+C] where 0:4 is [x1, y1, x2, y2] normalized 0-1
        new_targets = targets.copy()
        if len(targets) > 0:
            # Scale old normalized coordinates to resized pixels, add padding, re-normalize to 640
            new_targets[:, 0] = (targets[:, 0] * nw + dw) / self.img_size # x1
            new_targets[:, 1] = (targets[:, 1] * nh + dh) / self.img_size # y1
            new_targets[:, 2] = (targets[:, 2] * nw + dw) / self.img_size # x2
            new_targets[:, 3] = (targets[:, 3] * nh + dh) / self.img_size # y2
            
            # STABILITY CHECK: Prevent INF/NaN loss
            # Force x2 > x1 and y2 > y1 with a minimum 1-pixel width
            min_dim = 1.0 / self.img_size
            new_targets[:, 2] = np.maximum(new_targets[:, 2], new_targets[:, 0] + min_dim)
            new_targets[:, 3] = np.maximum(new_targets[:, 3], new_targets[:, 1] + min_dim)

        # 5. Final Normalization
        clip_tensor = normalize(new_clip)
        targets_tensor = torch.from_numpy(new_targets).float()
        
        return clip_tensor, targets_tensor

# Aliases for the dataset loader
Augmentation = A2SeekHighResTransform
A2Seek_transform = A2SeekHighResTransform