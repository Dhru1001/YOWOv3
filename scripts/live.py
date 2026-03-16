import torch
import torchvision.transforms.functional as FT

import os
import cv2

from utils.box import draw_bounding_box, non_max_suppression
from model.TSN.YOWOv3 import build_yowov3
from utils.build_config import build_config
from PIL import Image
from utils.flops import get_info


class live_transform():
    def __init__(self, img_size):
        self.img_size = img_size

    def to_tensor(self, image):
        return FT.to_tensor(image)

    def normalize(self, clip):
        mean = __import__('torch').FloatTensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std  = __import__('torch').FloatTensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        return (clip - mean) / std

    def __call__(self, img):
        img = img.resize([self.img_size, self.img_size])
        img = self.to_tensor(img)
        img = self.normalize(img)
        return img


def detect(config):
    """
    Headless video inference — no display, saves result to output_path.

    Required config keys:
        video_path  : path to input  .mp4 / .avi / etc.
        output_path : path to output .mp4
        img_size    : model input spatial size
        idx2name    : class index → name mapping

    Optional config keys:
        clip_length : temporal window  (default 16)
        conf_thresh : confidence threshold (default 0.5)
        iou_thresh  : NMS IoU threshold    (default 0.5)
    """

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_yowov3(config)
    get_info(config, model)
    model.to('cuda')
    model.eval()

    mapping     = config['idx2name']
    img_size    = config['img_size']
    clip_length = config.get('clip_length', 16)
    conf_thresh = config.get('conf_thresh', 0.5)
    iou_thresh  = config.get('iou_thresh',  0.5)

    # ── Video source ──────────────────────────────────────────────────────────
    video_path = config.get('video_path', None)
    if not video_path or not os.path.isfile(video_path):
        raise FileNotFoundError(f'Video file not found: {video_path}')

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {video_path}')

    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'[Video] Input  : {video_path}  ({total} frames @ {fps:.1f} fps)',
          flush=True)

    # ── Output writer ─────────────────────────────────────────────────────────
    output_path = config.get('output_path', 'output.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (img_size, img_size))
    if not writer.isOpened():
        raise RuntimeError(f'Cannot open output writer: {output_path}')
    print(f'[Video] Output : {output_path}', flush=True)

    # ── Inference loop ────────────────────────────────────────────────────────
    transform  = live_transform(img_size)
    frame_list = []
    frame_idx  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % 50 == 0 or frame_idx == 1:
            print(f'[Video] Processing frame {frame_idx}/{total}', flush=True)

        # Add to sliding window
        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_list.append(transform(pil_frame))
        if len(frame_list) > clip_length:
            frame_list.pop(0)

        vis_frame = cv2.resize(frame, (img_size, img_size))

        # Buffer not full yet — write raw frame
        if len(frame_list) < clip_length:
            writer.write(vis_frame)
            continue

        # Build clip [1, C, T, H, W]
        clip = torch.stack(frame_list, dim=0)         # [T, C, H, W]
        clip = clip.permute(1, 0, 2, 3).contiguous()  # [C, T, H, W]
        clip = clip.unsqueeze(0).to('cuda')            # [1, C, T, H, W]

        with torch.no_grad():
            outputs = model(clip)

        detections = non_max_suppression(
            outputs,
            conf_threshold=conf_thresh,
            iou_threshold=iou_thresh
        )[0]

        if detections is not None and len(detections):
            draw_bounding_box(
                vis_frame,
                detections[:, :4],
                detections[:, 5],
                detections[:, 4],
                mapping
            )

        writer.write(vis_frame)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    writer.release()
    print(f'[Video] Done — {frame_idx} frames written to: {output_path}',
          flush=True)


if __name__ == '__main__':
    config = build_config()
    detect(config)