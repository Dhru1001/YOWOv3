"""
A2Seek Dataset Loader for YOWOv3
Mirrors the UCF_dataset interface: returns (clip, boxes, labels)

A2Seek annotation format (per frame JSON):
{
  "frame_id": "00070",
  "video_name": "video_001",
  "anomaly_category": "fighting",
  "bboxes": [[x1_norm, y1_norm, x2_norm, y2_norm], ...],
  "labels": [class_id, ...]
}

Directory structure expected:
<data_root>/
  rgb-images/
    <video_name>/
      00001.jpg
      00002.jpg
      ...
  labels/
    <video_name>/
      00001.json   (or .txt in UCF format: class_id x1 y1 x2 y2)
  trainlist.txt    (one entry per keyframe: labels/<video_name>/00070.json)
  testlist.txt
"""

import torch
import torch.utils.data as data
import os
import json
import numpy as np
from PIL import Image
from .transforms import Augmentation, A2SeekHighResTransform


# 20 anomaly classes in A2Seek
A2SEEK_CLASSES = [
    "fighting",        # 0
    "robbery",         # 1
    "falling",         # 2
    "water_play",      # 3
    "jaywalking",      # 4
    "climbing_wall",   # 5
    "sneaking",        # 6
    "running",         # 7
    "gathering",       # 8
    "chasing",         # 9
    "trespassing",     # 10
    "vandalism",       # 11
    "loitering",       # 12
    "illegal_parking", # 13
    "red_light",       # 14
    "throwing",        # 15
    "carrying",        # 16
    "pushing",         # 17
    "kicking",         # 18
    "other",           # 19
]
NUM_CLASSES = len(A2SEEK_CLASSES)
CLASS2IDX = {c: i for i, c in enumerate(A2SEEK_CLASSES)}


class A2Seek_dataset(data.Dataset):

    def __init__(self, root_path, split_path, data_path, ann_path,
                 clip_length, sampling_rate, img_size, phase,
                 transform=None):
        self.root_path     = root_path
        self.split_path    = os.path.join(root_path, split_path)
        self.data_path     = os.path.join(root_path, data_path)   # rgb-images/
        self.ann_path      = os.path.join(root_path, ann_path)     # labels/
        self.clip_length   = clip_length
        self.sampling_rate = sampling_rate
        self.img_size      = img_size
        self.phase         = phase
        self.transform     = transform if transform is not None else Augmentation(img_size=img_size)

        with open(self.split_path, 'r') as f:
            self.lines = [l.rstrip() for l in f.readlines() if l.strip()]

        self.nSample = len(self.lines)

    def __len__(self):
        return self.nSample

    def __getitem__(self, index, get_origin_image=False):
        """
        Returns:
            clip   : Tensor [C, T, H, W]
            boxes  : Tensor [N, 4]  normalized xyxy
            labels : Tensor [N, NUM_CLASSES] one-hot (train) or [N] class_id (test)
        """
        ann_rel_path = self.lines[index]           # e.g. labels/video_001/00070.json
        parts        = ann_rel_path.split('/')
        video_name   = parts[-2]
        frame_str    = parts[-1].split('.')[0]     # e.g. "00070"
        key_frame_idx = int(frame_str)

        video_path = os.path.join(self.data_path, video_name)
        ann_dir    = os.path.join(self.ann_path,  video_name)

        # ── Build clip (T frames ending at key_frame_idx) ──────────────
        clip = []
        for i in reversed(range(self.clip_length)):
            # Calculate index based on the 16-frame stride
            cur_idx = key_frame_idx - i * self.sampling_rate
            
            # Ensure we don't go below 0 (since your files start at 00000.jpg)
            cur_idx = max(cur_idx, 0) 
            
            frame_path = os.path.join(video_path, '{:05d}.jpg'.format(cur_idx))
            
            # Fallback: if the specific frame doesn't exist, use the key_frame
            if not os.path.exists(frame_path):
                frame_path = os.path.join(video_path, '{:05d}.jpg'.format(key_frame_idx))
            
            img = Image.open(frame_path).convert('RGB')
            clip.append(img)

        if get_origin_image:
            kf_path = os.path.join(video_path, '{:05d}.jpg'.format(key_frame_idx))
            import cv2
            original_image = cv2.imread(kf_path)

        # ── Load annotation for key frame ──────────────────────────────
        ann_file = os.path.join(ann_dir, '{:05d}.json'.format(key_frame_idx))
        boxes, labels = self._parse_annotation(ann_file)
        if len(boxes) == 0:
            boxes = np.zeros((0,4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.float32)

        boxes  = np.array(boxes,  dtype=np.float32)   # [N, 4]
        labels = np.array(labels, dtype=np.float32)   # [N] class_ids

        # one-hot for train
        if self.phase == 'train':
            onehot = np.zeros((len(labels), NUM_CLASSES), dtype=np.float32)
            for i, lbl in enumerate(labels):
                onehot[i, int(lbl)] = 1.0
            labels = onehot                           # [N, NUM_CLASSES]
        else:
            labels = np.expand_dims(labels, axis=1)  # [N, 1]

        targets = np.concatenate((boxes, labels), axis=1)   # [N, 4+C]

        clip, targets = self.transform(clip, targets)

        boxes = targets[:, :4]
        if self.phase == 'train':
            labels = targets[:, 4:]
        else:
            labels = targets[:, -1]

        if get_origin_image:
            return original_image, clip, boxes, labels
        
        # if index == 0: # Check the first sample
            # print(f"DEBUG TARGETS: {targets[0, :4]}")

        return clip, boxes, labels

    # ────────────────────────────────────────────────────────────────────
    def _parse_annotation(self, ann_file):
        """
        Supports two formats:
          1) JSON:  {"bboxes": [[x1,y1,x2,y2],...], "labels": [class_id,...]}
          2) TXT (UCF-style): one line per box  → class_id x1 y1 x2 y2
        """
        boxes, labels = [], []

        if ann_file.endswith('.json'):
            with open(ann_file, 'r') as f:
                ann = json.load(f)
            for bbox, lbl in zip(ann['bboxes'], ann['labels']):
                boxes.append(bbox)
                # lbl can be int index or string class name
                if isinstance(lbl, str):
                    lbl = CLASS2IDX.get(lbl, 0)
                labels.append(float(lbl))
        else:
            # fallback: UCF-style txt  → class_id x1 y1 x2 y2
            txt_file = ann_file.replace('.json', '.txt')
            with open(txt_file, 'r') as f:
                for line in f.readlines():
                    parts = line.rstrip().split()
                    labels.append(float(int(parts[0]) - 1))  # 1-indexed → 0-indexed
                    boxes.append([float(p) for p in parts[1:5]])

        return boxes, labels


# ────────────────────────────────────────────────────────────────────────────
def build_a2seek_dataset(config, phase):
    root_path     = config['data_root']
    data_path     = 'rgb-images'
    ann_path      = 'labels'
    clip_length   = config['clip_length']
    sampling_rate = config['sampling_rate']
    img_size      = config['img_size']

    if phase == 'train':
        return A2Seek_dataset(
            root_path, 'trainlist.txt', data_path, ann_path,
            clip_length, sampling_rate, img_size, phase,
            transform=A2SeekHighResTransform(img_size=img_size)
        )
    else:
        return A2Seek_dataset(
            root_path, 'testlist.txt', data_path, ann_path,
            clip_length, sampling_rate, img_size, phase,
            transform=A2SeekHighResTransform(img_size=img_size)
        )