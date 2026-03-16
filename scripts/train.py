import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os
import shutil

from utils.EMA import EMA
from utils.build_config import build_config
from cus_datasets.collate_fn import collate_fn
from cus_datasets.build_dataset import build_dataset
from model.TSN.YOWOv3 import build_yowov3
from utils.loss import build_loss
from utils.warmup_lr import LinearWarmup
from utils.flops import get_info

# H100 opts — set once at module level
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32        = True
torch.backends.cudnn.benchmark         = True


def train_model(config):

    # ── 1. Distributed setup — identical to original ──────────────────────────
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    else:
        local_rank = 0
        device     = torch.device('cuda')

    is_main = (local_rank == 0)

    # ── 2. Save folder — identical to original ────────────────────────────────
    save_folder = config['save_folder']
    if is_main:
        os.makedirs(save_folder, exist_ok=True)
        shutil.copyfile(config['config_path'],
                        os.path.join(save_folder, 'config.yaml'))

    # ── 3. TensorBoard (new, main process only) ───────────────────────────────
    writer = None
    if is_main:
        tb_log_dir = os.path.join(save_folder, 'tensorboard')
        os.makedirs(tb_log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_log_dir)
        print(f'[TensorBoard] logdir : {tb_log_dir}', flush=True)
        print(f'[TensorBoard] run    : tensorboard --logdir {tb_log_dir}', flush=True)

    # ── 4. Dataset & DataLoader — same as original, +persistent_workers ───────
    dataset = build_dataset(config, phase='train')
    sampler = DistributedSampler(dataset) if dist.is_initialized() else None

    dataloader = data.DataLoader(
        dataset,
        batch_size         = config['batch_size'],
        shuffle            = (sampler is None),
        sampler            = sampler,
        collate_fn         = collate_fn,
        num_workers        = config['num_workers'],
        pin_memory         = True,
        persistent_workers = True,
        prefetch_factor    = 4,
    )

    # ── 5. Model — identical to original ─────────────────────────────────────
    model = build_yowov3(config)
    get_info(config, model)
    model.to(device)

    raw_model = model  # unwrapped ref for EMA + saving

    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    model.train()

    # ── 6. Loss — identical to original, NO barriers ─────────────────────────
    criterion = build_loss(model, config)

    # ── 7. Optimizer — identical to original ─────────────────────────────────
    bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)
    g  = [], [], []
    for v in model.modules():
        for p_name, p in v.named_parameters(recurse=0):
            if p_name == 'bias':
                g[2].append(p)
            elif p_name == 'weight' and isinstance(v, bn):
                g[1].append(p)
            else:
                g[0].append(p)

    optimizer = torch.optim.AdamW(g[0], lr=config['lr'],
                                  weight_decay=config['weight_decay'])
    optimizer.add_param_group({'params': g[1], 'weight_decay': 0.0})
    optimizer.add_param_group({'params': g[2], 'weight_decay': 0.0})

    # ── 8. AMP — BF16 on H100 (new) ──────────────────────────────────────────
    use_amp   = config.get('use_amp', True)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler    = torch.amp.GradScaler('cuda',
                    enabled=(use_amp and amp_dtype == torch.float16))
    if is_main:
        print(f'[AMP] enabled={use_amp}  dtype={amp_dtype}', flush=True)

    # ── 9. Warmup & EMA — identical to original ───────────────────────────────
    warmup_lr = LinearWarmup(config)
    ema       = EMA(raw_model)

    # ── 10. Hyper-params ──────────────────────────────────────────────────────
    adjustlr_schedule = config['adjustlr_schedule']
    acc_grad          = config['acc_grad']
    max_epoch         = config['max_epoch']
    lr_decay          = config['lr_decay']

    cur_epoch   = 1
    loss_acc    = 0.0
    global_step = 0

    # ── 11. Training loop — same flow as original ─────────────────────────────
    while cur_epoch <= max_epoch:

        if sampler:
            sampler.set_epoch(cur_epoch)

        cnt_param_update = 0
        epoch_loss       = 0.0
        epoch_steps      = 0
        grad_norm_accum  = 0.0

        for iteration, (batch_clip, batch_bboxes, batch_labels) in \
                enumerate(dataloader):

            batch_clip = batch_clip.to(device, non_blocking=True)
            for idx in range(batch_clip.shape[0]):
                batch_bboxes[idx] = batch_bboxes[idx].to(device, non_blocking=True)
                batch_labels[idx] = batch_labels[idx].to(device, non_blocking=True)

            # Build targets — identical to original but uses torch.zeros (not torch.Tensor)
            targets = []
            for i, (bboxes, labels) in enumerate(zip(batch_bboxes, batch_labels)):
                nbox   = bboxes.shape[0]
                nclass = labels.shape[1]
                t = torch.zeros(nbox, 5 + nclass, device=device)
                t[:, 0]   = i
                t[:, 1:5] = bboxes
                t[:, 5:]  = labels
                targets.append(t)
            targets = torch.cat(targets, dim=0)

            # Forward + loss with AMP (new)
            with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                outputs = model(batch_clip)
                loss    = criterion(outputs, targets) / acc_grad

            loss_acc   += loss.item()
            epoch_loss += loss.item() * acc_grad

            # Backward
            scaler.scale(loss).backward()

            if (iteration + 1) % acc_grad == 0:
                cnt_param_update += 1
                global_step      += 1

                if cur_epoch == 1:
                    warmup_lr(optimizer, cnt_param_update)

                scaler.unscale_(optimizer)

                # Grad norm before zero_grad
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.detach().norm(2).item() ** 2
                total_norm      = total_norm ** 0.5
                grad_norm_accum += total_norm

                nn.utils.clip_grad_value_(model.parameters(), clip_value=2.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema.update(raw_model)

                epoch_steps  += 1
                display_loss  = loss_acc * acc_grad

                if is_main:
                    print(
                        f'epoch: {cur_epoch:03d} | '
                        f'update: {cnt_param_update:05d} | '
                        f'loss: {display_loss:.4f} | '
                        f'grad_norm: {total_norm:.3f}',
                        flush=True)

                    writer.add_scalar('Loss/step',     display_loss, global_step)
                    writer.add_scalar('GradNorm/step', total_norm,   global_step)
                    writer.add_scalar('LR/step',
                        optimizer.param_groups[0]['lr'], global_step)

                    with open(os.path.join(save_folder, 'logging.txt'), 'a') as f:
                        f.write(
                            f'epoch: {cur_epoch:03d} | '
                            f'update: {cnt_param_update:05d} | '
                            f'loss: {display_loss:.4f} | '
                            f'grad_norm: {total_norm:.3f}\n')

                loss_acc = 0.0

        # ── End of epoch ──────────────────────────────────────────────────────

        if cur_epoch in adjustlr_schedule:
            for pg in optimizer.param_groups:
                pg['lr'] *= lr_decay

        # Per-epoch TensorBoard
        if is_main and epoch_steps > 0:
            writer.add_scalar('Loss/epoch',
                              epoch_loss / epoch_steps, cur_epoch)
            writer.add_scalar('GradNorm/epoch',
                              grad_norm_accum / epoch_steps, cur_epoch)
            writer.add_scalar('LR/epoch',
                              optimizer.param_groups[0]['lr'], cur_epoch)

        # Checkpoint — identical to original
        if is_main:
            torch.save(ema.ema.state_dict(),
                       os.path.join(save_folder, f'ema_epoch_{cur_epoch}.pth'))
            torch.save(raw_model.state_dict(),
                       os.path.join(save_folder, f'epoch_{cur_epoch}.pth'))
            print(f'[Checkpoint] Saved epoch {cur_epoch}', flush=True)

        cur_epoch += 1

    # ── Cleanup ───────────────────────────────────────────────────────────────
    if is_main and writer is not None:
        writer.close()

    if dist.is_initialized():
        dist.destroy_process_group()