import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from data_loader import get_dataloader, PROMPT_VIDEO
from model import build_model, AttentionConsistencyLoss

CONFIG = {
    "data_dir":           "data/",
    "vae_ckpt":           "checkpoints/vae-ft-mse-840000-ema-pruned.ckpt",
    "grounding_config":   "/home/work/.local/lib/python3.10/site-packages/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    "grounding_ckpt":     "checkpoints/groundingdino_swint_ogc.pth",
    "pretrained_ckpt":    None,
    "save_dir":           "checkpoints/",
    "device":             "cuda" if torch.cuda.is_available() else "cpu",
    "prompt":             PROMPT_VIDEO,

    "epochs":             15, # 30
    "batch_size":         8,
    "lr":                 1e-4, #5e-5
    "weight_decay":       1e-2,
    "lambda_attn":        0.3,   # 높이면 히트맵 집중도↑, 낮추면 정확도 우선

    "num_workers":        0,
    "max_frames":         16,
}

# 학습 1 에폭
def train_one_epoch(model, loader, optimizer, ce_loss, attn_loss_fn, config):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    device = config["device"]

    for batch_idx, (frames, labels) in enumerate(loader):
        B, N, C, H, W = frames.shape
        labels         = labels.to(device)

        frame_logits, frame_attns = [], []

        for n in range(N):
            frame          = frames[:, n].to(device)
            logits, attn   = model(frame)
            frame_logits.append(logits)
            if attn is not None:
                frame_attns.append(attn)

        logits_mean = torch.stack(frame_logits).mean(dim=0)
        loss_ce     = ce_loss(logits_mean, labels)

        # Attention Consistency Loss
        loss_attn = torch.tensor(0.0, device=device)
        if frame_attns and config["lambda_attn"] > 0:
            last_attn  = frame_attns[-1]
            last_frame = frames[:, -1].to(device)

            num_patches = last_attn.shape[1]
            p           = int(num_patches ** 0.5)
            if p * p == num_patches:
                attn_2d = last_attn.reshape(B, p, p)
            else:
                p       = int(num_patches ** 0.5)
                attn_2d = last_attn[:, :p*p].reshape(B, p, p)

            error_approx = last_frame.abs()
            loss_attn    = attn_loss_fn(attn_2d, error_approx)

        loss = loss_ce + config["lambda_attn"] * loss_attn

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds    = logits_mean.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += B
        total_loss += loss.item()

        if (batch_idx + 1) % 10 == 0:
            print(f"  step {batch_idx+1}/{len(loader)} "
                  f"| loss: {loss.item():.4f} "
                  f"(ce: {loss_ce.item():.4f}, attn: {loss_attn.item():.4f})")

    return total_loss / len(loader), correct / total

# 검증
@torch.no_grad()
def validate(model, loader, ce_loss, config):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    device = config["device"]

    for frames, labels in loader:
        B, N, C, H, W = frames.shape
        labels         = labels.to(device)

        frame_logits = [model(frames[:, n].to(device))[0] for n in range(N)]
        logits_mean  = torch.stack(frame_logits).mean(dim=0)
        loss         = ce_loss(logits_mean, labels)

        correct    += (logits_mean.argmax(dim=1) == labels).sum().item()
        total      += B
        total_loss += loss.item()

    return total_loss / len(loader), correct / total

# 메인 학습 루프
def train(config: dict = CONFIG):
    device = config["device"]
    print(f"디바이스: {device}")
    print(f"프롬프트: {config['prompt']}\n")

    train_loader, val_loader = get_dataloader(
        data_dir         = config["data_dir"],
        vae_ckpt         = config["vae_ckpt"],
        grounding_config = config["grounding_config"],
        grounding_ckpt   = config["grounding_ckpt"],
        prompt           = config["prompt"],
        batch_size       = config["batch_size"],
        num_workers      = config["num_workers"],
        device           = device,
    )

    model        = build_model(config["pretrained_ckpt"], freeze_backbone=True, device=device)
    ce_loss      = nn.CrossEntropyLoss()
    attn_loss_fn = AttentionConsistencyLoss(threshold=0.1)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["lr"], weight_decay=config["weight_decay"],
    )
    scheduler    = CosineAnnealingLR(optimizer, T_max=config["epochs"])

    os.makedirs(config["save_dir"], exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(1, config["epochs"] + 1):
        print(f"\n[Epoch {epoch}/{config['epochs']}]")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, ce_loss, attn_loss_fn, config
        )
        val_loss, val_acc = validate(model, val_loader, ce_loss, config)
        scheduler.step()

        print(f"train loss: {train_loss:.4f} | acc: {train_acc:.4f}")
        print(f"val   loss: {val_loss:.4f} | acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(config["save_dir"], "best_model.pth"))
            print(f"✓ best model 저장 (val_acc: {val_acc:.4f})")

    print(f"\n학습 완료 | best val acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    train()
