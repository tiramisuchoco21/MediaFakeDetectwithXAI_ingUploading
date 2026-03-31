import os
import base64
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from data_loader import (
    VAEErrorMapExtractor,
    RegionCropper,
    load_input,
    aggregate_scores,
)
from model import build_model

def frame_to_base64(frame_bgr: np.ndarray) -> str:
    _, buffer = cv2.imencode('.jpg', frame_bgr)
    return base64.b64encode(buffer).decode('utf-8')

def generate_heatmap(
    attention_map: torch.Tensor,
    error_map: torch.Tensor,
    output_size: tuple = (512, 512),
) -> np.ndarray:
    num_patches = attention_map.shape[0]
    p           = int(num_patches ** 0.5)
    if p * p != num_patches:
        attention_map = attention_map[:p*p]
    attn_2d = attention_map.reshape(p, p).cpu().numpy()

    attn_2d    = (attn_2d - attn_2d.min()) / (attn_2d.max() - attn_2d.min() + 1e-8)
    attn_up    = cv2.resize(attn_2d, output_size, interpolation=cv2.INTER_LINEAR)

    error_gray = error_map.mean(dim=0).cpu().numpy()
    error_gray = (error_gray - error_gray.min()) / (error_gray.max() - error_gray.min() + 1e-8)
    error_up   = cv2.resize(error_gray, output_size, interpolation=cv2.INTER_LINEAR)

    combined   = attn_up * error_up
    combined   = (combined - combined.min()) / (combined.max() - combined.min() + 1e-8)

    return cv2.applyColorMap((combined * 255).astype(np.uint8), cv2.COLORMAP_JET)

def overlay_heatmap(original_frame, heatmap, alpha=0.5):
    heatmap_r = cv2.resize(heatmap, (original_frame.shape[1], original_frame.shape[0]))
    return cv2.addWeighted(original_frame, 1 - alpha, heatmap_r, alpha, 0)

def build_fullframe_heatmap(
    frame_bgr: np.ndarray,
    crops: list,
    positions: list,
    attns: list,
    error_maps: list,
) -> np.ndarray:
    H, W      = frame_bgr.shape[:2]
    hmap_acc  = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)

    for attn, error_map, (x1, y1, x2, y2) in zip(attns, error_maps, positions):
        tile_hm   = generate_heatmap(attn, error_map, output_size=(512, 512))
        tile_gray = cv2.cvtColor(tile_hm, cv2.COLOR_BGR2GRAY).astype(np.float32)

        rw, rh  = x2 - x1, y2 - y1
        tile_r  = cv2.resize(tile_gray, (rw, rh))

        hmap_acc[y1:y2, x1:x2]  += tile_r
        count_map[y1:y2, x1:x2] += 1

    hmap_avg = hmap_acc / np.maximum(count_map, 1)
    hmap_avg = (hmap_avg / (hmap_avg.max() + 1e-8) * 255).astype(np.uint8)
    return cv2.applyColorMap(hmap_avg, cv2.COLORMAP_JET)

@torch.no_grad()
def infer(
    file_path: str,
    model,
    vae_extractor: VAEErrorMapExtractor,
    region_cropper: RegionCropper,
    device: str        = "cuda",
    sample_rate: float = 0.5,
    k_ratio: float     = 0.2,
    start_time: float  = 0.0,
    end_time: float    = None,
    save_crops: bool   = False,
    crop_dir: str      = "output_crops",
) -> dict:
    model.eval()
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    if save_crops:
        os.makedirs(crop_dir, exist_ok=True)

    frames = load_input(file_path, sample_rate)

    if start_time > 0.0 or end_time is not None:
        start_idx = int(start_time / sample_rate)
        end_idx   = int(end_time / sample_rate) if end_time is not None else len(frames)
        frames    = frames[start_idx:end_idx]

    scores   = []
    overlays = []

    for frame_idx, frame_bgr in enumerate(frames):
        crops, positions = region_cropper.crop(frame_bgr)

        if save_crops:
            for crop_idx, (crop, pos) in enumerate(zip(crops, positions)):
                x1, y1, x2, y2 = pos
                fname = f"frame{frame_idx:04d}_crop{crop_idx}_{x1}_{y1}_{x2}_{y2}.jpg"
                crop.save(os.path.join(crop_dir, fname))

        frame_attns      = []
        frame_error_maps = []
        crop_scores      = []

        for crop in crops:
            error_map = vae_extractor.get_error_map(crop)
            orig_t    = to_tensor(crop)
            fused     = (orig_t * error_map.cpu()).unsqueeze(0).to(device)

            logits, attn = model(fused)
            prob_fake    = torch.softmax(logits, dim=1)[0, 1].item()
            crop_scores.append(prob_fake)

            if attn is not None:
                frame_attns.append(attn[0])
                frame_error_maps.append(error_map)

        if crop_scores:
            scores.append(max(crop_scores))

        if frame_attns:
            full_hm  = build_fullframe_heatmap(
                frame_bgr, crops, positions,
                frame_attns, frame_error_maps,
            )
            overlay  = overlay_heatmap(frame_bgr, full_hm)
            overlays.append(overlay)

    final_score = aggregate_scores(scores, k_ratio)
    real_prob   = 1 - final_score

    return {
        "label":        "real" if real_prob >= 0.95 else "fake",
        "confidence":   round(final_score, 4),
        "real_prob":    round(real_prob, 4),
        "frame_scores": [round(s, 4) for s in scores],
        "overlays_b64": [frame_to_base64(f) for f in overlays],
    }

def save_heatmap_video(overlays_b64: list, output_path: str, fps: float = 1.0):
    if not overlays_b64:
        print("저장할 프레임 없음")
        return

    frames = []
    for b64 in overlays_b64:
        buffer = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
        frame  = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        frames.append(frame)

    h, w   = frames[0].shape[:2]
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()
    print(f"히트맵 영상 저장: {output_path}")

def evaluate_dataset(data_dir, model, vae_extractor, region_cropper, device="cuda"):
    from sklearn.metrics import accuracy_score, roc_auc_score

    y_true, y_score = [], []

    for label, subdir in [(0, "real"), (1, "fake")]:
        folder = os.path.join(data_dir, subdir)
        if not os.path.exists(folder):
            continue
        for fname in os.listdir(folder):
            path   = os.path.join(folder, fname)
            result = infer(path, model, vae_extractor, region_cropper, device)
            y_true.append(label)
            y_score.append(result["confidence"])
            print(f"[{subdir}] {fname} → {result['label']} "
                  f"(fake: {result['confidence']:.4f} / real: {result['real_prob']:.4f})")

    y_pred = [1 if s > 0.5 else 0 for s in y_score]
    print(f"\nAccuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"AUC      : {roc_auc_score(y_true, y_score):.4f}")

if __name__ == "__main__":
    import json

    DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
    VAE_CKPT         = "checkpoints/vae-ft-mse-840000-ema-pruned.ckpt"
    GROUNDING_CONFIG = "/home/work/.local/lib/python3.10/site-packages/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    GROUNDING_CKPT   = "checkpoints/groundingdino_swint_ogc.pth"
    MODEL_CKPT       = "checkpoints/best_model.pth"

    vae_extractor  = VAEErrorMapExtractor(VAE_CKPT, DEVICE)
    region_cropper = RegionCropper(GROUNDING_CONFIG, GROUNDING_CKPT, DEVICE)
    model          = build_model(pretrained_ckpt=None, freeze_backbone=True, device=DEVICE)
    model.load_state_dict(torch.load(MODEL_CKPT, map_location=DEVICE))

    result = infer("video.mp4", model, vae_extractor, region_cropper, DEVICE, start_time=10.0, end_time=20.0, save_crops=True)

    result = infer("SJW.jpg", model, vae_extractor, region_cropper, DEVICE,
                   save_crops=True)

    # JSON 출력 (overlays_b64 제외하고 출력)
    print_result = {k: v for k, v in result.items() if k != "overlays_b64"}
    print(json.dumps(print_result, indent=2, ensure_ascii=False))
    print(f"히트맵 프레임 수: {len(result['overlays_b64'])}개")

    # 로컬에서 히트맵 영상으로 저장해서 확인
    if result["overlays_b64"]:
        save_heatmap_video(result["overlays_b64"], "output_heatmap.mp4", fps=1.0)