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

# 1. 히트맵 생성
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


# ─────────────────────────────────────────────
# 2. 영역별 히트맵 → 원본 프레임 좌표 재매핑
# ─────────────────────────────────────────────
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


# ─────────────────────────────────────────────
# 3. 단일 파일 추론
# ─────────────────────────────────────────────
@torch.no_grad()
def infer(
    file_path: str,
    model,
    vae_extractor: VAEErrorMapExtractor,
    region_cropper: RegionCropper,
    device: str        = "cuda",
    sample_rate: float = 1.0,
    k_ratio: float     = 0.2,
    start_time: float  = 0.0,   # 분석 시작 시간 (초), 0이면 처음부터
    end_time: float    = None,  # 분석 끝 시간 (초), None이면 끝까지
) -> dict:
    """
    입력: 영상/이미지 로컬 경로 또는 URL
          URL 예시: https://www.youtube.com/watch?v=xxxx
                   https://www.instagram.com/reel/xxxx
                   https://www.tiktok.com/@user/video/xxxx

    구간 지정 예시:
          infer(url, ..., start_time=10.0, end_time=20.0)
          → 10초~20초 구간만 분석
          infer(url, ..., start_time=30.0)
          → 30초부터 끝까지 분석

    판정 기준:
          real 확률 0.95 이상 → "real"
          real 확률 0.95 미만 → "fake"

    출력: {
        label       : "real" 또는 "fake"
        confidence  : fake 확률 (0~1)
        real_prob   : real 확률 (0~1)
        frame_scores: 프레임별 fake 확률 리스트
        overlays    : 원본 위에 히트맵 오버레이된 프레임 리스트
    }
    """
    model.eval()
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    frames = load_input(file_path, sample_rate)

    # 구간 필터링
    if start_time > 0.0 or end_time is not None:
        start_idx = int(start_time / sample_rate)
        end_idx   = int(end_time / sample_rate) if end_time is not None else len(frames)
        frames    = frames[start_idx:end_idx]
        print(f"구간 분석: {start_time}초 ~ "
              f"{end_time if end_time else '끝'} "
              f"({len(frames)}프레임)")

    scores   = []
    overlays = []

    for frame_bgr in frames:
        crops, positions = region_cropper.crop(frame_bgr)

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
        "confidence":   final_score,
        "real_prob":    real_prob,
        "frame_scores": scores,
        "overlays":     overlays,
    }


# ─────────────────────────────────────────────
# 4. 히트맵 영상 저장
# ─────────────────────────────────────────────
def save_heatmap_video(overlays: list, output_path: str, fps: float = 10.0):
    if not overlays:
        print("저장할 프레임 없음")
        return
    h, w   = overlays[0].shape[:2]
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in overlays:
        writer.write(frame)
    writer.release()
    print(f"히트맵 영상 저장: {output_path}")


# ─────────────────────────────────────────────
# 5. 데이터셋 전체 평가
# ─────────────────────────────────────────────
def evaluate_dataset(data_dir, model, vae_extractor, region_cropper, device="cuda"):
    import os
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
                  f"(fake: {result['confidence']:.3f} / real: {result['real_prob']:.3f})")

    y_pred = [1 if s > 0.5 else 0 for s in y_score]
    print(f"\nAccuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"AUC      : {roc_auc_score(y_true, y_score):.4f}")


# ─────────────────────────────────────────────
# 실행 예시
# ─────────────────────────────────────────────
if __name__ == "__main__":
    DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
    VAE_CKPT         = "checkpoints/vae-ft-mse-840000-ema-pruned.ckpt"
    GROUNDING_CONFIG = "/home/work/.local/lib/python3.10/site-packages/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    GROUNDING_CKPT   = "checkpoints/groundingdino_swint_ogc.pth"
    MODEL_CKPT       = "checkpoints/best_model.pth"

    vae_extractor  = VAEErrorMapExtractor(VAE_CKPT, DEVICE)
    region_cropper = RegionCropper(GROUNDING_CONFIG, GROUNDING_CKPT, DEVICE)
    model          = build_model(pretrained_ckpt=None, freeze_backbone=True, device=DEVICE)
    model.load_state_dict(torch.load(MODEL_CKPT, map_location=DEVICE))

    # 예시 1: 전체 영상 분석
    #result = infer("DQB1KBdkoFc.mp4", model, vae_extractor, region_cropper, DEVICE)

    # 예시 2: 유튜브 링크 분석
    result = infer("https://www.youtube.com/shorts/nTsOh2w_JHE", model, vae_extractor, region_cropper, DEVICE)

    # 예시 3: 구간 지정 분석 (10초~20초)
    # result = infer("test_video.mp4", model, vae_extractor, region_cropper, DEVICE,
    #                start_time=10.0, end_time=20.0)

    # 예시 4: 이미지 분석
    #result = infer("SJW.jpg", model, vae_extractor, region_cropper, DEVICE)

    print(f"\n판정     : {result['label']}")
    print(f"fake 확률: {result['confidence']:.4f}")
    print(f"real 확률: {result['real_prob']:.4f}")
    print(f"프레임별 : {[f'{s:.3f}' for s in result['frame_scores']]}")

    if result["overlays"]:
        save_heatmap_video(result["overlays"], "output_heatmap.mp4")