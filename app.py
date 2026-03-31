"""
app.py
역할: FastAPI 웹 백엔드
      사용자 요청 받아서 infer() 호출 후 JSON 반환

실행: uvicorn app:app --host 0.0.0.0 --port 8000
"""

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from src.data_loader import VAEErrorMapExtractor, RegionCropper
from src.model import build_model
from src.evaluate import infer

# ─────────────────────────────────────────────
# 앱 초기화
# ─────────────────────────────────────────────
app = FastAPI(title="Deepfake Detector API")

# CORS 설정 (프론트엔드에서 호출 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ─────────────────────────────────────────────
# 모델 로드 (서버 시작 시 한 번만)
# ─────────────────────────────────────────────
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
VAE_CKPT         = "checkpoints/vae-ft-mse-840000-ema-pruned.ckpt"
GROUNDING_CONFIG = "/home/work/.local/lib/python3.10/site-packages/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_CKPT   = "checkpoints/groundingdino_swint_ogc.pth"
MODEL_CKPT       = "checkpoints/best_model.pth"

print("모델 로딩 중...")
vae_extractor  = VAEErrorMapExtractor(VAE_CKPT, DEVICE)
region_cropper = RegionCropper(GROUNDING_CONFIG, GROUNDING_CKPT, DEVICE)
model          = build_model(pretrained_ckpt=None, freeze_backbone=True, device=DEVICE)
model.load_state_dict(torch.load(MODEL_CKPT, map_location=DEVICE))
model.eval()
print("모델 로딩 완료")


# ─────────────────────────────────────────────
# 요청 스키마 (사용자가 웹에서 보내는 값)
# ─────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    file_path:   str            # 영상/이미지 경로 또는 URL
    start_time:  Optional[float] = 0.0    # 분석 시작 시간 (초)
    end_time:    Optional[float] = None   # 분석 끝 시간 (초), None이면 끝까지
    save_crops:  Optional[bool]  = False  # crop 저장 여부
    sample_rate: Optional[float] = 0.5   # 샘플링 간격 (초)


# ─────────────────────────────────────────────
# API 엔드포인트
# ─────────────────────────────────────────────
@app.post("/analyze")
async def analyze(request: AnalyzeRequest):
    """
    사용자 요청 → infer() 호출 → JSON 반환

    요청 예시:
    {
        "file_path":  "https://www.youtube.com/shorts/xxxx",
        "start_time": 10.0,
        "end_time":   20.0,
        "save_crops": true,
        "sample_rate": 0.5
    }

    응답 예시:
    {
        "label":        "fake",
        "confidence":   0.8732,
        "real_prob":    0.1268,
        "frame_scores": [0.91, 0.85, 0.88],
        "overlays_b64": ["base64문자열...", ...]
    }
    """
    result = infer(
        file_path      = request.file_path,
        model          = model,
        vae_extractor  = vae_extractor,
        region_cropper = region_cropper,
        device         = DEVICE,
        sample_rate    = request.sample_rate,
        start_time     = request.start_time,
        end_time       = request.end_time,
        save_crops     = request.save_crops,
    )
    return result


@app.get("/health")
async def health():
    """서버 상태 확인"""
    return {"status": "ok", "device": DEVICE}