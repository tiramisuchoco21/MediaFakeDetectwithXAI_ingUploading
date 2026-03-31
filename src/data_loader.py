import os
import cv2
import torch
import numpy as np
import yt_dlp

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.ops import box_convert
from groundingdino.util.inference import load_model, predict
from diffusers import AutoencoderKL

# 프롬프트 설정
PROMPT_VIDEO = (
    "face . person . animal . cat . dog . bird . "
    "background . sky . building . tree . texture . shadow"
)

PROMPT_IMAGE = (
    "object . animal . cat . dog . bird . tree . plant . "
    "texture . pattern . edge . shadow . reflection . background"
)

PROMPT_DEEPFAKE = (
    "face . neck . hair . skin . eye . "
    "blurry boundary . unnatural texture . artificial pattern"
)

# 1. VAE 오차맵 생성기
class VAEErrorMapExtractor:
    def __init__(self, ckpt_path: str, device: str = "cuda"):
        self.device = device
        self.vae    = AutoencoderKL.from_single_file(ckpt_path).to(device)
        self.vae.eval()

        self.to_tensor = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    @torch.no_grad()
    def get_error_map(self, pil_image: Image.Image) -> torch.Tensor:
        x      = self.to_tensor(pil_image).unsqueeze(0).to(self.device)
        latent = self.vae.encode(x).latent_dist.sample()
        recon  = self.vae.decode(latent).sample
        return torch.abs(x - recon).squeeze(0)

# 2. 범용 영역 Cropper (GroundingDINO)
class RegionCropper:
    def __init__(
        self,
        config_path: str      = "groundingdino/config/GroundingDINO_SwinT_OGC.py",
        ckpt_path: str        = "checkpoints/groundingdino_swint_ogc.pth",
        device: str           = "cuda",
        prompt: str           = None,
        box_threshold: float  = 0.3,
        text_threshold: float = 0.25,
        max_crops: int        = 5,
        padding: float        = 0.1,
    ):
        self.model          = load_model(config_path, ckpt_path, device=device)
        self.device         = device
        self.prompt         = prompt or PROMPT_VIDEO
        self.box_threshold  = box_threshold
        self.text_threshold = text_threshold
        self.max_crops      = max_crops
        self.padding        = padding

    def crop(self, frame_bgr: np.ndarray) -> tuple[list[Image.Image], list[tuple]]:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img   = Image.fromarray(frame_rgb)
        W, H      = pil_img.size

        import torchvision.transforms.functional as TF
        img_resized = pil_img.resize((800, 800))
        img_tensor = TF.to_tensor(img_resized)  # (3, H, W) float32

        boxes, scores, _ = predict(
            model          = self.model,
            image          = img_tensor,
            caption        = self.prompt,
            box_threshold  = self.box_threshold,
            text_threshold = self.text_threshold,
            device         = self.device,
        )

        if boxes is None or len(boxes) == 0:
            return [pil_img.resize((224, 224))], [(0, 0, W, H)]

        boxes_xyxy = box_convert(
            boxes * torch.tensor([W, H, W, H]),
            in_fmt="cxcywh",
            out_fmt="xyxy",
        )

        top_idx        = scores.argsort(descending=True)[:self.max_crops]
        crops, positions = [], []

        for i in top_idx:
            x1, y1, x2, y2 = boxes_xyxy[i].tolist()
            bw, bh = x2 - x1, y2 - y1

            x1p = max(0, x1 - bw * self.padding)
            y1p = max(0, y1 - bh * self.padding)
            x2p = min(W, x2 + bw * self.padding)
            y2p = min(H, y2 + bh * self.padding)

            crop = pil_img.crop((x1p, y1p, x2p, y2p)).resize((224, 224))
            crops.append(crop)
            positions.append((int(x1p), int(y1p), int(x2p), int(y2p)))

        return crops, positions

# 3. URL 다운로더
def download_video(url: str, save_dir: str = "tmp/") -> str:
    os.makedirs(save_dir, exist_ok=True)

    ydl_opts = {
        "outtmpl":        os.path.join(save_dir, "%(id)s.%(ext)s"),
        "format":         "mp4/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]",
        "quiet":          True,
        "no_warnings":    True,
        "merge_output_format": "mp4",
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info     = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        # 확장자가 .mp4로 끝나지 않으면 보정
        if not filename.endswith(".mp4"):
            filename = os.path.splitext(filename)[0] + ".mp4"

    print(f"다운로드 완료: {filename}")
    return filename


def load_input(path_or_url: str, sample_rate: float = 1.0) -> list[np.ndarray]:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        local_path = download_video(path_or_url)
    else:
        local_path = path_or_url

    return extract_frames(local_path, sample_rate)

# 4. 프레임 추출
def extract_frames(video_path: str, sample_rate: float = 1.0) -> list[np.ndarray]:
    if video_path.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
        frame = cv2.imread(video_path)
        return [frame] if frame is not None else []

    cap      = cv2.VideoCapture(video_path)
    fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    interval = int(fps * sample_rate)
    frames, idx = [], 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            frames.append(frame)
        idx += 1

    cap.release()
    return frames

# 5. 집계 함수
def aggregate_scores(scores: list[float], k_ratio: float = 0.2) -> float:
    if not scores:
        return 0.0
    k     = max(1, int(len(scores) * k_ratio))
    top_k = sorted(scores, reverse=True)[:k]
    return sum(top_k) / k

# 6. PyTorch Dataset
class DeepfakeDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        vae_extractor: VAEErrorMapExtractor,
        region_cropper: RegionCropper,
        sample_rate: float = 1.0,
        max_frames: int    = 16,
    ):
        self.vae_extractor  = vae_extractor
        self.region_cropper = region_cropper
        self.sample_rate    = sample_rate
        self.max_frames     = max_frames

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        self.samples = []
        for label, subdir in [(0, "real"), (1, "fake")]:
            folder = os.path.join(data_dir, subdir)
            if not os.path.exists(folder):
                continue
            for fname in os.listdir(folder):
                if fname.lower().endswith(('.mp4', '.avi', '.mov', '.jpg', '.jpeg', '.png', '.webp')):
                    self.samples.append((os.path.join(folder, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        frames      = extract_frames(path, self.sample_rate)[:self.max_frames]
        tensors     = []

        for frame in frames:
            crops, _ = self.region_cropper.crop(frame)
            for crop in crops:
                error_map = self.vae_extractor.get_error_map(crop).cpu()
                orig_t    = self.to_tensor(crop)
                fused     = orig_t * error_map
                tensors.append(fused)

        if not tensors:
            tensors = [torch.zeros(3, 224, 224)]

        max_len = self.max_frames
        if len(tensors) > max_len:
            tensors = tensors[:max_len]
        while len(tensors) < max_len:
            tensors.append(torch.zeros(3, 224, 224))

        return torch.stack(tensors), torch.tensor(label, dtype=torch.long)

# 7. DataLoader 헬퍼
def get_dataloader(
    data_dir: str,
    vae_ckpt: str,
    grounding_config: str,
    grounding_ckpt: str,
    prompt: str      = None,
    batch_size: int  = 8,
    num_workers: int = 4,
    device: str      = "cuda",
) -> tuple[DataLoader, DataLoader]:

    vae_extractor  = VAEErrorMapExtractor(vae_ckpt, device)
    region_cropper = RegionCropper(
        config_path = grounding_config,
        ckpt_path   = grounding_ckpt,
        device      = device,
        prompt      = prompt,
    )

    train_ds = DeepfakeDataset(os.path.join(data_dir, "train"), vae_extractor, region_cropper)
    val_ds   = DeepfakeDataset(os.path.join(data_dir, "val"),   vae_extractor, region_cropper)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )
