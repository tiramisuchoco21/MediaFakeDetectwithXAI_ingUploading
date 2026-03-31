import torch
import torch.nn as nn
from torchvision.models import vit_l_16, ViT_L_16_Weights

# 1. Attention Consistency Loss
class AttentionConsistencyLoss(nn.Module):
    def __init__(self, threshold: float = 0.1):
        super().__init__()
        self.threshold = threshold
        self.mse       = nn.MSELoss()

    def forward(self, attention_map: torch.Tensor, error_map: torch.Tensor) -> torch.Tensor:
        pseudo_mask = (error_map.mean(dim=1) > self.threshold).float()

        if attention_map.shape != pseudo_mask.shape:
            attention_map = nn.functional.interpolate(
                attention_map.unsqueeze(1),
                size=pseudo_mask.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

        return self.mse(attention_map, pseudo_mask)

# 2. Deepfake 탐지 모델
class DeepfakeDetector(nn.Module):
    def __init__(self, num_classes: int = 2, freeze_backbone: bool = True):
        super().__init__()

        vit = vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_V1)

        if freeze_backbone:
            for param in vit.parameters():
                param.requires_grad = False

        self.patch_embed = vit.conv_proj
        self.encoder     = vit.encoder
        self.class_token = vit.class_token
        self.hidden_dim  = vit.hidden_dim  # 1024 (ViT-L)

        self._attention_weights = []
        self._register_attention_hooks()

        # Classification Head
        self.head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def _register_attention_hooks(self):
        def hook_fn(module, input, output):
            attn = output[0] if isinstance(output, tuple) else output
            self._attention_weights.append(attn.detach())

        last_block = list(self.encoder.layers)[-1]
        last_block.self_attention.register_forward_hook(hook_fn)

    def get_attention_rollout(self) -> torch.Tensor | None:
        if not self._attention_weights:
            return None
        attn = self._attention_weights[-1]

        if attn.dim() == 4:
            attn     = attn.mean(dim=1)
            cls_attn = attn[:, 0, 1:]
        elif attn.dim() == 3:
            cls_attn = attn[:, 0, 1:]
        else:
            cls_attn = attn

        return cls_attn

    def forward(self, x: torch.Tensor):
        self._attention_weights.clear()
        B = x.shape[0]

        x   = self.patch_embed(x)
        x   = x.flatten(2).transpose(1, 2)
        cls = self.class_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        x   = self.encoder(x)

        logits        = self.head(x[:, 0])
        attention_map = self.get_attention_rollout()

        return logits, attention_map

# 3. 사전학습 가중치 로드 헬퍼
def load_pretrained_weights(model: DeepfakeDetector, ckpt_path: str):
    state_dict    = torch.load(ckpt_path, map_location="cpu")
    backbone_keys = {k: v for k, v in state_dict.items() if not k.startswith("head")}
    missing, unexpected = model.load_state_dict(backbone_keys, strict=False)
    print(f"가중치 로드 | missing: {len(missing)} | unexpected: {len(unexpected)}")
    return model

# 4. 모델 빌드 헬퍼
def build_model(
    pretrained_ckpt: str | None = None,
    freeze_backbone: bool       = True,
    device: str                 = "cuda",
) -> DeepfakeDetector:

    model = DeepfakeDetector(freeze_backbone=freeze_backbone).to(device)

    if pretrained_ckpt:
        model = load_pretrained_weights(model, pretrained_ckpt)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"학습 파라미터: {trainable:,} / 전체: {total:,} ({100*trainable/total:.1f}%)")

    return model
