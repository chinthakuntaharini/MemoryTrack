"""
Metric Re-Identification Feature Extractor
Extracts raw, unit-normalized 512-D appearance embeddings (CNN GeM + flip TTA, or a handcrafted fallback).

Design notes:
- Crops are letterboxed (aspect preserving) to 256x128 and padded with the ImageNet mean colour, so
  wide webcam / upper-body crops are not squashed.
- The last convolutional feature map is pooled with a padding-aware GeM (p=3), the horizontally
  flipped crop is pooled as well and both are averaged before L2 normalisation.
- 2048-D backbones are reduced to 512-D with a fixed, seeded orthogonal projection so stored
  embeddings stay comparable across sessions and machines (same `model_id` = same space).
- Embeddings are RAW (all-positive CNN activations share a large common component); centering and
  calibration are done downstream by `forensic_app.core.identity.IdentityManager`.
- No weights are ever downloaded: torchvision weights are only used when the cached checkpoint file
  already exists in `torch.hub.get_dir()/checkpoints`.
"""

import os
import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torchvision import models
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only on machines without torch
    torch = None
    nn = None
    models = None
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

# ImageNet normalisation constants (RGB)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# Letterbox padding colour = ImageNet mean (normalises to ~0 so padding is "neutral")
_PAD_RGB = tuple(int(round(v * 255)) for v in _IMAGENET_MEAN)

# Seed of the fixed orthogonal projection (changing it changes the embedding space -> bump version)
_PROJECTION_SEED = 20261026
_EMBEDDING_VERSION = "v1"

# Backbone preference order for backbone="auto" (benchmarked on a 16-core CPU, see report)
_AUTO_ORDER = ("osnet", "resnet50", "resnet18", "heuristic")
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _hub_checkpoint_dir() -> Optional[Path]:
    """Return torch hub's checkpoint directory (never creates or downloads anything)."""
    if not TORCH_AVAILABLE:
        return None
    try:
        return Path(torch.hub.get_dir()) / "checkpoints"
    except Exception:
        return None


def _cached_weight_file(weights_enum) -> Optional[Path]:
    """Return the first locally cached checkpoint file for a torchvision weights enum, else None."""
    ckpt_dir = _hub_checkpoint_dir()
    if ckpt_dir is None or not ckpt_dir.is_dir():
        return None
    for member in weights_enum:
        try:
            fname = member.url.split("/")[-1]
        except Exception:
            continue
        path = ckpt_dir / fname
        if path.is_file():
            return path
    return None


def seeded_orthogonal_projection(in_dim: int, out_dim: int, seed: int = _PROJECTION_SEED) -> np.ndarray:
    """Deterministic (in_dim, out_dim) matrix with orthonormal columns (QR of a seeded Gaussian)."""
    rng = np.random.default_rng(seed)
    gauss = rng.standard_normal((in_dim, out_dim))
    q, r = np.linalg.qr(gauss)
    # Sign-fix so the factorisation is unique regardless of LAPACK conventions
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    return (q * signs[None, :]).astype(np.float32)


def letterbox_crop(crop: np.ndarray, out_h: int = 256, out_w: int = 128,
                   pad_color: Tuple[int, int, int] = (104, 116, 124)) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Aspect-preserving resize of a BGR crop into an (out_h, out_w) canvas.
    Returns (canvas, (y0, x0, h, w)) where the last tuple is the valid (non-padded) region.
    """
    h, w = crop.shape[:2]
    scale = min(out_h / float(h), out_w / float(w))
    new_h = max(1, min(out_h, int(round(h * scale))))
    new_w = max(1, min(out_w, int(round(w * scale))))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(crop, (new_w, new_h), interpolation=interp)
    canvas = np.empty((out_h, out_w, 3), dtype=np.uint8)
    canvas[:] = pad_color
    y0 = (out_h - new_h) // 2
    x0 = (out_w - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas, (y0, x0, new_h, new_w)


class ReIDFeatureExtractor:
    """512-Dimensional appearance Re-ID extractor with batched, flip-augmented CNN inference."""

    INPUT_H = 256
    INPUT_W = 128
    MIN_CROP_SIDE = 10  # crops smaller than this (either side) produce an all-zero (invalid) row
    CENTRE_SIGMA = 0.35  # horizontal centre prior of the pooling (fraction of the crop width)

    def __init__(self,
                 embedding_dim: int = 512,
                 device: str = "cpu",
                 backbone: str = "auto",
                 flip_tta: bool = True,
                 gem_p: float = 3.0,
                 max_batch: int = 16):
        self.embedding_dim = int(embedding_dim)
        self.device = self._verify_device(device)
        self.flip_tta = bool(flip_tta)
        self.gem_p = float(gem_p)
        self.max_batch = max(1, int(max_batch))

        self.model = None                       # trunk returning the last conv feature map
        self.backbone_name = "heuristic"
        self.feature_stride = 32
        self._projection = None                 # torch (feat_dim, embedding_dim) or None
        self._mean_t = None
        self._std_t = None

        requested = (backbone or "auto").strip().lower()
        if requested == "auto":
            order = list(_AUTO_ORDER)
        else:
            # Explicit request first, then graceful fallback chain
            order = [requested] + [b for b in _AUTO_ORDER if b != requested]

        for name in order:
            # The heuristic descriptor is always available and ends the chain
            if name == "heuristic" or self._try_init_backbone(name):
                break

        if self.model is None:
            self.backbone_name = "heuristic"
            self.model_id = f"heuristic-stripes-{self.embedding_dim}-v2"
            if requested == "heuristic":
                logger.info("ReIDFeatureExtractor using handcrafted stripe descriptor (requested).")
            elif not TORCH_AVAILABLE:
                logger.warning("PyTorch not installed; ReID using handcrafted stripe-descriptor fallback.")
            else:
                logger.warning("No cached CNN weights found; ReID using handcrafted stripe-descriptor fallback.")
        else:
            # Flip TTA does not change the embedding space, so it is not part of the model id
            self.model_id = f"{self.backbone_name}-gem-{self.embedding_dim}-{_EMBEDDING_VERSION}"
            logger.info(f"ReIDFeatureExtractor initialized: {self.model_id} on {self.device} "
                        f"(flip TTA {'on' if self.flip_tta else 'off'})")

    # ------------------------------------------------------------------ backbone setup
    def _verify_device(self, req_device: str) -> str:
        if req_device == "cuda" and TORCH_AVAILABLE:
            return "cuda" if torch.cuda.is_available() else "cpu"
        return "cpu"

    def _try_init_backbone(self, name: str) -> bool:
        """Try to build the named backbone from LOCAL weights only. Returns True on success."""
        if not TORCH_AVAILABLE:
            return False
        try:
            if name == "osnet":
                return self._init_osnet()
            if name == "resnet50":
                return self._init_torchvision_resnet("resnet50", models.resnet50, models.ResNet50_Weights, 2048)
            if name == "resnet18":
                return self._init_torchvision_resnet("resnet18", models.resnet18, models.ResNet18_Weights, 512)
        except Exception as e:
            logger.warning(f"ReID backbone '{name}' unavailable: {e}")
        self.model = None
        return False

    def _init_torchvision_resnet(self, name: str, ctor, weights_enum, feat_dim: int) -> bool:
        """Build a torchvision ResNet trunk (conv1..layer4) from a cached checkpoint file."""
        weight_file = _cached_weight_file(weights_enum)
        if weight_file is None:
            logger.info(f"No cached weights for {name}; skipping (no download attempted).")
            return False
        net = ctor(weights=None)
        state = torch.load(str(weight_file), map_location="cpu", weights_only=True)
        net.load_state_dict(state)
        net.eval()
        trunk = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool,
                              net.layer1, net.layer2, net.layer3, net.layer4)
        trunk = self._fuse_resnet_trunk(trunk)
        self._finalise_model(trunk, name, feat_dim, stride=32)
        return True

    @staticmethod
    def _fuse_resnet_trunk(trunk):
        """Fold every BatchNorm into its preceding conv (identical outputs, ~10-30% faster on CPU)."""
        try:
            from torch.nn.utils.fusion import fuse_conv_bn_eval
            trunk[0] = fuse_conv_bn_eval(trunk[0], trunk[1])
            trunk[1] = nn.Identity()
            for layer in list(trunk)[4:]:
                for blk in layer:
                    for i in (1, 2, 3):
                        conv = getattr(blk, f"conv{i}", None)
                        bn = getattr(blk, f"bn{i}", None)
                        if conv is not None and isinstance(bn, nn.BatchNorm2d):
                            setattr(blk, f"conv{i}", fuse_conv_bn_eval(conv, bn))
                            setattr(blk, f"bn{i}", nn.Identity())
                    ds = getattr(blk, "downsample", None)
                    if ds is not None and len(ds) == 2 and isinstance(ds[1], nn.BatchNorm2d):
                        ds[0] = fuse_conv_bn_eval(ds[0], ds[1])
                        ds[1] = nn.Identity()
        except Exception as e:
            logger.debug(f"Conv-BN fusion skipped: {e}")
        return trunk

    def _init_osnet(self) -> bool:
        """Build torchreid OSNet only when the package is importable AND local weights exist."""
        try:
            import torchreid  # noqa: F401  (optional dependency)
            from torchreid.models import build_model
        except Exception:
            return False
        candidates: List[Path] = []
        for folder in (_PROJECT_ROOT / "models", _PROJECT_ROOT):
            if folder.is_dir():
                candidates.extend(sorted(folder.glob("osnet*.pth")) + sorted(folder.glob("osnet*.pth.tar")))
        if not candidates:
            return False
        net = build_model(name="osnet_x1_0", num_classes=1000, pretrained=False)
        ckpt = torch.load(str(candidates[0]), map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        state = {k.replace("module.", ""): v for k, v in state.items() if not k.startswith("classifier")}
        net.load_state_dict(state, strict=False)

        class _OSNetTrunk(nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, x):
                return self.inner.featuremaps(x)

        self._finalise_model(_OSNetTrunk(net), "osnet", 512, stride=16)
        return True

    def _finalise_model(self, trunk, name: str, feat_dim: int, stride: int) -> None:
        trunk = trunk.to(self.device).eval()
        if self.device == "cpu":
            # NHWC layout is markedly faster for oneDNN convolutions on CPU
            trunk = trunk.to(memory_format=torch.channels_last)
        for p in trunk.parameters():
            p.requires_grad_(False)
        self.model = trunk
        self.backbone_name = name
        self.feature_stride = stride
        self._mean_t = torch.tensor(_IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std_t = torch.tensor(_IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
        if feat_dim != self.embedding_dim:
            proj = seeded_orthogonal_projection(feat_dim, self.embedding_dim)
            self._projection = torch.from_numpy(proj).to(self.device)
        else:
            self._projection = None

    # ------------------------------------------------------------------ public API
    def extract(self, frame: np.ndarray, bbox: tuple) -> np.ndarray:
        """
        Extract a unit-normalized 512-D embedding for one person box.
        Returns an all-zero vector when the crop is invalid / too small.
        """
        return self.extract_batch(frame, [bbox])[0]

    def extract_batch(self, frame: np.ndarray, bboxes: Sequence[tuple]) -> np.ndarray:
        """
        Extract embeddings for several boxes of the same frame in one batched forward pass.
        Returns (N, 512) float32; rows of invalid / too-small crops are all zeros.
        """
        n = len(bboxes)
        out = np.zeros((n, self.embedding_dim), dtype=np.float32)
        if n == 0 or frame is None or getattr(frame, "ndim", 0) != 3:
            return out

        crops: List[np.ndarray] = []
        rows: List[int] = []
        for i, bbox in enumerate(bboxes):
            crop = self._safe_crop(frame, bbox)
            if crop is not None:
                crops.append(crop)
                rows.append(i)
        if not crops:
            return out

        if self.model is not None:
            try:
                for start in range(0, len(crops), self.max_batch):
                    chunk = crops[start:start + self.max_batch]
                    feats = self._cnn_embed(chunk)
                    out[rows[start:start + len(chunk)]] = feats
                return out
            except Exception as e:
                logger.debug(f"Neural ReID extraction error, using heuristic descriptor: {e}")

        for r, crop in zip(rows, crops):
            out[r] = self._heuristic_feature_vector(crop)
        return out

    # ------------------------------------------------------------------ internals
    def _safe_crop(self, frame: np.ndarray, bbox) -> Optional[np.ndarray]:
        """Clip a bbox to the frame; return None when the crop is too small."""
        try:
            x1, y1, x2, y2 = (int(round(float(v))) for v in bbox[:4])
        except Exception:
            return None
        h_f, w_f = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_f, x2), min(h_f, y2)
        if (x2 - x1) < self.MIN_CROP_SIDE or (y2 - y1) < self.MIN_CROP_SIDE:
            return None
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        if crop.dtype != np.uint8:
            crop = np.clip(crop, 0, 255).astype(np.uint8)
        return crop

    def _cnn_embed(self, crops: List[np.ndarray]) -> np.ndarray:
        """Letterbox + normalise + forward (with flip TTA) + masked GeM + projection + L2."""
        b = len(crops)
        batch = np.empty((b, self.INPUT_H, self.INPUT_W, 3), dtype=np.uint8)
        valid = []
        for i, crop in enumerate(crops):
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            canvas, region = letterbox_crop(rgb, self.INPUT_H, self.INPUT_W, pad_color=_PAD_RGB)
            batch[i] = canvas
            valid.append(region)

        with torch.inference_mode():
            x = torch.from_numpy(batch).to(self.device).permute(0, 3, 1, 2).float().div_(255.0)
            x = (x - self._mean_t) / self._std_t
            if self.flip_tta:
                x = torch.cat([x, torch.flip(x, dims=[3])], dim=0)
            if self.device == "cpu":
                x = x.contiguous(memory_format=torch.channels_last)
            fmap = self.model(x)                                   # (B[*2], C, h, w)
            mask = self._feature_mask(valid, fmap.shape[2], fmap.shape[3])
            if self.flip_tta:
                mask = torch.cat([mask, torch.flip(mask, dims=[3])], dim=0)
            pooled = self._masked_gem(fmap, mask)                  # (B[*2], C)
            if self.flip_tta:
                pooled = 0.5 * (pooled[:b] + pooled[b:])
            if self._projection is not None:
                pooled = pooled @ self._projection
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            return pooled.cpu().numpy().astype(np.float32)

    def _feature_mask(self, regions, fh: int, fw: int):
        """
        (B,1,fh,fw) pooling weights: zero on letterbox padding, and inside the crop a mild horizontal
        centre prior (people are centred in their box; the box sides are mostly background).
        """
        mask = torch.zeros((len(regions), 1, fh, fw), dtype=torch.float32, device=self.device)
        sy = self.INPUT_H / float(fh)
        sx = self.INPUT_W / float(fw)
        col_centres = (np.arange(fw) + 0.5) * sx
        for i, (y0, x0, h, w) in enumerate(regions):
            r0 = int(np.floor(y0 / sy))
            r1 = int(np.ceil((y0 + h) / sy))
            c0 = int(np.floor(x0 / sx))
            c1 = int(np.ceil((x0 + w) / sx))
            r0, c0 = max(0, min(r0, fh - 1)), max(0, min(c0, fw - 1))
            r1, c1 = max(r0 + 1, min(r1, fh)), max(c0 + 1, min(c1, fw))
            rel = (col_centres[c0:c1] - (x0 + 0.5 * w)) / max(float(w), 1.0)
            col_w = np.exp(-0.5 * (rel / self.CENTRE_SIGMA) ** 2).astype(np.float32)
            mask[i, 0, r0:r1, c0:c1] = torch.from_numpy(col_w).to(self.device).view(1, -1)
        return mask

    def _masked_gem(self, fmap, mask):
        """Generalised-mean pooling over the valid feature cells only (weighted by `mask`)."""
        p = self.gem_p
        x = fmap.clamp(min=1e-6).pow(p) * mask
        denom = mask.sum(dim=(2, 3)).clamp(min=1.0)
        return (x.sum(dim=(2, 3)) / denom).pow(1.0 / p)

    def _heuristic_feature_vector(self, crop: np.ndarray) -> np.ndarray:
        """
        Handcrafted 512-D part descriptor (no deep model):
        8 horizontal stripes x (HSV hist 32 + Lab mean/std 6 + gradient orientation 12)
        + global hue-saturation 2-D histogram (112). Blocks are Hellinger/L2 normalised and weighted.
        """
        canvas, (y0, x0, h, w) = letterbox_crop(crop, 128, 64, pad_color=(0, 0, 0))
        region = canvas[y0:y0 + h, x0:x0 + w]
        if region.shape[0] < 8 or region.shape[1] < 2:
            region = cv2.resize(crop, (64, 128))
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(region, cv2.COLOR_BGR2Lab).astype(np.float32)
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        ang = np.mod(ang, 180.0)  # unsigned orientation

        n_stripes = 8
        edges = np.linspace(0, region.shape[0], n_stripes + 1).astype(int)
        hist_blocks, lab_blocks, grad_blocks = [], [], []
        for s in range(n_stripes):
            r0, r1 = edges[s], max(edges[s] + 1, edges[s + 1])
            hs = hsv[r0:r1].reshape(-1, 3)
            sat_w = hs[:, 1].astype(np.float32) / 255.0 + 0.05
            h_hist = np.bincount((hs[:, 0].astype(np.int32) * 16) // 180, weights=sat_w, minlength=16)[:16]
            s_hist = np.bincount((hs[:, 1].astype(np.int32) * 8) // 256, minlength=8)[:8]
            v_hist = np.bincount((hs[:, 2].astype(np.int32) * 8) // 256, minlength=8)[:8]
            block = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)
            hist_blocks.append(np.sqrt(block / max(block.sum(), 1e-6)))

            ls = lab[r0:r1].reshape(-1, 3)
            mean = ls.mean(axis=0) / 255.0
            std = ls.std(axis=0) / 128.0
            lab_blocks.append(np.concatenate([mean, std]))

            m = mag[r0:r1].ravel()
            a = ang[r0:r1].ravel()
            g_hist = np.bincount(np.minimum((a / 15.0).astype(np.int32), 11), weights=m, minlength=12)[:12]
            grad_blocks.append(np.sqrt(g_hist / max(g_hist.sum(), 1e-6)))

        hs_all = hsv.reshape(-1, 3)
        h_idx = (hs_all[:, 0].astype(np.int32) * 16) // 180
        s_idx = np.minimum((hs_all[:, 1].astype(np.int32) * 7) // 256, 6)
        hs2d = np.bincount(h_idx * 7 + s_idx, minlength=112)[:112].astype(np.float32)
        hs2d = np.sqrt(hs2d / max(hs2d.sum(), 1e-6))

        def _unit(v: np.ndarray) -> np.ndarray:
            v = np.asarray(v, dtype=np.float32).ravel()
            n = float(np.linalg.norm(v))
            return v / n if n > 0 else v

        combined = np.concatenate([
            1.00 * _unit(np.concatenate(hist_blocks)),   # 256
            0.60 * _unit(np.concatenate(lab_blocks)),    # 48
            0.55 * _unit(np.concatenate(grad_blocks)),   # 96
            0.70 * _unit(hs2d),                          # 112
        ])
        if combined.size < self.embedding_dim:
            combined = np.pad(combined, (0, self.embedding_dim - combined.size))
        else:
            combined = combined[:self.embedding_dim]
        norm = float(np.linalg.norm(combined))
        if norm > 0:
            combined = combined / norm
        return combined.astype(np.float32)

    @staticmethod
    def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors mapped to 0..1 ((cos + 1) / 2)."""
        n1 = np.linalg.norm(vec1)
        n2 = np.linalg.norm(vec2)
        if n1 == 0 or n2 == 0:
            return 0.0
        dot = float(np.dot(vec1, vec2) / (n1 * n2))
        return max(0.0, min(1.0, (dot + 1.0) / 2.0))
