"""
Inference wrapper for the classifier service.

Loads the trained ConvNeXt-Small once at service startup, then exposes
classify_images() which takes a list of in-memory PIL images and returns
classification results.

This is the production version of the floorplan_classifier infer_v3.py logic,
adapted for service use:
  - Model loaded once at startup, not per request
  - Operates on PIL images directly (no PDF rendering, no disk I/O)
  - Returns Pydantic objects, not CSV rows
  - Stateless apart from the loaded model (safe to reuse across requests)
"""
import io
import logging
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import timm
from PIL import Image
from torchvision import transforms

from .config import (
    IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD, TILE_FRACTION,
    NUM_CLASSES, MULTICLASS_CLASS_NAMES, FLOOR_PLAN_IDX,
    FP_THRESHOLD, SNAKE_TO_LLM_LABEL,
)
from .schemas import PagePrediction

logger = logging.getLogger("classifier.inference")


def _build_model_arch(arch_name: str, num_classes: int = NUM_CLASSES) -> nn.Module:
    """
    Rebuild the ConvNeXt architecture so we can load weights.

    Mirrors model_v2.py:build_model exactly so the state_dict loads cleanly.
    """
    arch_to_timm = {
        "convnext_small": "convnext_small.fb_in22k_ft_in1k",
        "convnext_tiny":  "convnext_tiny.fb_in22k_ft_in1k",
    }
    if arch_name not in arch_to_timm:
        raise ValueError(f"Unknown arch: {arch_name}")

    model = timm.create_model(
        arch_to_timm[arch_name],
        pretrained=False,
        num_classes=num_classes,
        drop_path_rate=0.0,   # eval — disable stochastic depth
    )

    # Match training's head modification (swap head.drop to Dropout(0.3))
    if hasattr(model.head, "drop"):
        model.head.drop = nn.Dropout(p=0.3)

    return model


class PageClassifier:
    """
    Stateful classifier — loads the model once, classifies in-memory images.

    Thread safety: a single torch model is NOT thread-safe for concurrent
    forward passes (the CUDA context is shared). Don't share a single
    PageClassifier instance across threads doing inference in parallel.
    Cloud Run gives us 1 instance = 1 process with containerConcurrency=1,
    so this is not an issue.
    """

    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = Path(checkpoint_path)
        self._ready = False
        self.device = None
        self.model = None
        self.arch_name = None
        self.eval_transform = None
        self._load_model()

    def _load_model(self):
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found at {self.checkpoint_path}"
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            gpu_name = torch.cuda.get_device_name(self.device)
            vram_gb = torch.cuda.get_device_properties(self.device).total_memory / 1e9
            logger.info(f"GPU: {gpu_name} ({vram_gb:.1f}GB VRAM)")
        else:
            logger.warning("No GPU available — running on CPU (will be slow)")

        ckpt = torch.load(self.checkpoint_path, map_location=self.device,
                          weights_only=False)
        self.arch_name = ckpt.get("arch_name", "convnext_small")
        logger.info(f"Checkpoint arch: {self.arch_name}")
        if "best_macro_f1" in ckpt:
            logger.info(f"Checkpoint best val macro F1: {ckpt['best_macro_f1']:.4f}")

        self.model = _build_model_arch(self.arch_name, NUM_CLASSES)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

        self.eval_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        self._ready = True

    def is_ready(self) -> bool:
        return self._ready

    @torch.no_grad()
    def warmup(self):
        """Compile CUDA kernels with a dummy forward pass before serving traffic."""
        if not self._ready:
            return
        dummy = torch.randn(5, 3, IMG_SIZE, IMG_SIZE, device=self.device)
        with torch.amp.autocast(device_type=self.device.type):
            _ = self.model(dummy)
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    @staticmethod
    def _get_five_views(pil_img: Image.Image) -> List[Image.Image]:
        """Full page + 4 corner crops at TILE_FRACTION (the tile-max trick)."""
        w, h = pil_img.size
        tw, th = int(w * TILE_FRACTION), int(h * TILE_FRACTION)
        return [
            pil_img,
            pil_img.crop((0, 0, tw, th)),
            pil_img.crop((w - tw, 0, w, th)),
            pil_img.crop((0, h - th, tw, h)),
            pil_img.crop((w - tw, h - th, w, h)),
        ]

    @torch.no_grad()
    def _infer_one_image(self, pil_img: Image.Image) -> np.ndarray:
        """5-view tile-max → max softmax per class across views (shape: num_classes)."""
        views = self._get_five_views(pil_img)
        batch = torch.stack([self.eval_transform(v) for v in views]).to(self.device)
        with torch.amp.autocast(device_type=self.device.type):
            logits = self.model(batch)        # (5, num_classes)
        probs = torch.softmax(logits, dim=1)
        probs = probs.cpu().float().numpy()    # (5, num_classes)
        return probs.max(axis=0)

    def _apply_threshold(self, probs: np.ndarray) -> Tuple[int, int, bool]:
        """
        Apply the floor-plan threshold.
        Returns (final_idx, model_idx, was_demoted).
        """
        model_idx = int(probs.argmax())
        model_conf = float(probs[model_idx])

        if (FP_THRESHOLD > 0 and
                model_idx == FLOOR_PLAN_IDX and
                model_conf < FP_THRESHOLD):
            non_fp = probs.copy()
            non_fp[FLOOR_PLAN_IDX] = -1.0
            final_idx = int(non_fp.argmax())
            return final_idx, model_idx, True
        return model_idx, model_idx, False

    def classify_images(
        self,
        image_bytes_list: List[bytes],
        page_numbers: List[int],
    ) -> List[PagePrediction]:
        """
        Classify a list of in-memory PNG byte blobs.

        Args:
            image_bytes_list: each entry is the raw bytes of one PNG
            page_numbers: parallel list of page numbers (same length)

        Returns: list of PagePrediction (same order as input)
        """
        if len(image_bytes_list) != len(page_numbers):
            raise ValueError(
                f"image_bytes_list ({len(image_bytes_list)}) and "
                f"page_numbers ({len(page_numbers)}) must be same length"
            )

        results = []
        for img_bytes, page_num in zip(image_bytes_list, page_numbers):
            try:
                pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception as e:
                logger.error(f"Failed to decode image for page {page_num}: {e}")
                # Fallback: NOT_ARCHITECTURAL_PLAN at low confidence
                results.append(PagePrediction(
                    page_number=page_num,
                    plan_type="NOT_ARCHITECTURAL_PLAN",
                    confidence=0.0,
                    model_prediction="NOT_ARCHITECTURAL_PLAN",
                    threshold_demoted=False,
                ))
                continue

            t0 = time.time()
            probs = self._infer_one_image(pil_img)
            inference_time = time.time() - t0

            final_idx, model_idx, demoted = self._apply_threshold(probs)
            final_class = MULTICLASS_CLASS_NAMES[final_idx]
            model_class = MULTICLASS_CLASS_NAMES[model_idx]

            # Map snake_case → UPPER_SNAKE labels used by the rest of the pipeline
            results.append(PagePrediction(
                page_number=page_num,
                plan_type=SNAKE_TO_LLM_LABEL[final_class],
                confidence=float(probs[final_idx]),
                model_prediction=SNAKE_TO_LLM_LABEL[model_class],
                threshold_demoted=demoted,
            ))

            logger.debug(
                f"page {page_num}: {SNAKE_TO_LLM_LABEL[final_class]} "
                f"conf={probs[final_idx]:.3f} ({inference_time*1000:.0f}ms)"
                f"{' [DEMOTED]' if demoted else ''}"
            )

        return results
