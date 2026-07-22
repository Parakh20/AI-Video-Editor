"""Heuristic frame aesthetic scoring: contrast, colorfulness, and edge-energy composition.

Uses the Hasler & Susstrunk (2003) colorfulness metric plus luminance contrast
and a simple rule-of-thirds edge-energy proxy, combined into a single 0-100
score. This is a lightweight heuristic, not a learned aesthetic model — it is
meant to rank clips relative to each other, not to make absolute judgments.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ai_video_editor.utils.logging import get_logger

logger = get_logger(__name__)


def colorfulness(frame: np.ndarray) -> float:
    """Hasler-Susstrunk colorfulness metric."""
    b, g, r = (c.astype(np.float64) for c in cv2.split(frame.astype(np.float32)))
    rg = r - g
    yb = 0.5 * (r + g) - b
    std_rg, mean_rg = np.std(rg), np.mean(rg)
    std_yb, mean_yb = np.std(yb), np.mean(yb)
    std_root = np.sqrt(std_rg**2 + std_yb**2)
    mean_root = np.sqrt(mean_rg**2 + mean_yb**2)
    return float(std_root + 0.3 * mean_root)


def luminance_contrast(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
    return float(np.std(gray))


def rule_of_thirds_energy(frame: np.ndarray) -> float:
    """Fraction of Canny edge energy falling near the four rule-of-thirds lines."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    h, w = edges.shape
    band_h, band_w = max(1, h // 10), max(1, w // 10)

    thirds_mask = np.zeros_like(edges, dtype=bool)
    for frac in (1 / 3, 2 / 3):
        y = int(h * frac)
        x = int(w * frac)
        thirds_mask[max(0, y - band_h) : y + band_h, :] = True
        thirds_mask[:, max(0, x - band_w) : x + band_w] = True

    total_energy = np.count_nonzero(edges)
    if total_energy == 0:
        return 0.0
    thirds_energy = np.count_nonzero(edges & thirds_mask)
    return float(thirds_energy / total_energy)


@dataclass(frozen=True, slots=True)
class AestheticScore:
    score: float  # 0-100
    colorfulness: float
    contrast: float
    rule_of_thirds_energy: float


# Saturation points used to normalize each raw signal onto a comparable 0-1
# range before blending; picked from typical action-cam footage statistics.
_COLORFULNESS_SATURATION = 60.0
_CONTRAST_SATURATION = 70.0


def score_frame_aesthetics(frame: np.ndarray) -> AestheticScore:
    color = colorfulness(frame)
    contrast = luminance_contrast(frame)
    thirds = rule_of_thirds_energy(frame)

    color_norm = min(color / _COLORFULNESS_SATURATION, 1.0)
    contrast_norm = min(contrast / _CONTRAST_SATURATION, 1.0)
    thirds_norm = min(thirds / 0.6, 1.0)

    score = 100.0 * (0.4 * color_norm + 0.35 * contrast_norm + 0.25 * thirds_norm)
    return AestheticScore(
        score=float(score), colorfulness=color, contrast=contrast, rule_of_thirds_energy=thirds
    )
