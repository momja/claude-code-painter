"""
The critic. Deterministic stand-in for "LLM judge + pixel metric".

pixel: RMSE compared at full, 1/2 and 1/4 resolution. Coarse scales get more weight, so a stroke that
       lands 2px off costs less than a missing one.
style: distance between texture statistics (edge direction histogram, edge density, high-frequency
       energy, palette). Replace with pairwise VLM comparisons for real use.

`score()` and `report()` share `_pixel_scales()`, so the breakdown the dashboard shows is the arithmetic
that produced the score, not a separate reconstruction of it.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from conveyor.painting.canvas import Target
from conveyor.painting.canvas import _gray
from conveyor.painting.canvas import gradients
from conveyor.painting.canvas import heatmap
from conveyor.painting.canvas import to_png

PIXEL_WEIGHT = 0.6
STYLE_WEIGHT = 0.4
# (downsample factor, share of the pixel score). Coarser scales count for more.
SCALES = ((1, 0.25), (2, 0.35), (4, 0.40))
# RMSE at or above this scores 0 at that scale; 0 scores 1.
RMSE_CEILING = 0.35
SCALE_NAMES = {1: "Full", 2: "Half", 4: "Quarter"}


def _downsample(img: np.ndarray, s: int) -> np.ndarray:
    if s == 1:
        return img
    h, w = img.shape[0] // s * s, img.shape[1] // s * s
    return img[:h, :w].reshape(h // s, s, w // s, s, -1).mean(axis=(1, 3))


def _style_features(img: np.ndarray) -> dict[str, np.ndarray | float]:
    gx, gy = gradients(img)
    mag = np.hypot(gx, gy)
    ang = np.mod(np.arctan2(gy, gx), np.pi)
    hist, _ = np.histogram(ang, bins=8, range=(0, np.pi), weights=mag)
    hist = hist / (hist.sum() + 1e-8)
    g = _gray(img)
    lap = g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4 * g[1:-1, 1:-1]
    return {
        "orient": hist,
        "edge": float(mag.mean()),
        "hf": float(np.abs(lap).mean()),
        "mean": img.reshape(-1, 3).mean(axis=0),
        "std": img.reshape(-1, 3).std(axis=0),
    }


def _pixel_scales(canvas: np.ndarray, target: Target) -> list[dict]:
    """One entry per scale: the downsampled images, the per-pixel error map, and the numbers."""
    out = []
    for factor, weight in SCALES:
        a, b = _downsample(canvas, factor), _downsample(target.image, factor)
        err = np.sqrt(((a - b) ** 2).mean(axis=-1))  # per pixel, over the three channels
        rmse = float(np.sqrt((err**2).mean()))  # same value as sqrt of the mean over all pixels and channels
        score = max(0.0, 1.0 - rmse / RMSE_CEILING)
        out.append(dict(factor=factor, weight=weight, rmse=rmse, score=score, contribution=weight * score,
                        canvas=a, target=b, error=err))
    return out


class Critic:
    version = "v1 deterministic proxy"

    def __init__(self) -> None:
        self._target_features: dict[str, dict] = {}

    def _style(self, canvas: np.ndarray, target: Target) -> float:
        tf = self._target_features.get(target.name)
        if tf is None:
            tf = self._target_features[target.name] = _style_features(target.image)
        cf = _style_features(canvas)
        d_orient = 0.5 * float(np.abs(tf["orient"] - cf["orient"]).sum())
        d_edge = min(1.0, abs(tf["edge"] - cf["edge"]) / (tf["edge"] + 1e-3))
        d_hf = min(1.0, abs(tf["hf"] - cf["hf"]) / (tf["hf"] + 1e-3))
        d_color = min(1.0, 3.0 * float(np.abs(tf["mean"] - cf["mean"]).mean() + np.abs(tf["std"] - cf["std"]).mean()))
        return max(0.0, 1.0 - (0.35 * d_orient + 0.2 * d_edge + 0.2 * d_hf + 0.25 * d_color))

    def score(self, canvas: np.ndarray, target: Target) -> dict[str, float]:
        pixel = sum(s["contribution"] for s in _pixel_scales(canvas, target))
        style = self._style(canvas, target)
        return {"pixel": pixel, "style": style, "total": PIXEL_WEIGHT * pixel + STYLE_WEIGHT * style}

    def report(self, canvas: np.ndarray, target: Target, store: Callable[[bytes], str | None]) -> dict:
        """
        The score with its working shown: every scale's images and numbers. `store` turns PNG bytes into an
        artifact name (or None when nothing is recording). Target images repeat every call and deduplicate.
        """
        scales = _pixel_scales(canvas, target)
        pixel = sum(s["contribution"] for s in scales)
        style = self._style(canvas, target)
        rows = []
        for s in scales:
            h, w = s["error"].shape
            rows.append(dict(
                factor=s["factor"], name=SCALE_NAMES.get(s["factor"], f"1/{s['factor']}"), size=[w, h],
                weight=s["weight"], rmse=round(s["rmse"], 5), score=round(s["score"], 5),
                contribution=round(s["contribution"], 5),
                target=store(to_png(s["target"])), canvas=store(to_png(s["canvas"])),
                error=store(heatmap(s["error"], vmax=RMSE_CEILING)),
            ))
        return dict(
            target=target.name, pixel=round(pixel, 5), style=round(style, 5),
            total=round(PIXEL_WEIGHT * pixel + STYLE_WEIGHT * style, 5),
            pixel_weight=PIXEL_WEIGHT, style_weight=STYLE_WEIGHT, rmse_ceiling=RMSE_CEILING, scales=rows,
        )

    @staticmethod
    def patch_errors(canvas: np.ndarray, target: Target) -> np.ndarray:
        rows, cols = target.grid
        out = np.zeros((rows, cols), dtype=np.float32)
        for r in range(rows):
            for c in range(cols):
                ys, xs = target.patch_slice(r, c)
                out[r, c] = np.sqrt(((canvas[ys, xs] - target.image[ys, xs]) ** 2).mean())
        return out

    @staticmethod
    def pixel_error(canvas: np.ndarray, target: Target) -> np.ndarray:
        return np.sqrt(((canvas - target.image) ** 2).mean(axis=-1))
