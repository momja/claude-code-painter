"""
The critic: a deterministic score for a painting against its target.

pixel  RMSE at full, 1/2 and 1/4 resolution, coarse scales weighted more, so a mark 2 px off costs less than a
       missing one.
style  distance between texture statistics: edge-direction histogram, edge density, high-frequency energy, and
       palette. A proxy for "does it look painted the same way"; a VLM judge would do this better.

total = 0.6 pixel + 0.4 style, in 0..1, higher is better.
"""

from __future__ import annotations

import numpy as np

from conveyor.painting.canvas import Target

PIXEL_WEIGHT, STYLE_WEIGHT = 0.6, 0.4
SCALES = ((1, 0.25), (2, 0.35), (4, 0.40))  # (downsample factor, share of the pixel score)
RMSE_CEILING = 0.35  # RMSE at or above this scores 0 at that scale


def _gray(img: np.ndarray) -> np.ndarray:
    return img @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _downsample(img: np.ndarray, s: int) -> np.ndarray:
    if s == 1:
        return img
    h, w = img.shape[0] // s * s, img.shape[1] // s * s
    return img[:h, :w].reshape(h // s, s, w // s, s, -1).mean(axis=(1, 3))


def _features(img: np.ndarray) -> dict:
    g = _gray(img)
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:-1] = (g[:, 2:] - g[:, :-2]) * 0.5
    gy[1:-1, :] = (g[2:, :] - g[:-2, :]) * 0.5
    mag = np.hypot(gx, gy)
    hist, _ = np.histogram(np.mod(np.arctan2(gy, gx), np.pi), bins=8, range=(0, np.pi), weights=mag)
    lap = g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4 * g[1:-1, 1:-1]
    return {"orient": hist / (hist.sum() + 1e-8), "edge": float(mag.mean()), "hf": float(np.abs(lap).mean()),
            "mean": img.reshape(-1, 3).mean(axis=0), "std": img.reshape(-1, 3).std(axis=0)}


class Critic:
    version = "pixel x3 + style proxy"

    def __init__(self) -> None:
        self._target_features: dict[str, dict] = {}

    def score(self, canvas: np.ndarray, target: Target) -> dict:
        scales = []
        for factor, weight in SCALES:
            a, b = _downsample(canvas, factor), _downsample(target.image, factor)
            rmse = float(np.sqrt(((a - b) ** 2).mean()))
            s = max(0.0, 1.0 - rmse / RMSE_CEILING)
            scales.append({"factor": factor, "weight": weight, "rmse": round(rmse, 4), "score": round(s, 4)})
        pixel = sum(s["weight"] * s["score"] for s in scales)

        tf = self._target_features.get(target.name)
        if tf is None:
            tf = self._target_features[target.name] = _features(target.image)
        cf = _features(canvas)
        parts = {
            "orientation": 0.5 * float(np.abs(tf["orient"] - cf["orient"]).sum()),
            "edges": min(1.0, abs(tf["edge"] - cf["edge"]) / (tf["edge"] + 1e-3)),
            "detail": min(1.0, abs(tf["hf"] - cf["hf"]) / (tf["hf"] + 1e-3)),
            "palette": min(1.0, 3.0 * float(np.abs(tf["mean"] - cf["mean"]).mean() + np.abs(tf["std"] - cf["std"]).mean())),
        }
        weights = {"orientation": 0.35, "edges": 0.2, "detail": 0.2, "palette": 0.25}
        style = max(0.0, 1.0 - sum(weights[k] * v for k, v in parts.items()))
        return {"total": round(PIXEL_WEIGHT * pixel + STYLE_WEIGHT * style, 5), "pixel": round(pixel, 5),
                "style": round(style, 5), "scales": scales,
                "style_distance": {k: round(v, 4) for k, v in parts.items()}}

    @staticmethod
    def pixel_error(canvas: np.ndarray, target: Target) -> np.ndarray:
        return np.sqrt(((canvas - target.image) ** 2).mean(axis=-1))
