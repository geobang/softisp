"""
FastISP: consume payload (from envelope.payload or envelope.meta['payload_obj']) and render preview into write_views[0].
"""
from typing import List, Any
import numpy as np
import time
from isp_core.types import CompletionEvent, CanonicalPayload


def _demosaic_bilinear(raw: np.ndarray):
    H, W = raw.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    r = raw[0::2, 0::2].astype(np.float32)
    g1 = raw[0::2, 1::2].astype(np.float32)
    g2 = raw[1::2, 0::2].astype(np.float32)
    b = raw[1::2, 1::2].astype(np.float32)
    rgb[0::2, 0::2, 0] = r
    rgb[0::2, 1::2, 1] = g1
    rgb[1::2, 0::2, 1] = g2
    rgb[1::2, 1::2, 2] = b
    # fill missing via simple interpolation
    for c in range(3):
        ch = rgb[:, :, c]
        mask = ch == 0
        kernel_sum = np.zeros_like(ch)
        kernel_count = np.zeros_like(ch)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                y0 = max(0, dy)
                y1 = ch.shape[0] + min(0, dy)
                x0 = max(0, dx)
                x1 = ch.shape[1] + min(0, dx)
                src = ch[max(0, -dy):min(ch.shape[0], ch.shape[0] - dy), max(0, -dx):min(ch.shape[1], ch.shape[1] - dx)]
                kernel_sum[y0:y1, x0:x1] += src
                kernel_count[y0:y1, x0:x1] += (src != 0).astype(np.float32)
        kernel_count[kernel_count == 0] = 1.0
        filled = kernel_sum / kernel_count
        ch[mask] = filled[mask]
        rgb[:, :, c] = ch
    return rgb


def process(envelope: Any, read_views: List[np.ndarray], write_views: List[np.ndarray], ctx: Any) -> CompletionEvent:
    t0 = time.perf_counter()
    raw = read_views[0]
    # obtain payload
    payload_obj = None
    if isinstance(envelope.payload, CanonicalPayload):
        payload_obj = envelope.payload
    else:
        payload_meta = envelope.meta.get("payload_obj") or envelope.meta.get("prev_payload")
        if payload_meta:
            try:
                if isinstance(payload_meta, dict):
                    payload_obj = CanonicalPayload(**payload_meta)
                elif hasattr(payload_meta, "dict"):
                    payload_obj = CanonicalPayload(**(payload_meta.dict()))
            except Exception:
                payload_obj = None
    # demosaic
    rgb = _demosaic_bilinear(raw)
    # apply WB if available
    if payload_obj:
        wb = payload_obj.wb_gains
        rgb[:, :, 0] *= wb.get("r", 1.0)
        rgb[:, :, 1] *= wb.get("g", 1.0)
        rgb[:, :, 2] *= wb.get("b", 1.0)
    # simple tonemap/gamma
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb = np.power(rgb, 1.0 / 2.2, where=rgb > 0, out=rgb)
    written = []
    if write_views:
        out = write_views[0]
        # resize naive to out shape via block mean
        oh = out.shape[0]
        ow = out.shape[1] if out.ndim == 3 else out.shape[0]
        sh = max(1, rgb.shape[0] // oh)
        sw = max(1, rgb.shape[1] // ow)
        small = rgb[: oh * sh, : ow * sw].reshape(oh, sh, ow, sw, 3).mean(axis=(1, 3))
        small8 = np.clip(small * 255.0, 0, 255).astype(out.dtype)
        try:
            if out.shape == small8.shape:
                out[:] = small8
            else:
                out_view = out.reshape(-1)
                src_view = small8.reshape(-1)[: out_view.size]
                out_view[:] = src_view
            written = envelope.writehandles.copy()
        except Exception:
            written = []
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    ce = CompletionEvent(
        id=envelope.id,
        ok=True,
        writtenhandles=written,
        metrics={"computelatencyms": elapsed_ms},
        provenance={
            "workername": "FastISP",
            "modelhash": envelope.meta.get("modelhash", "heuristic"),
            "deterministic_seed": envelope.meta.get("deterministic_seed"),
            "zero_copy": False,
            "fallback_used": False,
        },
    )
    return ce
