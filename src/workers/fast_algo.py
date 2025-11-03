"""
FastAlgo: compute AWB (gray world + smoothing), AGC (exponential smoothing), motion estimate, and produce CanonicalPayload.
"""
from typing import List, Any, Dict
import time
import numpy as np
from isp_core.types import CompletionEvent, CanonicalPayload


def _seed_from_meta(meta: Dict[str, Any]):
    s = meta.get("deterministic_seed")
    if s is None:
        return None
    try:
        return int(s) & 0xFFFFFFFF
    except Exception:
        return None


def _gray_world_awb(raw: np.ndarray, bayer_pattern: str = "RGGB"):
    # Estimate per-channel gains from a simple gray-world on demosaiced approximation
    H, W = raw.shape
    # simple split by Bayer positions
    r = raw[0::2, 0::2].astype(np.float32)
    g1 = raw[0::2, 1::2].astype(np.float32)
    g2 = raw[1::2, 0::2].astype(np.float32)
    b = raw[1::2, 1::2].astype(np.float32)
    g = (g1 + g2) * 0.5
    rm = np.median(r)
    gm = np.median(g)
    bm = np.median(b)
    # avoid zeros
    gm = max(gm, 1e-6)
    r_gain = float(np.clip(gm / max(rm, 1e-6), 0.6, 2.0))
    g_gain = 1.0
    b_gain = float(np.clip(gm / max(bm, 1e-6), 0.6, 2.0))
    return {"r": r_gain, "g": g_gain, "b": b_gain}


def _compute_brightness(raw: np.ndarray):
    if raw.size == 0:
        return 0.0
    return float(np.mean(raw.astype(np.float32)))


def process(envelope: Any, read_views: List[np.ndarray], write_views: List[np.ndarray], ctx: Any) -> CompletionEvent:
    t0 = time.perf_counter()
    seed = _seed_from_meta(envelope.meta)
    rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState(None)

    raw = read_views[0]
    if np.issubdtype(raw.dtype, np.integer):
        maxv = float(np.iinfo(raw.dtype).max)
        raw_f = raw.astype(np.float32) / maxv
    else:
        raw_f = raw.astype(np.float32)

    # AWB estimate
    awb = _gray_world_awb(raw)

    # motion estimate (very simple): use small center patch variance vs full-frame
    H, W = raw.shape
    ch = H // 4
    cw = W // 4
    center = raw[H // 2 - ch // 2:H // 2 + ch // 2, W // 2 - cw // 2:W // 2 + cw // 2]
    motion_score = float(np.var(center)) / (np.var(raw) + 1e-9)

    # AGC: compute brightness and target exposure using exponential smoothing
    measured_brightness = _compute_brightness(raw)
    # retrieve previous payload if provided in meta
    prev_payload = envelope.meta.get("prev_payload") or envelope.meta.get("payload_obj")
    prev_exp = None
    if prev_payload and isinstance(prev_payload, dict):
        prev_exp = prev_payload.get("exposure", {}).get("exposureusec")
    elif prev_payload and hasattr(prev_payload, "exposure"):
        prev_exp = prev_payload.exposure.get("exposureusec")
    # target brightness is 0.45 normalized, compute simple scale
    target = 0.45
    if measured_brightness <= 0:
        scale = 1.0
    else:
        scale = target / measured_brightness
    # compute naive exposure target (clamp to sensible range)
    base_exposure = 1000
    exposure_target = int(np.clip((prev_exp or base_exposure) * (1.0 + 0.5 * (scale - 1.0)), 50, 30000))

    # smoothing for stability: exponential smoothing with alpha depending on motion
    motion_factor = np.clip(1.0 - motion_score, 0.1, 1.0)
    alpha = 0.2 * motion_factor
    if prev_exp:
        smoothed = int(prev_exp * (1.0 - alpha) + exposure_target * alpha)
    else:
        smoothed = exposure_target

    exposure = {"exposureusec": smoothed, "analoggain": 1.0, "ae_gain": 1.0}

    # LSC coarse grid: simple flat identity values for now
    grid_w, grid_h = 8, 8
    lsc_vals = [1.0] * (grid_w * grid_h)

    payload = CanonicalPayload(
        frameid=envelope.meta.get("frameid", "unknown"),
        cameraid=envelope.meta.get("cameraid", "cam0"),
        timestamp=time.time(),
        exposure=exposure,
        wb_gains=awb,
        lscgrid={"grid_w": grid_w, "grid_h": grid_h, "values": lsc_vals},
        ccm_matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    ce = CompletionEvent(
        id=envelope.id,
        ok=True,
        writtenhandles=[],
        metrics={"computelatencyms": elapsed_ms},
        provenance={
            "workername": "FastAlgo",
            "modelhash": "heuristic",
            "deterministic_seed": seed,
            "zero_copy": False,
            "fallback_used": False,
            # include payload for dispatcher/mainloop to persist
            "payload_obj": payload.dict(),
            "measured_brightness": measured_brightness,
            "motion_score": motion_score,
        },
    )
    return ce
