# softisp_full_pipeline.py
import os
import json
import time
import threading
import queue
import numpy as np
import onnxruntime as ort

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
GAMMA_N = int(os.environ.get("GAMMA_N", "256"))
HOUSEKEEP_FILE = os.environ.get("HOUSEKEEP_FILE", "/dev/null")
MODEL_DIR = os.environ.get("MODEL_DIR", "")

# ONNX graph paths (adjust to your repo layout)
ALGO_ONNX   = os.path.join(MODEL_DIR, "isp_algo_coeffs_stride_crop.onnx")           # stage 2 (raw coeffs)
RULE_ONNX   = os.path.join(MODEL_DIR, "ruleengine_full.onnx")       # stage 3 (stabilization)
ISP_ONNX    = os.path.join(MODEL_DIR, "isp_onnx_stride_crop.onnx")              # single-graph ISP

# Providers (use GPU providers if configured)
PROVIDERS = ["CPUExecutionProvider"]

# Simulated camera
HEIGHT = int(os.environ.get("HEIGHT", "1080"))
WIDTH  = int(os.environ.get("WIDTH", "1920"))
BAYER_DTYPE = np.uint16

# Queue sizes
Q_MAX = 8

# ------------------------------------------------------------
# Identity/golden coeffs
# ------------------------------------------------------------
def generate_identity_coeffs(gammaN: int = GAMMA_N):
    wb = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    ccm = np.array([
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    ], dtype=np.float32)
    gamma = np.linspace(0.0, 1.0, gammaN, dtype=np.float32)
    sharp = np.array([0.0], dtype=np.float32)
    nr    = np.array([0.0], dtype=np.float32)
    return {"wb": wb, "ccm": ccm, "gamma": gamma, "sharp": sharp, "nr": nr}

IDENTITY = generate_identity_coeffs(GAMMA_N)
GOLDEN   = dict(IDENTITY)  # last-good stabilized snapshot

# Rule params (align with your builder)
RULE_PARAMS = {
    "alpha_wb": np.array([0.2], dtype=np.float32),
    "alpha_ccm": np.array([0.2], dtype=np.float32),
    "alpha_gamma": np.array([0.2], dtype=np.float32),
    "alpha_sharp": np.array([0.2], dtype=np.float32),
    "alpha_nr": np.array([0.2], dtype=np.float32),
    "alpha_fast": np.array([0.8], dtype=np.float32),
    "wb_step": np.array([0.05], dtype=np.float32),
    "sharp_step": np.array([0.05], dtype=np.float32),
    "wb_min": np.array([0.6], dtype=np.float32),
    "wb_max": np.array([1.6], dtype=np.float32),
    "gamma_min": np.array([0.0], dtype=np.float32),
    "gamma_max": np.array([1.0], dtype=np.float32),
    "ccm_min": np.array([0.5], dtype=np.float32),
    "ccm_max": np.array([1.5], dtype=np.float32),
    "nr_min": np.array([0.0], dtype=np.float32),
    "nr_max": np.array([1.0], dtype=np.float32),
}

# ------------------------------------------------------------
# Queues (each thread owns an input queue)
# ------------------------------------------------------------
camera_in_q = queue.Queue(maxsize=Q_MAX)     # external trigger if needed (unused in sim)
algos_in_q  = queue.Queue(maxsize=Q_MAX)     # frames from Camera -> Algos
coord_in_q  = queue.Queue(maxsize=Q_MAX)     # raw coeffs from Algos -> Coordinator
isp_in_q    = queue.Queue(maxsize=Q_MAX)     # frame bundles from Camera -> ISP
house_in_q  = queue.Queue(maxsize=Q_MAX * 4) # output frames from ISP -> Housekeeping

stop_event = threading.Event()

def drop_oldest_and_put(q: queue.Queue, item):
    try:
        q.put(item, timeout=0.02)
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        q.put(item)

def house_emit(msg: dict):
    msg["ts"] = time.time()
    drop_oldest_and_put(house_in_q, msg)

# ------------------------------------------------------------
# ONNX sessions
# ------------------------------------------------------------
algo_sess   = ort.InferenceSession(ALGO_ONNX, providers=PROVIDERS)
rule_sess   = ort.InferenceSession(RULE_ONNX, providers=PROVIDERS)
isp_sess    = ort.InferenceSession(ISP_ONNX,  providers=PROVIDERS)

# ------------------------------------------------------------
# Threads
# ------------------------------------------------------------
def camera_thread():
    """
    Dispatcher and holder of latest coeffs.
    - On coeff arrival (from coordinator): update latest_coeffs.
    - On new frame: bundle bayer + latest_coeffs + meta; enqueue to Algos and ISP.
    """
    latest_coeffs = dict(GOLDEN)  # start with golden/identity
    frame_id = 0

    # Optional: receive coeff updates via a side-channel queue if needed
    # For simplicity, we update latest_coeffs inside this thread by reading coord_in_q? No:
    # Coordinator sends coeffs back via a dedicated "coeffs_to_camera_q" to avoid mixing.
    coeffs_to_camera_q = queue.Queue(maxsize=Q_MAX)

    def consume_coeff_updates():
        # Non-blocking drain; update latest_coeffs when coordinator posts
        try:
            while True:
                update = coeffs_to_camera_q.get_nowait()
                latest_coeffs.update(update["coeffs"])
        except queue.Empty:
            pass

    # Provide the coordinator a reference to this queue (closure trick)
    camera_thread.coeffs_to_camera_q = coeffs_to_camera_q

    while not stop_event.is_set():
        # 1) Drain any pending coeff updates
        consume_coeff_updates()

        # 2) Simulate a Bayer frame
        bayer = np.random.randint(0, 1024, (HEIGHT, WIDTH), dtype=BAYER_DTYPE)
        meta = {
            "gain": 4.0,
            "exposure": 10.0,
            "temp": 40.0,
            "scene_change": 0.1,
        }

        bundle = {"frame_id": frame_id, "bayer": bayer, "meta": meta, "coeffs": dict(latest_coeffs)}

        # 3) Dispatch to Algos for raw coeff updates
        drop_oldest_and_put(algos_in_q, bundle)

        # 4) Dispatch to ISP with latest coeffs
        drop_oldest_and_put(isp_in_q, bundle)

        house_emit({"thread": "camera", "event": "dispatch", "frame_id": frame_id})
        frame_id += 1
        time.sleep(0.01)

def algos_thread():
    """
    Consumes frame bundles from Camera, runs stride→algo coeffs ONNX, sends raw coeffs to Coordinator.
    """
    while not stop_event.is_set():
        try:
            bundle = algos_in_q.get(timeout=0.1)
        except queue.Empty:
            continue

        # Align input names to test_integration_algos_stride.py
        bayer_raw_f32 = bundle["bayer"].astype(np.float32)
        bayer_cropped = stride_sess.run(None, {"bayer_raw": bayer_raw_f32})[0]

        outs = algo_sess.run(None, {"bayer_cropped": bayer_cropped})
        raw = {
            "frame_id": bundle["frame_id"],
            "raw_wb": outs[0],
            "raw_ccm": outs[1],
            "raw_gamma": outs[2],
            "raw_sharpness": outs[3],
            "raw_nr": outs[4],
            "meta": bundle["meta"],
        }
        drop_oldest_and_put(coord_in_q, raw)
        house_emit({"thread": "algos", "event": "raw_coeffs", "frame_id": bundle["frame_id"]})

def coordinator_thread():
    """
    Consumes raw coeffs, runs rule engine ONNX to stabilize them,
    holds latest stabilized coeffs internally, and posts updates back to Camera.
    Ignores frames (operates only on coeff updates).
    """
    prev = dict(IDENTITY)
    failures = 0

    # Access camera’s coeff update queue
    coeffs_to_camera_q = camera_thread.coeffs_to_camera_q

    while not stop_event.is_set():
        try:
            raw = coord_in_q.get(timeout=0.1)
        except queue.Empty:
            continue

        sensor_meta = {
            "analog_gain": np.array([raw["meta"]["gain"]], dtype=np.float32),
            "exposure_time": np.array([raw["meta"]["exposure"]], dtype=np.float32),
            "sensor_temp": np.array([raw["meta"]["temp"]], dtype=np.float32),
            "scene_change": np.array([raw["meta"]["scene_change"]], dtype=np.float32),
        }
        inputs = {
            "raw_wb": raw["raw_wb"], "prev_wb": prev["wb"],
            "raw_ccm": raw["raw_ccm"], "prev_ccm": prev["ccm"],
            "raw_gamma": raw["raw_gamma"], "prev_gamma": prev["gamma"],
            "raw_sharpness": raw["raw_sharpness"], "prev_sharpness": prev["sharp"],
            "raw_nr": raw["raw_nr"], "prev_nr": prev["nr"],
            **sensor_meta, **RULE_PARAMS
        }

        try:
            outs = rule_sess.run(None, inputs)
            stab = {"wb": outs[0], "ccm": outs[1], "gamma": outs[2], "sharp": outs[3], "nr": outs[4]}
            GOLDEN.update(stab)
            prev.update(stab)
            failures = 0
            # Post stabilized coeff update back to Camera (dispatcher)
            drop_oldest_and_put(coeffs_to_camera_q, {"frame_id": raw["frame_id"], "coeffs": stab})
            house_emit({"thread": "coord", "event": "stab_update", "frame_id": raw["frame_id"]})
        except Exception:
            failures += 1
            base = GOLDEN if failures < 10 else IDENTITY
            stab = dict(base)
            drop_oldest_and_put(coeffs_to_camera_q, {"frame_id": raw["frame_id"], "coeffs": stab})
            house_emit({"thread": "coord", "event": "stab_fallback", "frame_id": raw["frame_id"], "failures": failures})

def isp_thread():
    """
    Consumes bundles from Camera (bayer + latest coeffs + meta), runs single ISP ONNX,
    and pushes output frames to Housekeeping.
    """
    while not stop_event.is_set():
        try:
            bundle = isp_in_q.get(timeout=0.1)
        except queue.Empty:
            continue

        bayer_f32 = bundle["bayer"].astype(np.float32)
        coeffs = bundle["coeffs"]
        meta = bundle["meta"]

        inputs = {
            # Align these names to your ISP ONNX inputs
            "bayer_raw": bayer_f32,
            "wb": coeffs["wb"],
            "ccm": coeffs["ccm"],
            "gamma": coeffs["gamma"],
            "sharp": coeffs["sharp"],
            "nr": coeffs["nr"],
            "analog_gain": np.array([meta["gain"]], dtype=np.float32),
            "exposure_time": np.array([meta["exposure"]], dtype=np.float32),
            "sensor_temp": np.array([meta["temp"]], dtype=np.float32),
            "scene_change": np.array([meta["scene_change"]], dtype=np.float32),
        }

        try:
            # Expect ISP ONNX to output RGB/YUV
            out_img = isp_sess.run(None, inputs)[0]
            # Push minimal record into housekeeping queue; attach frame_id for traceability
            house_emit({"thread": "isp", "event": "output", "frame_id": bundle["frame_id"],
                        "shape": list(out_img.shape)})
        except Exception as e:
            # If ISP ONNX fails, emit a log and continue
            house_emit({"thread": "isp", "event": "onnx_fail", "frame_id": bundle["frame_id"], "error": str(e)})

def housekeeping_thread():
    """
    Writes ISP outputs (status only) to file; default is /dev/null.
    """
    while not stop_event.is_set():
        try:
            msg = house_in_q.get(timeout=0.5)
        except queue.Empty:
            continue

        try:
            with open(HOUSEKEEP_FILE, "a") as f:
                f.write(json.dumps(msg) + "\n")
        except Exception:
            pass

# ------------------------------------------------------------
# Boot / teardown
# ------------------------------------------------------------
def main():
    threads = [
        threading.Thread(target=camera_thread, daemon=True),
        threading.Thread(target=algos_thread, daemon=True),
        threading.Thread(target=coordinator_thread, daemon=True),
        threading.Thread(target=isp_thread, daemon=True),
        threading.Thread(target=housekeeping_thread, daemon=True),
    ]
    for t in threads:
        t.start()
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        stop_event.set()
        for t in threads:
            t.join(timeout=1.0)

if __name__ == "__main__":
    main()
