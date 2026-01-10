# Trunk 1: Imports, configuration, helpers
import os, json, time, threading, queue, csv, logging
import numpy as np
import onnxruntime as ort

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
MODEL_DIR = os.environ.get("MODEL_DIR", "")
ALGO_ONNX = os.path.join(MODEL_DIR, "isp_algo_coeffs_full.onnx")
RULE_ONNX = os.path.join(MODEL_DIR, "isp_rule_engine.onnx")
ISP_ONNX  = os.path.join(MODEL_DIR, "isp_rggb_full.onnx")

PROVIDERS = ["CPUExecutionProvider"]

# FHD active area and stride
HEIGHT = 1080
WIDTH_ACTIVE = 1920
STRIDE = 2048

BIT_DEPTH = 10
BLC_OFFSET = int(os.environ.get("BLC_OFFSET", "64"))

HOUSEKEEP_FILE = os.environ.get("HOUSEKEEP_FILE", "/dev/null")
Q_MAX = 8

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ------------------------------------------------------------
# Identity/golden coeffs aligned to new spec
# ------------------------------------------------------------
def generate_identity_coeffs():
    return {
        "wb": np.array([1.0, 1.0, 1.0], dtype=np.float32),                   # (3,)
        "ccm": np.array([1,0,0, 0,1,0, 0,0,1], dtype=np.float32),            # (9,)
        "gamma": np.array([2.2], dtype=np.float32),                          # (1,)
        "nr_strength": np.array([0.0], dtype=np.float32),                    # (1,)
        "sharp_strength": np.array([0.0], dtype=np.float32),                 # (1,)
    }

IDENTITY = generate_identity_coeffs()
GOLDEN   = dict(IDENTITY)

# ------------------------------------------------------------
# Queue utilities, housekeeping emit
# ------------------------------------------------------------
def drop_oldest_and_put(q, item):
    try:
        q.put(item, timeout=0.02)
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        q.put(item)

def house_emit(msg, house_q):
    msg["ts"] = time.time()
    drop_oldest_and_put(house_q, msg)
# Trunk 2: Stats collection
class PipelineStats:
    def __init__(self):
        self.data = {"Camera":[], "Algos":[], "Coord":[], "ISP":[]}
        self.per_frame = []
        self.frames_skipped_isp_before_ready = 0

    def record(self, stage, duration):
        if stage in self.data:
            self.data[stage].append(duration)

    def record_frame_summary(self, frame_id, cam, alg, coord, isp):
        self.per_frame.append({
            "FrameID": frame_id,
            "Camera": cam,
            "Algos": alg,
            "Coord": coord,
            "ISP": isp
        })

    def summarize(self):
        logging.info("=== Aggregate Performance Summary ===")
        for stage, durations in self.data.items():
            if durations:
                avg, mx, mn = float(np.mean(durations)), float(np.max(durations)), float(np.min(durations))
                logging.info(f"{stage}: avg={avg:.3f} ms, min={mn:.3f} ms, max={mx:.3f} ms, count={len(durations)}")
        logging.info(f"ISP frames skipped before algo_ready: {self.frames_skipped_isp_before_ready}")

    def export_csv(self, filename="pipeline_stats.csv"):
        with open(filename,"w",newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["FrameID","Camera","Algos","Coord","ISP"])
            writer.writeheader()
            for row in self.per_frame:
                writer.writerow(row)
        logging.info(f"Per-frame stats exported to {filename}")
# Trunk 3: ONNX sessions factory
def create_sessions():
    algo_sess = ort.InferenceSession(ALGO_ONNX, providers=PROVIDERS)
    rule_sess = ort.InferenceSession(RULE_ONNX, providers=PROVIDERS)
    isp_sess  = ort.InferenceSession(ISP_ONNX,  providers=PROVIDERS)
    return algo_sess, rule_sess, isp_sess
# Trunk 4: Strided RGGB Bayer emulation
def emulate_rggb_bayer_10bit_strided(H, W_active, W_stride, bit_depth=BIT_DEPTH):
    """
    Returns float32 array of shape (H, W_stride) with RGGB CFA.
    Active area columns [0:W_active]; right padding [W_active:W_stride] is zero.
    """
    assert W_stride >= W_active
    max_val = (1 << bit_depth) - 1

    base = np.linspace(0, max_val, H*W_active, dtype=np.float32).reshape(H, W_active)
    base = 0.6*base + 0.4*np.roll(base, shift=H//12, axis=0)
    noise = np.random.normal(0, 3.0, size=(H, W_active)).astype(np.float32)
    raw_active = np.clip(base + noise, 0, max_val).astype(np.float32)

    raw = np.zeros((H, W_stride), dtype=np.float32)
    raw[:, :W_active] = raw_active

    # CFA masks over stride width (float for easy multiply)
    yy, xx = np.meshgrid(np.arange(H), np.arange(W_stride), indexing='ij')
    r_mask = ((yy % 2 == 0) & (xx % 2 == 0)).astype(np.float32)
    g_mask = (((yy % 2 == 0) & (xx % 2 == 1)) | ((yy % 2 == 1) & (xx % 2 == 0))).astype(np.float32)
    b_mask = ((yy % 2 == 1) & (xx % 2 == 1)).astype(np.float32)

    bayer = (raw * r_mask) + (raw * g_mask) + (raw * b_mask)
    return bayer.astype(np.float32)
# Trunk 5: Camera thread
def camera_thread(algos_in_q, isp_in_q, house_q, stats, coeffs_to_camera_q, stop_event):
    latest_coeffs = dict(GOLDEN)
    frame_id = 0

    def consume_coeff_updates():
        try:
            while True:
                update = coeffs_to_camera_q.get_nowait()
                latest_coeffs.update(update["coeffs"])
        except queue.Empty:
            pass

    while not stop_event.is_set():
        arrival_ts = time.time()
        logging.info(f"[Camera][Frame {frame_id}] Arrival at {arrival_ts:.6f}")

        consume_coeff_updates()
        logging.info(f"[Camera][Frame {frame_id}] Processing started")

        bayer_strided = emulate_rggb_bayer_10bit_strided(HEIGHT, WIDTH_ACTIVE, STRIDE, BIT_DEPTH)
        meta = {
            "analog_gain": np.array([2.0], dtype=np.float32),
            "exposure_time": np.array([0.01], dtype=np.float32),
            "sensor_temp": np.array([35.0], dtype=np.float32),
            "scene_change": np.array([0.0], dtype=np.float32),
        }

        bundle = {"frame_id": frame_id, "bayer": bayer_strided, "meta": meta, "coeffs": dict(latest_coeffs)}
        drop_oldest_and_put(algos_in_q, bundle)
        drop_oldest_and_put(isp_in_q, bundle)
        house_emit({"thread":"camera","event":"dispatch","frame_id":frame_id}, house_q)

        dur = (time.time()-arrival_ts)*1000.0
        logging.info(f"[Camera][Frame {frame_id}] Completed in {dur:.3f} ms")
        stats.record("Camera", dur)
        frame_id += 1
        time.sleep(0.01)
# Trunk 6: Algos thread
def algos_thread(algos_in_q, coord_in_q, algo_sess, house_q, stats, stop_event):
    while not stop_event.is_set():
        try:
            bundle = algos_in_q.get(timeout=0.1)
        except queue.Empty:
            continue

        frame_id = bundle["frame_id"]
        arrival_ts = time.time()
        logging.info(f"[Algos][Frame {frame_id}] Arrival at {arrival_ts:.6f}")
        logging.info(f"[Algos][Frame {frame_id}] Processing started")

        bayer_nchw = np.expand_dims(np.expand_dims(bundle["bayer"].astype(np.float32), 0), 0)  # (1,1,H,W_stride)
        meta = bundle["meta"]

        # If algos model accepts crop tensors, feed them; else remove.
        crop_starts = np.array([0, 0], dtype=np.int64)
        crop_ends   = np.array([HEIGHT, WIDTH_ACTIVE], dtype=np.int64)
        crop_axes   = np.array([2, 3], dtype=np.int64)

        feed = {
            "input.bayer": bayer_nchw,
            "analog_gain": meta["analog_gain"],
            "exposure_time": meta["exposure_time"],
            "sensor_temp": meta["sensor_temp"],
            "scene_change": meta["scene_change"],
            "height_active": np.array([HEIGHT], dtype=np.int64),
            "width_active": np.array([WIDTH_ACTIVE], dtype=np.int64),
        }

        # Optional crop inputs if declared in the model
        try:
            input_names = {i.name for i in algo_sess.get_inputs()}
        except Exception:
            input_names = set()
        if {"crop_starts","crop_ends","crop_axes"}.issubset(input_names):
            feed.update({"crop_starts": crop_starts, "crop_ends": crop_ends, "crop_axes": crop_axes})

        outs = algo_sess.run(None, feed)

        raw = {
            "frame_id": frame_id,
            "raw_wb": outs[0],                   # (3,)
            "raw_ccm": outs[1],                  # (9,)
            "raw_gamma": outs[2],                # (1,)
            "raw_nr_strength": outs[3],          # (1,)
            "raw_sharp_strength": outs[4],       # (1,)
            "meta": meta,
        }
        drop_oldest_and_put(coord_in_q, raw)
        house_emit({"thread":"algos","event":"raw_coeffs","frame_id":frame_id}, house_q)

        dur = (time.time()-arrival_ts)*1000.0
        logging.info(f"[Algos][Frame {frame_id}] Completed in {dur:.3f} ms")
        stats.record("Algos", dur)
# Trunk 7: Coordinator thread
def coordinator_thread(coord_in_q, rule_sess, coeffs_to_camera_q, house_q, stats, stop_event, flags):
    prev = dict(IDENTITY)
    failures = 0
    flags["algo_ready"] = False

    while not stop_event.is_set():
        try:
            raw = coord_in_q.get(timeout=0.1)
        except queue.Empty:
            continue

        frame_id = raw["frame_id"]
        arrival_ts = time.time()
        logging.info(f"[Coord][Frame {frame_id}] Arrival at {arrival_ts:.6f}")
        logging.info(f"[Coord][Frame {frame_id}] Processing started")

        sensor_meta = raw["meta"]
        inputs = {
            "wb_prev": prev["wb"], "wb_next": raw["raw_wb"],
            "ccm_prev": prev["ccm"], "ccm_next": raw["raw_ccm"],
            "gamma_prev": prev["gamma"], "gamma_next": raw["raw_gamma"],
            "nr_prev": prev["nr_strength"], "nr_next": raw["raw_nr_strength"],
            "sharp_prev": prev["sharp_strength"], "sharp_next": raw["raw_sharp_strength"],
            **sensor_meta,
        }

        try:
            outs = rule_sess.run(None, inputs)
            stab = {
                "wb": outs[0], "ccm": outs[1], "gamma": outs[2],
                "nr_strength": outs[3], "sharp_strength": outs[4]
            }
            GOLDEN.update(stab); prev.update(stab); failures = 0
            flags["algo_ready"] = True
            drop_oldest_and_put(coeffs_to_camera_q, {"frame_id": frame_id, "coeffs": stab})
            house_emit({"thread":"coord","event":"stab_update","frame_id":frame_id}, house_q)
        except Exception as e:
            failures += 1
            base = GOLDEN if failures < 10 else IDENTITY
            stab = dict(base)
            drop_oldest_and_put(coeffs_to_camera_q, {"frame_id": frame_id, "coeffs": stab})
            house_emit({"thread":"coord","event":"stab_fallback","frame_id":frame_id,"failures":failures,"error":str(e)}, house_q)

        dur = (time.time()-arrival_ts)*1000.0
        logging.info(f"[Coord][Frame {frame_id}] Completed in {dur:.3f} ms")
        stats.record("Coord", dur)
# Trunk 8: ISP thread
def isp_thread(isp_in_q, isp_sess, house_q, stats, stop_event, flags):
    while not stop_event.is_set():
        try:
            bundle = isp_in_q.get(timeout=0.1)
        except queue.Empty:
            continue

        frame_id = bundle["frame_id"]
        arrival_ts = time.time()
        logging.info(f"[ISP][Frame {frame_id}] Arrival at {arrival_ts:.6f}")

        if not flags.get("algo_ready", False):
            logging.info(f"[ISP][Frame {frame_id}] Skipped — no valid coeffs yet")
            stats.frames_skipped_isp_before_ready += 1
            continue

        logging.info(f"[ISP][Frame {frame_id}] Processing started")

        bayer_nchw = np.expand_dims(np.expand_dims(bundle["bayer"].astype(np.float32), 0), 0)
        coeffs = bundle["coeffs"]

        feed = {
            "input.bayer": bayer_nchw,
            "bit_depth": np.array([BIT_DEPTH], dtype=np.float32),
            "blc_offset": np.array([BLC_OFFSET], dtype=np.float32),
            "lsc_gain_map": np.ones_like(bayer_nchw, dtype=np.float32),
            "wb": coeffs["wb"].astype(np.float32),
            "ccm": coeffs["ccm"].astype(np.float32),
            "gamma": coeffs["gamma"].astype(np.float32),
            "nr_strength": coeffs["nr_strength"].astype(np.float32),
            "sharp_strength": coeffs["sharp_strength"].astype(np.float32),
        }

        # If ISP model expects crop tensors, feed them; else skip.
        try:
            input_names = {i.name for i in isp_sess.get_inputs()}
        except Exception:
            input_names = set()
        if {"crop_starts","crop_ends","crop_axes"}.issubset(input_names):
            feed.update({
                "crop_starts": np.array([0, 0], dtype=np.int64),
                "crop_ends":   np.array([HEIGHT, WIDTH_ACTIVE], dtype=np.int64),
                "crop_axes":   np.array([2, 3], dtype=np.int64),
            })

        try:
            outputs = isp_sess.run(None, feed)
            # Adapt logging based on model outputs: RGB, YUV, etc.
            house_emit({"thread":"isp","event":"output",
                        "frame_id":frame_id,
                        "out_shapes":[list(o.shape) for o in outputs],
                        "out_means":[float(np.mean(o)) for o in outputs]},
                       house_q)
        except Exception as e:
            house_emit({"thread":"isp","event":"onnx_fail","frame_id":frame_id,"error":str(e)}, house_q)

        dur = (time.time()-arrival_ts)*1000.0
        logging.info(f"[ISP][Frame {frame_id}] Completed in {dur:.3f} ms")
        stats.record("ISP", dur)
# Trunk 9: Housekeeping thread
def housekeeping_thread(house_q, stop_event):
    while not stop_event.is_set():
        try:
            msg = house_q.get(timeout=0.5)
        except queue.Empty:
            continue

        arrival_ts = time.time()
        logging.info(f"[Housekeeping] Message arrival at {arrival_ts:.6f}")
        logging.info(f"[Housekeeping] Processing started")

        try:
            with open(HOUSEKEEP_FILE, "a") as f:
                f.write(json.dumps(msg) + "\n")
        except Exception:
            pass

        dur = (time.time()-arrival_ts)*1000.0
        logging.info(f"[Housekeeping] Completed in {dur:.3f} ms")
# Trunk 10: Boot / teardown
def main():
    stats = PipelineStats()
    stop_event = threading.Event()
    flags = {}

    # Queues
    algos_in_q  = queue.Queue(maxsize=Q_MAX)
    coord_in_q  = queue.Queue(maxsize=Q_MAX)
    isp_in_q    = queue.Queue(maxsize=Q_MAX)
    house_q     = queue.Queue(maxsize=Q_MAX*4)
    coeffs_to_camera_q = queue.Queue(maxsize=Q_MAX)

    # Sessions
    algo_sess, rule_sess, isp_sess = create_sessions()

    # Threads
    threads = [
        threading.Thread(target=camera_thread, daemon=True,
                         args=(algos_in_q, isp_in_q, house_q, stats, coeffs_to_camera_q, stop_event)),
        threading.Thread(target=algos_thread, daemon=True,
                         args=(algos_in_q, coord_in_q, algo_sess, house_q, stats, stop_event)),
        threading.Thread(target=coordinator_thread, daemon=True,
                         args=(coord_in_q, rule_sess, coeffs_to_camera_q, house_q, stats, stop_event, flags)),
        threading.Thread(target=isp_thread, daemon=True,
                         args=(isp_in_q, isp_sess, house_q, stats, stop_event, flags)),
        threading.Thread(target=housekeeping_thread, daemon=True,
                         args=(house_q, stop_event)),
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

    stats.summarize()
    stats.export_csv("pipeline_stats.csv")

if __name__ == "__main__":
    main()
