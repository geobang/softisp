# softisp_full_pipeline_profiled.py
import os, json, time, threading, queue, csv, logging
import numpy as np
import onnxruntime as ort

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
GAMMA_N = int(os.environ.get("GAMMA_N", "256"))
HOUSEKEEP_FILE = os.environ.get("HOUSEKEEP_FILE", "/dev/null")
MODEL_DIR = os.environ.get("MODEL_DIR", "")

ALGO_ONNX = os.path.join(MODEL_DIR, "isp_algo_coeffs_stride_crop.onnx")
RULE_ONNX = os.path.join(MODEL_DIR, "ruleengine_full.onnx")
ISP_ONNX  = os.path.join(MODEL_DIR, "isp_onnx_stride_crop.onnx")

PROVIDERS = ["CPUExecutionProvider"]

HEIGHT = int(os.environ.get("HEIGHT", "1080"))
WIDTH  = int(os.environ.get("WIDTH", "1920"))
BAYER_DTYPE = np.uint16
Q_MAX = 8

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ------------------------------------------------------------
# Identity/golden coeffs
# ------------------------------------------------------------
def generate_identity_coeffs(gammaN: int = GAMMA_N):
    return {
        "wb": np.array([1.0, 1.0, 1.0], dtype=np.float32),
        "ccm": np.array([1,0,0,0,1,0,0,0,1], dtype=np.float32),
        "gamma": np.linspace(0.0, 1.0, gammaN, dtype=np.float32),
        "sharp": np.array([0.0], dtype=np.float32),
        "nr": np.array([0.0], dtype=np.float32),
    }

IDENTITY = generate_identity_coeffs(GAMMA_N)
GOLDEN   = dict(IDENTITY)

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
# Queues + housekeeping
# ------------------------------------------------------------
algos_in_q  = queue.Queue(maxsize=Q_MAX)
coord_in_q  = queue.Queue(maxsize=Q_MAX)
isp_in_q    = queue.Queue(maxsize=Q_MAX)
house_in_q  = queue.Queue(maxsize=Q_MAX*4)
stop_event  = threading.Event()

def drop_oldest_and_put(q, item):
    try: q.put(item, timeout=0.02)
    except queue.Full:
        try: q.get_nowait()
        except queue.Empty: pass
        q.put(item)

def house_emit(msg):
    msg["ts"] = time.time()
    drop_oldest_and_put(house_in_q, msg)

# ------------------------------------------------------------
# Stats collector
# ------------------------------------------------------------
class PipelineStats:
    def __init__(self):
        self.data = {"Camera":[],"Algos":[],"Coord":[],"ISP":[]}
        self.per_frame = []

    def record(self, stage, duration):
        if stage in self.data: self.data[stage].append(duration)

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
                avg, maxd, mind = np.mean(durations), np.max(durations), np.min(durations)
                logging.info(f"{stage}: avg={avg:.3f} ms, min={mind:.3f} ms, max={maxd:.3f} ms, count={len(durations)}")

    def export_csv(self, filename="pipeline_stats.csv"):
        with open(filename,"w",newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["FrameID","Camera","Algos","Coord","ISP"])
            writer.writeheader()
            for row in self.per_frame: writer.writerow(row)
        logging.info(f"Per-frame stats exported to {filename}")

STATS = PipelineStats()
# ------------------------------------------------------------
# ONNX sessions
# ------------------------------------------------------------
algo_sess = ort.InferenceSession(ALGO_ONNX, providers=PROVIDERS)
rule_sess = ort.InferenceSession(RULE_ONNX, providers=PROVIDERS)
isp_sess  = ort.InferenceSession(ISP_ONNX,  providers=PROVIDERS)

def camera_thread():
    latest_coeffs = dict(GOLDEN)
    frame_id = 0
    coeffs_to_camera_q = queue.Queue(maxsize=Q_MAX)

    def consume_coeff_updates():
        try:
            while True:
                update = coeffs_to_camera_q.get_nowait()
                latest_coeffs.update(update["coeffs"])
        except queue.Empty:
            pass

    camera_thread.coeffs_to_camera_q = coeffs_to_camera_q

    while not stop_event.is_set():
        arrival_ts = time.time()
        logging.info(f"[Camera][Frame {frame_id}] Arrival at {arrival_ts:.6f}")

        consume_coeff_updates()
        logging.info(f"[Camera][Frame {frame_id}] Processing started")

        bayer = np.random.randint(0,1024,(HEIGHT,WIDTH),dtype=BAYER_DTYPE)
        meta = {"gain":4.0,"exposure":10.0,"temp":40.0,"scene_change":0.1}
        bundle = {"frame_id":frame_id,"bayer":bayer,"meta":meta,"coeffs":dict(latest_coeffs)}
        drop_oldest_and_put(algos_in_q,bundle)
        drop_oldest_and_put(isp_in_q,bundle)
        house_emit({"thread":"camera","event":"dispatch","frame_id":frame_id})

        dur = (time.time()-arrival_ts)*1000.0
        logging.info(f"[Camera][Frame {frame_id}] Completed in {dur:.3f} ms")
        STATS.record("Camera", dur)

        frame_id += 1
        time.sleep(0.01)

def algos_thread():
    while not stop_event.is_set():
        try:
            bundle = algos_in_q.get(timeout=0.1)
        except queue.Empty:
            continue

        frame_id = bundle["frame_id"]
        arrival_ts = time.time()
        logging.info(f"[Algos][Frame {frame_id}] Arrival at {arrival_ts:.6f}")
        logging.info(f"[Algos][Frame {frame_id}] Processing started")

        bayer_raw_f32 = np.expand_dims(np.expand_dims(bundle["bayer"].astype(np.float32),0),0)
        outs = algo_sess.run(None,{
            "input.bayer": bayer_raw_f32,
            "crop_starts": np.array([0,0],dtype=np.int64),
            "crop_ends": np.array([HEIGHT,WIDTH],dtype=np.int64),
            "crop_axes": np.array([2,3],dtype=np.int64)
        })
        raw = {
            "frame_id": frame_id,
            "raw_wb": outs[1],
            "raw_ccm": outs[2],
            "raw_gamma": outs[3],
            "raw_sharpness": outs[6],
            "raw_nr": outs[5],
            "meta": bundle["meta"],
        }
        drop_oldest_and_put(coord_in_q, raw)
        house_emit({"thread":"algos","event":"raw_coeffs","frame_id":frame_id})

        dur = (time.time()-arrival_ts)*1000.0
        logging.info(f"[Algos][Frame {frame_id}] Completed in {dur:.3f} ms")
        STATS.record("Algos", dur)

def coordinator_thread():
    prev = dict(IDENTITY)
    failures = 0
    coeffs_to_camera_q = camera_thread.coeffs_to_camera_q

    while not stop_event.is_set():
        try:
            raw = coord_in_q.get(timeout=0.1)
        except queue.Empty:
            continue

        frame_id = raw["frame_id"]
        arrival_ts = time.time()
        logging.info(f"[Coord][Frame {frame_id}] Arrival at {arrival_ts:.6f}")
        logging.info(f"[Coord][Frame {frame_id}] Processing started")

        sensor_meta = {
            "analog_gain": np.array([raw["meta"]["gain"]],dtype=np.float32),
            "exposure_time": np.array([raw["meta"]["exposure"]],dtype=np.float32),
            "sensor_temp": np.array([raw["meta"]["temp"]],dtype=np.float32),
            "scene_change": np.array([raw["meta"]["scene_change"]],dtype=np.float32),
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
            GOLDEN.update(stab); prev.update(stab); failures=0
            drop_oldest_and_put(coeffs_to_camera_q, {"frame_id": frame_id, "coeffs": stab})
            house_emit({"thread":"coord","event":"stab_update","frame_id":frame_id})
        except Exception:
            failures += 1
            base = GOLDEN if failures<10 else IDENTITY
            stab = dict(base)
            drop_oldest_and_put(coeffs_to_camera_q, {"frame_id": frame_id, "coeffs": stab})
            house_emit({"thread":"coord","event":"stab_fallback","frame_id":frame_id,"failures":failures})

        dur = (time.time()-arrival_ts)*1000.0
        logging.info(f"[Coord][Frame {frame_id}] Completed in {dur:.3f} ms")
        STATS.record("Coord", dur)

def isp_thread():
    while not stop_event.is_set():
        try:
            bundle = isp_in_q.get(timeout=0.1)
        except queue.Empty:
            continue

        frame_id = bundle["frame_id"]
        arrival_ts = time.time()
        logging.info(f"[ISP][Frame {frame_id}] Arrival at {arrival_ts:.6f}")
        logging.info(f"[ISP][Frame {frame_id}] Processing started")

        bayer_f32 = bundle["bayer"].astype(np.float32)
        coeffs = bundle["coeffs"]
        meta   = bundle["meta"]
        inputs = {
            "input.bayer": bayer_f32,
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
            out_img = isp_sess.run(None, inputs)[0]
            house_emit({"thread":"isp","event":"output","frame_id":frame_id,"shape":list(out_img.shape)})
        except Exception as e:
            house_emit({"thread":"isp","event":"onnx_fail","frame_id":frame_id,"error":str(e)})

        dur = (time.time()-arrival_ts)*1000.0
        logging.info(f"[ISP][Frame {frame_id}] Completed in {dur:.3f} ms")
        STATS.record("ISP", dur)

def housekeeping_thread():
    while not stop_event.is_set():
        try:
            msg = house_in_q.get(timeout=0.5)
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
    # Summarize and export stats at shutdown
    STATS.summarize()
    STATS.export_csv("pipeline_stats.csv")

if __name__ == "__main__":
    main()
