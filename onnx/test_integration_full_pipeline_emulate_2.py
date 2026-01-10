# test_integration_full_pipeline_emulate_1.py
# Integration test for full SoftISP pipeline emulation with ONNX Runtime
# Includes logging and profiling instrumentation

import os
import sys
import time
import logging
import unittest

import numpy as np
import onnxruntime as ort

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

def log_frame_arrival(frame_id):
    ts = time.time()
    logging.info(f"[Frame {frame_id}] Arrival at {ts:.6f}")
    return ts

def timed_stage(stage_name, func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    duration = (end - start) * 1000.0  # ms
    logging.info(f"[{stage_name}] Duration: {duration:.3f} ms")
    return result, duration

def log_frame_completion(frame_id, arrival_ts):
    total = (time.time() - arrival_ts) * 1000.0
    logging.info(f"[Frame {frame_id}] Completed in {total:.3f} ms")
# --- Trunk 2: ONNX session setup + Rule Engine wrapper ---

def create_onnx_session(model_path, use_cuda=True):
    """Create an ONNX Runtime session with optional CUDA/TensorRT provider."""
    providers = ["CUDAExecutionProvider", "TensorrtExecutionProvider", "CPUExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]
    so = ort.SessionOptions()
    so.enable_profiling = False  # can be set True for deeper profiling
    session = ort.InferenceSession(model_path, sess_options=so, providers=providers)
    logging.info(f"ONNX session created for {model_path} with providers {providers}")
    return session

def run_rule_engine(session, inputs, frame_id):
    """
    - If inputs is None: use defaults.
    - If inputs is dict: merge with defaults.
    - Filter out unknown keys (e.g., 'coeffs') that aren't in the model input signature.
    - Log merged and dropped keys, run with timing.
    """
    # Discover valid inputs from the model
    model_inputs = session.get_inputs()
    valid_names = {i.name for i in model_inputs}

    # Build defaults that match the model's required shapes/dtypes
    # Adjust as per your earlier fixes:
    defaults = {
        "raw_wb":     np.array([1.0, 1.0, 1.0], dtype=np.float32),        # 3
        "prev_wb":    np.array([1.0, 1.0, 1.0], dtype=np.float32),        # 3
        "raw_ccm":    np.ones(9, dtype=np.float32),                       # 9 (flattened 3x3)
        "prev_ccm":   np.ones(9, dtype=np.float32),                       # 9
        "raw_gamma":  np.linspace(0.0, 1.0, 256).astype(np.float32),      # 256
        "prev_gamma": np.linspace(0.0, 1.0, 256).astype(np.float32),      # 256
        "raw_sharpness": np.array([0.5], dtype=np.float32),               # 1
        "raw_nr":        np.array([0.5], dtype=np.float32),               # 1
        "prev_sharpness": np.array([0.5], dtype=np.float32),              # 1
        "prev_nr":        np.array([0.5], dtype=np.float32),              # 1

        "analog_gain":   np.array([1.0], dtype=np.float32),               # 1
        "exposure_time": np.array([0.01], dtype=np.float32),              # 1
        "sensor_temp":   np.array([25.0], dtype=np.float32),              # 1
        "scene_change":  np.array([0.0], dtype=np.float32),               # 1

        "alpha_wb":    np.array([0.5], dtype=np.float32),                 # 1
        "alpha_ccm":   np.array([0.5], dtype=np.float32),                 # 1
        "alpha_gamma": np.array([0.5], dtype=np.float32),                 # 1
        "alpha_sharp": np.array([0.5], dtype=np.float32),                 # 1
        "alpha_nr":    np.array([0.5], dtype=np.float32),                 # 1
        "alpha_fast":  np.array([0.5], dtype=np.float32),                 # 1

        "wb_step":    np.array([1.0], dtype=np.float32),                  # 1
        "sharp_step": np.array([1.0], dtype=np.float32),                  # 1

        "wb_min":    np.array([0.8, 0.8, 0.8], dtype=np.float32),         # 3 (if model wants 3)
        "wb_max":    np.array([1.2, 1.2, 1.2], dtype=np.float32),         # 3
        "gamma_min": np.array([0.8], dtype=np.float32),                   # 1 (scalar, not 256)
        "gamma_max": np.array([1.2], dtype=np.float32),                   # 1
        "ccm_min":   np.array([0.0], dtype=np.float32),                   # 1 (scalar, not 9)
        "ccm_max":   np.array([1.0], dtype=np.float32),                   # 1
        "nr_min":    np.array([0.0], dtype=np.float32),                   # 1
        "nr_max":    np.array([1.0], dtype=np.float32),                   # 1
    }

    # Use defaults if inputs is None
    provided = {} if inputs is None else dict(inputs)

    # Filter out unknown keys first (e.g., 'coeffs')
    dropped = [k for k in provided.keys() if k not in valid_names]
    for k in dropped:
        provided.pop(k, None)

    # Merge defaults + sanitized overrides
    feed = defaults.copy()
    feed.update(provided)

    # Optional: ensure we only pass valid names to ORT
    feed = {k: v for k, v in feed.items() if k in valid_names}

    # Logging
    logging.info(f"[Frame {frame_id}] RuleEngine merged inputs: {list(provided.keys())}")
    if dropped:
        logging.warning(f"[Frame {frame_id}] RuleEngine dropped unknown inputs: {dropped}")

    # Timing + inference
    start = time.time()
    outputs = session.run(None, feed)
    end = time.time()
    duration = (end - start) * 1000.0
    logging.info(f"[Frame {frame_id}] RuleEngine inference time: {duration:.3f} ms")

    return outputs, duration

# Example: initialize sessions for stride/crop and rule engine
ONNX_MODELS_DIR = os.path.join(os.path.dirname(__file__), "")

stride_crop_model = os.path.join(ONNX_MODELS_DIR, "isp_onnx_stride_crop.onnx")
rule_engine_model = os.path.join(ONNX_MODELS_DIR, "ruleengine_full.onnx")

stride_crop_session = create_onnx_session(stride_crop_model)
rule_engine_session = create_onnx_session(rule_engine_model)
# --- Trunk 3: Pipeline emulation loop ---

def emulate_pipeline(num_frames=5):
    """Run a simplified full pipeline emulation for a fixed number of frames."""
    for frame_id in range(1, num_frames + 1):
        # Log arrival
        arrival_ts = log_frame_arrival(frame_id)

        # Dummy input frame (simulate Bayer pattern)
        frame = np.random.randint(0, 255, (1080, 1920), dtype=np.uint8)

        # Stage 1: Stride/Crop via ONNX
        def stride_crop_func(f):
            # Convert to float32 and add batch + channel dimensions
            bayer_input = f.astype(np.float32)[None, None, :, :]  # shape (1,1,H,W)

            inputs = {
                "input.bayer": bayer_input,
                "crop_starts": np.array([0, 0], dtype=np.int64),
                "crop_ends": np.array([f.shape[0], f.shape[1]], dtype=np.int64),
                "crop_axes": np.array([2, 3], dtype=np.int64)  # crop along H and W axes
            }
            outputs = stride_crop_session.run(None, inputs)
            return outputs[0]

        cropped, dur_stride = timed_stage("StrideCrop", stride_crop_func, frame)

        # Stage 2: Rule Engine
        inputs_rule = {
            "coeffs": np.array([1.0], dtype=np.float32),
            "raw_wb": np.array([1.0, 1.0, 1.0], dtype=np.float32),
            "raw_ccm": np.ones(9, dtype=np.float32),
            "raw_gamma": np.linspace(0.0, 1.0, 256).astype(np.float32),
        }  # placeholder coeffs

        rule_outputs, dur_rule = run_rule_engine(rule_engine_session, inputs_rule, frame_id)

        # Stage 3: Tone Mapping (dummy function for illustration)
        def tone_map_func(f):
            return np.clip(f * 1.1, 0, 255).astype(np.uint8)

        toned, dur_tone = timed_stage("ToneMap", tone_map_func, cropped)

        # Stage 4: YUV Conversion (dummy function)
        def yuv_convert_func(f):
            return np.stack([f, f, f], axis=-1)  # fake YUV as 3-channel copy

        yuv_frame, dur_yuv = timed_stage("YUVConvert", yuv_convert_func, toned)

        # Log completion
        log_frame_completion(frame_id, arrival_ts)

        # Optionally collect stats
        logging.info(
            f"[Frame {frame_id}] Summary: StrideCrop={dur_stride:.3f} ms, "
            f"RuleEngine={dur_rule:.3f} ms, ToneMap={dur_tone:.3f} ms, "
            f"YUVConvert={dur_yuv:.3f} ms"
        )
# --- Trunk 4: Unit test harness + main entry point ---

class TestFullPipelineEmulation(unittest.TestCase):
    def test_pipeline_runs(self):
        """Basic test to ensure pipeline emulation runs without errors."""
        try:
            emulate_pipeline(num_frames=3)
        except Exception as e:
            self.fail(f"Pipeline emulation failed with exception: {e}")

if __name__ == "__main__":
    # Run as standalone script
    logging.info("Starting full pipeline emulation test...")
    emulate_pipeline(num_frames=5)
    logging.info("Pipeline emulation test completed.")
# --- Trunk 5: Aggregate statistics collector ---

class PipelineStats:
    def __init__(self):
        self.data = {
            "StrideCrop": [],
            "RuleEngine": [],
            "ToneMap": [],
            "YUVConvert": [],
            "TotalFrame": []
        }

    def record(self, stage, duration):
        if stage in self.data:
            self.data[stage].append(duration)

    def summarize(self):
        logging.info("=== Aggregate Performance Summary ===")
        for stage, durations in self.data.items():
            if durations:
                avg = np.mean(durations)
                maxd = np.max(durations)
                mind = np.min(durations)
                logging.info(
                    f"{stage}: avg={avg:.3f} ms, min={mind:.3f} ms, max={maxd:.3f} ms, count={len(durations)}"
                )

# Modify emulate_pipeline to use stats
def emulate_pipeline(num_frames=5):
    stats = PipelineStats()
    for frame_id in range(1, num_frames + 1):
        arrival_ts = log_frame_arrival(frame_id)
        frame = np.random.randint(0, 255, (1080, 1920), dtype=np.uint8)

        def stride_crop_func(f):
            inputs = {"input": f.astype(np.float32)}
            outputs = stride_crop_session.run(None, inputs)
            return outputs[0]

        cropped, dur_stride = timed_stage("StrideCrop", stride_crop_func, frame)
        stats.record("StrideCrop", dur_stride)

        inputs_rule = {"coeffs": np.array([1.0], dtype=np.float32)}
        rule_outputs, dur_rule = run_rule_engine(rule_engine_session, inputs_rule, frame_id)
        stats.record("RuleEngine", dur_rule)

        def tone_map_func(f):
            return np.clip(f * 1.1, 0, 255).astype(np.uint8)

        toned, dur_tone = timed_stage("ToneMap", tone_map_func, cropped)
        stats.record("ToneMap", dur_tone)

        def yuv_convert_func(f):
            return np.stack([f, f, f], axis=-1)

        yuv_frame, dur_yuv = timed_stage("YUVConvert", yuv_convert_func, toned)
        stats.record("YUVConvert", dur_yuv)

        log_frame_completion(frame_id, arrival_ts)
        stats.record("TotalFrame", (time.time() - arrival_ts) * 1000.0)

        logging.info(
            f"[Frame {frame_id}] Summary: StrideCrop={dur_stride:.3f} ms, "
            f"RuleEngine={dur_rule:.3f} ms, ToneMap={dur_tone:.3f} ms, "
            f"YUVConvert={dur_yuv:.3f} ms"
        )

    # Print aggregate stats at the end
    stats.summarize()
# --- Trunk 6: CSV export for performance stats ---

import csv

class PipelineStats:
    def __init__(self):
        self.data = {
            "StrideCrop": [],
            "RuleEngine": [],
            "ToneMap": [],
            "YUVConvert": [],
            "TotalFrame": []
        }
        self.per_frame = []  # store per-frame summaries

    def record(self, stage, duration):
        if stage in self.data:
            self.data[stage].append(duration)

    def record_frame_summary(self, frame_id, stride, rule, tone, yuv, total):
        self.per_frame.append({
            "FrameID": frame_id,
            "StrideCrop": stride,
            "RuleEngine": rule,
            "ToneMap": tone,
            "YUVConvert": yuv,
            "TotalFrame": total
        })

    def summarize(self):
        logging.info("=== Aggregate Performance Summary ===")
        for stage, durations in self.data.items():
            if durations:
                avg = np.mean(durations)
                maxd = np.max(durations)
                mind = np.min(durations)
                logging.info(
                    f"{stage}: avg={avg:.3f} ms, min={mind:.3f} ms, max={maxd:.3f} ms, count={len(durations)}"
                )

    def export_csv(self, filename="pipeline_stats.csv"):
        with open(filename, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["FrameID","StrideCrop","RuleEngine","ToneMap","YUVConvert","TotalFrame"])
            writer.writeheader()
            for row in self.per_frame:
                writer.writerow(row)
        logging.info(f"Per-frame stats exported to {filename}")
