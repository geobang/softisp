import time
import onnxruntime as ort
import numpy as np
import logging

class ProcessItem:
    """
    Unified work unit for passing through pipeline queues.
    Mirrors the C++ struct design in src/.
    """
    def __init__(self, frame_id, bayer, meta, coeffs=None):
        self.frame_id = frame_id
        self.bayer = bayer
        self.meta = meta
        self.coeffs = coeffs or {}
        self.ts = time.time()

    def __repr__(self):
        return f"<ProcessItem frame={self.frame_id} coeffs={list(self.coeffs.keys())}>"

class SessionManager:
    """
    Wraps ONNX Runtime session with provider setup, input validation, and timing.
    """
    def __init__(self, model_path, providers=None):
        self.sess = ort.InferenceSession(model_path, providers=providers or ["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}
        logging.info(f"SessionManager initialized for {model_path} with inputs {self.inputs}")

    def run(self, feed):
        missing = self.inputs - set(feed.keys())
        if missing:
            raise ValueError(f"Missing inputs: {missing}")
        start = time.time()
        outs = self.sess.run(None, feed)
        dur = (time.time() - start) * 1000.0
        logging.info(f"ONNX run completed in {dur:.3f} ms")
        return outs

class Monitor:
    """
    Collects per-stage timings and exports summaries.
    """
    def __init__(self):
        self.data = {}

    def record(self, stage, duration):
        self.data.setdefault(stage, []).append(duration)

    def summarize(self):
        logging.info("=== Performance Summary ===")
        for stage, durations in self.data.items():
            if durations:
                avg = float(np.mean(durations))
                mx = float(np.max(durations))
                mn = float(np.min(durations))
                logging.info(f"{stage}: avg={avg:.3f} ms, min={mn:.3f} ms, max={mx:.3f} ms, count={len(durations)}")
