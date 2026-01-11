from pipeline.pipeline import Pipeline
from pipeline.process_item import ProcessItem
import queue

def dispatcher(manifest, q, frame_buf):
    pipeline = Pipeline(manifest)
    # Example coeffs set
    pipeline.set_coeffs("wbblock", {"gain_r": 1.1, "gain_g": 1.0, "gain_b": 0.9})
    coeffs_bulk = pipeline.get_all_coeffs()
    item = ProcessItem(frame_id=42, image=frame_buf, coeffs_bulk=coeffs_bulk)
    q.put(item)

def isp_thread(q, isp_session):
    item = q.get()
    isp_session.run([], item.coeffs_bulk)
