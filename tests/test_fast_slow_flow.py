"""
Simple integration test exercising fast and slow submission flows.

Run with: pytest tests/test_fast_slow_flow.py
"""
import time
from isp_pipeline.types import Envelope
from isp_pipeline.resource_manager import ResourceManager
from isp_pipeline.platform_allocator import LinuxAllocatorMock
from isp_pipeline.telemetry_manager import TelemetryManager
from isp_pipeline.completion_dispatcher import CompletionDispatcher
from isp_pipeline.threading_manager import ThreadingManager
from isp_pipeline import fastalgo, fastisp, slowalgo, slowisp, rawalgo, rawisp

def test_fast_flow_accepts_and_generates_completion(tmp_path):
    rm = ResourceManager(global_quota_bytes=10*1024*1024)
    tm = TelemetryManager()
    cd = CompletionDispatcher(rm, tm)
    tmgr = ThreadingManager(cd, fast_capacity=2, bg_capacity=4)

    # Simulate allocate/pin preview buffer
    preview_hdl = rm.allocate(1024*16, "RGBA", None)
    rm.pin(preview_hdl)

    env = Envelope.create(
        selected_frames=[{"camera_id":"cam0","frame_id":"f0"}],
        read_handles=["raw-handle-1"],
        write_handles=[preview_hdl],
        payload={},
        meta={"origin":"test", "qos":"realtime"}
    )

    res = tmgr.submit_compute(env, lambda e: fastalgo.run(e), qos="realtime")
    assert res.accepted

    res2 = tmgr.submit_compute(env, lambda e: fastisp.run(e), qos="realtime")
    assert res2.accepted

    # allow workers to run
    time.sleep(0.5)
    records = tm.query_all()
    assert len(records) >= 2

    tmgr.shutdown()

def test_realtime_queue_rejects_when_full(tmp_path):
    rm = ResourceManager(global_quota_bytes=10*1024*1024)
    tm = TelemetryManager()
    cd = CompletionDispatcher(rm, tm)
    tmgr = ThreadingManager(cd, fast_capacity=1, bg_capacity=1)

    env = Envelope.create(
        selected_frames=[{"camera_id":"cam0","frame_id":"f1"}],
        read_handles=[],
        write_handles=["w1"],
        payload={},
        meta={"origin":"test", "qos":"realtime"}
    )

    # fill the fast queue
    res1 = tmgr.submit_compute(env, lambda e: (time.sleep(0.5) or fastalgo.run(e)), qos="realtime")
    assert res1.accepted
    res2 = tmgr.submit_compute(env, lambda e: fastalgo.run(e), qos="realtime")
    assert not res2.accepted
    tmgr.shutdown()
