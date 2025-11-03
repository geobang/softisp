"""
PlatformAllocator (Phase A linux mock) — provides safe file-backed native allocations
using numpy.memmap and tempfile. Keeps interface narrow.
"""
import tempfile
import os
import numpy as np
from typing import Tuple, Dict, Any


class NativeHandle(Dict):
    """
    Dict-like container with:
      - path: filesystem path backing the buffer
      - size: bytes
      - memmap: numpy.memmap object
    """


class PlatformAllocator:
    def __init__(self, base_dir: str = None):
        self.base_dir = base_dir

    def allocatenative(self, size: int, format: str = None, hints: Dict = None) -> NativeHandle:
        tmp = tempfile.NamedTemporaryFile(delete=False, dir=self.base_dir)
        path = tmp.name
        tmp.close()
        # create file of requested size
        with open(path, "wb") as f:
            f.truncate(size)
        memmap = np.memmap(path, dtype=np.uint8, mode="r+")
        return {"path": path, "size": size, "memmap": memmap}

    def exportdmabuf(self, nativehandle: NativeHandle) -> int:
        # Phase A mock: use os.stat inode as an int token (non-privileged)
        try:
            st = os.stat(nativehandle["path"])
            return st.st_ino & 0xFFFF
        except Exception:
            return -1

    def syncfordevice(self, nativehandle: NativeHandle) -> int:
        return -1

    def syncforcpu(self, nativehandle: NativeHandle) -> int:
        return -1

    def freenative(self, nativehandle: NativeHandle):
        try:
            memmap = nativehandle.get("memmap")
            if memmap is not None:
                memmap._mmap.close()
        except Exception:
            pass
        try:
            os.remove(nativehandle["path"])
        except Exception:
            pass
"