"""
PlatformAllocator mock (Linux placeholder).

This file contains a mock stub for allocatenative/exportdmabuf. Replace with a real
memfd/dmabuf implementation in production.
"""
from typing import Any, Dict

class PlatformAllocator:
    def allocatenative(self, size: int, format: str, hints: Dict = None) -> Any:
        """
        Allocate a native handle. In a real linux_allocator this would create a memfd
        and possibly call ioctl to export dmabuf with the right format.
        Return a platform-native opaque object.
        """
        raise NotImplementedError

    def exportdmabuf(self, nativehandle: Any) -> int:
        """
        Export a dmabuf FD for zero-copy import into other runtimes.
        """
        raise NotImplementedError

    def syncfordevice(self, nativehandle: Any) -> int:
        raise NotImplementedError

    def syncforcpu(self, nativehandle: Any) -> int:
        raise NotImplementedError

    def freenative(self, nativehandle: Any) -> None:
        raise NotImplementedError


class LinuxAllocatorMock(PlatformAllocator):
    def allocatenative(self, size: int, format: str, hints: Dict = None) -> Dict:
        # Return a dict as a fake native handle
        return {"fake_memfd": True, "size": size, "format": format, "hints": hints}

    def exportdmabuf(self, nativehandle: Dict) -> int:
        # Return a fake fd integer
        return 42

    def syncfordevice(self, nativehandle: Dict) -> int:
        return -1

    def syncforcpu(self, nativehandle: Dict) -> int:
        return -1

    def freenative(self, nativehandle: Dict) -> None:
        return
