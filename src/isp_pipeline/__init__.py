"""
softisp (isp_pipeline) package entrypoint.
Expose main modules for convenience.
"""
from .types import Envelope, CompletionEvent, SelectedFrame
from .resource_manager import ResourceManager
from .platform_allocator import PlatformAllocator, LinuxAllocatorMock
from .threading_manager import ThreadingManager, SubmitResult
from .completion_dispatcher import CompletionDispatcher
from .telemetry_manager import TelemetryManager
from .housekeeper import Housekeeper
# expose worker modules
from . import fastalgo, fastisp, slowalgo, slowisp, rawalgo, rawisp
from .model_manager import ModelManager
