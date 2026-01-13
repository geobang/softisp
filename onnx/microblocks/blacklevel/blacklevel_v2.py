# onnx/microblocks/blacklevel/blacklevel_v2.py

from .blacklevel_base import BlackLevelBase

class BlackLevelV2(BlackLevelBase):
    name = "blacklevel"
    version = "v2"

    # Inherits all build_* methods from BlackLevelBase.
    # If you want to override behavior, you can redefine _common_outputs.
    # For now, it reuses the same contract and node structure.
