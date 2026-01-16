from microblocks.base import BuildResult
from microblocks.blacklevel.blacklevel_base import BlackLevelBase

class BlackLevelV2(BlackLevelBase):
    """
    Black level correction microblock (v2).
    Inherits from BlackLevelBase and overrides version/logic as needed.
    """
    name = 'blacklevel'
    version = 'v2'

    def build_applier(self, stage: str, prev_stages=None):
        return super().build_applier(stage, prev_stages)

    def build_algo(self, stage: str, prev_stages=None):
        return super().build_applier(stage, prev_stages)
