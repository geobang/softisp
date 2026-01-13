from microblocks.blacklevel.blacklevel_base import BlackLevelBase

class BlackLevelV2(BlackLevelBase):
    """
    Black level correction microblock (v2).
    Inherits from BlackLevelBase and overrides version/logic as needed.
    """

    name = "blacklevel"
    version = "v2"

    def build_applier(self, stage: str, prev_stages=None):
        # Call the base implementation first
        outputs, nodes, inits, vis = super().build_applier(stage, prev_stages)

        # --- v2 adjustments go here ---
        # For now, this is just a stub. You can modify nodes or add new ones.
        # Example: replace Sub with Add, or insert a Clip node, etc.

        return outputs, nodes, inits, vis
