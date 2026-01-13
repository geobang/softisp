from microblocks.base import MicroblockBase
from onnx import helper









@register_block
class CCMBlockV1(MicroBlock):
    def input_names(self):
        return ["input"]

    def output_names(self):
        return ["output"]

    def build_applier_node(self, prev_out=None):
        from onnx import helper
        node = helper.make_node(
            "Identity",
            inputs=[prev_out or "input"],
            outputs=["output"],
            name="ApplierStub"
        )
        return node

    def build_coordinator_node(self, prev_out=None):
        from onnx import helper
        node = helper.make_node(
            "Identity",
            inputs=[prev_out or "input"],
            outputs=["output"],
            name="CoordinatorStub"
        )
        return node

    name = "ccm"
    version = "v1"
    coeff_names = ['matrix']

    # stub
    def build_algo_node(self, prev_out="awb_out"):
        """
        Build the Algo node for CCM.
        Produces both the corrected image and the CCM matrix coefficients.
        """

        # Step 1: Generate CCM coefficients (example: daylight CCM)
        # In a real implementation, these would come from sensor metadata or AWB statistics.
        daylight_ccm = np.array([
            [1.2, -0.1, -0.1],
            [-0.05, 1.1, -0.05],
            [-0.05, -0.1, 1.15]
        ], dtype=np.float32).reshape(-1)

        # Step 2: Create initializer for default CCM
        ccm_init = helper.make_tensor(
            name="ccm_matrix",
            data_type=TensorProto.FLOAT,
            dims=[3,3],
            vals=daylight_ccm
        )

        # Step 3: Algo node applies CCM to image
        node = helper.make_node(
            "MatMul",
            inputs=[prev_out, "ccm_matrix"],
            outputs=["ccm_out", "ccm_matrix"],  # corrected image + coeffs
            name="CcmV1Algo"
        )

        return node, [ccm_init]

    # stub
    def build_algo_initializer(self):
        """
        Provide a default CCM initializer (identity matrix).
        This ensures the graph runs even if no dynamic coeffs are fed.
        """
        identity_ccm = np.eye(3, dtype=np.float32).reshape(-1)
        ccm_init = helper.make_tensor(
            name="ccm_matrix",
            data_type=TensorProto.FLOAT,
            dims=[3,3],
            vals=identity_ccm
        )
        return [ccm_init]
