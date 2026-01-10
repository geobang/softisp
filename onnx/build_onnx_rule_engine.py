# build_onnx_rule_engine.py
import os
import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as nh
from onnx import TensorProto

def const_tensor(name, arr, dtype=np.float32):
    return nh.from_array(np.array(arr, dtype=dtype), name)

def make_tensor_info(name, shape, t=TensorProto.FLOAT):
    return oh.make_tensor_value_info(name, t, shape)
def build_rule_engine_model():
    nodes, initializers, inputs, outputs = [], [], [], []

    # Inputs: previous + new coefficients
    wb_prev    = make_tensor_info('wb_prev', [3], TensorProto.FLOAT)
    wb_next    = make_tensor_info('wb_next', [3], TensorProto.FLOAT)
    ccm_prev   = make_tensor_info('ccm_prev', [9], TensorProto.FLOAT)
    ccm_next   = make_tensor_info('ccm_next', [9], TensorProto.FLOAT)
    gamma_prev = make_tensor_info('gamma_prev', [1], TensorProto.FLOAT)
    gamma_next = make_tensor_info('gamma_next', [1], TensorProto.FLOAT)
    nr_prev    = make_tensor_info('nr_prev', [1], TensorProto.FLOAT)
    nr_next    = make_tensor_info('nr_next', [1], TensorProto.FLOAT)
    sharp_prev = make_tensor_info('sharp_prev', [1], TensorProto.FLOAT)
    sharp_next = make_tensor_info('sharp_next', [1], TensorProto.FLOAT)
    inputs.extend([wb_prev, wb_next, ccm_prev, ccm_next,
                   gamma_prev, gamma_next, nr_prev, nr_next,
                   sharp_prev, sharp_next])

    # Outputs: stabilized coefficients
    wb_out    = make_tensor_info('wb', [3], TensorProto.FLOAT)
    ccm_out   = make_tensor_info('ccm', [9], TensorProto.FLOAT)
    gamma_out = make_tensor_info('gamma', [1], TensorProto.FLOAT)
    nr_out    = make_tensor_info('nr_strength', [1], TensorProto.FLOAT)
    sharp_out = make_tensor_info('sharp_strength', [1], TensorProto.FLOAT)
    outputs.extend([wb_out, ccm_out, gamma_out, nr_out, sharp_out])

    # Constants for blending and bounds
    alpha_const = oh.make_node('Constant', inputs=[], outputs=['alpha_const'],
                               value=const_tensor('alpha_val', [0.7], dtype=np.float32))  # smoothing factor
    zero_const  = oh.make_node('Constant', inputs=[], outputs=['zero_const'],
                               value=const_tensor('zero_val', [0.0], dtype=np.float32))
    one_const   = oh.make_node('Constant', inputs=[], outputs=['one_const'],
                               value=const_tensor('one_val', [1.0], dtype=np.float32))
    gmin_const  = oh.make_node('Constant', inputs=[], outputs=['gmin_const'],
                               value=const_tensor('gmin_val', [1.8], dtype=np.float32))
    gmax_const  = oh.make_node('Constant', inputs=[], outputs=['gmax_const'],
                               value=const_tensor('gmax_val', [2.6], dtype=np.float32))
    nodes += [alpha_const, zero_const, one_const, gmin_const, gmax_const]
    # --- WB stabilization: wb = alpha*wb_prev + (1-alpha)*wb_next ---
    one_minus_alpha = oh.make_node('Sub', inputs=['one_const','alpha_const'], outputs=['one_minus_alpha'])
    wb_prev_scaled  = oh.make_node('Mul', inputs=['alpha_const','wb_prev'], outputs=['wb_prev_scaled'])
    wb_next_scaled  = oh.make_node('Mul', inputs=['one_minus_alpha','wb_next'], outputs=['wb_next_scaled'])
    wb_blend        = oh.make_node('Add', inputs=['wb_prev_scaled','wb_next_scaled'], outputs=['wb'])
    nodes += [one_minus_alpha, wb_prev_scaled, wb_next_scaled, wb_blend]

    # --- CCM stabilization: same blend ---
    ccm_prev_scaled = oh.make_node('Mul', inputs=['alpha_const','ccm_prev'], outputs=['ccm_prev_scaled'])
    ccm_next_scaled = oh.make_node('Mul', inputs=['one_minus_alpha','ccm_next'], outputs=['ccm_next_scaled'])
    ccm_blend       = oh.make_node('Add', inputs=['ccm_prev_scaled','ccm_next_scaled'], outputs=['ccm'])
    nodes += [ccm_prev_scaled, ccm_next_scaled, ccm_blend]

    # --- Gamma stabilization + bounds ---
    gamma_prev_scaled = oh.make_node('Mul', inputs=['alpha_const','gamma_prev'], outputs=['gamma_prev_scaled'])
    gamma_next_scaled = oh.make_node('Mul', inputs=['one_minus_alpha','gamma_next'], outputs=['gamma_next_scaled'])
    gamma_blend       = oh.make_node('Add', inputs=['gamma_prev_scaled','gamma_next_scaled'], outputs=['gamma_blend'])
    gamma_clip        = oh.make_node('Clip', inputs=['gamma_blend','gmin_const','gmax_const'], outputs=['gamma'])
    nodes += [gamma_prev_scaled, gamma_next_scaled, gamma_blend, gamma_clip]

    # --- NR stabilization + bounds [0,1] ---
    nr_prev_scaled = oh.make_node('Mul', inputs=['alpha_const','nr_prev'], outputs=['nr_prev_scaled'])
    nr_next_scaled = oh.make_node('Mul', inputs=['one_minus_alpha','nr_next'], outputs=['nr_next_scaled'])
    nr_blend       = oh.make_node('Add', inputs=['nr_prev_scaled','nr_next_scaled'], outputs=['nr_blend'])
    nr_clip        = oh.make_node('Clip', inputs=['nr_blend','zero_const','one_const'], outputs=['nr_strength'])
    nodes += [nr_prev_scaled, nr_next_scaled, nr_blend, nr_clip]

    # --- Sharp stabilization + bounds [0,1] ---
    sharp_prev_scaled = oh.make_node('Mul', inputs=['alpha_const','sharp_prev'], outputs=['sharp_prev_scaled'])
    sharp_next_scaled = oh.make_node('Mul', inputs=['one_minus_alpha','sharp_next'], outputs=['sharp_next_scaled'])
    sharp_blend       = oh.make_node('Add', inputs=['sharp_prev_scaled','sharp_next_scaled'], outputs=['sharp_blend'])
    sharp_clip        = oh.make_node('Clip', inputs=['sharp_blend','zero_const','one_const'], outputs=['sharp_strength'])
    nodes += [sharp_prev_scaled, sharp_next_scaled, sharp_blend, sharp_clip]
    graph = oh.make_graph(
        nodes=nodes,
        name='ISP_RULE_ENGINE',
        inputs=inputs,
        outputs=outputs,
        initializer=initializers
    )
    model = oh.make_model(
        graph,
        producer_name='softisp_rule_engine_builder',
        ir_version=11,
        opset_imports=[onnx.helper.make_opsetid("", 13)]
    )
    onnx.checker.check_model(model)
    return model

if __name__ == '__main__':
    out_path = os.environ.get('OUT_ONNX','isp_rule_engine.onnx')
    model = build_rule_engine_model()
    onnx.save(model,out_path)
    print(f"Saved {out_path}")
