# Trunk 1: Imports and helpers
import os
import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as nh
from onnx import TensorProto

def make_tensor_info(name, shape, t=TensorProto.FLOAT):
    return oh.make_tensor_value_info(name, t, shape)

def const_f32(name, val):
    return nh.from_array(np.array(val, dtype=np.float32), name)

def const_i64(name, val):
    return nh.from_array(np.array(val, dtype=np.int64), name)
# Trunk 2: Inputs and outputs
def create_inputs_outputs():
    inputs, outputs = [], []

    # Inputs
    bayer         = make_tensor_info('input.bayer', [1,1,'H','W_stride'], TensorProto.FLOAT)
    analog_gain   = make_tensor_info('analog_gain', [1], TensorProto.FLOAT)
    exposure_time = make_tensor_info('exposure_time', [1], TensorProto.FLOAT)
    sensor_temp   = make_tensor_info('sensor_temp', [1], TensorProto.FLOAT)
    scene_change  = make_tensor_info('scene_change', [1], TensorProto.FLOAT)

    # Geometry inputs (INT64)
    height_active = oh.make_tensor_value_info('height_active', TensorProto.INT64, [1])
    width_active  = oh.make_tensor_value_info('width_active',  TensorProto.INT64, [1])

    inputs.extend([bayer, analog_gain, exposure_time, sensor_temp,
                   scene_change, height_active, width_active])

    # Outputs
    wb_out    = make_tensor_info('wb',            [3], TensorProto.FLOAT)
    ccm_out   = make_tensor_info('ccm',           [9], TensorProto.FLOAT)
    gamma_out = make_tensor_info('gamma',         [1], TensorProto.FLOAT)
    nr_out    = make_tensor_info('nr_strength',   [1], TensorProto.FLOAT)
    sharp_out = make_tensor_info('sharp_strength',[1], TensorProto.FLOAT)
    outputs.extend([wb_out, ccm_out, gamma_out, nr_out, sharp_out])

    return inputs, outputs
# Trunk 3: Constants and shapes (INT64 defaults for geometry)
def add_constants_and_shapes(nodes, initializers):
    # Float constants
    eps_val   = const_f32('eps_val',   [1e-6])
    one_val   = const_f32('one_val',   [1.0])
    zero_val  = const_f32('zero_val',  [0.0])
    three_val = const_f32('three_val', [3.0])
    ln10_val  = const_f32('ln10_val',  [np.log(10.0)])
    initializers += [eps_val, one_val, zero_val, three_val, ln10_val]

    nodes += [
        oh.make_node('Constant', inputs=[], outputs=['eps_const'],   value=eps_val),
        oh.make_node('Constant', inputs=[], outputs=['one_const'],   value=one_val),
        oh.make_node('Constant', inputs=[], outputs=['zero_const'],  value=zero_val),
        oh.make_node('Constant', inputs=[], outputs=['three_const'], value=three_val),
        oh.make_node('Constant', inputs=[], outputs=['ln10_const'],  value=ln10_val),
    ]

    # INT64 constants
    one_i64_val = const_i64('one_i64_val', [1])
    initializers.append(one_i64_val)
    nodes.append(oh.make_node('Constant', inputs=[], outputs=['one_i64'], value=one_i64_val))

    # Slice params
    slice_starts = nh.from_array(np.array([0,0,0,0], dtype=np.int64), 'slice_starts')
    slice_axes   = nh.from_array(np.array([0,1,2,3], dtype=np.int64), 'slice_axes')
    initializers += [slice_starts, slice_axes]

    # Axes tensors
    axes_hw    = const_i64('axes_hw',    [2, 3])
    unsq_axis0 = const_i64('unsq_axis0', [0])
    initializers += [axes_hw, unsq_axis0]

    return nodes, initializers
# Trunk 4: Active crop from stride
def add_active_crop(nodes):
    nodes.append(oh.make_node('Concat',
                              inputs=['one_i64','one_i64','height_active','width_active'],
                              outputs=['slice_ends'], axis=0))
    nodes.append(oh.make_node('Slice',
                              inputs=['input.bayer','slice_starts','slice_ends','slice_axes'],
                              outputs=['bayer_active']))
    return nodes
# Trunk 5: RGGB masks and weighted sums
def add_masks_and_sums(nodes, initializers, H, W):
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    r_mask_arr = ((yy % 2 == 0) & (xx % 2 == 0)).astype(np.float32)[None,None,:,:]
    g_mask_arr = (((yy % 2 == 0) & (xx % 2 == 1)) | ((yy % 2 == 1) & (xx % 2 == 0))).astype(np.float32)[None,None,:,:]
    b_mask_arr = ((yy % 2 == 1) & (xx % 2 == 1)).astype(np.float32)[None,None,:,:]

    r_mask_init = nh.from_array(r_mask_arr, 'r_mask_init')
    g_mask_init = nh.from_array(g_mask_arr, 'g_mask_init')
    b_mask_init = nh.from_array(b_mask_arr, 'b_mask_init')
    initializers += [r_mask_init, g_mask_init, b_mask_init]

    nodes += [
        oh.make_node('Constant', inputs=[], outputs=['r_mask'], value=r_mask_init),
        oh.make_node('Constant', inputs=[], outputs=['g_mask'], value=g_mask_init),
        oh.make_node('Constant', inputs=[], outputs=['b_mask'], value=b_mask_init),
    ]

    nodes += [
        oh.make_node('Mul', inputs=['bayer_active','r_mask'], outputs=['r_weighted']),
        oh.make_node('Mul', inputs=['bayer_active','g_mask'], outputs=['g_weighted']),
        oh.make_node('Mul', inputs=['bayer_active','b_mask'], outputs=['b_weighted']),
    ]
    nodes += [
        oh.make_node('ReduceSum', inputs=['r_weighted','axes_hw'], outputs=['r_sum']),
        oh.make_node('ReduceSum', inputs=['g_weighted','axes_hw'], outputs=['g_sum']),
        oh.make_node('ReduceSum', inputs=['b_weighted','axes_hw'], outputs=['b_sum']),
    ]
    nodes += [
        oh.make_node('ReduceSum', inputs=['r_mask','axes_hw'], outputs=['r_cnt']),
        oh.make_node('ReduceSum', inputs=['g_mask','axes_hw'], outputs=['g_cnt']),
        oh.make_node('ReduceSum', inputs=['b_mask','axes_hw'], outputs=['b_cnt']),
    ]
    return nodes, initializers
# Trunk 6: WB computation
def add_wb_computation(nodes):
    nodes += [
        oh.make_node('Add', inputs=['r_cnt','eps_const'], outputs=['r_cnt_eps']),
        oh.make_node('Add', inputs=['g_cnt','eps_const'], outputs=['g_cnt_eps']),
        oh.make_node('Add', inputs=['b_cnt','eps_const'], outputs=['b_cnt_eps']),
        oh.make_node('Div', inputs=['r_sum','r_cnt_eps'], outputs=['r_mean']),
        oh.make_node('Div', inputs=['g_sum','g_cnt_eps'], outputs=['g_mean']),
        oh.make_node('Div', inputs=['b_sum','b_cnt_eps'], outputs=['b_mean']),
        oh.make_node('Add', inputs=['r_mean','g_mean'], outputs=['rg_sum']),
        oh.make_node('Add', inputs=['rg_sum','b_mean'], outputs=['rgb_sum']),
        oh.make_node('Div', inputs=['rgb_sum','three_const'], outputs=['gray']),
        oh.make_node('Div', inputs=['gray','r_mean'], outputs=['wb_r']),
        oh.make_node('Div', inputs=['gray','g_mean'], outputs=['wb_g']),
        oh.make_node('Div', inputs=['gray','b_mean'], outputs=['wb_b']),
        oh.make_node('Unsqueeze', inputs=['wb_r','unsq_axis0'], outputs=['wb_r_u']),
        oh.make_node('Unsqueeze', inputs=['wb_g','unsq_axis0'], outputs=['wb_g_u']),
        oh.make_node('Unsqueeze', inputs=['wb_b','unsq_axis0'], outputs=['wb_b_u']),
        oh.make_node('Concat', inputs=['wb_r_u','wb_g_u','wb_b_u'], outputs=['wb'], axis=0),
    ]
    return nodes

# Trunk 7: Identity CCM and gamma/nr/sharpen
def add_identity_ccm(nodes, initializers):
    # Identity CCM (3x3 identity matrix flattened to 9)
    ccm_base = nh.from_array(np.eye(3, dtype=np.float32).reshape(-1), 'ccm_base')
    initializers.append(ccm_base)
    nodes.append(oh.make_node('Identity', inputs=['ccm_base'], outputs=['ccm']))
    return nodes, initializers

def add_gamma_nr_sharpen(nodes, initializers):
    # Gamma parameters
    base_gamma_val = const_f32('base_gamma_val', [2.2])
    k1_val = const_f32('k1_val', [0.2])
    k2_val = const_f32('k2_val', [0.3])
    gmin_val = const_f32('gmin_val', [1.8])
    gmax_val = const_f32('gmax_val', [2.6])
    initializers += [base_gamma_val, k1_val, k2_val, gmin_val, gmax_val]

    nodes += [
        oh.make_node('Constant', inputs=[], outputs=['base_gamma_const'], value=base_gamma_val),
        oh.make_node('Constant', inputs=[], outputs=['k1_const'], value=k1_val),
        oh.make_node('Constant', inputs=[], outputs=['k2_const'], value=k2_val),
        oh.make_node('Constant', inputs=[], outputs=['gmin_const'], value=gmin_val),
        oh.make_node('Constant', inputs=[], outputs=['gmax_const'], value=gmax_val),
        oh.make_node('Add', inputs=['analog_gain','eps_const'], outputs=['ag_eps']),
        oh.make_node('Log', inputs=['ag_eps'], outputs=['ln_ag']),
        oh.make_node('Div', inputs=['ln_ag','ln10_const'], outputs=['log10_ag']),
        oh.make_node('Mul', inputs=['k1_const','log10_ag'], outputs=['k1_log']),
        oh.make_node('Mul', inputs=['k2_const','scene_change'], outputs=['k2_scene']),
        oh.make_node('Add', inputs=['base_gamma_const','k1_log'], outputs=['base_plus']),
        oh.make_node('Sub', inputs=['base_plus','k2_scene'], outputs=['gamma_raw']),
        oh.make_node('Clip', inputs=['gamma_raw','gmin_const','gmax_const'], outputs=['gamma']),
    ]

    # NR parameters
    nr_k1_val = const_f32('nr_k1_val', [0.15])
    nr_k2_val = const_f32('nr_k2_val', [0.002])
    initializers += [nr_k1_val, nr_k2_val]

    nodes += [
        oh.make_node('Constant', inputs=[], outputs=['nr_k1'], value=nr_k1_val),
        oh.make_node('Constant', inputs=[], outputs=['nr_k2'], value=nr_k2_val),
        oh.make_node('Mul', inputs=['nr_k1','analog_gain'], outputs=['nr_ag']),
        oh.make_node('Mul', inputs=['nr_k2','sensor_temp'], outputs=['nr_tp']),
        oh.make_node('Add', inputs=['nr_ag','nr_tp'], outputs=['nr_sum']),
        oh.make_node('Clip', inputs=['nr_sum','zero_const','one_const'], outputs=['nr_strength']),
    ]

    # Sharpen parameters
    sh_k1_val = const_f32('sh_k1_val', [0.6])
    half_val  = const_f32('half_val',  [0.5])
    initializers += [sh_k1_val, half_val]

    nodes += [
        oh.make_node('Constant', inputs=[], outputs=['sh_k1'], value=sh_k1_val),
        oh.make_node('Constant', inputs=[], outputs=['half_const'], value=half_val),
        oh.make_node('Sub', inputs=['one_const','nr_strength'], outputs=['one_minus_nr']),
        oh.make_node('Mul', inputs=['half_const','scene_change'], outputs=['half_scene']),
        oh.make_node('Sub', inputs=['one_const','half_scene'], outputs=['one_minus_half_scene']),
        oh.make_node('Mul', inputs=['one_minus_nr','one_minus_half_scene'], outputs=['sharpen_prod']),
        oh.make_node('Mul', inputs=['sh_k1','sharpen_prod'], outputs=['sharp_raw']),
        oh.make_node('Clip', inputs=['sharp_raw','zero_const','one_const'], outputs=['sharp_strength']),
    ]
    return nodes, initializers
# Trunk 8: Model assembly
def assemble_model(nodes, inputs, outputs, initializers):
    graph = oh.make_graph(
        nodes=nodes,
        name='ISP_ALGOS_COEFFS_FULL',
        inputs=inputs,
        outputs=outputs,
        initializer=initializers
    )
    model = oh.make_model(
        graph,
        producer_name='softisp_algos_builder',
        ir_version=11,
        opset_imports=[onnx.helper.make_opsetid("", 13)]
    )
    onnx.checker.check_model(model)
    return model
# Trunk 9: Orchestration
def build_algos_model(H=1080, W=1920):
    nodes, initializers = [], []
    inputs, outputs = create_inputs_outputs()

    # Constants and shapes
    nodes, initializers = add_constants_and_shapes(nodes, initializers)

    # Crop active area
    nodes = add_active_crop(nodes)

    # Masks and weighted sums
    nodes, initializers = add_masks_and_sums(nodes, initializers, H, W)

    # WB computation
    nodes = add_wb_computation(nodes)

    # CCM identity and gamma/nr/sharpen
    nodes, initializers = add_identity_ccm(nodes, initializers)
    nodes, initializers = add_gamma_nr_sharpen(nodes, initializers)

    return assemble_model(nodes, inputs, outputs, initializers)
# Trunk 10: Save model
if __name__ == '__main__':
    H = int(os.environ.get('HEIGHT', '1080'))
    W = int(os.environ.get('WIDTH', '1920'))
    out_path = os.environ.get('OUT_ONNX', 'isp_algo_coeffs_full.onnx')
    model = build_algos_model(H=H, W=W)
    onnx.save(model, out_path)
    print(f"Saved {out_path} (H={H}, W={W})")
