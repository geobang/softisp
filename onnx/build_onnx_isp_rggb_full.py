# Trunk 1: Imports and helpers
import os
import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as nh
from onnx import TensorProto

def make_tensor_info(name, shape, t=TensorProto.FLOAT):
    return oh.make_tensor_value_info(name, t, shape)

def add_init(initializers, name, array):
    initializers.append(nh.from_array(np.array(array), name))
    return name  # return the name for convenience
# Trunk 2: Inputs and outputs
def create_inputs_outputs():
    inputs, outputs = [], []

    # Stride-aware inputs
    bayer          = make_tensor_info('input.bayer', [1,1,'H','W_stride'], TensorProto.FLOAT)
    blc_offset     = make_tensor_info('blc_offset', [1], TensorProto.FLOAT)
    bit_depth      = make_tensor_info('bit_depth', [1], TensorProto.FLOAT)
    lsc_gain_map   = make_tensor_info('lsc_gain_map', [1,1,'H','W_stride'], TensorProto.FLOAT)
    wb             = make_tensor_info('wb', [3], TensorProto.FLOAT)   # (3,)
    ccm            = make_tensor_info('ccm', [9], TensorProto.FLOAT)  # row-major 3x3
    gamma          = make_tensor_info('gamma', [1], TensorProto.FLOAT)
    nr_strength    = make_tensor_info('nr_strength', [1], TensorProto.FLOAT)
    sharp_strength = make_tensor_info('sharp_strength', [1], TensorProto.FLOAT)

    # Geometry coeffs (INT64 for Slice/Concat)
    height_active  = oh.make_tensor_value_info('height_active', TensorProto.INT64, [1])
    width_active   = oh.make_tensor_value_info('width_active',  TensorProto.INT64, [1])

    inputs.extend([
        bayer, blc_offset, bit_depth, lsc_gain_map,
        wb, ccm, gamma, nr_strength, sharp_strength,
        height_active, width_active
    ])

    # Outputs: Y, U, V planes from active area
    Y = make_tensor_info('Y', [1,1,'H','W_active'], TensorProto.FLOAT)
    U = make_tensor_info('U', [1,1,'H','W_active'], TensorProto.FLOAT)
    V = make_tensor_info('V', [1,1,'H','W_active'], TensorProto.FLOAT)
    outputs.extend([Y, U, V])

    return inputs, outputs

# Trunk 3: Constants and shapes (INT64 by default for geometry)
def add_constants_and_shapes(nodes, initializers):
    # Geometry constants (INT64)
    one_i64   = nh.from_array(np.array([1], dtype=np.int64), 'one_i64')
    zero_i64  = nh.from_array(np.array([0], dtype=np.int64), 'zero_i64')
    two_i64   = nh.from_array(np.array([2], dtype=np.int64), 'two_i64')
    initializers += [one_i64, zero_i64, two_i64]

    # Slice params (INT64)
    slice_starts = nh.from_array(np.array([0,0,0,0], dtype=np.int64), 'slice_starts')
    slice_axes   = nh.from_array(np.array([0,1,2,3], dtype=np.int64), 'slice_axes')
    initializers += [slice_starts, slice_axes]

    # Reshape shapes (INT64)
    shape_wb4d      = nh.from_array(np.array([1,3,1,1], dtype=np.int64), 'shape_wb4d')
    shape_scalar4d  = nh.from_array(np.array([1,1,1,1], dtype=np.int64), 'shape_scalar4d')
    shape_ccm_conv  = nh.from_array(np.array([3,3,1,1], dtype=np.int64), 'shape_ccm_conv')
    shape_yuv_conv  = nh.from_array(np.array([3,3,1,1], dtype=np.int64), 'shape_yuv_conv')
    initializers += [shape_wb4d, shape_scalar4d, shape_ccm_conv, shape_yuv_conv]

    # Split sizes (INT64)
    split_sizes = nh.from_array(np.array([1,1,1], dtype=np.int64), 'split_sizes')
    initializers.append(split_sizes)

    # Arithmetic constants (FLOAT32) — only for math ops
    one_f32   = nh.from_array(np.array([1.0], dtype=np.float32), 'one_f32')
    zero_f32  = nh.from_array(np.array([0.0], dtype=np.float32), 'zero_f32')
    two_f32   = nh.from_array(np.array([2.0], dtype=np.float32), 'two_f32')
    initializers += [one_f32, zero_f32, two_f32]

    # Average kernel for interpolation
    avg3x3 = np.ones((1,1,3,3), dtype=np.float32)/9.0
    avg3x3_w = nh.from_array(avg3x3, 'avg3x3_w')
    initializers.append(avg3x3_w)

    return nodes, initializers

# Trunk 4: BLC + normalization + LSC
def add_blc_norm_lsc(nodes):
    # Subtract black level
    nodes.append(oh.make_node('Sub', inputs=['input.bayer','blc_offset'], outputs=['raw_minus_blc']))

    # Normalize to [0,1]: raw_minus_blc / ((2^bit_depth)-1)
    nodes.append(oh.make_node('Pow', inputs=['two_f32','bit_depth'], outputs=['fullscale_plus']))
    nodes.append(oh.make_node('Sub', inputs=['fullscale_plus','one_f32'], outputs=['fullscale']))
    nodes.append(oh.make_node('Div', inputs=['raw_minus_blc','fullscale'], outputs=['raw_norm_preclip']))
    nodes.append(oh.make_node('Clip', inputs=['raw_norm_preclip','zero_f32','one_f32'], outputs=['raw_norm']))

    # Lens shading correction on stride buffer
    nodes.append(oh.make_node('Mul', inputs=['raw_norm','lsc_gain_map'], outputs=['raw_lsc']))
    return nodes
# Trunk 5: Slice to active area
def add_active_slice(nodes):
    # ends = [1,1,height_active,width_active] (INT64) via Concat of initializers and inputs
    nodes.append(oh.make_node('Concat',
                              inputs=['one_i64','one_i64','height_active','width_active'],
                              outputs=['slice_ends'], axis=0))
    # Slice raw_lsc to active area
    nodes.append(oh.make_node('Slice',
                              inputs=['raw_lsc','slice_starts','slice_ends','slice_axes'],
                              outputs=['raw_lsc_active']))
    return nodes
# Trunk 6: RGGB masks and known samples
def add_masks_and_known(nodes, initializers, HMAX=1080, WSMAX=2048):
    # Precompute stride-wide boolean masks, then Slice to active area
    yy, xx = np.meshgrid(np.arange(HMAX), np.arange(WSMAX), indexing='ij')
    r_mask_full = ((yy%2==0)&(xx%2==0))[None,None,:,:].astype(np.bool_)
    g_mask_full = (((yy%2==0)&(xx%2==1))|((yy%2==1)&(xx%2==0)))[None,None,:,:].astype(np.bool_)
    b_mask_full = ((yy%2==1)&(xx%2==1))[None,None,:,:].astype(np.bool_)

    initializers.append(nh.from_array(r_mask_full, 'r_mask_full'))
    initializers.append(nh.from_array(g_mask_full, 'g_mask_full'))
    initializers.append(nh.from_array(b_mask_full, 'b_mask_full'))

    # Slice masks to active area
    nodes.append(oh.make_node('Slice', inputs=['r_mask_full','slice_starts','slice_ends','slice_axes'], outputs=['r_mask']))
    nodes.append(oh.make_node('Slice', inputs=['g_mask_full','slice_starts','slice_ends','slice_axes'], outputs=['g_mask']))
    nodes.append(oh.make_node('Slice', inputs=['b_mask_full','slice_starts','slice_ends','slice_axes'], outputs=['b_mask']))

    # Cast masks to float for arithmetic (known samples)
    nodes.append(oh.make_node('Cast', inputs=['r_mask'], outputs=['r_mask_f32'], to=TensorProto.FLOAT))
    nodes.append(oh.make_node('Cast', inputs=['g_mask'], outputs=['g_mask_f32'], to=TensorProto.FLOAT))
    nodes.append(oh.make_node('Cast', inputs=['b_mask'], outputs=['b_mask_f32'], to=TensorProto.FLOAT))

    # Known samples from the mosaic
    nodes.append(oh.make_node('Mul', inputs=['raw_lsc_active','r_mask_f32'], outputs=['r_known']))
    nodes.append(oh.make_node('Mul', inputs=['raw_lsc_active','g_mask_f32'], outputs=['g_known']))
    nodes.append(oh.make_node('Mul', inputs=['raw_lsc_active','b_mask_f32'], outputs=['b_known']))

    return nodes, initializers
# Trunk 7: Demosaic interpolation and channel selection
def add_demosaic(nodes):
    # Interpolate each channel with 3x3 average (simple demo)
    nodes.append(oh.make_node('Conv', inputs=['raw_lsc_active','avg3x3_w'], outputs=['r_interp'],
                              kernel_shape=[3,3], pads=[1,1,1,1], strides=[1,1]))
    nodes.append(oh.make_node('Conv', inputs=['raw_lsc_active','avg3x3_w'], outputs=['g_interp'],
                              kernel_shape=[3,3], pads=[1,1,1,1], strides=[1,1]))
    nodes.append(oh.make_node('Conv', inputs=['raw_lsc_active','avg3x3_w'], outputs=['b_interp'],
                              kernel_shape=[3,3], pads=[1,1,1,1], strides=[1,1]))

    # Select known vs interpolated via Where (bool masks)
    nodes.append(oh.make_node('Where', inputs=['r_mask','r_known','r_interp'], outputs=['r_chan']))
    nodes.append(oh.make_node('Where', inputs=['g_mask','g_known','g_interp'], outputs=['g_chan']))
    nodes.append(oh.make_node('Where', inputs=['b_mask','b_known','b_interp'], outputs=['b_chan']))

    # Concat RGB
    nodes.append(oh.make_node('Concat', inputs=['r_chan','g_chan','b_chan'], outputs=['rgb_raw'], axis=1))
    return nodes

# Trunk 8: Color pipeline (after CCM stage)
def add_color_pipeline(nodes, initializers):
    # NR: scale (reshape scalar to [1,1,1,1] for broadcast)
    nodes.append(oh.make_node('Reshape',
                              inputs=['nr_strength','shape_scalar4d'],
                              outputs=['nr_4d']))
    nodes.append(oh.make_node('Mul',
                              inputs=['rgb_ccm','nr_4d'],
                              outputs=['rgb_nr']))

    # Sharpen: simple scale
    nodes.append(oh.make_node('Reshape',
                              inputs=['sharp_strength','shape_scalar4d'],
                              outputs=['sharp_4d']))
    nodes.append(oh.make_node('Mul',
                              inputs=['rgb_nr','sharp_4d'],
                              outputs=['rgb_sharp']))

    # Gamma: exponent per pixel
    nodes.append(oh.make_node('Reshape',
                              inputs=['gamma','shape_scalar4d'],
                              outputs=['gamma_4d']))
    nodes.append(oh.make_node('Pow',
                              inputs=['rgb_sharp','gamma_4d'],
                              outputs=['rgb_gamma']))

    return nodes, initializers

# Trunk 9: RGB to YUV and split outputs
def add_rgb_to_yuv_and_split(nodes, initializers):
    # BT.601 conversion matrix as 1x1 conv weights [3,3,1,1]
    rgb2yuv = np.array([[0.299,  0.587,  0.114],
                        [-0.147, -0.289,  0.436],
                        [0.615, -0.515, -0.100]], dtype=np.float32)
    rgb2yuv_w = nh.from_array(rgb2yuv.reshape(3,3,1,1), 'rgb2yuv_w')
    initializers.append(rgb2yuv_w)

    # Apply conversion
    nodes.append(oh.make_node('Conv',
                              inputs=['rgb_gamma','rgb2yuv_w'],
                              outputs=['yuv'],
                              kernel_shape=[1,1],
                              pads=[0,0,0,0],
                              strides=[1,1]))

    # Split into Y, U, V using the existing split_sizes initializer
    nodes.append(oh.make_node('Split',
                              inputs=['yuv','split_sizes'],
                              outputs=['Y','U','V'],
                              axis=1))
    return nodes, initializers

# Trunk 10: Model assembly
def assemble_model(nodes, inputs, outputs, initializers):
    graph = oh.make_graph(
        nodes=nodes,
        name='ISP_RGGB_FULL',
        inputs=inputs,
        outputs=outputs,
        initializer=initializers
    )
    model = oh.make_model(
        graph,
        producer_name='softisp_builder',
        ir_version=11,
        opset_imports=[onnx.helper.make_opsetid("", 13)]
    )
    onnx.checker.check_model(model)
    return model

# Trunk X: White Balance stage
def add_white_balance(nodes):
    # Reshape wb (3,) → [1,3,1,1] for broadcast
    nodes.append(oh.make_node('Reshape',
                              inputs=['wb','shape_wb4d'],
                              outputs=['wb_4d_stage']))
    # Apply WB scaling
    nodes.append(oh.make_node('Mul',
                              inputs=['rgb_raw','wb_4d_stage'],
                              outputs=['rgb_wb']))
    return nodes
# Trunk Y: CCM stage
def add_ccm(nodes):
    # Reshape CCM (9,) → [3,3,1,1] for 1x1 Conv
    nodes.append(oh.make_node('Reshape',
                              inputs=['ccm','shape_ccm_conv'],
                              outputs=['ccm_w']))
    # Apply CCM as 1x1 Conv
    nodes.append(oh.make_node('Conv',
                              inputs=['rgb_wb','ccm_w'],
                              outputs=['rgb_ccm'],
                              kernel_shape=[1,1],
                              pads=[0,0,0,0],
                              strides=[1,1]))
    return nodes

def build_isp_model(HMAX: int, WSMAX: int):
    nodes, initializers = [], []

    inputs, outputs = create_inputs_outputs()
    nodes, initializers = add_constants_and_shapes(nodes, initializers)
    nodes = add_blc_norm_lsc(nodes)
    nodes = add_active_slice(nodes)
    nodes, initializers = add_masks_and_known(nodes, initializers, HMAX=HMAX, WSMAX=WSMAX)
    nodes = add_demosaic(nodes)

    # Call modular WB and CCM methods
    nodes = add_white_balance(nodes)
    nodes = add_ccm(nodes)

    # Continue with NR, Sharpen, Gamma
    nodes, initializers = add_color_pipeline(nodes, initializers)

    # RGB→YUV split
    nodes, initializers = add_rgb_to_yuv_and_split(nodes, initializers)

    return assemble_model(nodes, inputs, outputs, initializers)

# Trunk 12: Save model
if __name__ == '__main__':
    HMAX = int(os.environ.get('HMAX', '1080'))
    WSMAX = int(os.environ.get('WSMAX', '2048'))
    out_path = os.environ.get('OUT_ONNX', 'isp_rggb_full.onnx')
    model = build_isp_model(HMAX=HMAX, WSMAX=WSMAX)
    onnx.save(model, out_path)
    print(f"Saved {out_path} (HMAX={HMAX}, WSMAX={WSMAX})")
