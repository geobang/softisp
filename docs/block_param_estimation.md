# Block Parameter Estimation

This document lists all parameters required by each microblock in the SoftISP pipeline.

## image_desc_base
- **image**: Raw input tensor [n,c,h,stride]
- **width**: Scalar width metadata

## bayernorm_base
- **norm_scale**: Normalization divisor (e.g. 4095.0 for 12-bit)

## stride_remove
- **crop_starts**: INT64 array start indices
- **crop_ends**: INT64 array end indices
- **axes**: (recommended) dimensions to crop, typically [2,3]
- **steps**: (recommended) stride steps, typically [1,1]

## blacklevel
- **offset**: Per-channel black level offset

## demosaic
- **kernels**: Convolution kernels for Bayer → RGB

## awb
- **wb_gains**: White balance gains per channel

## ccm
- **ccm**: 3×3 color correction matrix

## lcs
- **lcs_gain_map**: Lens shading correction gain map [1,3,h,w]

## resize
- **target_h**: Target height
- **target_w**: Target width
- **mode**: (recommended) interpolation mode

## tonemap
- **tonemap_curve**: Tone mapping curve or LUT

## gamma
- **gamma_value**: Gamma exponent

## yuv
- **rgb2yuv_matrix**: RGB→YUV conversion matrix
- **offsets**: (recommended) codec offsets [16,128,128]
- **range_mode**: (recommended) full vs limited range

## chroma
- **subsample_scale**: Chroma subsampling factor
- **mode**: (recommended) subsampling scheme (4:4:4, 4:2:2, 4:2:0)
