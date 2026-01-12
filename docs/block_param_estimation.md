Here’s the **full updated block inventory** with semantic parameter names, incorporating the stride handling in Crop and the mandatory coeffs array convention we agreed on:

---

## 🔹 Full Block Inventory (Semantic Parameters)

| Block              | Inputs                          | Outputs            | Mandatory Parameters / Coeffs |
|--------------------|---------------------------------|--------------------|-------------------------------|
| **BlackLevel**     | Raw sensor image                | `BlackLevel.image` | `BlackLevel.offset` (per‑channel black level values) |
| **Demosaic**       | `BlackLevel.image`              | `Demosaic.image`   | `Demosaic.pattern` (Bayer pattern enum) |
| **AWB (WB_V1)**    | `Demosaic.image`                | `AWB.image`        | `AWB.gain` (RGB gain vector) |
| **CCM**            | `AWB.image`, `AWB.gain`         | `CCM.image`        | `CCM.matrix` (3×3 color correction matrix) |
| **LensCorrection** | `CCM.image`                     | `Lens.image`       | `Lens.distortion_map`, `Lens.chromatic_shift` |
| **Crop**           | `Lens.image`                    | `Crop.image`       | `Crop.roi` (x,y,w,h), `Crop.stride_mode` (remove/preserve/align) |
| **Resize**         | `Crop.image`                    | `Resize.image`     | `Resize.target_size`, `Resize.interp_mode` |
| **Tone**           | `Resize.image`                  | `Tone.image`       | `Tone.curve` (tone LUT) |
| **YUVConv**        | `Tone.image`                    | `YUV.image`        | `YUV.matrix` (RGB→YUV conversion matrix) |
| **ChromaSubsample**| `YUV.image`                     | `Chroma.image`     | `Chroma.mode` (4:2:0, 4:2:2) |
| **EEH (Edge Enhance)** | `Tone.image`                | `EEH.image`        | `EEH.kernel`, `EEH.strength` |
| **BBC (Brightness/Contrast)** | `Tone.image`         | `BBC.image`        | `BBC.contrast`, `BBC.brightness` |
| **GAC (Gamma/Adaptive Curve)** | `Tone.image`        | `GAC.image`        | `GAC.gamma_lut`, `GAC.gamma_exp` |

---

## 🔹 Key Points
- **Inputs:** Always specified by block name in JSON (`inputs: ["AWB"]`), coordinator expands to semantic outputs (`AWB.image`, `AWB.gain`).  
- **Outputs:** Declared inside each microblock body, not in JSON.  
- **Mandatory coeffs array:** Every block must expose a `coeffs` array with semantic names (`gain`, `matrix`, `curve`, etc.).  
- **Crop:** Explicitly includes stride handling (`stride_mode`) in its parameters.  
- **Compatibility:** CCM declares dependency on AWB (or WB_V1), meta‑blocks can inherit WB_V1 to insert quirks while still advertising compatibility.  

---

✅ This list is now the canonical reference for your pipeline design. Each block has clear semantic parameters, mandatory coeffs, and explicit inputs/outputs.  

Would you like me to **draw a dependency DAG** (textual tree) showing how these blocks connect, so you can visualize the flow from Raw → BlackLevel → … → ChromaSubsample?
