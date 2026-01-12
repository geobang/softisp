Here’s the **canonical processing chain** for your SoftISP pipeline, ordered from earliest to latest stage. This reflects the semantic block inventory we defined, with each block’s role and why it sits in that position:

---

## 🔹 Early Stages (Sensor → Linearization)
1. **BlackLevel**  
   - First step: subtract sensor offsets.  
   - Ensures raw values are normalized before any color processing.

2. **Demosaic**  
   - Converts Bayer mosaic into RGB image.  
   - Must happen immediately after black level correction.

---

## 🔹 Mid Stages (Color → Geometry → ROI)
3. **AWB (WB_V1)**  
   - Applies per‑channel gains to balance white.  
   - Needs demosaiced image as input.

4. **CCM**  
   - Applies color correction matrix.  
   - Depends on AWB outputs (image + gain).

5. **LensCorrection**  
   - Corrects geometric distortion and chromatic aberration.  
   - Operates on color‑corrected image.

6. **Crop**  
   - Selects ROI and removes stride/padding.  
   - Prepares a contiguous buffer for downstream blocks.

---

## 🔹 Later Stages (Resampling → Tone → Output)
7. **Resize**  
   - Scales cropped image to target resolution.  
   - Must follow crop to avoid wasted computation.

8. **Tone**  
   - Applies tone curve (LUT).  
   - Operates on resized image.

9. **YUVConv**  
   - Converts RGB → YUV.  
   - Typically after tone mapping to preserve perceptual balance.

10. **ChromaSubsample**  
    - Downsamples chroma channels (e.g., 4:2:0).  
    - Final step before encoding or storage.

---

## 🔹 Optional Enhancement Blocks (branching from Tone)
- **EEH (Edge Enhance)** → `Tone.image` → sharpened output.  
- **BBC (Brightness/Contrast)** → `Tone.image` → adjusted output.  
- **GAC (Gamma/Adaptive Curve)** → `Tone.image` → gamma‑corrected output.  

These can run in parallel with YUVConv depending on pipeline design.

---

## 🔹 Chain Summary

**Raw → BlackLevel → Demosaic → AWB → CCM → LensCorrection → Crop → Resize → Tone → YUVConv → ChromaSubsample**

With optional branches after **Tone** for EEH, BBC, GAC.

---

✅ This gives you a clear **early vs late stage chain**: sensor normalization and demosaic are always first, color and geometry corrections are middle, ROI and resampling follow, and tone/YUV/subsampling are last.  

Would you like me to sketch this as a **dependency DAG diagram in text form** (tree view) so you can visualize the branching more clearly?
