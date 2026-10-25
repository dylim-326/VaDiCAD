# VaDiCAD: Trichomonas vaginalis Detection in Microscopic Smears

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

> **B.S. Thesis Project (HUFS Biomedical Engineering, 2026)**  
> Benchmark of **Object Detection (YOLO)** vs. **Camouflaged Object Detection (COD)** for detecting semi-transparent *Trichomonas vaginalis* protozoa in vaginal smears.

---

## 📌 Problem & Motivation

- **Target**: *Trichomonas vaginalis* (5–15 µm parasitic protozoan).
- **Challenge**: The organism has semi-transparent cytoplasm and faint cell borders, visually blending into epithelial cells and debris (Camouflaged Object).
- **Goal**: Benchmark off-the-shelf **YOLO** (bbox-level) against **COD** (pixel-level) models to identify optimal architectures and failure modes in clinical microscopic screening.

<p align="center">
  <img src="assets/fig2_trichomonas_sample.png" width="45%" alt="Trichomonas Morphology">
  <br>
  <em>Figure 1. Microscopic appearance of Trichomonas vaginalis with semi-transparent cytoplasm.</em>
</p>

---

## 🔬 Benchmark & Pipeline

1. **Dataset**: 912 expert-annotated smear images from MVDI25K $\rightarrow$ expanded to **5,460 images** via geometric & optical augmentations (leak-free grouped split: 80% train / 10% val / 10% test).
2. **Unified Evaluation**: To compare segmentation (COD) and detection (YOLO), predicted binary masks were converted into bounding boxes using OpenCV contour tracking ($IoU \ge 0.5$).

<p align="center">
  <img src="assets/fig4_mask_to_bbox.png" width="85%" alt="Mask to BBox Conversion">
  <br>
  <em>Figure 2. Unified comparison pipeline: (a) Input, (b) DGNet prediction mask, (c) Converted bounding box.</em>
</p>

---

## 📊 Key Results

### 1. Unified Detection Benchmark ($IoU \ge 0.5$)
| Model | Paradigm | Precision | Recall | $F_1$-Score | $\text{AP}_{50}$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **YOLOv8-s** | YOLO | 0.676 | 0.859 | 0.757 | **0.850** |
| **YOLOv5-s** | YOLO | **0.677** | **0.870** | **0.762** | 0.826 |
| **SINet_V2** | COD | 0.374 | 0.503 | 0.429 | 0.198 |
| **BGNet** | COD | 0.413 | 0.462 | 0.436 | 0.196 |
| **DGNet** | COD | 0.376 | 0.515 | 0.435 | 0.196 |
| **ZoomNet** | COD | 0.086 | 0.079 | 0.083 | 0.007 |

### 2. COD Boundary Segmentation Benchmark
| Model | <span style="white-space:nowrap">S-measure ($S_m \uparrow$)</span> | <span style="white-space:nowrap">F-measure ($F_\beta \uparrow$)</span> | <span style="white-space:nowrap">MAE ($\downarrow$)</span> | Key Characteristics |
| :--- | :---: | :---: | :---: | :--- |
| **DGNet** | **0.751** | **0.654** | **0.038** | Best boundary preservation with lowest pixel error |
| **BGNet** | 0.734 | 0.618 | 0.043 | Edge-guided boundary refinement |
| **SINet_V2** | 0.722 | 0.613 | 0.046 | Detects main targets but retains slight noise |
| **ZoomNet** | 0.695 | 0.570 | 0.052 | Severe mask fragmentation |

<p align="center">
  <img src="assets/fig3_cod_comparison.png" width="80%" alt="COD Model Comparison">
  <br>
  <em>Figure 3. Qualitative comparison of COD models: (a) Input, (b) GT, (c) ZoomNet, (d) SINet_V2, (e) BGNet, (f) DGNet.</em>
</p>

---

## 💡 Practical Takeaways

- **Fast Screening**: **YOLOv8-s** demonstrated the highest localization precision and $\text{AP}_{50}$ (0.850), proving most effective as a reliable single screening detector.
- **Fine Morphology**: **DGNet** captured faint protozoan membranes best ($S_m$ 0.751, MAE 0.038) under complex background clutter and cellular debris.
- **Hybrid Pipeline Suggestion**: Fast global candidate screening via **YOLOv8-s** followed by localized fine segmentation via **DGNet** inside candidate ROIs.

---

## 🖥 Streamlit Demo

A real-time diagnostic demo app running the fine-tuned **YOLOv8-s** detector. Upload a smear image to visualize predicted bounding boxes in real time.

```bash
# Clone and install
git clone [https://github.com/dylim-326/VaDiCAD.git](https://github.com/dylim-326/VaDiCAD.git)
cd VaDiCAD
pip install -r requirements.txt

# Run demo
streamlit run app.py