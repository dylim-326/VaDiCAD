# VaDiCAD: Trichomonas vaginalis Detection in Microscopic Smears

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

> **B.S. Thesis Project (HUFS Biomedical Engineering, 2026)**[cite: 33]  
> Benchmark of **Object Detection (YOLO)** vs. **Camouflaged Object Detection (COD)** for detecting semi-transparent *Trichomonas vaginalis* protozoa in vaginal smears[cite: 33, 35].

---

## 📌 Problem & Motivation

* **Target**: *Trichomonas vaginalis* (5–15 µm parasitic protozoan)[cite: 38].
* **Challenge**: The organism has semi-transparent cytoplasm and faint cell borders, visually blending into epithelial cells and debris (Camouflaged Object)[cite: 35, 42].
* **Goal**: Benchmark off-the-shelf **YOLO** (bbox-level) against **COD** (pixel-level) models to identify optimal architectures and failure modes in clinical microscopic screening[cite: 35, 39, 43].

<p align="center">
  <img src="assets/fig2_trichomonas_sample.png" width="45%" alt="Trichomonas Morphology">
  <br>
  <em>Figure 1. Microscopic appearance of Trichomonas vaginalis with semi-transparent cytoplasm[cite: 42].</em>
</p>

---

## 🔬 Benchmark & Pipeline

1. **Dataset**: 912 expert-annotated smear images from MVDI25K $\rightarrow$ expanded to **5,460 images** via geometric & optical augmentations (leak-free grouped split: 80% train / 10% val / 10% test)[cite: 43, 44].
2. **Unified Evaluation**: To compare segmentation (COD) and detection (YOLO), predicted binary masks were converted into bounding boxes using OpenCV contour tracking ($IoU \ge 0.5$)[cite: 35, 48, 51].

<p align="center">
  <img src="assets/fig4_mask_to_bbox.png" width="85%" alt="Mask to BBox Conversion">
  <br>
  <em>Figure 2. Unified comparison pipeline: (a) Input, (b) DGNet prediction mask, (c) Converted bounding box[cite: 57].</em>
</p>

---

## 📊 Key Results

### 1. Unified Detection Benchmark ($IoU \ge 0.5$)
| Model | Paradigm | Precision | Recall | $F_1$-Score | $\text{AP}_{50}$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **YOLOv8-n** | YOLO[cite: 58] | **0.703**[cite: 58] | 0.857[cite: 58] | **0.772**[cite: 58] | **0.853**[cite: 58] |
| **YOLOv8-s** | YOLO[cite: 58] | 0.676[cite: 58] | 0.859[cite: 58] | 0.757[cite: 58] | 0.850[cite: 58] |
| **YOLOv5-s** | YOLO[cite: 58] | 0.677[cite: 58] | **0.870**[cite: 58] | 0.762[cite: 58] | 0.826[cite: 58] |
| **BGNet** | COD[cite: 58] | 0.413[cite: 58] | 0.462[cite: 58] | 0.436[cite: 58] | 0.196[cite: 58] |
| **DGNet** | COD[cite: 58] | 0.376[cite: 58] | 0.515[cite: 58] | 0.435[cite: 58] | 0.196[cite: 58] |
| **SINet_V2** | COD[cite: 58] | 0.374[cite: 58] | 0.503[cite: 58] | 0.429[cite: 58] | 0.198[cite: 58] |
| **ZoomNet** | COD[cite: 58] | 0.086[cite: 58] | 0.079[cite: 58] | 0.083[cite: 58] | 0.007[cite: 58] |

### 2. COD Boundary Segmentation Benchmark
| Model | <span style="white-space:nowrap">S-measure ($S_m \uparrow$)</span> | <span style="white-space:nowrap">F-measure ($F_\beta \uparrow$)</span> | <span style="white-space:nowrap">MAE ($\downarrow$)</span> | Key Characteristics |
| :--- | :---: | :---: | :---: | :--- |
| **DGNet** | **0.751**[cite: 55] | **0.654**[cite: 55] | **0.038**[cite: 55] | Best boundary preservation with lowest pixel error[cite: 55, 56] |
| **BGNet** | 0.734[cite: 55] | 0.618[cite: 55] | 0.043[cite: 55] | Edge-guided boundary refinement[cite: 47, 56] |
| **SINet_V2** | 0.722[cite: 55] | 0.613[cite: 55] | 0.046[cite: 55] | Detects main targets but retains slight noise[cite: 55] |
| **ZoomNet** | 0.695[cite: 55] | 0.570[cite: 55] | 0.052[cite: 55] | Severe mask fragmentation[cite: 55] |

<p align="center">
  <img src="assets/fig3_cod_comparison.png" width="80%" alt="COD Model Comparison">
  <br>
  <em>Figure 3. Qualitative comparison of COD models: (a) Input, (b) GT, (c) ZoomNet, (d) SINet_V2, (e) BGNet, (f) DGNet[cite: 54].</em>
</p>

---

## 💡 Practical Takeaways

* **Fast Screening**: **YOLOv8-n** demonstrated the highest bounding box precision and $\text{AP}_{50}$ (0.853), proving most effective as a lightweight single screening detector[cite: 52, 58, 59].
* **Fine Morphology**: **DGNet** captured faint protozoan membranes best ($S_m$ 0.751, MAE 0.038) under complex background clutter[cite: 55, 56].
* **Hybrid Pipeline Suggestion**: Fast global candidate screening via **YOLO** followed by localized fine segmentation via **DGNet** inside candidate ROIs[cite: 59, 61].

---

## 🖥 Streamlit Demo

A simple demo app running the best-performing **YOLOv8** model[cite: 58]. Upload a smear image to visualize predicted bounding boxes in real time.

```bash
# Clone and install
git clone [https://github.com/dylim-326/VaDiCAD.git](https://github.com/dylim-326/VaDiCAD.git)
cd VaDiCAD
pip install -r requirements.txt

# Run demo
streamlit run app.py
```

---

## 📂 Project Structure

```bash
VaDiCAD/
├── assets/                  # Figures & samples
├── data/                    # Preprocessing & augmentation scripts
├── models/                  # YOLO and COD model wrappers
├── utils/                   # Mask-to-bbox converter & metric evaluators
├── app.py                   # Streamlit inference demo
├── requirements.txt
└── README.md
```

---

## 🔗 Acknowledgements
* **Dataset**: [MVDI25K (Zenodo)](https://zenodo.org/records/5523661) (BenchCouncil 2021)
* **Frameworks**: [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), [DGNet](https://github.com/GewelsJI/DGNet), [BGNet](https://github.com/thograce/BGNet), [SINet-V2](https://github.com/GewelsJI/SINet-V2), [ZoomNet](https://github.com/lartpang/ZoomNet)[cite: 45, 47, 62, 63]