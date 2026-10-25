# VaDiCAD: Trichomonas vaginalis Detection in Microscopic Smears

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

> **Official repository for the undergraduate thesis (June 2026)**  
> **Title**: *Trichomonas vaginalis Detection in Microscopic Smear Samples: A Performance Comparison of Deep Learning-Based Models*[cite: 33]  
> **Author**: **Doyeon Lim** (HUFS Department of Biomedical Engineering)[cite: 33, 34]  
> **Advisor**: **Prof. Bo-Hyoung Kim**[cite: 34]  
> **Inference Demo**: Lightweight Streamlit Demo powered by the best-performing **YOLOv8** model for automated *Trichomonas vaginalis* detection[cite: 58]

---

## 📌 Overview

*Trichomonas vaginalis* is a parasitic protozoan responsible for trichomoniasis, one of the most common sexually transmitted infections (STIs)[cite: 38]. In clinical microscopic smear examinations, detecting *Trichomonas vaginalis* is extremely challenging because its semi-transparent cytoplasm and ambiguous cell borders blend seamlessly into complex background debris, such as epithelial cells, leukocytes, and mucus[cite: 35, 38, 42].

This repository conducts a rigorous comparative benchmark specifically evaluating the ***Trichomonas vaginalis* detection performance** between standard **Object Detection (YOLO series)** and **Camouflaged Object Detection (COD series)** models on identical microscopic smear datasets[cite: 35, 43]. 

Furthermore, this repository provides a **lightweight Streamlit web demo** powered by the best-performing **YOLOv8** detector[cite: 58]: users can upload a microscopic smear image, and the model instantly detects *Trichomonas vaginalis* and displays the predicted bounding boxes.

<p align="center">
  <img src="assets/fig2_trichomonas_sample.png" width="55%" alt="Trichomonas Morphology">
  <br>
  <em>Figure 1. Microscopic smear examples of Trichomonas vaginalis showing semi-transparent cytoplasm and indistinct boundaries that cause visual assimilation into the background[cite: 42].</em>
</p>

---

## ⚠️ Problem Formulation & Motivation

Previous large-scale benchmarks (e.g., MVDI25K) demonstrated that off-the-shelf YOLO detectors often suffer from high False Positive rates and missed detections (False Negatives) specifically when identifying camouflaged *Trichomonas vaginalis* protozoa[cite: 39, 40, 41].

<p align="center">
  <img src="assets/fig1_mvdi25k_detection.png" width="90%" alt="MVDI25K Detection Results">
  <br>
  <img src="assets/fig1_mvdi25k_legend.png" width="90%" alt="MVDI25K Class Legend">
  <br>
  <em>Figure 2. Object detection results from the MVDI25K benchmark across YOLOv5 scales. Genuine Trichomonas vaginalis targets (light green bounding boxes) are frequently missed (False Negatives) across all YOLOv5 variants due to visual camouflage into complex background debris[cite: 41, 65, 66].</em>
</p>

To address this morphological constraint, this study benchmarks YOLO detectors against COD models specifically engineered to extract camouflaged boundaries, directly contrasting their efficacy in *Trichomonas vaginalis* detection[cite: 35, 42, 43].

---

## 🔄 Mask-to-Bounding Box Conversion Protocol

Because COD models output pixel-level binary masks while YOLO outputs bounding boxes, a contour-based projection is required for fair comparison[cite: 35, 48]. Using OpenCV contour tracking, the minimum enclosing bounding boxes are reverse-computed from COD prediction masks and evaluated using standardized object detection metrics ($IoU \ge 0.5$)[cite: 48, 51].

<p align="center">
  <img src="assets/fig4_mask_to_bbox.png" width="90%" alt="Mask to BBox Conversion">
  <br>
  <em>Figure 3. Mask-to-bounding box projection pipeline: <strong>(a)</strong> Input microscopic smear image, <strong>(b)</strong> DGNet predicted binary segmentation mask, and <strong>(c)</strong> Reverse-calculated bounding box coordinates overlaid on the original image for unified evaluation[cite: 57].</em>
</p>

---

## 📊 Experimental Results

### 1. Unified Detection Comparison for *Trichomonas vaginalis* (Bounding Box, $IoU \ge 0.5$)
All 7 models evaluated on the identical test set under the bounding box framework[cite: 51, 58]:

| Model | Paradigm | Precision | Recall | $F_1$-Score | $\text{AP}_{50}$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **YOLOv8-n** | YOLO[cite: 58] | **0.703**[cite: 58] | 0.857[cite: 58] | **0.772**[cite: 58] | **0.853**[cite: 58] |
| YOLOv8-s | YOLO[cite: 58] | 0.676[cite: 58] | 0.859[cite: 58] | 0.757[cite: 58] | 0.850[cite: 58] |
| **YOLOv5-s** | YOLO[cite: 58] | 0.677[cite: 58] | **0.870**[cite: 58] | 0.762[cite: 58] | 0.826[cite: 58] |
| BGNet | COD[cite: 58] | 0.413[cite: 58] | 0.462[cite: 58] | 0.436[cite: 58] | 0.196[cite: 58] |
| DGNet | COD[cite: 58] | 0.376[cite: 58] | 0.515[cite: 58] | 0.435[cite: 58] | 0.196[cite: 58] |
| SINet_V2 | COD[cite: 58] | 0.374[cite: 58] | 0.503[cite: 58] | 0.429[cite: 58] | 0.198[cite: 58] |
| ZoomNet | COD[cite: 58] | 0.086[cite: 58] | 0.079[cite: 58] | 0.083[cite: 58] | 0.007[cite: 58] |

### 2. COD Model Segmentation Benchmark (Pixel-Level)
Intrinsic segmentation metrics evaluate boundary fidelity, structural similarity ($S_m$), and absolute pixel error (MAE)[cite: 50, 51]:

| Model | S-measure&nbsp;($S_m\uparrow$) | F-measure&nbsp;($F_\beta\uparrow$) | MAE&nbsp;($\downarrow$) | Remark |
| :--- | :---: | :---: | :---: | :--- |
| **DGNet** | **0.751**[cite: 55] | **0.654**[cite: 55] | **0.038**[cite: 55] | Superior boundary gradient & texture preservation[cite: 47, 56] |
| **BGNet** | 0.734[cite: 55] | 0.618[cite: 55] | 0.043[cite: 55] | Edge-guided boundary refinement[cite: 47, 56] |
| **SINet_V2** | 0.722[cite: 55] | 0.613[cite: 55] | 0.046[cite: 55] | Detects main targets but retains fine noise[cite: 55] |
| **ZoomNet** | 0.695[cite: 55] | 0.570[cite: 55] | 0.052[cite: 55] | Mask fragmentation due to high background interference[cite: 55, 56] |

<p align="center">
  <img src="assets/fig3_cod_comparison.png" width="85%" alt="Qualitative Comparison of COD Models">
  <br>
  <em>Figure 4. Visual comparison of COD segmentation masks: <strong>(a)</strong> Original input image, <strong>(b)</strong> Ground Truth (GT), <strong>(c)</strong> ZoomNet mask, <strong>(d)</strong> SINet_V2 mask, <strong>(e)</strong> BGNet mask, and <strong>(f)</strong> DGNet mask[cite: 54].</em>
</p>

---

## 💡 Key Findings & Proposed Hybrid Pipeline

1. **Superior Single-Model Detection**:
   * **YOLOv8-n** achieved the highest overall detection performance ($\text{AP}_{50}$ 0.853, Precision 0.703, $F_1$-Score 0.772), effectively suppressing false alarms while maintaining high sensitivity[cite: 52, 58].
   * Therefore, **YOLOv8** was chosen as the primary inference backbone for the interactive demo[cite: 58].
2. **High-Fidelity Boundary Delineation**:
   * **DGNet** demonstrated unmatched pixel-level boundary extraction ($S_m$ 0.751, MAE 0.038), separating semi-transparent protozoan membranes from debris[cite: 55, 56].
3. **Serial Hybrid Pipeline Proposal**:
   * A sequential ensemble combining **YOLO (fast global screening of ROIs)** followed by **DGNet (localized pixel-level membrane segmentation)** is proposed as an optimal clinical diagnostic aid[cite: 59, 61].

---

## 🖥 Streamlit Demo (YOLOv8-Based Detection)

A simple and intuitive **Streamlit demo page** is included to showcase the detection capability of the trained YOLOv8 model[cite: 58]. Users can upload raw microscopic smear images to immediately inspect predicted bounding boxes marking detected *Trichomonas vaginalis* targets.

### How to Run
```bash
# 1. Clone repository
git clone [https://github.com/dylim-326/VaDiCAD.git](https://github.com/dylim-326/VaDiCAD.git)
cd VaDiCAD

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the Streamlit detection app
streamlit run app.py
```

---

## 📂 Repository Structure

```bash
VaDiCAD/
├── assets/                  # Qualitative figures and thesis assets
│   ├── fig1_mvdi25k_detection.png
│   ├── fig1_mvdi25k_legend.png
│   ├── fig2_trichomonas_sample.png
│   ├── fig3_cod_comparison.png
│   └── fig4_mask_to_bbox.png
├── data/                    # Dataset split & preprocessing logic
├── models/
│   ├── yolo/                # YOLO configs and trained weights
│   └── cod/                 # COD (ZoomNet, SINet_V2, BGNet, DGNet) architectures
├── utils/
│   ├── mask2bbox.py         # Contour-to-bounding-box conversion
│   └── metrics.py           # Segmentation & detection evaluators
├── app.py                   # Streamlit demo page (YOLOv8 inference)
├── requirements.txt
└── README.md
```

---

## 📖 Citation

```bibtex
@thesis{lim2026trichomonas,
  title     = {현미경 도말 검체 내 트리코모나스 탐지: 딥러닝 기반 모델 성능 비교 (Trichomonas vaginalis Detection in Microscopic Smear Samples: A Performance Comparison of Deep Learning-Based Models)},
  author    = {Doyeon Lim},
  school    = {Hankuk University of Foreign Studies, Department of Biomedical Engineering},
  year      = {2026},
  note      = {Undergraduate Thesis, Advisor: Prof. Bo-Hyoung Kim}
}
```

## 🔗 References & Acknowledgements
- **Dataset**: [MVDI25K](https://doi.org/10.1016/j.bmtse.2021.100008) — A large-scale dataset of microscopic vaginal discharge images (BenchCouncil 2021)[cite: 62]
- **Architectures**: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics), [DGNet](https://github.com/GewelsJI/DGNet)[cite: 63]