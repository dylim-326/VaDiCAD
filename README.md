# VaDiCAD: Trichomonas vaginalis Detection in Microscopic Smear Samples

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

> **Official repository for the undergraduate thesis (June 2026)**  
> *"Trichomonas vaginalis Detection in Microscopic Smear Samples: A Performance Comparison of Deep Learning-Based Models"*[cite: 33]  
> Author: **Doyeon Lim** (HUFS BME)[cite: 33, 34] | Advisor: **Prof. Bo-Hyoung Kim**[cite: 34]

---

## 📌 Overview

질 분비물 현미경 도말 검체 내에서 주변 불순물(상피세포, 백혈구 등)과 시각적으로 동화되는 **트리코모나스 원충(*Trichomonas vaginalis*)**을 자동 탐지하기 위해 **YOLO(Object Detection)** 군과 **COD(Camouflaged Object Detection)** 군의 성능을 비교 분석한 연구 프로젝트입니다[cite: 35, 38, 42].

* **Objective**: 빠른 위치 특정(YOLO)과 픽셀 단위 미세 경계 분할(COD) 간의 공학적 성능 대조[cite: 35]
* **Demo**: 모델별 탐지 및 분할 결과를 인터랙티브하게 비교할 수 있는 경량 **Streamlit Result Viewer** 제공

<p align="center">
  <img src="assets/cod_comparison.png" width="80%" alt="Qualitative Comparison">
  <br>
  <em>Figure: Qualitative comparison of COD models ((a) Image, (b) Ground Truth, (c) ZoomNet, (d) SINet_V2, (e) BGNet, (f) DGNet)[cite: 54]</em>
</p>

---

## 📊 Benchmark Results

### 1. Unified Detection Comparison (Bounding Box, $IoU \ge 0.5$)
COD의 이진 마스크를 OpenCV 외곽선 기반 바운딩 박스로 역산출하여 통일된 환경에서 비교한 결과입니다[cite: 35, 48].

| Model | Paradigm | Precision | Recall | $F_1$-Score | $\text{AP}_{50}$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **YOLOv8-n** | YOLO | **0.703** | 0.857 | **0.772** | **0.853** |
| YOLOv8-s | YOLO | 0.676 | 0.859 | 0.757 | 0.850 |
| **YOLOv5-s** | YOLO | 0.677 | **0.870** | 0.762 | 0.826 |
| BGNet | COD | 0.413 | 0.462 | 0.436 | 0.196 |
| DGNet | COD | 0.376 | 0.515 | 0.435 | 0.196 |
| SINet_V2 | COD | 0.374 | 0.503 | 0.429 | 0.198 |
| ZoomNet | COD | 0.086 | 0.079 | 0.083 | 0.007 |

*전체 7개 모델 평가 결과[cite: 58]*

### 2. COD Model Segmentation Benchmark (Pixel-Level)
| Model | S-measure ($S_m \uparrow$) | F-measure ($F_\beta \uparrow$) | MAE ($\downarrow$) |
| :--- | :---: | :---: | :---: |
| **DGNet** | **0.751** | **0.654** | **0.038** |
| BGNet | 0.734 | 0.618 | 0.043 |
| SINet_V2 | 0.722 | 0.613 | 0.046 |
| ZoomNet | 0.695 | 0.570 | 0.052 |

*COD 분할 정량 지표[cite: 55]*

---

## 💡 Key Takeaways

1. **단일 위치 특정**: **YOLOv8-n**이 $\text{AP}_{50}$ 0.853, $F_1$ 0.772로 가장 우수한 탐지 밸런스를 달성[cite: 52, 58].
2. **미세 경계 분할**: **DGNet**이 불순물 노이즈 환경에서도 가장 낮은 오차율(MAE 0.038)로 세포막 경계를 정밀 복원[cite: 55, 56].
3. **제언 (하이브리드 시스템)**: 빠른 YOLO로 대략적인 관심 영역(ROI)을 1차 스크리닝한 후, DGNet을 통해 세포막을 정밀 분할하는 **직렬 앙상블 파이프라인** 제안[cite: 59, 61].

---

## 🚀 Quick Start (Streamlit Demo)

```bash
# Clone repository
git clone [https://github.com/dylim-326/VaDiCAD.git](https://github.com/dylim-326/VaDiCAD.git)
cd VaDiCAD

# Install dependencies
pip install -r requirements.txt

# Run demo viewer
streamlit run app.py


📂 Repository Structure

├── assets/                  # Qualitative result figures
├── data/                    # Dataset loaders & preprocessing scripts
├── models/                  # YOLO and COD model architectures
├── utils/                   # Mask-to-BBox conversion & evaluation metrics
├── app.py                   # Streamlit demo viewer
├── requirements.txt
└── README.md

📖 Citation
@thesis{lim2026trichomonas,
  title     = {현미경 도말 검체 내 트리코모나스 탐지: 딥러닝 기반 모델 성능 비교},
  author    = {임도연},
  school    = {한국외국어대학교 바이오메디컬공학부},
  year      = {2026},
  note      = {지도교수: 김보형}


[cite: 33, 34, 35, 38, 42, 48, 52, 54, 55, 56, 58, 59, 61]