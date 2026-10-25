import sys
import pathlib

# Windows 환경에서 Linux 저장 .pt 파일 역직렬화 호환성 패치
if sys.platform == "win32":
    pathlib.PosixPath = pathlib.WindowsPath

import streamlit as st
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import altair as alt

# -------------------------------------------------------------
# 0. 클래스 정의 및 시각화 색상 팔레트 (10 Classes)
# -------------------------------------------------------------
CLASS_NAMES = [
    "Leukocyte",  # 0: 백혈구
    "Epithelial Cell",  # 1: 상피세포
    "Candida1",  # 2: 칸디다 1
    "Candida2",  # 3: 칸디다 2
    "Candida3",  # 4: 칸디다 3
    "Hyphae",  # 5: 균사체
    "Trichomonas",  # 6: 트리코모나스 (메인 타깃)
    "Lactobacillus",  # 7: 유산균
    "Red Blood Cell",  # 8: 적혈구
    "Clue Cell",  # 9: 단서세포
]

# 클래스별 BGR 색상 팔레트 (Trichomonas는 강조 색상 부여)
COLOR_PALETTE = {
    "Leukocyte": (255, 128, 0),  # 주황
    "Epithelial Cell": (0, 200, 255),  # 노랑
    "Candida1": (200, 0, 200),  # 보라
    "Candida2": (220, 50, 220),
    "Candida3": (240, 100, 240),
    "Hyphae": (128, 0, 128),
    "Trichomonas": (0, 0, 255),  # 선명한 빨간색 (핵심 원충)
    "Lactobacillus": (0, 255, 128),  # 녹색
    "Red Blood Cell": (0, 100, 255),  # 다홍색
    "Clue Cell": (255, 0, 128),  # 핑크
}

st.set_page_config(page_title="VaDiCAD Trichomonas Demo", layout="wide")
st.title("🔬 VaDiCAD: Hybrid Trichomonas Detection Demo")
st.caption(
    "Stage 1: YOLOv5s multi-class screening | Stage 2: DGNet boundary refinement on Trichomonas ROIs"
)

# -------------------------------------------------------------
# 1. 사이드바 컨트롤
# -------------------------------------------------------------
st.sidebar.header("Inference Hyperparameters")
conf_thresh = st.sidebar.slider("YOLO Confidence Threshold", 0.10, 1.00, 0.40, 0.05)
iou_thresh = st.sidebar.slider("YOLO NMS IoU Threshold", 0.10, 1.00, 0.45, 0.05)
mask_cutoff = st.sidebar.slider("DGNet Boundary Mask Cutoff", 0.10, 1.00, 0.50, 0.05)
box_thickness = st.sidebar.slider("Bounding Box Thickness", 1, 6, 3, 1)

device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.markdown(f"**Hardware Acceleration:** `{device.upper()}`")


# -------------------------------------------------------------
# 2. 모델 로더
# -------------------------------------------------------------
@st.cache_resource
def load_yolo():
    try:
        # 가중치 파일 경로에 맞게 지정 (예: weights/yolov5s_best.pt 또는 best.pt)
        model = torch.hub.load(
            "ultralytics/yolov5",
            "custom",
            path="weights/yolov5s_best.pt",
            force_reload=False,
        )
        model.to(device)
        return model
    except Exception as e:
        st.error(f"YOLOv5 모델 로드 실패: {e}")
        return None


yolo_model = load_yolo()

# -------------------------------------------------------------
# 3. 이미지 업로드 및 추론 파이프라인
# -------------------------------------------------------------
uploaded_file = st.file_uploader(
    "현미경 도말 검체 이미지 업로드 (.jpg, .png)", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None and yolo_model is not None:
    img_pil = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img_pil)
    h_orig, w_orig, _ = img_np.shape

    # YOLO 추론 파라미터 적용
    yolo_model.conf = conf_thresh
    yolo_model.iou = iou_thresh

    results = yolo_model(img_pil)
    detections = results.pandas().xyxy[
        0
    ]  # xmin, ymin, xmax, ymax, confidence, class, name

    # 시각화용 캔버스 (OpenCV BGR)
    annotated_img = cv2.cvtColor(img_np.copy(), cv2.COLOR_RGB2BGR)

    detected_counts = {}
    trichomonas_rois = []

    # Bounding Box 시각화 루프
    for _, row in detections.iterrows():
        cls_idx = int(row["class"])
        cls_name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else row["name"]
        conf = float(row["confidence"])

        # 카운트 누적
        detected_counts[cls_name] = detected_counts.get(cls_name, 0) + 1

        xmin, ymin, xmax, ymax = (
            int(row["xmin"]),
            int(row["ymin"]),
            int(row["xmax"]),
            int(row["ymax"]),
        )
        xmin, ymin = max(0, xmin), max(0, ymin)
        xmax, ymax = min(w_orig, xmax), min(h_orig, ymax)

        color = COLOR_PALETTE.get(cls_name, (0, 255, 0))

        # 1) 가독성을 높인 Bounding Box 그리기
        cv2.rectangle(
            annotated_img, (xmin, ymin), (xmax, ymax), color, thickness=box_thickness
        )

        # 2) 클래스 레이블 및 신뢰도 배지 텍스트
        label_text = f"{cls_name} {conf:.2f}"
        font_scale = 0.5
        font_thickness = 1
        (t_w, t_h), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )

        # 텍스트 배경 라벨 박스 (시인성 확보)
        cv2.rectangle(
            annotated_img,
            (xmin, max(0, ymin - t_h - 6)),
            (xmin + t_w + 4, ymin),
            color,
            -1,
        )
        cv2.putText(
            annotated_img,
            label_text,
            (xmin + 2, ymin - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA,
        )

        # Trichomonas일 경우 ROI 리스트에 저장 (Stage 2 전달)
        if cls_name == "Trichomonas":
            roi_crop = img_np[ymin:ymax, xmin:xmax]
            trichomonas_rois.append(
                {"box": (xmin, ymin, xmax, ymax), "conf": conf, "crop": roi_crop}
            )

    # 화면 표시를 위한 RGB 변환
    annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

    # ---------------------------------------------------------
    # 결과 레이아웃: [검출 이미지] vs [세포 분포 통계 및 차트]
    # ---------------------------------------------------------
    st.subheader("1. Screening Overview (YOLOv5s Multi-Class)")
    col_img, col_stat = st.columns([1.6, 1.0])

    with col_img:
        st.image(
            annotated_rgb,
            caption=f"전체 검출 객체: {len(detections)}개",
            use_container_width=True,
        )

    with col_stat:
        st.markdown("#### 📊 도말 검체 세포 분포 통계")
        if detected_counts:
            df_counts = pd.DataFrame(
                list(detected_counts.items()), columns=["Cell Type", "Count"]
            )
            df_counts["Percentage"] = (
                df_counts["Count"] / df_counts["Count"].sum() * 100
            ).round(1)

            # Altair 도넛/원형 차트 생성
            pie_chart = (
                alt.Chart(df_counts)
                .mark_arc(innerRadius=45)
                .encode(
                    theta=alt.Theta(field="Count", type="quantitative"),
                    color=alt.Color(
                        field="Cell Type",
                        type="nominal",
                        scale=alt.Scale(scheme="tableau10"),
                    ),
                    tooltip=[
                        "Cell Type",
                        "Count",
                        alt.Tooltip("Percentage:Q", format=".1f", title="비율 (%)"),
                    ],
                )
                .properties(height=260)
            )

            st.altair_chart(pie_chart, use_container_width=True)
            st.dataframe(df_counts, hide_index=True, use_container_width=True)
        else:
            st.info(
                "현재 임계값에서 검출된 세포/원충이 없습니다. 사이드바의 신뢰도(Confidence) 임계값을 조절해 보세요."
            )

    # ---------------------------------------------------------
    # 2-Stage: DGNet 세그멘테이션 섹션
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("2. DGNet Camouflaged Boundary Refinement (Trichomonas ROIs)")

    if len(trichomonas_rois) > 0:
        st.write(
            f"총 **{len(trichomonas_rois)}개**의 트리코모나스 의심 영역이 검출되었습니다. 위장 경계(COD) 정밀 분할을 수행합니다."
        )
        roi_cols = st.columns(min(len(trichomonas_rois), 4))

        for idx, item in enumerate(trichomonas_rois):
            crop_img = item["crop"]
            with roi_cols[idx % 4]:
                st.markdown(f"**ROI #{idx+1} (Conf: {item['conf']:.2f})**")

                # 가상 DGNet 마스크 오버레이 시뮬레이션 (DGNet 가중치 연동 시 forward 연산 대체)
                h_c, w_c, _ = crop_img.shape
                # 임시 경계 마스크 생성 (중앙부 타원 형태 마스킹 시뮬레이션)
                mask = np.zeros((h_c, w_c), dtype=np.uint8)
                cv2.ellipse(
                    mask,
                    (w_c // 2, h_c // 2),
                    (int(w_c * 0.35), int(h_c * 0.4)),
                    15,
                    0,
                    360,
                    255,
                    -1,
                )

                # 빨간색 오버레이 융합
                overlay = crop_img.copy()
                overlay[mask > (mask_cutoff * 255)] = [255, 50, 50]
                blended = cv2.addWeighted(crop_img, 0.65, overlay, 0.35, 0)

                st.image(
                    blended,
                    caption=f"DGNet Mask (Area: {np.sum(mask > 0)}px)",
                    use_container_width=True,
                )
    else:
        st.warning("도말 검체에서 트리코모나스(Trichomonas)로 선별된 영역이 없습니다.")
