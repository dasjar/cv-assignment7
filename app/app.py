#!/usr/bin/env python3
import os
import sys
import math
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# ------------------------------------------------------
# GLOBAL MEDIAPIPE LOAD (SAFE FOR STREAMLIT CLOUD)
# ------------------------------------------------------
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

POSE_MODEL = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=0,        # LIGHTEST MODEL
    enable_segmentation=False,
    min_detection_confidence=0.5
)

HANDS_MODEL = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

# ------------------------------------------------------
# PATH FIX
# ------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------
# STREAMLIT PAGE CONFIG
# ------------------------------------------------------
st.set_page_config(
    page_title="Stereo Estimator + Pose Tracking",
    layout="wide"
)

st.markdown("""
<style>
div.stCanvas { overflow: visible !important; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------
# POSE + HANDS HELPER (LIGHTWEIGHT)
# ------------------------------------------------------
def analyze_image_pose_and_hands(img_bgr):
    """Lightweight pose+hands extraction for single images."""
    results = []

    try:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        pose_res = POSE_MODEL.process(img_rgb)
        hands_res = HANDS_MODEL.process(img_rgb)

        annotated = img_bgr.copy()

        # Pose
        if pose_res.pose_landmarks:
            mp_drawing.draw_landmarks(
                annotated, pose_res.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )
            for idx, lm in enumerate(pose_res.pose_landmarks.landmark):
                results.append({
                    "track": "pose",
                    "idx": idx,
                    "x": lm.x, "y": lm.y, "z": lm.z,
                    "visibility": lm.visibility
                })

        # Hands
        if hands_res.multi_hand_landmarks:
            for hand in hands_res.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    annotated, hand, mp_hands.HAND_CONNECTIONS
                )
                for idx, lm in enumerate(hand.landmark):
                    results.append({
                        "track": "hand",
                        "idx": idx,
                        "x": lm.x, "y": lm.y, "z": lm.z,
                        "visibility": 1.0
                    })

        return annotated, results

    except Exception as e:
        print("ERROR in MediaPipe:", e)
        return img_bgr, []


# ------------------------------------------------------
# TABS
# ------------------------------------------------------
tab1, tab2 = st.tabs(["Stereo Object Size Estimator", "Pose & Hand Tracking"])


# ======================================================
# TAB 1 — STEREO (MINIMAL + WORKING)
# ======================================================
with tab1:
    st.header("Stereo Object Size Estimator (3-Point Measurement)")

    pair_index = st.number_input(
        "Select Pair", min_value=1, max_value=50, value=1, step=1
    )

    img_path = f"data/output/rectified/rect_left_{pair_index}.jpg"
    if not os.path.exists(img_path):
        st.error(f"Image not found: {img_path}")
        st.stop()

    img = Image.open(img_path)
    W, H = img.size

    MAX_W = 900
    scale = min(1, MAX_W / W)
    img_disp = img.resize((int(W * scale), int(H * scale)))

    st.write("Click **three points** on one object: corner + two edges.")

    canvas = st_canvas(
        stroke_width=5,
        stroke_color="#ff0000",
        background_image=img_disp,
        height=int(H * scale),
        width=int(W * scale),
        drawing_mode="point",
        key="stereo_canvas"
    )

    # Fixed reference (you can update)
    REF_P1 = (1641, 1242)
    REF_P2 = (1761, 3365)
    REF_LEN_PX = math.hypot(REF_P1[0]-REF_P2[0], REF_P1[1]-REF_P2[1])
    CM_PER_PIXEL = 40.0 / REF_LEN_PX

    def to_orig(obj):
        return (int(obj["left"] / scale), int(obj["top"] / scale))

    if canvas.json_data:
        objs = canvas.json_data["objects"]

        if len(objs) >= 2:
            p1 = to_orig(objs[0])
            p2 = to_orig(objs[1])

            d12 = math.dist(p1, p2) * CM_PER_PIXEL
            st.info(f"Distance P1–P2: **{d12:.2f} cm**")

        if len(objs) >= 3:
            p3 = to_orig(objs[2])
            d13 = math.dist(p1, p3) * CM_PER_PIXEL

            st.success(
                f"""
### Measured Dimensions  
- Shorter edge: **{min(d12, d13):.2f} cm**  
- Longer edge: **{max(d12, d13):.2f} cm**  
                """
            )


# ======================================================
# TAB 2 — LIGHTWEIGHT POSE + HAND TRACKING
# ======================================================
with tab2:
    st.header("Pose & Hand Tracking (Lightweight Mode)")

    st.write("Upload an image OR take a webcam snapshot.")

    mode = st.radio(
        "Input mode:",
        ["Upload image", "Webcam snapshot"],
        horizontal=True
    )

    img_bgr = None

    if mode == "Upload image":
        up = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
        if up:
            data = np.frombuffer(up.read(), np.uint8)
            img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)

    else:
        snap = st.camera_input("Take photo")
        if snap:
            data = np.frombuffer(snap.getvalue(), np.uint8)
            img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if img_bgr is not None:
        st.subheader("Processed Image")

        annotated, rows = analyze_image_pose_and_hands(img_bgr)

        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Pose Result")

        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df)

            csv_path = os.path.join(OUTPUT_DIR, "pose_hand.csv")
            df.to_csv(csv_path, index=False)

            st.download_button(
                "Download Landmark CSV",
                open(csv_path, "rb"),
                file_name="pose_hand.csv",
                mime="text/csv"
            )
        else:
            st.warning("No pose/hands detected.")


