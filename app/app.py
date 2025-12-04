#!/usr/bin/env python3
# ============================================================
# CSc 8830 – Assignment 7
# Stereo Size Estimator + Pose & Hand Tracking (MediaPipe)
# Fully patched for Render deployment
# ============================================================

import os
import sys
import math
import base64
from io import BytesIO
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import mediapipe as mp

# ------------------------------------------------------------
# FIX PATH FOR MODULE IMPORTS
# ------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.size_estimation import compute_3d_distance

OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# MEDIAPIPE SETUP
# ============================================================
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands


def analyze_frame_pose_and_hands(frame_bgr, frame_index: int):
    """
    Run MediaPipe Pose + Hands on one BGR frame.
    Returns:
        annotated frame
        list of CSV-rows {timestamp, frame_index, x,y,z,...}
    """
    rows = []
    ts = datetime.utcnow().isoformat()

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose, mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:

        pose_results = pose.process(frame_rgb)
        hands_results = hands.process(frame_rgb)

        annotated = frame_bgr.copy()

        # Pose landmarks
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(
                annotated,
                pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
            )

            for idx, lm in enumerate(pose_results.pose_landmarks.landmark):
                rows.append(
                    {
                        "timestamp": ts,
                        "frame_index": frame_index,
                        "track": "pose",
                        "landmark_index": idx,
                        "x": lm.x,
                        "y": lm.y,
                        "z": lm.z,
                        "visibility": lm.visibility,
                    }
                )

        # Hand landmarks
        if hands_results.multi_hand_landmarks and hands_results.multi_handedness:
            for hand_lms, handedness in zip(
                hands_results.multi_hand_landmarks,
                hands_results.multi_handedness,
            ):
                label = handedness.classification[0].label.lower()
                track_name = f"{label}_hand"

                mp_drawing.draw_landmarks(
                    annotated,
                    hand_lms,
                    mp_hands.HAND_CONNECTIONS,
                )

                for idx, lm in enumerate(hand_lms.landmark):
                    rows.append(
                        {
                            "timestamp": ts,
                            "frame_index": frame_index,
                            "track": track_name,
                            "landmark_index": idx,
                            "x": lm.x,
                            "y": lm.y,
                            "z": lm.z,
                            "visibility": 1.0,
                        }
                    )

    return annotated, rows


# ============================================================
# STREAMLIT PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Assignment 7 – Stereo + Pose Tracking", layout="wide")

st.markdown(
    """
    <style>
    div.stCanvas { overflow: visible !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# HELPER: Convert PIL to Base64 data URL (REPLACES image_to_url)
# ============================================================
def pil_to_data_url(pil_img):
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    data = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{data}"


# ============================================================
# SAFE IMAGE DISPLAY
# ============================================================
def safe_display_image(bgr_img, caption=""):
    """Prevent huge images from causing Render 502 errors."""
    max_width = 650
    h, w = bgr_img.shape[:2]

    if w > max_width:
        scale = max_width / w
        bgr_img = cv2.resize(
            bgr_img,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )

    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    st.image(rgb, caption=caption)


# ============================================================
# TABS
# ============================================================
tab_stereo, tab_pose = st.tabs([
    "Stereo Object Size Estimator",
    "Pose & Hand Tracking",
])

# ============================================================
# TAB 1 — STEREO SIZE ESTIMATOR
# ============================================================
with tab_stereo:

    st.title("Stereo Object Size Estimator (3-Point Measurement)")

    pair_index = st.number_input("Select Pair", min_value=1, max_value=50, value=1)

    left_path = f"data/output/rectified/rect_left_{pair_index}.jpg"

    if not os.path.exists(left_path):
        st.error(f"Rectified image not found: {left_path}")
        st.stop()

    img = Image.open(left_path)
    img_w, img_h = img.size

    MAX_WIDTH = 900
    if img_w > MAX_WIDTH:
        scale_factor = MAX_WIDTH / img_w
        disp_w = MAX_WIDTH
        disp_h = int(img_h * scale_factor)
    else:
        scale_factor = 1.0
        disp_w = img_w
        disp_h = img_h

    img_display = img.resize((disp_w, disp_h))

    st.write(
        "Click **three points** on one object: corner + two edges.\n"
        "Shorter edge = width, longer edge = height."
    )

    EXTRA_PAD = int(0.15 * disp_h)

    # Convert image to data URL for Canvas (fixes Render crash)
    img_display_url = pil_to_data_url(img_display)

    canvas = st_canvas(
        stroke_width=5,
        stroke_color="#ff0000",
        background_image=None,
        background_image_url=img_display_url,     # FIXED
        height=disp_h + EXTRA_PAD,
        width=disp_w,
        drawing_mode="point",
        key="canvas_measure",
    )

    # calibration
    REF_P1 = (1641, 1242)
    REF_P2 = (1761, 3365)
    REF_LEN_PX = math.hypot(REF_P2[0] - REF_P1[0], REF_P2[1] - REF_P1[1])
    CM_PER_PIXEL = 40.0 / REF_LEN_PX

    def to_original_coords(obj):
        x_s = int(obj["left"])
        y_s = int(obj["top"])
        return (x_s, y_s), (int(x_s / scale_factor), int(y_s / scale_factor))

    if canvas.json_data:
        objs = canvas.json_data.get("objects", [])

        if len(objs) == 1:
            st.info("Click two more points.")
        elif len(objs) == 2:
            st.info("Click one more point.")

        if len(objs) >= 2:
            p1_s, p1 = to_original_coords(objs[0])
            p2_s, p2 = to_original_coords(objs[1])

            dx12 = p2[0] - p1[0]
            dy12 = p2[1] - p1[1]
            pix12 = math.hypot(dx12, dy12)
            len12 = pix12 * CM_PER_PIXEL

            if len(objs) == 2:
                st.success(f"Edge (P1–P2): {len12:.2f} cm")

        if len(objs) >= 3:
            p3_s, p3 = to_original_coords(objs[2])

            dx13 = p3[0] - p1[0]
            dy13 = p3[1] - p1[1]
            pix13 = math.hypot(dx13, dy13)
            len13 = pix13 * CM_PER_PIXEL

            width = min(len12, len13)
            height = max(len12, len13)

            st.success(
                f"**Width:** {width:.2f} cm\n\n"
                f"**Height:** {height:.2f} cm"
            )


# ============================================================
# TAB 2 — POSE + HAND TRACKING
# ============================================================
with tab_pose:
    st.title("Pose & Hand Tracking (MediaPipe)")

    if "pose_log" not in st.session_state:
        st.session_state.pose_log = []
    if "frame_counter" not in st.session_state:
        st.session_state.frame_counter = 0
    if "preview_frames" not in st.session_state:
        st.session_state.preview_frames = []

    st.write(
        "**Upload a short video or use webcam.**\n"
        "For deployment stability: max **300 processed frames**."
    )

    mode = st.radio(
        "Select mode",
        ["Webcam snapshot", "Upload video"],
        horizontal=True,
    )

    # -------------------------------------------------------
    # MODE 1 — WEBCAM
    # -------------------------------------------------------
    if mode == "Webcam snapshot":
        cam = st.camera_input("Capture frame")

        if cam:
            data = np.asarray(bytearray(cam.getvalue()), dtype=np.uint8)
            frame = cv2.imdecode(data, cv2.IMREAD_COLOR)

            st.session_state.frame_counter += 1
            idx = st.session_state.frame_counter

            annotated, rows = analyze_frame_pose_and_hands(frame, idx)
            st.session_state.pose_log.extend(rows)

            safe_display_image(annotated, f"Webcam frame #{idx}")

    # -------------------------------------------------------
    # MODE 2 — VIDEO
    # -------------------------------------------------------
    else:
        video_file = st.file_uploader(
            "Upload video", type=["mp4", "mov", "avi", "mkv"]
        )

        if video_file:
            tmp = Path(ROOT_DIR) / "tmp_video.mp4"
            with open(tmp, "wb") as f:
                f.write(video_file.getbuffer())

            if st.button("Process uploaded video"):
                cap = cv2.VideoCapture(str(tmp))
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
                frame_idx = 0
                progress = st.progress(0.0)
                last_annot = None

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_idx += 1

                    # PROCESS EVERY 2nd FRAME
                    if frame_idx % 2 != 0:
                        continue

                    if frame_idx > 600:  # protects Render free plan
                        break

                    st.session_state.frame_counter += 1
                    idx = st.session_state.frame_counter

                    annotated, rows = analyze_frame_pose_and_hands(frame, idx)
                    st.session_state.pose_log.extend(rows)
                    last_annot = annotated

                    # sample preview
                    if (frame_idx // 2) % 10 == 0:
                        rgb_small = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        st.session_state.preview_frames.append(rgb_small)
                        if len(st.session_state.preview_frames) > 30:
                            st.session_state.preview_frames.pop(0)

                    progress.progress(min(frame_idx / total, 1.0))

                cap.release()
                st.success(f"Processed {frame_idx} frames.")

                if last_annot is not None:
                    safe_display_image(last_annot, "Last processed frame")

                if st.session_state.preview_frames:
                    st.write("### Playback (every 10th processed frame)")
                    cols = st.columns(3)
                    for i, fr in enumerate(st.session_state.preview_frames):
                        with cols[i % 3]:
                            st.image(fr, width=250)

    # -------------------------------------------------------
    # CSV OUTPUT
    # -------------------------------------------------------
    st.subheader("Pose / Hand Tracking Log")

    if st.session_state.pose_log:
        df = pd.DataFrame(st.session_state.pose_log)
        st.dataframe(df.head(100))

        csv_path = os.path.join(OUTPUT_DIR, "pose_hand_tracks.csv")
        df.to_csv(csv_path, index=False)

        with open(csv_path, "rb") as f:
            st.download_button(
                "Download CSV",
                f,
                "pose_hand_tracks.csv",
                mime="text/csv",
            )

        if st.button("Clear log"):
            st.session_state.pose_log = []
            st.session_state.preview_frames = []
            st.session_state.frame_counter = 0
            st.experimental_rerun()
    else:
        st.info("No pose/hand data logged yet.")
