#!/usr/bin/env python3
# ============================================================
# CSc 8830 – Assignment 7
# Stereo Object Size Estimator (Tab 1) + Pose & Hand Tracking (Tab 2)
# ============================================================

import os
import sys
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import mediapipe as mp

# --------------------------------------------------------------------
# FIX: Add project root BEFORE importing from src
# --------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.size_estimation import compute_3d_distance  # kept for debugging/info

OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# MEDIAPIPE (POSE + HANDS) SETUP
# ============================================================
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands


def analyze_frame_pose_and_hands(frame_bgr, frame_index: int):
    """
    Run MediaPipe Pose + Hands on a single BGR frame.

    Returns:
        annotated_bgr: BGR image with landmarks drawn
        rows: list[dict] describing each landmark, suitable for CSV.
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
                label = handedness.classification[0].label.lower()  # 'left' or 'right'
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
                            "visibility": 1.0,  # hands do not expose visibility
                        }
                    )

    return annotated, rows


# ============================================================
# STREAMLIT PAGE CONFIG + GLOBAL STYLE
# ============================================================
st.set_page_config(page_title="Stereo + Pose App – Assignment 7", layout="wide")

# Prevent Streamlit from clipping the canvas (from your original code)
st.markdown(
    """
<style>
div.stCanvas { overflow: visible !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# TABS
# ============================================================
tab_stereo, tab_pose = st.tabs(
    ["Stereo Object Size Estimator", "Pose & Hand Tracking"]
)

# ============================================================
# TAB 1 – YOUR ORIGINAL STEREO OBJECT SIZE ESTIMATOR
# (logic kept the same, only wrapped in a tab)
# ============================================================
with tab_stereo:

    st.title("Stereo Object Size Estimator – Click Three Points")

    pair_index = st.number_input(
        "Select Stereo Pair", min_value=1, max_value=50, value=1
    )

    # Load rectified left image
    left_path = f"data/output/rectified/rect_left_{pair_index}.jpg"

    if not os.path.exists(left_path):
        st.error(f"Rectified image not found: {left_path}")
        st.stop()

    # Load original full-resolution image
    img = Image.open(left_path)
    img_w, img_h = img.size

    # -----------------------------------------------------
    # SCALE IMAGE DOWN ONLY FOR DISPLAY
    # (clicks get re-scaled back to original resolution)
    # -----------------------------------------------------
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
        "### Click **three points** on the object:\n"
        "- Point 1: common corner\n"
        "- Point 2 & Point 3: other two corners along edges\n"
        "The shorter distance from Point 1 is reported as *width*, the longer as *height*."
    )

    # Add padding to avoid cropping
    EXTRA_PAD = int(0.15 * disp_h)

    canvas = st_canvas(
        stroke_width=5,
        stroke_color="#ff0000",
        background_image=img_display,
        height=disp_h + EXTRA_PAD,
        width=disp_w,
        drawing_mode="point",
        key="canvas_measure",
    )

    # 2D calibration: pixel length that corresponds to 40 cm in your setup
    # From your example: P1=(1641,1242), P2=(1761,3365)
    REF_P1 = (1641, 1242)
    REF_P2 = (1761, 3365)
    REF_LEN_PX = math.hypot(REF_P2[0] - REF_P1[0], REF_P2[1] - REF_P1[1])
    CM_PER_PIXEL = 40.0 / REF_LEN_PX  # ≈ 0.01881 cm/px

    def to_original_coords(obj):
        """Convert a Fabric.js point object to original image coords."""
        x_s = int(obj["left"])
        y_s = int(obj["top"])
        x = int(x_s / scale_factor)
        y = int(y_s / scale_factor)
        return (x_s, y_s), (x, y)

    if canvas.json_data:
        objs = canvas.json_data.get("objects", [])

        if len(objs) == 1:
            st.info("Click two more points to measure width and height.")
        elif len(objs) == 2:
            st.info("Click one more point (total 3) so we can compute width and height.")

        if len(objs) >= 2:
            # ---- Point 1 and Point 2 ----
            p1_s, p1 = to_original_coords(objs[0])
            p2_s, p2 = to_original_coords(objs[1])

            # distance P1-P2, used if only two points
            dx12 = p2[0] - p1[0]
            dy12 = p2[1] - p1[1]
            pix_12 = math.hypot(dx12, dy12)
            len12_cm = pix_12 * CM_PER_PIXEL

            if len(objs) == 2:
                st.success(f"Single edge length (P1–P2): **{len12_cm:.2f} cm**")

        # If we have 3 or more points, use P1 as common corner and P2, P3 as edges
        if len(objs) >= 3:
            p3_s, p3 = to_original_coords(objs[2])
            st.write("Point 3 (display):", p3_s, "  (original):", p3)

            # distance P1-P3
            dx13 = p3[0] - p1[0]
            dy13 = p3[1] - p1[1]
            pix_13 = math.hypot(dx13, dy13)
            len13_cm = pix_13 * CM_PER_PIXEL

            st.write(f"Edge from P1–P2: {len12_cm:.2f} cm")
            st.write(f"Edge from P1–P3: {len13_cm:.2f} cm")

            # decide which is width vs height
            if len12_cm <= len13_cm:
                width_cm = len12_cm
                height_cm = len13_cm
            else:
                width_cm = len13_cm
                height_cm = len12_cm

            st.success(
                f"Estimated dimensions:\n\n"
                f"- **Width** (shorter): {width_cm:.2f} cm\n"
                f"- **Height** (longer): {height_cm:.2f} cm"
            )

def safe_display_image(bgr_img, caption=""):
    """
    Safely displays an image in Streamlit WITHOUT Axios 400 errors.
    Downscales large frames before converting to RGB.
    """
    max_width = 650
    h, w = bgr_img.shape[:2]

    if w > max_width:
        scale = max_width / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        bgr_img = cv2.resize(bgr_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    st.image(rgb, caption=caption)   # <=== FIXED



# ============================================================
# TAB 2 – POSE + HAND TRACKING WITH CSV LOGGING + FRAME GALLERY
# ============================================================
with tab_pose:
    st.title("Real-Time Pose and Hand Tracking (MediaPipe)")

    # ---- Session state for logs ----
    if "pose_hand_log" not in st.session_state:
        st.session_state.pose_hand_log = []
    if "pose_frame_counter" not in st.session_state:
        st.session_state.pose_frame_counter = 0
    if "pose_preview_frames" not in st.session_state:
        st.session_state.pose_preview_frames = []   # stores sampled annotated frames

    st.markdown(
        """
This tab performs **pose estimation** + **hand tracking** using MediaPipe.
All detected landmarks are logged to a CSV file.

**NEW:** When processing a video, the app now displays **every 10th frame**
as a mini “video playback” gallery.
        """
    )

    mode = st.radio(
        "Select input mode:",
        ["Webcam snapshot (browser)", "Upload video file"],
        horizontal=True
    )

    # ----------------------------------------------------------
    # MODE 1 — Webcam snapshot
    # ----------------------------------------------------------
    if mode == "Webcam snapshot (browser)":
        st.info("Capture webcam snapshots. Each frame is logged.")

        cam_image = st.camera_input("Capture a frame")

        if cam_image is not None:
            file_bytes = np.asarray(bytearray(cam_image.getvalue()), dtype=np.uint8)
            frame_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            st.session_state.pose_frame_counter += 1
            frame_idx = st.session_state.pose_frame_counter

            annotated_bgr, rows = analyze_frame_pose_and_hands(frame_bgr, frame_idx)
            st.session_state.pose_hand_log.extend(rows)

            safe_display_image(annotated_bgr, f"Annotated webcam frame #{frame_idx}")

    # ----------------------------------------------------------
    # MODE 2 — Upload video
    # ----------------------------------------------------------
    else:
        st.info("Upload a video. Every frame is logged. Every 10th frame is displayed.")

        video_file = st.file_uploader(
            "Upload a video file", type=["mp4", "mov", "avi", "mkv"]
        )

        if video_file is not None:

            tmp_video_path = Path(ROOT_DIR) / "tmp_pose_video.mp4"
            with open(tmp_video_path, "wb") as f:
                f.write(video_file.read())

            if st.button("Process uploaded video"):
                cap = cv2.VideoCapture(str(tmp_video_path))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

                frame_idx = 0
                progress = st.progress(0.0)
                last_annotated = None
                st.session_state.pose_preview_frames = []   # RESET gallery

                while True:
                    ret, frame_bgr = cap.read()
                    if not ret:
                        break

                    frame_idx += 1
                    st.session_state.pose_frame_counter += 1
                    global_idx = st.session_state.pose_frame_counter

                    # process with mediapipe
                    annotated_bgr, rows = analyze_frame_pose_and_hands(
                        frame_bgr, global_idx
                    )
                    st.session_state.pose_hand_log.extend(rows)
                    last_annotated = annotated_bgr

                    # sample every 10 frames for gallery
                    if frame_idx % 10 == 0:
                        rgb_small = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                        st.session_state.pose_preview_frames.append(rgb_small)

                    if frame_idx % 5 == 0:
                        progress.progress(min(frame_idx / total_frames, 1.0))

                cap.release()
                progress.progress(1.0)

                st.success(f"Processed {frame_idx} frames.")

                # ---- Show last annotated frame ----
                if last_annotated is not None:
                    safe_display_image(
                        last_annotated,
                        caption="Annotated last frame of video"
                    )

                # ---- Display gallery of every 10th frame ----
                if st.session_state.pose_preview_frames:
                    st.markdown("### Playback Preview (every 10th frame)")
                    cols = st.columns(3)

                    for i, frame_rgb in enumerate(st.session_state.pose_preview_frames):
                        with cols[i % 3]:
                            st.image(frame_rgb, caption=f"Frame {i*10}", width=250)

    # ----------------------------------------------------------
    # CSV LOG SECTION
    # ----------------------------------------------------------
    st.markdown("### Pose + Hand Landmark CSV Log")

    if st.session_state.pose_hand_log:
        df = pd.DataFrame(st.session_state.pose_hand_log)

        st.dataframe(df.head(100))

        csv_path = os.path.join(OUTPUT_DIR, "pose_hand_tracks.csv")
        df.to_csv(csv_path, index=False)

        with open(csv_path, "rb") as f:
            st.download_button(
                "Download CSV",
                f,
                file_name="pose_hand_tracks.csv",
                mime="text/csv"
            )

        if st.button("Clear logged data"):
            st.session_state.pose_hand_log = []
            st.session_state.pose_preview_frames = []
            st.session_state.pose_frame_counter = 0
            st.experimental_rerun()
    else:
        st.info("No data logged yet.")
