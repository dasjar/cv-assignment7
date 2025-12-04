import cv2
import numpy as np
import glob
import os
import re

# ============================================================
# CORRECT CHECKERBOARD PARAMETERS FOR YOUR PRINTED PATTERN
# 7×10 squares  →  inner corners = (10-1) × (7-1) = (9, 6)
# ============================================================
PATTERN_SIZE = (9, 6)        # (cols, rows)
SQUARE_SIZE = 0.015          # 16 mm = 0.016 meters

LEFT_PATH  = "data/calib_images/left/*.jpg"
RIGHT_PATH = "data/calib_images/right/*.jpg"

OUTPUT_FILE = "data/stereo_calib_params.npz"


# ----------------------------
# Numeric sorting for 1.jpg, 2.jpg, ... 10.jpg
# ----------------------------
def numeric_sort(paths):
    def extract_number(f):
        nums = re.findall(r'\d+', os.path.basename(f))
        return int(nums[0]) if nums else -1
    return sorted(paths, key=extract_number)


# ----------------------------
# Generate world 3D checkerboard corner coordinates
# ----------------------------
def generate_object_points():
    objp = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
    objp[:, :2] = np.indices(PATTERN_SIZE).T.reshape(-1, 2)
    objp *= SQUARE_SIZE
    return objp


# ----------------------------
# Detect rotated boards (optional but useful)
# ----------------------------
def is_rotated(corners):
    if corners is None:
        return False
    dx = abs(corners[-1][0][0] - corners[0][0][0])
    dy = abs(corners[-1][0][1] - corners[0][0][1])
    return dy > dx   # Taller than wide = likely rotated


# ----------------------------
# MAIN
# ----------------------------
def main():

    # Load + sort filenames
    left_images = numeric_sort(glob.glob(LEFT_PATH))
    right_images = numeric_sort(glob.glob(RIGHT_PATH))

    print("Left order:",  [os.path.basename(p) for p in left_images])
    print("Right order:", [os.path.basename(p) for p in right_images])
    print()

    if len(left_images) != len(right_images):
        print("ERROR: Left / Right count mismatch!")
        return

    # Prepare data containers
    objpoints = []
    imgpoints_l = []
    imgpoints_r = []
    objp = generate_object_points()

    # -----------------------------------------------------------
    # Detect corners pair-by-pair
    # -----------------------------------------------------------
    for l_path, r_path in zip(left_images, right_images):

        img_l = cv2.imread(l_path)
        img_r = cv2.imread(r_path)

        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

        ret_l, corners_l = cv2.findChessboardCorners(gray_l, PATTERN_SIZE)
        ret_r, corners_r = cv2.findChessboardCorners(gray_r, PATTERN_SIZE)

        if not ret_l or not ret_r:
            print("Corner detection FAILED:", l_path, r_path)
            continue

        # Optional orientation sanity check
        if is_rotated(corners_l):
            print("WARNING: LEFT image appears rotated →", l_path)
        if is_rotated(corners_r):
            print("WARNING: RIGHT image appears rotated →", r_path)

        # Refine corners
        corners_l = cv2.cornerSubPix(
            gray_l, corners_l, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

        corners_r = cv2.cornerSubPix(
            gray_r, corners_r, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

        # Add to lists
        objpoints.append(objp)
        imgpoints_l.append(corners_l)
        imgpoints_r.append(corners_r)

    if len(objpoints) < 3:
        print("ERROR: Too few valid stereo pairs to calibrate.")
        return

    # -----------------------------------------------------------
    # Calibrate each camera individually
    # -----------------------------------------------------------
    print("\nCalibrating LEFT camera…")
    ret_l, K_l, D_l, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints_l, gray_l.shape[::-1], None, None)

    print("Calibrating RIGHT camera…")
    ret_r, K_r, D_r, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints_r, gray_r.shape[::-1], None, None)

    # -----------------------------------------------------------
    # Stereo calibration (extrinsics)
    # -----------------------------------------------------------
    print("\nRunning stereoCalibrate()…")

    flags = cv2.CALIB_FIX_INTRINSIC

    ret_s, K_l, D_l, K_r, D_r, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_l, imgpoints_r,
        K_l, D_l, K_r, D_r,
        gray_l.shape[::-1],
        criteria=(cv2.TERM_CRITERIA_MAX_ITER +
                  cv2.TERM_CRITERIA_EPS, 100, 1e-6),
        flags=flags
    )

    print("\nStereo RMS error:", ret_s)
    print("Baseline (meters):", np.linalg.norm(T))

    # -----------------------------------------------------------
    # Rectification
    # -----------------------------------------------------------
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K_l, D_l, K_r, D_r, gray_l.shape[::-1], R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)

    # -----------------------------------------------------------
    # Save calibration
    # -----------------------------------------------------------
    np.savez(OUTPUT_FILE,
             K_l=K_l, D_l=D_l,
             K_r=K_r, D_r=D_r,
             R=R, T=T, E=E, F=F,
             R1=R1, R2=R2, P1=P1, P2=P2,
             Q=Q)

    print("\nSaved stereo calibration to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
