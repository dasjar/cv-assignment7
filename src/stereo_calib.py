import cv2
import numpy as np
import glob
import argparse
import os

def stereo_calibrate(
    left_pattern="data/calib/left*.jpg",
    right_pattern="data/calib/right*.jpg",
    chessboard_size=(9,6),
    square_size=0.025,  # 2.5 cm squares (change to your measured size)
    save_path="data/calib/stereo_calib.npz"
):
    """
    Performs stereo calibration, rectification, and saves:
    - K1, K2 intrinsics
    - distortion coeffs
    - R, T between cameras
    - rectification transforms
    - Q disparity-to-depth reproject matrix
    """

    print("[i] Starting stereo calibration...")
    left_images = sorted(glob.glob(left_pattern))
    right_images = sorted(glob.glob(right_pattern))

    if len(left_images) != len(right_images):
        raise ValueError("Left and right image counts do not match.")

    # chessboard pattern
    cols, rows = chessboard_size
    pattern_points = np.zeros((rows * cols, 3), np.float32)
    pattern_points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    pattern_points *= square_size

    objpoints = []
    imgpoints_left = []
    imgpoints_right = []

    for lf, rf in zip(left_images, right_images):
        img_l = cv2.imread(lf)
        img_r = cv2.imread(rf)

        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

        found_l, corners_l = cv2.findChessboardCorners(gray_l, (cols, rows))
        found_r, corners_r = cv2.findChessboardCorners(gray_r, (cols, rows))

        if found_l and found_r:
            corners_l = cv2.cornerSubPix(
                gray_l, corners_l, (11,11), (-1,-1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )
            corners_r = cv2.cornerSubPix(
                gray_r, corners_r, (11,11), (-1,-1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )

            objpoints.append(pattern_points)
            imgpoints_left.append(corners_l)
            imgpoints_right.append(corners_r)

    print(f"[i] Collected {len(objpoints)} valid stereo pairs for calibration")

    img_shape = gray_l.shape[::-1]

    # Intrinsics from single camera calibration
    ret_l, K_l, dist_l, _, _ = cv2.calibrateCamera(objpoints, imgpoints_left, img_shape, None, None)
    ret_r, K_r, dist_r, _, _ = cv2.calibrateCamera(objpoints, imgpoints_right, img_shape, None, None)

    print("[i] Single camera calibrations done.")

    # Stereo calibration
    flags = cv2.CALIB_FIX_INTRINSIC
    stereocalib_criteria = (
        cv2.TERM_CRITERIA_MAX_ITER +
        cv2.TERM_CRITERIA_EPS,
        100,
        1e-5
    )

    retval, K_l, dist_l, K_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
        objpoints,
        imgpoints_left,
        imgpoints_right,
        K_l,
        dist_l,
        K_r,
        dist_r,
        img_shape,
        criteria=stereocalib_criteria,
        flags=flags
    )

    print("[i] Stereo calibration complete.")
    print("[i] R:", R)
    print("[i] T:", T)

    # Stereo rectification
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K_l, dist_l,
        K_r, dist_r,
        img_shape,
        R,
        T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=-1
    )

    print("[i] Stereo rectification complete.")

    # Save calibration
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(
        save_path,
        K_l=K_l, dist_l=dist_l,
        K_r=K_r, dist_r=dist_r,
        R=R, T=T,
        R1=R1, R2=R2,
        P1=P1, P2=P2,
        Q=Q
    )

    print(f"[✓] Saved calibration → {save_path}")


if __name__ == "__main__":
    stereo_calibrate()
