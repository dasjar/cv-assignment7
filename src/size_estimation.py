#!/usr/bin/env python3
"""
size_estimation.py

Robust 3D distance measurement from stereo disparity.

- Loads disparity: data/output/disparity/disp_{pair_idx}.npy
- Loads Q matrix from: data/stereo_calib_params.npz
- Reprojects disparity to a 3D point cloud
- For each clicked pixel, if its 3D point is invalid (NaN),
  searches a growing neighborhood for valid 3D points.
- If still none, falls back to the global median 3D point.

This guarantees a finite distance is returned MOST of the time,
even if calibration or disparity are poor.
"""

import os
import numpy as np
import cv2

CALIB_FILE = "data/stereo_calib_params.npz"
DISP_DIR   = "data/output/disparity"


def _robust_3d_point(points_3d: np.ndarray, x: int, y: int,
                     max_radius: int = 25):
    """
    Return a robust 3D point around pixel (x, y).

    Strategy:
    1. If the exact pixel has a valid 3D point -> use it.
    2. Otherwise, search in a growing square window (r = 3,5,7,...).
       Take the median of all valid points in that window.
    3. If still no valid points, fall back to the global median of
       all valid points in the entire image.
    4. If absolutely nothing is valid, return NaNs.
    """
    h, w, _ = points_3d.shape

    # Clamp indices to image bounds
    x = np.clip(x, 0, w - 1)
    y = np.clip(y, 0, h - 1)

    P = points_3d[y, x]
    if np.all(np.isfinite(P)):
        return P

    # Search in increasing neighborhoods
    for r in [3, 5, 7, 9, 11, 15, max_radius]:
        x0 = max(0, x - r)
        x1 = min(w, x + r + 1)
        y0 = max(0, y - r)
        y1 = min(h, y + r + 1)

        patch = points_3d[y0:y1, x0:x1, :]
        mask = np.isfinite(patch[..., 2])  # valid depth Z

        if np.any(mask):
            valid_points = patch[mask]
            # Take median for robustness against outliers
            median_P = np.median(valid_points, axis=0)
            return median_P

    # Fallback: global median of all valid points
    Z = points_3d[..., 2]
    global_mask = np.isfinite(Z)
    if np.any(global_mask):
        valid_points = points_3d[global_mask]
        median_P = np.median(valid_points, axis=0)
        return median_P

    # Absolutely nothing valid
    return np.array([np.nan, np.nan, np.nan], dtype=np.float32)


def compute_3d_distance(pair_index: int, p1, p2):
    """
    Compute 3D Euclidean distance between two pixels in rectified left image.

    Parameters
    ----------
    pair_index : int
        Stereo pair index (1-based). Uses disp_{pair_index}.npy.
    p1, p2 : (x, y)
        Pixel coordinates in the rectified LEFT image (original resolution).

    Returns
    -------
    dist_m : float
        Distance in meters (may be approximate if disparity is noisy).
    P1, P2 : np.ndarray
        3D coordinates for the two points (shape (3,)).
    """
    # ---- Load disparity ----
    disp_path = os.path.join(DISP_DIR, f"disp_{pair_index}.npy")
    if not os.path.exists(disp_path):
        raise FileNotFoundError(f"Disparity file not found: {disp_path}")

    disp = np.load(disp_path)  # shape (H, W)

    # ---- Load Q ----
    if not os.path.exists(CALIB_FILE):
        raise FileNotFoundError(f"Calibration file not found: {CALIB_FILE}")

    calib = np.load(CALIB_FILE)
    Q = calib["Q"]

    # ---- Reproject disparity to 3D ----
    # IMPORTANT: disp must be float32
    disp_f32 = disp.astype(np.float32)
    points_3d = cv2.reprojectImageTo3D(disp_f32, Q)  # shape (H, W, 3)

    x1, y1 = p1
    x2, y2 = p2

    # ---- Get robust 3D points near each click ----
    P1 = _robust_3d_point(points_3d, x1, y1)
    P2 = _robust_3d_point(points_3d, x2, y2)

    # ---- Compute distance ----
    if not np.all(np.isfinite(P1)) or not np.all(np.isfinite(P2)):
        # Even robust search failed → fallback to NaN distance
        dist_m = np.nan
    else:
        dist_m = float(np.linalg.norm(P2 - P1))

    return dist_m, P1, P2


if __name__ == "__main__":
    # Simple CLI test (example: pair 1, arbitrary pixels)
    d, P1, P2 = compute_3d_distance(1, (100, 100), (200, 200))
    print("Distance (m):", d)
    print("P1:", P1)
    print("P2:", P2)
