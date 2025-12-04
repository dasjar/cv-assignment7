import cv2
import numpy as np
import glob
import os
import re

CALIB_FILE = "data/stereo_calib_params.npz"

RECT_LEFT_PATH  = "data/output/rectified/rect_left_*.jpg"
RECT_RIGHT_PATH = "data/output/rectified/rect_right_*.jpg"

DISP_DIR = "data/output/disparity"
os.makedirs(DISP_DIR, exist_ok=True)


def numeric_sort(paths):
    def extract_number(f):
        nums = re.findall(r"\d+", os.path.basename(f))
        return int(nums[0]) if nums else -1
    return sorted(paths, key=extract_number)


def create_stereo_matcher():
    # Reasonable default SGBM parameters; you can tune later if needed
    min_disp = 0
    num_disp = 128  # must be divisible by 16

    block_size = 5
    P1 = 8 * 3 * block_size ** 2
    P2 = 32 * 3 * block_size ** 2

    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=P1,
        P2=P2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )
    return stereo


def main():
    # ---- load calibration Q matrix ----
    calib = np.load(CALIB_FILE)
    Q = calib["Q"]
    print("[OK] Loaded Q matrix from calibration.")

    # ---- load rectified pairs ----
    left_imgs  = numeric_sort(glob.glob(RECT_LEFT_PATH))
    right_imgs = numeric_sort(glob.glob(RECT_RIGHT_PATH))

    if len(left_imgs) == 0:
        print("No rectified images found.")
        return

    if len(left_imgs) != len(right_imgs):
        print("Rectified left/right count mismatch.")
        return

    print(f"[OK] Found {len(left_imgs)} rectified stereo pairs.\n")

    stereo = create_stereo_matcher()

    for i, (l_path, r_path) in enumerate(zip(left_imgs, right_imgs), start=1):
        imgL = cv2.imread(l_path)
        imgR = cv2.imread(r_path)

        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

        print(f"Computing disparity for pair {i}...")

        # raw disparity (fixed-point 16x)
        disp_raw = stereo.compute(grayL, grayR).astype(np.float32)
        disp = disp_raw / 16.0  # convert to real disparity values

        # mask invalid disparity (<=0)
        disp[disp <= 0] = np.nan

        # save raw disparity map as .npy
        np.save(os.path.join(DISP_DIR, f"disp_{i}.npy"), disp)

        # visualize disparity (for debugging)
        disp_vis = disp.copy()
        # replace nan with 0 for visualization
        disp_vis = np.nan_to_num(disp_vis, nan=0.0)

        disp_vis_norm = cv2.normalize(
            disp_vis, None, alpha=0, beta=255,
            norm_type=cv2.NORM_MINMAX
        )
        disp_vis_norm = disp_vis_norm.astype(np.uint8)
        cv2.imwrite(os.path.join(DISP_DIR, f"disp_{i}_vis.png"), disp_vis_norm)

        # ---- reproject to 3D to obtain depth (Z) ----
        points_3D = cv2.reprojectImageTo3D(disp, Q)
        Z = points_3D[:, :, 2]  # depth in meters

        np.save(os.path.join(DISP_DIR, f"depth_Z_{i}.npy"), Z)

        # simple depth visualization (near = bright)
        Z_vis = Z.copy()
        # ignore very far / invalid depths
        Z_vis[~np.isfinite(Z_vis)] = 0
        Z_vis[Z_vis > 5.0] = 5.0  # clamp at 5 meters for display

        Z_vis_norm = cv2.normalize(
            Z_vis, None, alpha=0, beta=255,
            norm_type=cv2.NORM_MINMAX
        )
        Z_vis_norm = Z_vis_norm.astype(np.uint8)
        cv2.imwrite(os.path.join(DISP_DIR, f"depth_{i}_vis.png"), Z_vis_norm)

        print(f"[OK] Saved disparity and depth for pair {i}")

    print("\n[DONE] Disparity and depth maps saved in", DISP_DIR)


if __name__ == "__main__":
    main()
