import cv2
import numpy as np
import glob
import os
import re

CALIB_PATH = "data/stereo_calib_params.npz"
LEFT_DIR = "data/stereo_pairs/left/"
RIGHT_DIR = "data/stereo_pairs/right/"
OUT_DIR = "data/output/rectified"
os.makedirs(OUT_DIR, exist_ok=True)


def numeric_sort(paths):
    """Sort file names naturally: 1.jpg, 2.jpg, 10.jpg."""
    def key(f):
        nums = re.findall(r"\d+", os.path.basename(f))
        return int(nums[0]) if nums else -1
    return sorted(paths, key=key)


def load_pairs():
    left = numeric_sort(glob.glob(LEFT_DIR + "*.jpg"))
    right = numeric_sort(glob.glob(RIGHT_DIR + "*.jpg"))
    if len(left) != len(right):
        raise ValueError("Left/right count mismatch")
    print(f"Found {len(left)} stereo pairs.")
    return left, right


def crop_to_valid(img):
    """Crop away pure black borders."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray > 5
    coords = cv2.findNonZero(mask.astype(np.uint8))
    x, y, w, h = cv2.boundingRect(coords)
    return img[y:y+h, x:x+w]


def main():
    # Load calibration
    data = np.load(CALIB_PATH)
    K_l, D_l = data["K_l"], data["D_l"]
    K_r, D_r = data["K_r"], data["D_r"]
    R, T = data["R"], data["T"]

    left_list, right_list = load_pairs()

    # Use real image size
    sample = cv2.imread(left_list[0])
    H, W = sample.shape[:2]
    img_size = (W, H)

    print("Image size:", img_size)

    # Rectify using alpha=1 first
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K_l, D_l, K_r, D_r,
        img_size,
        R, T,
        alpha=1,
        flags=cv2.CALIB_ZERO_DISPARITY
    )

    # Build remap matrices
    map1x, map1y = cv2.initUndistortRectifyMap(
        K_l, D_l, R1, P1, img_size, cv2.CV_32FC1
    )
    map2x, map2y = cv2.initUndistortRectifyMap(
        K_r, D_r, R2, P2, img_size, cv2.CV_32FC1
    )

    for idx, (l_path, r_path) in enumerate(zip(left_list, right_list), 1):
        imgL = cv2.imread(l_path)
        imgR = cv2.imread(r_path)

        # Apply rectification (may cause borders)
        rL = cv2.remap(imgL, map1x, map1y, cv2.INTER_LINEAR)
        rR = cv2.remap(imgR, map2x, map2y, cv2.INTER_LINEAR)

        # Crop away black borders
        rL_c = crop_to_valid(rL)
        rR_c = crop_to_valid(rR)

        # Resize back to original (ensures full-size output)
        rL_final = cv2.resize(rL_c, (W, H), interpolation=cv2.INTER_LINEAR)
        rR_final = cv2.resize(rR_c, (W, H), interpolation=cv2.INTER_LINEAR)

        cv2.imwrite(f"{OUT_DIR}/rect_left_{idx}.jpg", rL_final)
        cv2.imwrite(f"{OUT_DIR}/rect_right_{idx}.jpg", rR_final)

        print(f"[OK] Rectified pair {idx} saved (cropped + restored).")

    print("\nDONE! Full-size rectified images WITHOUT black borders saved to:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
