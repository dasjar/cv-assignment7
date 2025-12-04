import cv2
import glob
import os
import re
import numpy as np

# Correct inner-corner pattern for your checkerboard:
# 7 × 10 squares → (10-1) × (7-1) = (9,6)
PATTERN_SIZE = (9, 6)
OUTPUT_DIR = "data/output/corner_checks"

LEFT_PATH  = "data/calib_images/left/*.jpg"
RIGHT_PATH = "data/calib_images/right/*.jpg"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def numeric_sort(path_list):
    """Sort files like 1.jpg → 2.jpg → ... → 10.jpg."""
    def extract_number(f):
        nums = re.findall(r'\d+', os.path.basename(f))
        return int(nums[0]) if nums else -1
    return sorted(path_list, key=extract_number)


def is_rotated(corners):
    """Detect whether corner pattern appears rotated 90 degrees."""
    if corners is None:
        return False
    x_range = abs(corners[0][0][0] - corners[-1][0][0])
    y_range = abs(corners[0][0][1] - corners[-1][0][1])
    return y_range > x_range   # taller than wide = likely rotated


def main():
    left_images = numeric_sort(glob.glob(LEFT_PATH))
    right_images = numeric_sort(glob.glob(RIGHT_PATH))

    print("Left order:", [os.path.basename(f) for f in left_images])
    print("Right order:", [os.path.basename(f) for f in right_images])
    print()

    if len(left_images) != len(right_images):
        print("ERROR: mismatch in number of left/right images")
        return

    print(f"Found {len(left_images)} stereo pairs\n")

    for i, (l_path, r_path) in enumerate(zip(left_images, right_images), start=1):

        img_l = cv2.imread(l_path)
        img_r = cv2.imread(r_path)

        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

        ret_l, corners_l = cv2.findChessboardCorners(gray_l, PATTERN_SIZE)
        ret_r, corners_r = cv2.findChessboardCorners(gray_r, PATTERN_SIZE)

        print(f"Pair {i}: {os.path.basename(l_path)} & {os.path.basename(r_path)}")

        if ret_l and ret_r:
            print("  Both OK")

            if is_rotated(corners_l):
                print("  WARNING: Left image appears rotated")
            if is_rotated(corners_r):
                print("  WARNING: Right image appears rotated")

            # Draw and save annotated corners
            l_draw = img_l.copy()
            r_draw = img_r.copy()

            cv2.drawChessboardCorners(l_draw, PATTERN_SIZE, corners_l, ret_l)
            cv2.drawChessboardCorners(r_draw, PATTERN_SIZE, corners_r, ret_r)

            cv2.imwrite(f"{OUTPUT_DIR}/left_{i}_corners.jpg", l_draw)
            cv2.imwrite(f"{OUTPUT_DIR}/right_{i}_corners.jpg", r_draw)

        else:
            if not ret_l:
                print("  LEFT  → detection FAILED")
            if not ret_r:
                print("  RIGHT → detection FAILED")

    print("\nAnnotated corner images saved to:", OUTPUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
