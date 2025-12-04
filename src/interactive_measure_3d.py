import cv2
import numpy as np
import os

CALIB_FILE = "data/stereo_calib_params.npz"
RECT_DIR   = "data/output/rectified"
DISP_DIR   = "data/output/disparity"

# Choose which pair to use (1..7 in your case)
PAIR_INDEX = 1  # change this as needed


def load_Q():
    calib = np.load(CALIB_FILE)
    return calib["Q"]


def load_rectified_left(pair_index: int):
    path = os.path.join(RECT_DIR, f"rect_left_{pair_index}.jpg")
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read rectified image: {path}")
    return img


def load_disparity(pair_index: int):
    path = os.path.join(DISP_DIR, f"disp_{pair_index}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Disparity file not found: {path}")
    disp = np.load(path)
    return disp


def main():
    Q = load_Q()
    img = load_rectified_left(PAIR_INDEX)
    disp = load_disparity(PAIR_INDEX)

    # Reproject disparity to 3D
    points_3D = cv2.reprojectImageTo3D(disp, Q)

    # Copy of the image for drawing
    draw_img = img.copy()
    clicked_points = []

    def on_mouse(event, x, y, flags, param):
        nonlocal draw_img, clicked_points, points_3D

        if event == cv2.EVENT_LBUTTONDOWN:
            if len(clicked_points) >= 2:
                # ignore extra clicks until reset
                return

            clicked_points.append((x, y))
            # draw a small circle at the clicked point
            cv2.circle(draw_img, (x, y), 5, (0, 0, 255), -1)

            if len(clicked_points) == 2:
                # We have two points: compute 3D distance
                (u1, v1), (u2, v2) = clicked_points

                P1 = points_3D[v1, u1]  # (X1, Y1, Z1)
                P2 = points_3D[v2, u2]  # (X2, Y2, Z2)

                if not np.all(np.isfinite(P1)):
                    print(f"P1 at {(u1, v1)} is invalid (NaN/inf): {P1}")
                    return
                if not np.all(np.isfinite(P2)):
                    print(f"P2 at {(u2, v2)} is invalid (NaN/inf): {P2}")
                    return

                diff = P1 - P2
                dist_m = float(np.linalg.norm(diff))

                # Draw a line between the two points
                cv2.line(draw_img, (u1, v1), (u2, v2), (0, 255, 0), 2)

                # Put text on the image (distance in cm)
                text = f"{dist_m*100:.1f} cm"
                mid_x = int((u1 + u2) / 2)
                mid_y = int((v1 + v2) / 2)
                cv2.putText(
                    draw_img, text,
                    (mid_x, mid_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 0, 0), 2
                )

                print(f"P1 (u={u1}, v={v1}) -> {P1}")
                print(f"P2 (u={u2}, v={v2}) -> {P2}")
                print(f"Distance: {dist_m:.4f} m ({dist_m*100:.2f} cm)")

    cv2.namedWindow("measure_3d", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("measure_3d", on_mouse)

    print("Instructions:")
    print(f"  - Viewing rectified left image for pair {PAIR_INDEX}")
    print("  - Left-click two points along the object edge to measure length")
    print("  - Press 'r' to reset points and measure again")
    print("  - Press 'q' to quit")

    while True:
        cv2.imshow("measure_3d", draw_img)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r'):
            # Reset drawing and clicks
            draw_img = img.copy()
            clicked_points = []
            print("Reset clicks.")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
