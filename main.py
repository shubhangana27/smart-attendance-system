import argparse
import os
from datetime import datetime
import cv2
import numpy as np
import pandas as pd


def extract_features(img_gray):
    # Resize to standard feature resolution
    resized = cv2.resize(img_gray, (100, 100))
    # Compute image gradients (Sobel filter representation)
    gx = cv2.Sobel(resized, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(resized, cv2.CV_32F, 0, 1, ksize=3)
    magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    # Normalize feature vector
    norm_mag = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return norm_mag.flatten()


def load_known_profiles(known_dir):
    profiles = {}

    if not os.path.exists(known_dir):
        print(f"[ERROR] Directory '{known_dir}' does not exist.")
        return profiles

    for file_name in os.listdir(known_dir):
        if file_name.lower().endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(known_dir, file_name)
            name = os.path.splitext(file_name)[0]

            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                features = extract_features(img)
                profiles[name] = features

    print(
        f"[INFO] Successfully loaded {len(profiles)} known profile(s) from '{known_dir}'."
    )
    return profiles


def process_attendance(image_path, profiles, output_csv):
    if not profiles:
        print("[ERROR] No known profiles available for matching.")
        return

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[ERROR] Could not read image at '{image_path}'.")
        return

    test_features = extract_features(img)

    best_match = "Unknown"
    min_dist = float("inf")

    # Measure Euclidean distance between feature representations
    for name, known_features in profiles.items():
        dist = np.linalg.norm(test_features - known_features)
        if dist < min_dist:
            min_dist = dist
            best_match = name

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    records = [{"Name": best_match, "Timestamp": timestamp}]

    df = pd.DataFrame(records)
    if os.path.exists(output_csv):
        df.to_csv(output_csv, mode="a", header=False, index=False)
    else:
        df.to_csv(output_csv, index=False)

    print(
        f"[SUCCESS] Identified match: '{best_match}'. Logged to '{output_csv}'."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Smart Attendance System CLI"
    )
    parser.add_argument(
        "--known_dir",
        type=str,
        default="known_faces",
        help="Path to known faces directory",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input image for attendance",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="attendance.csv",
        help="Path to output CSV log file",
    )

    args = parser.parse_args()

    known_profiles = load_known_profiles(args.known_dir)
    process_attendance(args.input, known_profiles, args.output)