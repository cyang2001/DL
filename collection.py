"""
Author: Chen YANG
Date: 2025-05-27
Description: This file is used to collect the data for the project.
"""

import os
import cv2
import numpy as np
import time
import mediapipe as mp
from utils import mediapipe_detection, draw_styled_landmarks, draw_landmarks, extract_keypoints

def main():
    print("Collecting data...")
    SEQUENCE_LENGTH = 30
    CAPTURE_TIMES = 10
    DATA_PATH = 'MP_Data'

    word = input("Please input the word you want to capture(e.g. hello):").strip().lower()

    word_path = os.path.join(DATA_PATH, word)
    os.makedirs(word_path, exist_ok=True)
            
    existing = sorted([int(f) for f in os.listdir(word_path) if f.isdigit()])
    start_index = existing[-1] + 1 if existing else 0

    cap = cv2.VideoCapture(0)
    with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

        for i in range(CAPTURE_TIMES):
            sequence = []
            collecting = False
            print(f"\nCapturing {word} {start_index+i}th sequence. Press space to start/stop...")

            while True:
                ret, frame = cap.read()
                image, results = mediapipe_detection(frame, holistic)
                draw_styled_landmarks(image, results)

                key = cv2.waitKey(10)
                if key == 32:  # 空格键
                    collecting = not collecting
                    if collecting:
                        print(f"Start capturing {start_index+i}th sequence...")
                        sequence = []
                    else:
                        print(f"End capturing {start_index+i}th sequence.")
                        break

                if collecting:
                    keypoints = extract_keypoints(results)
                    sequence.append(keypoints)
                    cv2.putText(image, f"Recording: Frame {len(sequence)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

                    if len(sequence) >= SEQUENCE_LENGTH:
                        print(f"Automatically completed: {SEQUENCE_LENGTH} frames")
                        break

                cv2.imshow('OpenCV Feed', image)

            save_path = os.path.join(word_path, str(start_index+i))
            os.makedirs(save_path, exist_ok=True)
            for j, frame in enumerate(sequence):
                np.save(os.path.join(save_path, f"{j}.npy"), frame)
            print(f"Data saved to {save_path}")

        cap.release()
        cv2.destroyAllWindows()
        print(f"\nFinished capturing all {CAPTURE_TIMES} sequences of {word}")

if __name__ == "__main__":
    main()
