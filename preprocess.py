import sys

import cv2
import numpy as np


def _eyes_centers(landmark):
    centers = []
    for start in [36, 42]:
        eye = landmark[0][start:start + 6]
        centers.append(np.take(eye, [1, 2, 4, 5], axis=0).mean(axis=0))
    return centers


def preprocess(image_path):
    image = cv2.imread(image_path)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    haarcascade = "haarcascade_frontalface_alt2.xml"
    detector = cv2.CascadeClassifier(haarcascade)
    faces = detector.detectMultiScale(image_gray)
    print("Faces:\n", faces)

    landmark_detector = cv2.face.createFacemarkLBF()
    LBFmodel = "LFBmodel.yaml"
    landmark_detector.loadModel(LBFmodel)
    _, landmarks = landmark_detector.fit(image_gray, faces)
    for landmark in landmarks:
        eyes_centers = _eyes_centers(landmark)
        for x, y in landmark[0]:
            cv2.circle(image, (int(x), int(y)), 1, (255, 255, 255), 1)
        for x, y in eyes_centers:
            cv2.circle(image, (int(x), int(y)), 1, (0, 0, 255), 1)
    cv2.imwrite("output.png", image)


if __name__ == "__main__":
    preprocess(sys.argv[1])
