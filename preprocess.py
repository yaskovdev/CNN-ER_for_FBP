import sys

import cv2


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
        for x, y in landmark[0]:
            cv2.circle(image, (int(x), int(y)), 1, (255, 255, 255), 1)
    cv2.imwrite("output.png", image)


if __name__ == "__main__":
    preprocess(sys.argv[1])
