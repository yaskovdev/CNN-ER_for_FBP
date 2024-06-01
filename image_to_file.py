import sys

import cv2
import torch

if __name__ == "__main__":
    index = int(sys.argv[1])
    data, targets, sigma = torch.load("test_fold6.pt")
    cv2.imwrite("output_" + str(index) + ".jpg", data[index].numpy().transpose((1, 2, 0)))
