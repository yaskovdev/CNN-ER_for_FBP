import numpy as np
import torch
from PIL import Image
import cv2

data, targets, sigma = torch.load("test_fold6.pt")

# image = Image.fromarray(data[11].numpy().transpose((1, 2, 0)).astype(np.uint8), "RGB")
cv2.imwrite("output.jpg", data[139].numpy().transpose((1, 2, 0)))
# image.save("output.jpg")
