import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import preprocess
from models import REXINCET


def _res_next_model():
    model = torchvision.models.resnext50_32x4d(pretrained=True)
    model.fc = nn.Linear(2048, 1)
    model.load_state_dict(torch.load(os.path.join("Models", "ResneXt_6_MSE.pt")))
    model.eval()
    return model


def _inception_model():
    model = torchvision.models.inception_v3(pretrained=True)
    model.fc = nn.Linear(2048, 1)
    model.load_state_dict(torch.load(os.path.join("Models", "Inception_6_Dy_Huber.pt")))
    model.eval()
    return model


def _rex_incet_model(file_name):
    model = REXINCET()
    model.load_state_dict(torch.load(os.path.join("Models", file_name)))
    model.eval()
    return model


def _rex_incet_mse_model():
    return _rex_incet_model("REXINCET_6_MSE.pt")


def _rex_incet_dy_param_smooth_l1_model():
    return _rex_incet_model("REXINCET_6_Dy_ParamSmoothL1.pt")


def _rex_incet_dy_huber_model():
    return _rex_incet_model("REXINCET_6_Dy_Huber.pt")


def _rex_incet_tukey_model():
    return _rex_incet_model("REXINCET_6_Dy_Tukey.pt")


def infer(image_path):
    res_next_model = _res_next_model()
    inception_model = _inception_model()
    rex_incet_models = [_rex_incet_mse_model(), _rex_incet_dy_param_smooth_l1_model(), _rex_incet_dy_huber_model(), _rex_incet_tukey_model()]

    res_next_transform = transforms.Compose([
        transforms.ToPILImage(mode='RGB'),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    inception_transform = transforms.Compose([
        transforms.ToPILImage(mode='RGB'),
        transforms.Resize((299, 299)),
        transforms.ToTensor()
    ])

    rex_incet_transform = transforms.Compose([
        transforms.ToPILImage(mode='RGB'),
        transforms.ToTensor()
    ])

    image = cv2.imread(image_path)
    aligned_image = preprocess.preprocess(image)
    cv2.imwrite("output.png", aligned_image)
    test_res_next_image = res_next_transform(np.array(aligned_image).astype(np.uint8))
    test_inception_image = inception_transform(np.array(aligned_image).astype(np.uint8))
    test_rex_incet_image_0 = rex_incet_transform(cv2.resize(np.array(aligned_image), (224, 224)).astype(np.uint8))
    test_rex_incet_image_1 = rex_incet_transform(cv2.resize(np.array(aligned_image), (299, 299)).astype(np.uint8))
    res_next_model_prediction = res_next_model(torch.unsqueeze(test_res_next_image, 0))
    inception_model_prediction = inception_model(torch.unsqueeze(test_inception_image, 0))
    res_incet_model_predictions = [model(torch.unsqueeze(test_rex_incet_image_0, 0), torch.unsqueeze(test_rex_incet_image_1, 0)) for model in
                                   rex_incet_models]
    predictions = [torch.squeeze(p).item() for p in [res_next_model_prediction, inception_model_prediction] + res_incet_model_predictions]
    prediction = np.mean(predictions)

    print("Predicted value is", prediction)


if __name__ == "__main__":
    infer(sys.argv[1])
