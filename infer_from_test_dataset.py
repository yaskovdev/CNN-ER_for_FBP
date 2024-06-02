import os

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import ResNeXt50_32X4D_Weights, Inception_V3_Weights

from FBP_Dataloader import Beauty_Db, Beauty_Db2im
from models import REXINCET


def _res_next_model():
    model = torchvision.models.resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.DEFAULT)
    model.fc = nn.Linear(2048, 1)
    model.load_state_dict(torch.load(os.path.join("Models", "ResneXt_6_MSE.pt")))
    model.eval()
    return model


def _inception_model():
    model = torchvision.models.inception_v3(weights=Inception_V3_Weights.DEFAULT)
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


def infer():
    res_next_model = _res_next_model()
    inception_model = _inception_model()
    rex_incet_models = [_rex_incet_mse_model(), _rex_incet_dy_param_smooth_l1_model(), _rex_incet_dy_huber_model(), _rex_incet_tukey_model()]

    res_next_transform = transforms.Compose([
        transforms.ToPILImage(mode='RGB'),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    res_next_test_set = Beauty_Db(root='./', train='test_fold6.pt', transform=res_next_transform)

    inception_transform = transforms.Compose([
        transforms.ToPILImage(mode='RGB'),
        transforms.Resize((299, 299)),
        transforms.ToTensor()
    ])
    inception_test_set = Beauty_Db(root='./', train='test_fold6.pt', transform=inception_transform)

    rex_incet_transform = transforms.Compose([
        transforms.ToPILImage(mode='RGB'),
        transforms.ToTensor()
    ])
    rex_incet_test_set = Beauty_Db2im(root='./', train='test_fold6.pt', transform=rex_incet_transform)

    for test_image_index in range(len(res_next_test_set)):
        test_res_next_image, test_res_next_target, _ = res_next_test_set[test_image_index]
        test_inception_image, test_inception_target, _ = inception_test_set[test_image_index]
        test_rex_incet_image_0, test_rex_incet_image_1, test_rex_incet_target, _ = rex_incet_test_set[test_image_index]

        res_next_model_prediction = res_next_model(torch.unsqueeze(test_res_next_image, 0))
        inception_model_prediction = inception_model(torch.unsqueeze(test_inception_image, 0))
        res_incet_model_predictions = [model(torch.unsqueeze(test_rex_incet_image_0, 0), torch.unsqueeze(test_rex_incet_image_1, 0)) for model in
                                       rex_incet_models]
        predictions = [torch.squeeze(p).item() for p in [res_next_model_prediction, inception_model_prediction] + res_incet_model_predictions]
        prediction = np.mean(predictions)

        print("For image with index", test_image_index, "predicted value is", prediction, ", expected value is", test_res_next_target.item())


if __name__ == "__main__":
    infer()
