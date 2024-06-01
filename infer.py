import os

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from FBP_Dataloader import Beauty_Db, Beauty_Db2im
from models import REXINCET


def _load_res_next_model():
    model = torchvision.models.resnext50_32x4d(pretrained=True)
    model.fc = nn.Linear(2048, 1)
    model.load_state_dict(torch.load(os.path.join("Models", "ResneXt_6_MSE.pt")))
    model.eval()
    return model


def _load_inception_model():
    model = torchvision.models.inception_v3(pretrained=True)
    model.fc = nn.Linear(2048, 1)
    model.load_state_dict(torch.load(os.path.join("Models", "Inception_6_Dy_Huber.pt")))
    model.eval()
    return model


def _load_rex_incet_model(file_name):
    model = REXINCET()
    model.load_state_dict(torch.load(os.path.join("Models", file_name)))
    model.eval()
    return model


def _load_rex_incet_mse_model():
    return _load_rex_incet_model("REXINCET_6_MSE.pt")


def _load_rex_incet_dy_param_smooth_l1_model():
    return _load_rex_incet_model("REXINCET_6_Dy_ParamSmoothL1.pt")


def _load_rex_incet_dy_huber_model():
    return _load_rex_incet_model("REXINCET_6_Dy_Huber.pt")


def _load_rex_incet_tukey_model():
    return _load_rex_incet_model("REXINCET_6_Dy_Tukey.pt")


def _as_model_input(image):
    return torch.from_numpy(np.expand_dims(image.transpose((2, 0, 1)), axis=0)).float()


def infer():
    res_next_model = _load_res_next_model()
    inception_model = _load_inception_model()
    rex_incet_mse_model = _load_rex_incet_mse_model()
    rex_incet_dy_param_smooth_l1_model = _load_rex_incet_dy_param_smooth_l1_model()
    rex_incet_dy_huber_model = _load_rex_incet_dy_huber_model()
    rex_incet_tukey_model = _load_rex_incet_tukey_model()

    test_transform = transforms.Compose([
        transforms.ToPILImage(mode='RGB'),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    test_set = Beauty_Db(root='./', train='test_fold6.pt', transform=test_transform)
    # test_set_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)
    rex_incet_test_transform = transforms.Compose([
        transforms.ToPILImage(mode='RGB'),
        transforms.ToTensor()
    ])
    rex_incet_test_set = Beauty_Db2im(root='./', train='test_fold6.pt', transform=rex_incet_test_transform)
    # rex_incet_test_set_loader = torch.utils.data.DataLoader(rex_incet_test_set, batch_size=1, shuffle=False)

    test_image, test_target, _ = test_set[0]
    test_rex_incet_image_0, test_rex_incet_image_1, test_rex_incet_target, _ = rex_incet_test_set[0]
    # cv2.imwrite("output_test.jpg", test_image)
    # cv2.imwrite("output_test_rex_incet.jpg", test_rex_incet_image_0)

    res_next_model_prediction = res_next_model(torch.unsqueeze(test_image, 0))
    rex_incet_mse_model_prediction = rex_incet_mse_model(torch.unsqueeze(test_rex_incet_image_0, 0), torch.unsqueeze(test_rex_incet_image_1, 0))

    print(res_next_model_prediction, rex_incet_mse_model_prediction)

    pass


if __name__ == "__main__":
    infer()
