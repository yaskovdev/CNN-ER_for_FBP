## Deep Learning Based Face Beauty Prediction via Dynamic Robust Losses and Ensemble Regression

A fork of [this repository](https://github.com/faresbougourzi/CNN-ER_for_FBP).

Facial landmark detection is heavily inspired
by [this repository](https://github.com/Danotsonof/facial-landmark-detection).

### How To Run

1. Install Miniconda from [the official website](https://docs.conda.io/projects/miniconda/en/latest/).
2. Create a new Conda environment using `conda create -n torch-cnn python=3.12`. If you want to first delete previously
   created environment, run `conda deactivate`, then `conda remove -n torch-cnn --all`.
3. Activate the new environment with `conda activate torch-cnn`.
4. If you are using PyCharm, make sure to [pick the Conda environment](https://stackoverflow.com/a/46133678/1862286) you
   have just created.
5. Install the dependencies:
    * Install PyTorch with `conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia`,
      see [this guide](https://pytorch.org/get-started/locally/) for more details.
    * Install OpenCV with `pip3 install opencv-contrib-python`.
    * Install scikit-image with `pip3 install scikit-image`.
    * Install tqdm with `pip3 install tqdm`.
    * Install Matplotlib with `conda install matplotlib`.
    * Install openpyxl with `pip3 install openpyxl`.
6. Download the archive using the Google Drive link
   from [the official README.md](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release). Unzip it to a folder
   called `SCUT-FBP5500_v2.1`.
7. Run `create_dataset.py` specifying the folder that contains the `SCUT-FBP5500_v2.1` folder from the above paragraph
   as the value of the `--data_path` argument. Note that this and below commands assume the 60/40 data splitting (the
   fold is 6).
8. Train the necessary models:
    * `python train_test_FBP.py --model ResneXt --LossFnc MSE --Nepochs 40`
    * `python train_test_FBP.py --model Inception --LossFnc Dy_Huber --Nepochs 40`
    * `python train_test_FBP.py --model REXINCET --LossFnc MSE --Nepochs 40`
    * `python train_test_FBP.py --model REXINCET --LossFnc Dy_ParamSmoothL1 --Nepochs 40`
    * `python train_test_FBP.py --model REXINCET --LossFnc Dy_Huber --Nepochs 40`
    * `python train_test_FBP.py --model REXINCET --LossFnc Dy_Tukey --Nepochs 40`
9. Download https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml
   to `haarcascade_frontalface_alt2.xml`.
10. Download https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml to `LBFmodel.yaml`.
11. Run `infer_from_image_file.py` and pass a full path to an image as a parameter. The image can have an arbitrary
    size. The image must contain exactly one face. The preprocessed image will be saved as `output.png`.
    You can also infer against the test dataset, to do so run `infer_from_test_dataset.py` without parameters.
