## Deep Learning Based Face Beauty Prediction via Dynamic Robust Losses and Ensemble Regression

1. Install Miniconda from [the official website](https://docs.conda.io/projects/miniconda/en/latest/).
2. Create a new Conda environment using `conda create -n torch-cnn python=3.12`. If you want to first delete previously
   created environment, run `conda deactivate`, then `conda remove -n torch-cnn --all`.
3. Activate the new environment with `conda activate torch-cnn`.
4. If you are using PyCharm, make sure to [pick the Conda environment](https://stackoverflow.com/a/46133678/1862286) you
   have just created.
5. Install PyTorch with `conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia`,
   see [this guide](https://pytorch.org/get-started/locally/) for more details.
6. Install OpenCV with `pip3 install opencv-python`.
7. Install scikit-image with `pip3 install scikit-image`.
8. Install tqdm with `pip3 install tqdm`.
9. Install Matplotlib with `conda install matplotlib`.
10. Install openpyxl with `pip3 install openpyxl`.
11. Download the archive using the Google Drive link
    from [the official README.md](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release). Unzip it to a folder
    called `SCUT-FBP5500_v2.1`.
12. Run `create_dataset.py` specifying the folder that contains the `SCUT-FBP5500_v2.1` folder from the above paragraph
    as the value of the `--data_path` argument.
