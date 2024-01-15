# Overview
This repository contains a binary X-ray classifier implemented in PyTorch, designed to classify X-ray images into two categories: normal and pneumonia-infected lungs. 
The model utilizes a Convolutional Neural Network (CNN) architecture and is trained on a dataset containing labeled X-ray images of normal and pneumonia cases.

# Prerequisites:
- Python 3.7 or later
- PyTorch
- torch.optim
- torch.nn
- torchvision
- typer
- pathlib

## Install the required dependencies using the following command:
pip install torch torchvision typer

# Usage

## Clone the repository:
https://github.com/RobyIm/binary-xray-classifier.git

## Or type this in the terminal:
git clone https://github.com/RobyIm/binary-xray-classifier.git

## Run the classifier using the provided run.py file:
python run.py run --save_dir /path/to/save/model --model_type X_Ray-cnn --train_model --enable_checkpoints --test_model

## Parameters:
- save_dir: Directory to save the model and checkpoints.
 model_type: Choose the model type (currently only 'X_Ray-cnn' is available).
- train_model: Train the model (default: True).
- enable_checkpoints: Enable saving checkpoints during training (default: True).
- test_model: Test the model on the test set (default: True).
- test_model_path: Path to a pre-trained model for testing. If not provided, the latest trained model will be used.

# Model Architecture
The model architecture is defined in BinaryX_RayCNN.py. It is a Convolutional Neural Network designed for binary classification of X-ray images.

# Training and Testing
Training is performed using the X_Ray_cnn_train_fn function, and testing is performed using the X_Ray_cnn_test_fn function. Checkpoints are saved 
during training in the specified save_dir.

# Citation

If you use the provided dataset, please cite the original source:
Kermany, Daniel; Zhang, Kang; Goldbaum, Michael (2018), “Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification”, 
Mendeley Data, V2, doi: 10.17632/rscbjbr9sj.2
