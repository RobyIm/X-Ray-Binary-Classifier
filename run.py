import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from datetime import datetime
from enum import Enum
import typer

from pathlib import Path

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from dataset import get_loaders
import hyperparameters as hp
from BinaryX_RayCNN import BinaryX_RayCNN
from train import X_Ray_cnn_train_fn
from test import X_Ray_cnn_test_fn

# Define a Typer application
app = typer.Typer()

# Enum for different model types
class Model(str, Enum):
    BinaryX_RayCNN = 'BinaryX-RayCNN'

# Define a CLI command for running the model
@app.command()
def run(
    save_dir: Path = Path('best_model/'),
    model_type: Model = 'BinaryX-RayCNN',
    train_model: bool = True,
    enable_checkpoints: bool = True,
    test_model: bool = True,
    test_model_path: Path = None,
):
    
    # Create a path for saving the results
    SAVE_PATH = Path(f'{save_dir}/{model_type}/{datetime.now().strftime("%Y-%m-%d_%H%M%S")}')  

    if test_model_path is None:
        test_model_path = Path('C:/Users/robya/Documents/Term Folders/MTE 2B/model_checkpoint-Copy.pth')

    # Set the paths to the dataset
    train_data_dir = './train'
    validation_data_dir = './val'
    test_data_dir = './test'

    # Load the data
    train_loader, validation_loader, test_loader = get_loaders(train_data_dir, validation_data_dir, test_data_dir)
    
    if model_type == 'BinaryX-RayCNN':
        print("hi")
        # Initialize the BinaryX_RayCNN model
        model = BinaryX_RayCNN()

        # Move the model to GPU if available
        model = model.to(hp.DEVICE)

        # Define loss function and optimizer
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())
        if(train_model) :
            # Train the model
            X_Ray_cnn_train_fn(
                train_loader,
                validation_loader,
                model,
                optimizer,
                loss_fn,
                enable_checkpoints,
                SAVE_PATH,
                is_training=True
            )

            # Set test_model_path to the best_model generated from the training, if it exists
            new_model_path = Path(f'{SAVE_PATH}/best/best_model.pth')
            if new_model_path.exists():
                test_model_path = new_model_path

        if(test_model) :
          # Test the model
            X_Ray_cnn_test_fn(
            test_loader,
            test_model_path,
            model,
            )

if __name__ == '__main__':
    app()

    '''
    save a model:
    torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()
    }, 'C:/Users/robya/Documents/Term Folders/MTE 2B/model_checkpoint.pth')

    load a model:
    import torch
    checkpoint = torch.load('C:/Users/robya/Documents/Term Folders/MTE 2B/model_checkpoint.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    '''

    '''
    Data from:
    https://data.mendeley.com/datasets/rscbjbr9sj/2
    Kermany, Daniel; Zhang, Kang; Goldbaum, Michael (2018), “Labeled Optical 
    Coherence Tomography (OCT) and Chest X-Ray Images for Classification”, 
    Mendeley Data, V2, doi: 10.17632/rscbjbr9sj.2
    '''