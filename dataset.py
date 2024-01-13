from torch.utils.data import DataLoader
from torchvision import transforms, datasets

import hyperparameters as hp

def get_loaders(train_data_dir, validation_data_dir, test_data_dir) :

    """
    Retrieves DataLoader instances for training, validation, and test datasets.

    Parameters
    ----------
    train_data_dir : str
        Directory path containing the training dataset.
    
    validation_data_dir : str
        Directory path containing the validation dataset.

    test_data_dir : str
        Directory path containing the test dataset.

    Returns
    -------
    train_loader : torch.utils.data.DataLoader
        DataLoader for the training dataset with data augmentation.

    validation_loader : torch.utils.data.DataLoader
        DataLoader for the validation dataset without data augmentation.

    test_loader : torch.utils.data.DataLoader
        DataLoader for the test dataset without data augmentation.

    """
    # Define preprocessing transformations
    preprocessing_transforms = transforms.Compose([
        transforms.Resize(hp.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load and augment the training data
    train_dataset = datasets.ImageFolder(train_data_dir, transform=preprocessing_transforms)
    train_loader = DataLoader(train_dataset, batch_size=hp.BATCH_SIZE, shuffle=True)

    # Load the validation and test data without data augmentation
    validation_dataset = datasets.ImageFolder(validation_data_dir, transform=preprocessing_transforms)
    validation_loader = DataLoader(validation_dataset, batch_size=hp.BATCH_SIZE)

    test_dataset = datasets.ImageFolder(test_data_dir, transform=preprocessing_transforms)
    test_loader = DataLoader(test_dataset, batch_size=hp.BATCH_SIZE)

    return train_loader, validation_loader, test_loader