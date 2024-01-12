from torch.utils.data import DataLoader
from torchvision import transforms, datasets

import hyperparameters as hp

def get_loaders(train_data_dir, validation_data_dir, test_data_dir) :

    data_transforms = transforms.Compose([
        transforms.Resize(hp.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load and augment the training data
    train_dataset = datasets.ImageFolder(train_data_dir, transform=data_transforms)
    train_loader = DataLoader(train_dataset, batch_size=hp.BATCH_SIZE, shuffle=True)

    # Load the validation and test data without data augmentation
    validation_dataset = datasets.ImageFolder(validation_data_dir, transform=data_transforms)
    validation_loader = DataLoader(validation_dataset, batch_size=hp.BATCH_SIZE)

    test_dataset = datasets.ImageFolder(test_data_dir, transform=data_transforms)
    test_loader = DataLoader(test_dataset, batch_size=hp.BATCH_SIZE)
    return train_loader, validation_loader, test_loader