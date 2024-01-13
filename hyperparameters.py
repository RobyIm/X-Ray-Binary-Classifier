import torch

LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16
NUM_EPOCHS = 10
IMAGE_SIZE = (224, 224)
NUM_CLASSES = 2