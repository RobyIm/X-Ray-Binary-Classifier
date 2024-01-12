import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn.functional as F

import hyperparameters as hp

def X_Ray_cnn_test_fn(
    test_loader,
    test_model_path,
    model,
):
    model_path = torch.load(test_model_path, map_location=torch.device('cpu'))
    model.load_state_dict(model_path['model_state_dict'])

    print(f"\nDeploying model on test images\nModel used: {test_model_path}")

    # Evaluate the model on the test set
    correct = 0
    total = 0
        
    model.eval()
        
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(hp.DEVICE), labels.to(hp.DEVICE)
            outputs = torch.sigmoid(model(images))
                
            labels_onehot = F.one_hot(labels, num_classes=hp.NUM_CLASSES).float()
            predicted = torch.round(outputs.squeeze()).long()
            total += labels.size(0)
            predicted_onehot = torch.argmax(predicted, dim=1)
            correct += (predicted_onehot == labels.float()).sum().item()

    test_accuracy = correct / total
    print('Test accuracy:', test_accuracy)