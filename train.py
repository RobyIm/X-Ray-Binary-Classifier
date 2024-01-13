import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

import hyperparameters as hp

def X_Ray_cnn_train_fn(
    train_loader,
    validation_loader,
    model,
    optimizer,
    loss_fn,
    enable_checkpoints: bool,
    save_path,
    is_training=True
):
    
    """
    Training function for the BinaryX_RayCNN model.

    Parameters
    ----------
    train_loader : torch.utils.data.DataLoader
        DataLoader for the training dataset.
    
    validation_loader : torch.utils.data.DataLoader
        DataLoader for the validation dataset.

    model : torch.nn.Module
        Instance of the X-Ray CNN model.

    optimizer : torch.optim.Optimizer
        Optimization algorithm for model parameter updates.

    loss_fn : torch.nn.Module
        Loss function for evaluating the model's performance.

    enable_checkpoints : bool
        Flag to enable saving model checkpoints during training.

    save_path : str
        Path to the directory where model checkpoints and results will be saved.

    is_training : bool, optional
        Flag indicating whether the model is in training mode. Default is True.

    Returns
    -------
    None

    """
    train_accuracies = []
    validation_accuracies = []

    best_loss = float('inf') # Trying to minimize MSE loss
    best_epoch = 1

    for epoch in range(hp.NUM_EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        # Training loop
        for images, labels in train_loader:
            images, labels = images.to(hp.DEVICE), labels.to(hp.DEVICE)

            optimizer.zero_grad()
            outputs = torch.sigmoid(model(images))

            # Compute training losses
            loss = loss_fn(outputs.squeeze(), labels.long())
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

            # Print the range of values in the input tensors
            print(f"Min value: {torch.min(images)}, Max value: {torch.max(images)}")

            # Round the predicted outputs to obtain binary predictions   
            predicted = torch.round(outputs.squeeze()).long()

            # Calculate images classified correctly
            total += labels.size(0)
            predicted_onehot = torch.argmax(predicted, dim=1)
            correct += (predicted_onehot == labels).sum().item()

        # Calculate and training loss and accuracy
        train_loss = train_loss / len(train_loader)
        train_accuracy = correct / total

        # Validation
        model.eval()
        validation_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in validation_loader:
                images, labels = images.to(hp.DEVICE), labels.to(hp.DEVICE)
                outputs = torch.sigmoid(model(images))
                
                # Compute validation losses
                loss = loss_fn(outputs.squeeze(), labels.long())
                validation_loss += loss.item() * images.size(0)

                # Round the predicted outputs to obtain binary predictions 
                predicted = torch.round(outputs.squeeze()).long()

                # Calculate images classified correctly
                total += labels.size(0)
                predicted_onehot = torch.argmax(predicted, dim=1)
                correct += (predicted_onehot == labels).sum().item()

        # Calculate and validation loss and accuracy
        validation_loss = validation_loss / len(validation_loader)
        validation_accuracy = correct / total

        # Print progress after each epoch
        print(f"Epoch {epoch+1}/{hp.NUM_EPOCHS} - Train Loss: {train_loss:.4f} - Train Accuracy: {train_accuracy:.4f} - "
            f"Validation Loss: {validation_loss:.4f} - Validation Accuracy: {validation_accuracy:.4f}")
        train_accuracies.append(train_accuracy)
        validation_accuracies.append(validation_accuracy)
    
        if enable_checkpoints:
            # Save checkpoint
            if (epoch+1) % 5 == 0:
                if not os.path.exists(os.path.join(save_path, 'checkpoint')):
                    os.makedirs(os.path.join(save_path, 'checkpoint'))
                torch.save({
                        'epoch': epoch+1,
                        'model_state_dict': model.state_dict(),
                        'optim_state_dict': optimizer.state_dict(),
                        'train_loss': train_loss,
                        'val_loss': validation_loss},
                    os.path.join(save_path, 'checkpoint', 'checkpoint_{}.pth'.format(epoch+1))
                    )
            
            # Save best model
            if not os.path.exists(os.path.join(save_path, 'best')):
                os.makedirs(os.path.join(save_path, 'best'))
            if validation_loss < best_loss: 
                best_loss = validation_loss
                best_epoch = epoch+1
                torch.save({
                    'best_epoch': best_epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, 
                    os.path.join(save_path, 'best', f'best_model.pth'))
            print(f'Best Epoch: {best_epoch}\tBest Val Loss: {best_loss}\n\n')
        
    # Plot training accuracy and validation accuracy when complete
    plt.plot(range(1, hp.NUM_EPOCHS+1), train_accuracies, 'b', label='Training Accuracy')
    plt.plot(range(1, hp.NUM_EPOCHS+1), validation_accuracies, 'r', label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy after Each Epoch')
    plt.legend()
    plt.show()