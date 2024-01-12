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

    # Training loop
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

        for images, labels in train_loader:
            images, labels = images.to(hp.DEVICE), labels.to(hp.DEVICE)

            optimizer.zero_grad()
            outputs = torch.sigmoid(model(images))
            #outputs = torch.sigmoid(outputs)
            #outputs = torch.clamp(outputs, 0.0, 1.0)
            # Convert labels to one-hot encoding
            labels_onehot = F.one_hot(labels, num_classes=hp.NUM_CLASSES).float()

            loss = loss_fn(outputs.squeeze(), labels.long())
            loss.backward()
            optimizer.step()
            
            # Print the range of values in the input tensors
            print(f"Min value: {torch.min(images)}, Max value: {torch.max(images)}")

            train_loss += loss.item() * images.size(0)
            predicted = torch.round(outputs.squeeze()).long()
            total += labels.size(0)
            predicted_onehot = torch.argmax(predicted, dim=1)
            #print(f"Predicted size: {predicted.size()}, Labels size: {labels.size()}")
            correct += (predicted_onehot == labels).sum().item()

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
                #outputs = torch.sigmoid(outputs)
                #outputs = torch.clamp(outputs, 0.0, 1.0)
                labels_onehot = F.one_hot(labels, num_classes=hp.NUM_CLASSES).float()
                
                loss = loss_fn(outputs.squeeze(), labels.long())

                validation_loss += loss.item() * images.size(0)
                predicted = torch.round(outputs.squeeze()).long()
                total += labels.size(0)
                predicted_onehot = torch.argmax(predicted, dim=1)
                correct += (predicted_onehot == labels).sum().item()

        validation_loss = validation_loss / len(validation_loader)
        validation_accuracy = correct / total

        # Print progress
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
        
    # Plot training accuracy and validation accuracy
    plt.plot(range(1, hp.NUM_EPOCHS+1), train_accuracies, 'b', label='Training Accuracy')
    plt.plot(range(1, hp.NUM_EPOCHS+1), validation_accuracies, 'r', label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy after Each Epoch')
    plt.legend()
    plt.show()