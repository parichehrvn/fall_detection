import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
from fall_detection.models.tcn import TemporalConvNet
from fall_detection import FallDetectionDataset
from fall_detection import evaluate_model
import os

def training_loop(model, train_loader, val_loader, criterion, optimizer, epochs, patience, save_path, device):
    """
    Trains the model with early stopping and saves the best model.
    Includes evaluation metrics for validation set.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        criterion: Loss function.
        optimizer: Optimizer.
        epochs (int): Maximum number of epochs.
        patience (int): Early stopping patience.
        save_path (str): Path to save the best model.
        device (str): Set device for loading data and model ("cuda" or "cpu").

    Returns:
        best_model: The model with the best validation performance.
    """
    best_loss = float('inf')
    epochs_no_improve = 0
    best_model = None

    # For visualization
    train_losses, val_losses = [], []
    val_accuracies = []

    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            inputs, labels = batch['keypoints'], batch['label']
            inputs = inputs.permute(0, 2, 1)
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation Phase
        val_metrics = evaluate_model(model, val_loader, criterion, device)
        val_loss = val_metrics['val_loss']
        val_losses.append(val_loss)
        val_accuracies.append(val_metrics['accuracy'])

        print(f"########## Epoch {epoch + 1}/{epochs} #########")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['val_loss']:.4f}")
        print(f"Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Precision: {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f} | F1: {val_metrics['f1']:.4f}")
        print(f"Confusion Matrix:\n{val_metrics['confusion_matrix']}")

        # Check for improvement
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
            best_model = model.state_dict()
            torch.save(best_model, save_path/'best_tcn.pt')  # Save the best model
            print(f"Validation loss improved to {val_loss:.4f}. Saved model.")
        else:
            epochs_no_improve += 1
            # print(f"Validation loss did not improve.")

        # Early stopping
        if epochs_no_improve == patience:
            print(f"Early stopping triggered after {patience} epochs with no improvement.")
            break

    # Evaluate the best model
    print("########## Evaluate the best model ##########")
    model.load_state_dict(torch.load(save_path/'best_tcn.pt'))
    eval_metrics = evaluate_model(model=model, data_loader=val_loader, device=device)
    print(f"Validation Accuracy: {eval_metrics['accuracy']:.4f}")
    print(f"Validation Precision: {eval_metrics['precision']:.4f}")
    print(f"Validation Recall: {eval_metrics['recall']:.4f}")
    print(f"Validation F1: {eval_metrics['f1']:.4f}")
    print(f"Confusion Matrix:\n{eval_metrics['confusion_matrix']}")

    # Plot training progress
    plot_training_progress(train_losses, val_losses, val_accuracies, save_path)

    # return model

def plot_training_progress(train_losses, val_losses, val_accuracies, save_path):
    """Plots the training and validation loss and accuracy curves."""
    plt.figure(figsize=(12, 5))

    # Loss Curve
    plt.plot(train_losses, label="Train Loss", color="blue")
    plt.plot(val_losses, label="Val Loss", color="orange")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig(save_path/'train_val_loss.jpg')
    plt.close()
    # Accuracy Curve
    plt.plot(val_accuracies, label="Val Accuracy", color="green")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.legend()
    plt.savefig(save_path/'val_acc.jpg')
    plt.close()

    # plt.tight_layout()
    # plt.show()

def train(dataset_dir, epochs, batch_size, lr, patience, save_path, num_keypoints=34):
    # Dataset and DataLoader
    dataset_dir = Path(dataset_dir)
    ADL_dir = dataset_dir / 'train' / 'ADL'
    # val_dir = Path(dataset_dir) / 'val'

    Path.mkdir(save_path, parents=True)

    seq_len = len(os.listdir(os.path.join(ADL_dir, os.listdir(ADL_dir)[0]))) - 1

    train_dataset = FallDetectionDataset(dataset_dir, split='train', seq_len=seq_len, num_keypoints=num_keypoints)
    val_dataset = FallDetectionDataset(dataset_dir, split='val', seq_len=seq_len, num_keypoints=num_keypoints)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalConvNet(num_inputs=34, num_channels=[64, 128, 256]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    training_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=epochs,
        patience=patience,
        save_path=save_path,
        device=device
    )


