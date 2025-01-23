from pathlib import Path
import os
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

from fall_detection.models.tcn import TemporalConvNet
from fall_detection.scripts.data_loader import FallDetectionDataset
from torch.utils.data import DataLoader


def evaluate_model(model, data_loader, criterion=None, device=torch.device('cpu')):
    """
    Evaluate the model on the validation set.

    Args:
        model (torch.nn.Module): The model to evaluate.
        data_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        criterion (torch.nn.Module, optional): Loss function. Defaults to None.
        device (torch.device): Device to perform computation (CPU/GPU).

    Returns:
        dict: A dictionary containing validation loss (if criterion is provided),
              metrics (accuracy, precision, recall, F1), and the confusion matrix.
    """
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():  # Disable gradient computation
        for batch in data_loader:
            inputs, labels = batch['keypoints'], batch['label']
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.permute(0, 2, 1)

            # Forward pass
            outputs = model(inputs)

            # Compute loss if criterion is provided
            if criterion:
                loss = criterion(outputs, labels)
                val_loss += loss.item()

            # Predictions and labels
            preds = torch.argmax(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # If criterion is provided, compute average loss
    if criterion:
        val_loss /= len(data_loader)

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Return metrics as a dictionary
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
    }

    # Include loss in results if criterion is provided
    if criterion:
        results['val_loss'] = val_loss

    return results


def evaluate(model_path, dataset_dir, batch_size, split='val'):
    # Prepare the validation data
    data_dir = Path(dataset_dir)
    ADL_dir = data_dir / 'val' / 'ADL'
    seq_len = len(os.listdir(os.path.join(ADL_dir, os.listdir(ADL_dir)[0]))) - 1
    dataset = FallDetectionDataset(data_dir, split=split, seq_len=seq_len, num_keypoints=34)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = TemporalConvNet(num_inputs=34, num_channels=[64, 128, 256]).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode

    eval_metrics = evaluate_model(model=model, data_loader=data_loader, device=device)

    return eval_metrics
