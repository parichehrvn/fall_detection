import torch
from fall_detection.models.tcn import TemporalConvNet


def load_tcn_model(model_path):
    model = TemporalConvNet(num_inputs=34, num_channels=[64, 128, 256])
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def tcn_inference(keypoints_tensor, model, device):
    model.to(device)
    with torch.no_grad():
        keypoints_tensor.to(device)
        outputs = model(keypoints_tensor)
        predicted_label = torch.argmax(outputs, dim=0).item()  # Get the predicted class (ADL or Fall)

    return predicted_label



