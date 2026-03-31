import h5py
import torch
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch import nn
from tqdm import tqdm
import argparse

# Read features from h5 file
def load_features_from_h5(file_path):
    with h5py.File(file_path, 'r') as f:
        features = np.array(f['features']) 
        coord = np.array(f['coords'])
        return features, coord

# Save new features to h5 file
def save_features_to_h5(file_path, predictions, coord):
    with h5py.File(file_path, 'w') as f:
        f.create_dataset('coords', data=coord)  # Save original coordinates
        f.create_dataset('features', data=predictions)  # Save prediction results as new features

# Define simple MLP model
class SimpleMLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleMLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.fc(x)

# Load trained model
def load_model(model_path, model, device):
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    return model

# Use model for prediction
def predict(model, features, device):
    model.eval()
    with torch.no_grad():
        features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
        outputs = model(features_tensor)
        _, preds = outputs.max(1)  # Get max value index as predicted class
    return preds.cpu().numpy()

# Iterate through folder to process all files
def process_directory(input_dir, output_dir, model_path, device, input_size, num_classes):
    # Load model
    model = SimpleMLP(input_size, num_classes)
    model = load_model(model_path, model, device)
    
    # Iterate through all .h5 files in folder
    for filename in tqdm(os.listdir(input_dir), desc="Predicting", leave=True):
        if filename.endswith(".h5"):
            input_h5_path = os.path.join(input_dir, filename)
            output_h5_path = os.path.join(output_dir, filename)

            # Read features
            features, coord = load_features_from_h5(input_h5_path)

            # Standardize features
            scaler = StandardScaler()
            features = scaler.fit_transform(features)

            # Perform prediction
            predictions = predict(model, features, device)

            # Save prediction results to file
            save_features_to_h5(output_h5_path, predictions, coord)

            print(f"Processed: {filename}, saved predictions to: {output_h5_path}")

# Run code
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Tissue classification prediction script")
    parser.add_argument('--input_dir', type=str,
                        help='Input h5 folder path')
    parser.add_argument('--output_dir', type=str,
                        help='Output prediction results folder path')
    parser.add_argument('--model_path', type=str,
                        help='Trained model file path')
    parser.add_argument('--input_size', type=int, default=1536, help='Feature dimension')
    parser.add_argument('--num_classes', type=int, default=7, help='Number of classes')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Run device (auto/cuda/cpu)')
    args = parser.parse_args()

    # Automatically select device
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Create output folder (if not exists)
    os.makedirs(args.output_dir, exist_ok=True)
    process_directory(args.input_dir, args.output_dir, args.model_path, device, args.input_size, args.num_classes)

