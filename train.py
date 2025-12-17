import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os
import glob
import warnings
from scipy.io import arff

warnings.filterwarnings('ignore')

class CNNClassifier(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)
        
        final_length = num_features // 4
        if final_length < 1: final_length = 1
        self.fc = nn.Linear(64 * final_length, num_classes)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)

class LSTMClassifier(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(num_features, 128, 2, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(128, num_classes)
        
    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

class HybridCNNLSTM(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(64, 128, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

def load_iscx_data(data_dir='data/processed'):
    print("Loading ISCX VPN-NonVPN dataset...")
    scenario_paths = {
        'A1': os.path.join(data_dir, 'ScenarioA1', 'Scenario A1-ARFF'),
        'A2': os.path.join(data_dir, 'ScenarioA2', 'Scenario A2-ARFF'),
        'B': os.path.join(data_dir, 'ScenarioB', 'Scenario B-ARFF')
    }
    
    all_data = []
    for scenario_name, scenario_path in scenario_paths.items():
        if not os.path.exists(scenario_path):
            print(f"Warning: Path not found {scenario_path}")
            continue
            
        arff_files = glob.glob(os.path.join(scenario_path, "*.arff"))
        print(f"  Scenario {scenario_name}: Found {len(arff_files)} files")
        
        for arff_file in arff_files:
            try:
                data, meta = arff.loadarff(arff_file)
                df = pd.DataFrame(data)
                df['Scenario'] = scenario_name
                all_data.append(df)
            except Exception as e:
                continue

    if not all_data:
        raise ValueError("No valid data files found. Check data/processed structure.")
    
    df = pd.concat(all_data, ignore_index=True)
    print(f"Total flows loaded: {len(df)}")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns[:8].tolist()
    print(f"Using features: {numeric_cols}")
    
    df['Label'] = (df['Scenario'] == 'A1').astype(int)
    
    X = df[numeric_cols].fillna(0).values
    y = df['Label'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_reshaped, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def train_model(model, train_loader, test_loader, epochs=20):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    train_losses = []
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_loss = train_loss / len(train_loader)
        train_losses.append(avg_loss)
        if epoch % 5 == 0:
            print(f'  Epoch {epoch}: Loss {avg_loss:.4f}')
            
    model.eval()
    preds, true = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy())
            true.extend(y_batch.cpu().numpy())
            
    acc = accuracy_score(true, preds)
    f1 = f1_score(true, preds, average='weighted')
    return acc, f1, train_losses

def main():
    X_train, X_test, y_train, y_test = load_iscx_data()
    
    batch_size = 64
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    num_features = X_train.shape[2]
    num_classes = len(np.unique(y_train))
    
    print(f"Features: {num_features}, Classes: {num_classes}")
    
    models = {
        'CNN': CNNClassifier(num_features, num_classes),
        'LSTM': LSTMClassifier(num_features, num_classes),
        'Hybrid': HybridCNNLSTM(num_features, num_classes)
    }
    
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        acc, f1, losses = train_model(model, train_loader, test_loader)
        results[name] = {'accuracy': acc, 'f1': f1, 'losses': losses}
        print(f"{name} Results -> Accuracy: {acc:.4f}, F1: {f1:.4f}")
        
        if name == 'Hybrid':
            torch.save(model.state_dict(), 'hybrid_model.pth')
            print("Saved hybrid_model.pth")
    
    plt.figure(figsize=(15, 5))
    for i, (name, res) in enumerate(results.items()):
        plt.subplot(1, 3, i+1)
        plt.plot(res['losses'])
        plt.title(f'{name} Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    print("\nSaved training_results.png")
    
    print("\n" + "="*60)
    print(f"{'Model':<15} {'Accuracy':<15} {'F1-Score'}")
    print("-" * 60)
    for name, res in results.items():
        print(f"{name:<15} {res['accuracy']:<15.4f} {res['f1']:<10.4f}")

if __name__ == "__main__":
    main()
    