import torch
import torch.nn as nn
import yaml

class CNNClassifier(nn.Module):
    def __init__(self, num_features, num_classes, config_path='config.yaml'):
        super().__init__()
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)['models']['cnn']
        
        self.conv1 = nn.Conv1d(1, config['filters'][0], config['kernel_size'])
        self.conv2 = nn.Conv1d(config['filters'][0], config['filters'][1], config['kernel_size'])
        self.conv3 = nn.Conv1d(config['filters'][1], config['filters'][2], config['kernel_size'])
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(config['dropout'])
        self.fc = nn.Linear(config['filters'][2], num_classes)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)

class LSTMClassifier(nn.Module):
    def __init__(self, num_features, num_classes, config_path='config.yaml'):
        super().__init__()
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)['models']['lstm']
        
        self.lstm = nn.LSTM(num_features, config['hidden_size'], 
                           config['num_layers'], batch_first=True, dropout=config['dropout'])
        self.fc = nn.Linear(config['hidden_size'], num_classes)
        
    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

class HybridCNNLSTM(nn.Module):
    def __init__(self, num_features, num_classes, config_path='config.yaml'):
        super().__init__()
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)['models']['hybrid']
        
        self.cnn = nn.Sequential(
            nn.Conv1d(1, config['cnn_filters'][0], 3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(config['cnn_filters'][0], config['cnn_filters'][1], 3),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(config['cnn_filters'][1], config['lstm_hidden'], 
                           batch_first=True, dropout=0.3)
        self.fc = nn.Linear(config['lstm_hidden'], num_classes)
        
    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])
