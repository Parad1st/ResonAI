# backend/model.py
import torch
import torch.nn as nn

class AudioUpscaleModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Модель типа "Autoencoder" или "Refinement Network"
        # Она сохраняет размерность [Length -> Length]
        
        self.enc1 = nn.Conv1d(2, 64, kernel_size=9, padding=4)
        self.enc2 = nn.Conv1d(64, 128, kernel_size=9, padding=4)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
        # "Бутылочное горлышко" для выделения признаков
        self.mid = nn.Conv1d(128, 128, kernel_size=9, padding=4)
        
        self.dec1 = nn.Conv1d(128, 64, kernel_size=9, padding=4)
        self.dec2 = nn.Conv1d(64, 2, kernel_size=9, padding=4) 


    def forward(self, x):
        # x shape: [Batch, 2, Length]
        
        x1 = self.relu(self.enc1(x))
        x2 = self.relu(self.enc2(x1))
        
        xm = self.relu(self.mid(x2))
        
        x3 = self.relu(self.dec1(xm))
        
        # Residual connection: складываем вход с результатом, 
        # чтобы сеть учила только "разницу" (потерянные детали)
        out = self.dec2(x3) + x 
        
        return out