import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from model import AudioUpscaleModel

# === НАСТРОЙКИ DEEP RESTORE ===
TARGET_SR = 48000      # Итог: 48 кГц
INPUT_SR = 22000       # Вход: 22 кГц (очень глухо)
BATCH_SIZE = 8         # Если слабая видюха или мало видеопамяти - понижай
NUM_EPOCHS = 15        # Эпохи
LEARNING_RATE = 1e-4
MODEL_SAVE_PATH = "model_restore_beta.pth"
TRAIN_DIR = "/home/user/Рабочий стол/ResonAI/train/" # Путь к датасету

class RestoreDegrader:
    """Имитация 22kHz, 16bit, ~92kbps (плохое качество)"""
    def __init__(self, device):
        self.down = T.Resample(TARGET_SR, INPUT_SR).to(device)
        self.up = T.Resample(INPUT_SR, TARGET_SR).to(device)
    
    def __call__(self, waveform):
        # 1. Жестко режем частоту до 22 кГц
        degraded = self.down(waveform)
        
        # 2. 16 бит (тут можно было бы и 12, но ты просил 16)
        scale = 32767.0
        degraded = torch.round(degraded * scale) / scale
        
        # 3. Имитация 92 кбит/с (СИЛЬНЫЙ шум)
        # Уровень шума 0.008-0.01 создает эффект "песка" как в плохом MP3
        noise = torch.randn_like(degraded) * 0.008 
        degraded = degraded + noise
        
        return self.up(degraded)

class RestoreDataset(Dataset):
    def __init__(self, root_dir):
        self.filepaths = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith((".flac", ".wav", ".mp3")):
                    self.filepaths.append(os.path.join(root, file))

    def __len__(self): return len(self.filepaths) * 3

    def __getitem__(self, idx):
        filepath = self.filepaths[idx % len(self.filepaths)]
        try:
            waveform, sr = torchaudio.load(filepath)
        except: return torch.zeros(2, TARGET_SR*3)

        if sr != TARGET_SR:
            resampler = T.Resample(sr, TARGET_SR)
            waveform = resampler(waveform)

        if waveform.shape[0] == 1: waveform = waveform.repeat(2, 1)
        waveform = waveform[:2, :]

        # Берем кусок 3 секунды для лучшего контекста
        chunk_len = TARGET_SR * 3
        if waveform.shape[1] < chunk_len:
            waveform = torch.nn.functional.pad(waveform, (0, chunk_len - waveform.shape[1]))
        else:
            start = torch.randint(0, waveform.shape[1] - chunk_len, (1,)).item()
            waveform = waveform[:, start : start + chunk_len]

        return waveform

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training Deep Restore (22k->48k) on {device}")
    
    degrader = RestoreDegrader(device)
    dataset = RestoreDataset(TRAIN_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    model = AudioUpscaleModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler()
    criterion = nn.L1Loss()

    for epoch in range(NUM_EPOCHS):
        model.train()
        for i, target_cpu in enumerate(dataloader):
            target = target_cpu.to(device)
            
            with torch.no_grad():
                inp = degrader(target)
            
            optimizer.zero_grad()
            with autocast():
                output = model(inp)
                loss = criterion(output, target)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if i % 10 == 0: print(f"[Ep {epoch+1}] Step {i}, Loss: {loss.item():.5f}")

        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"💾 Saved {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()
