import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from model import AudioUpscaleModel

# === НАСТРОЙКИ STANDARD ===
TARGET_SR = 48000      # Итог: 48 кГц
INPUT_SR = 32000       # Вход: 32 кГц
BATCH_SIZE = 16        # Если слабая видюха или мало видеопамяти - понижай
NUM_EPOCHS = 350       # Эпохи
LEARNING_RATE = 1e-4
MODEL_SAVE_PATH = "upscale_model_standard.pth"
TRAIN_DIR = "/home/user/Рабочий стол/ResonAI/train/" # Путь к датасету

class StandardDegrader:
    """Имитация 32kHz, 16bit, ~192kbps"""
    def __init__(self, device):
        self.down = T.Resample(TARGET_SR, INPUT_SR).to(device)
        self.up = T.Resample(INPUT_SR, TARGET_SR).to(device)
    
    def __call__(self, waveform):
        # 1. Режем частоту до 32 кГц
        degraded = self.down(waveform)
        
        # 2. Округляем до 16 бит (CD качество)
        scale = 32767.0 # 2^15 - 1
        degraded = torch.round(degraded * scale) / scale
        
        # 3. Имитация 192 кбит/с (Легкий шум сжатия)
        # Шум небольшой, так как 192кбит качество "под пиво пойдёт"
        noise = torch.randn_like(degraded) * 0.0015 
        degraded = degraded + noise
        
        # 4. Возвращаем размер 48кГц для входа в сеть
        return self.up(degraded)

class StandardDataset(Dataset):
    def __init__(self, root_dir):
        self.filepaths = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith((".flac", ".wav", ".mp3")):
                    self.filepaths.append(os.path.join(root, file))

    def __len__(self): return len(self.filepaths) * 4

    def __getitem__(self, idx):
        filepath = self.filepaths[idx % len(self.filepaths)]
        try:
            waveform, sr = torchaudio.load(filepath)
        except: return torch.zeros(2, TARGET_SR*2), torch.zeros(2, TARGET_SR*2)

        if sr != TARGET_SR:
            resampler = T.Resample(sr, TARGET_SR)
            waveform = resampler(waveform)

        if waveform.shape[0] == 1: waveform = waveform.repeat(2, 1)
        waveform = waveform[:2, :]

        # Чанк 2 секунды
        chunk_len = TARGET_SR * 2
        if waveform.shape[1] < chunk_len:
            waveform = torch.nn.functional.pad(waveform, (0, chunk_len - waveform.shape[1]))
        else:
            start = torch.randint(0, waveform.shape[1] - chunk_len, (1,)).item()
            waveform = waveform[:, start : start + chunk_len]

        return waveform # Возвращаем чистый (target), ломать будем на GPU

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training Standard (32k->48k) on {device}")
    
    degrader = StandardDegrader(device)
    dataset = StandardDataset(TRAIN_DIR)
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