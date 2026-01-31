import io
import os
import torch
import torchaudio
import torchaudio.transforms as T
import tempfile
import math
import urllib.parse
from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from model import AudioUpscaleModel

app = FastAPI()

# === КОНФИГУРАЦИЯ МОДЕЛЕЙ ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir, "frontend"))

# Все модели
MODELS_CONFIG = {
    "standard": {
        "path": os.path.join(BASE_DIR, "upscale_model_standard.pth"),
        "sr": 48000,
        "name": "Standard (48kHz)"
    },
    "audiophile": {
        "path": os.path.join(BASE_DIR, "model_192k_beta.pth"),
        "sr": 192000,
        "name": "Audiophile (192kHz Beta)"
    },
    "restore": {
        "path": os.path.join(BASE_DIR, "model_restore_beta.pth"),
        "sr": 48000,
        "name": "Deep Restore (Beta)"
    }
}

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
templates = Jinja2Templates(directory=FRONTEND_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loaded_models = {} # Кэш загруженных моделей

def get_model(model_key):
    """Загружает модель в память, если она еще не там"""
    if model_key not in MODELS_CONFIG:
        return None, None
        
    cfg = MODELS_CONFIG[model_key]
    
    # Если уже загружена - возвращаем
    if model_key in loaded_models:
        return loaded_models[model_key], cfg["sr"]

    print(f"🔄 Загрузка модели: {cfg['name']}...")
    model = AudioUpscaleModel()
    
    if os.path.exists(cfg["path"]):
        try:
            state = torch.load(cfg["path"], map_location=device)
            model.load_state_dict(state)
            model.to(device)
            model.eval()
            loaded_models[model_key] = model
            print(f"✅ Модель {model_key} успешно загружена")
            return model, cfg["sr"]
        except Exception as e:
            print(f"❌ Ошибка загрузки {model_key}: {e}")
            return None, None
    else:
        print(f"⚠️ Файл весов не найден: {cfg['path']}")
        return None, None

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def process_audio(
    file: UploadFile = File(...), 
    model_type: str = Form("standard") # Получаем выбор из формы
):
    print(f"Запрос на обработку: {file.filename} с моделью {model_type}")
    
    model, target_sr = get_model(model_type)
    
    if not model:
        return {"error": f"Модель '{model_type}' не найдена или файл весов отсутствует. Сначала обучи её!"}

    try:
        contents = await file.read()
        with tempfile.NamedTemporaryFile(suffix=".tmp", delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
        
        try:
            waveform, sr = torchaudio.load(tmp_path)
        finally:
            if os.path.exists(tmp_path): os.unlink(tmp_path)

        # Ресемплинг под целевую частоту модели
        if sr != target_sr:
            resampler = T.Resample(sr, target_sr).to(waveform.device)
            waveform = resampler(waveform)
        
        if waveform.shape[0] == 1: waveform = waveform.repeat(2, 1)
        elif waveform.shape[0] > 2: waveform = waveform[:2, :]
        
        waveform = waveform.to(device)

        # Параметры чанкинга
        CHUNK_SEC = 2 if target_sr > 96000 else 4 # Для 192к берем куски поменьше
        chunk_len = target_sr * CHUNK_SEC
        num_chunks = math.ceil(waveform.shape[1] / chunk_len)
        output_chunks = []
        
        with torch.no_grad():
            for i in range(num_chunks):
                start = i * chunk_len
                end = min(start + chunk_len, waveform.shape[1])
                chunk = waveform[:, start:end].unsqueeze(0)
                
                if chunk.shape[-1] == 0: continue

                original_len = chunk.shape[-1]
                pad_size = 0
                if original_len < 1000: 
                    pad_size = 1000 - original_len
                    chunk = torch.nn.functional.pad(chunk, (0, pad_size))
                
                processed = model(chunk)
                
                if pad_size > 0: processed = processed[..., :original_len]
                output_chunks.append(processed.squeeze(0).cpu())

        full_output = torch.cat(output_chunks, dim=1)
        full_output = torch.clamp(full_output, -1.0, 1.0)

        out_filename = f"Upscaled_{model_type}_{file.filename}.flac"
        encoded_filename = urllib.parse.quote(out_filename)
        
        with tempfile.NamedTemporaryFile(suffix=".flac", delete=False) as out_tmp:
            out_path = out_tmp.name
            
        torchaudio.save(out_path, full_output, target_sr, backend="soundfile", bits_per_sample=24)
        
        with open(out_path, "rb") as f:
            out_data = f.read()
        if os.path.exists(out_path): os.unlink(out_path)

        return StreamingResponse(
            io.BytesIO(out_data),
            media_type="audio/flac",
            headers={"Content-Disposition": f"attachment; filename*=utf-8''{encoded_filename}"}
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)