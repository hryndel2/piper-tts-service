from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import io
import wave
from piper import PiperVoice
import os
import logging
import requests
from pathlib import Path

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Глобальная переменная для голоса
voice = None

def download_voice_model():
    """Скачивает голосовую модель с Hugging Face"""
    model_name = "ru_RU-irina-medium"
    model_path = f"{model_name}.onnx"
    config_path = f"{model_name}.onnx.json"
    
    # Hugging Face URL для модели
    model_url = f"https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/irina/medium/{model_name}.onnx"
    config_url = f"https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/irina/medium/{model_name}.onnx.json"
    
    # Проверяем, существует ли уже модель
    if os.path.exists(model_path) and os.path.exists(config_path):
        logger.info("Голосовая модель уже существует")
        return True
    
    try:
        logger.info("Скачиваем русскую голосовую модель с Hugging Face...")
        
        # Скачиваем модель
        logger.info("Скачиваем модель...")
        response = requests.get(model_url, stream=True)
        response.raise_for_status()
        
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Скачиваем конфиг
        logger.info("Скачиваем конфигурацию...")
        response = requests.get(config_url)
        response.raise_for_status()
        
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        # Проверяем размер файла
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # в MB
        logger.info(f"✓ Модель скачана успешно. Размер: {file_size:.1f} MB")
        return True
        
    except Exception as e:
        logger.error(f"✗ Ошибка скачивания голосовой модели: {e}")
        
        # Удаляем частично скачанные файлы
        for file_path in [model_path, config_path]:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Удален поврежденный файл: {file_path}")
        
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global voice
    try:
        model_path = "ru_RU-irina-medium.onnx"
        
        # Скачиваем голос если нужно
        if not download_voice_model():
            raise Exception("Не удалось скачать голосовую модель с Hugging Face")
        
        logger.info("Загружаем голос...")
        voice = PiperVoice.load(model_path, use_cuda=False)
        logger.info("✓ Голос успешно загружен")
        
    except Exception as e:
        logger.error(f"✗ Ошибка загрузки голоса: {e}")
        # Не прерываем запуск, но логируем ошибку
    
    yield
    # Shutdown

app = FastAPI(title="Piper TTS Service", lifespan=lifespan)

# Разрешаем все CORS запросы
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "status": "Piper TTS Service is running",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if voice else "unhealthy",
        "service": "piper-tts",
        "voice_loaded": voice is not None
    }

@app.get("/synthesize")
async def synthesize_text(
    text: str = Query(..., description="Текст для синтеза")
):
    """
    Синтез речи из текста
    """
    if not voice:
        raise HTTPException(status_code=503, detail="Голос не загружен")
    
    if not text or len(text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Текст не может быть пустым")
    
    try:
        # Ограничиваем длину текста для безопасности
        if len(text) > 1000:
            text = text[:1000] + "..."
        
        # Создаем аудио в памяти
        wav_io = io.BytesIO()
        with wave.open(wav_io, 'wb') as wav_file:
            voice.synthesize_wav(text, wav_file)
        
        wav_io.seek(0)
        audio_data = wav_io.getvalue()
        
        logger.info(f"Синтезирован текст: {text[:50]}...")
        
        # Возвращаем аудио как поток
        return StreamingResponse(
            io.BytesIO(audio_data),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "inline; filename=speech.wav",
                "Access-Control-Expose-Headers": "*"
            }
        )
        
    except Exception as e:
        logger.error(f"Ошибка синтеза: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка синтеза: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
