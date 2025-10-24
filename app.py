from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import io
import wave
import os
import logging
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

voice = None

def download_russian_voice():
    """Скачивает русскую голосовую модель"""
    voice_path = "ru_RU-irina-medium.onnx"
    
    # Проверяем существующий файл
    if os.path.exists(voice_path):
        file_size = os.path.getsize(voice_path)
        if file_size > 50000000:  # > 50MB - нормальный размер
            logger.info(f"✓ Русская модель уже существует ({file_size/1024/1024:.1f} MB)")
            return True
        else:
            logger.warning(f"Файл поврежден ({file_size} bytes), удаляем")
            os.remove(voice_path)
    
    logger.info("Скачиваем русскую голосовую модель...")
    
    # Способ 1: Прямое скачивание
    try:
        import requests
        model_url = "https://github.com/rhasspy/piper/releases/download/voices_2023.11/ru_RU-irina-medium.onnx"
        
        logger.info("Пробуем прямое скачивание...")
        response = requests.get(model_url, stream=True, timeout=300)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        logger.info(f"Размер файла: {total_size/1024/1024:.1f} MB")
        
        with open(voice_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info("✓ Модель скачана напрямую")
        return True
        
    except Exception as e:
        logger.error(f"Прямое скачивание не удалось: {e}")
    
    # Способ 2: Через piper (резервный)
    try:
        logger.info("Пробуем через piper...")
        result = subprocess.run([
            "python", "-m", "piper.download_voices", 
            "ru_RU-irina-medium"
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            logger.info("✓ Модель скачана через piper")
            return True
    except Exception as e:
        logger.error(f"Piper скачивание не удалось: {e}")
    
    return False

def load_voice():
    """Загружает голосовую модель"""
    global voice
    try:
        from piper import PiperVoice
        
        voice_path = "ru_RU-irina-medium.onnx"
        
        if not os.path.exists(voice_path):
            logger.error(f"Файл {voice_path} не существует")
            return None
            
        file_size = os.path.getsize(voice_path)
        logger.info(f"Размер файла модели: {file_size / (1024*1024):.1f} MB")
        
        if file_size < 50000000:  # Меньше 50MB - поврежден
            logger.error(f"Файл слишком маленький, вероятно поврежден")
            return None
        
        logger.info("Загружаем модель...")
        voice = PiperVoice.load(voice_path, use_cuda=False)
        logger.info("✓ Русская голосовая модель загружена!")
        return voice
        
    except Exception as e:
        logger.error(f"Ошибка загрузки модели: {e}")
        return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global voice
    logger.info("Запуск Piper TTS сервиса...")
    
    # Скачиваем модель
    if download_russian_voice():
        # Загружаем модель
        voice = load_voice()
    else:
        logger.error("Не удалось скачать голосовую модель")
    
    yield

app = FastAPI(title="Piper TTS Service", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "Piper TTS Service"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if voice else "unhealthy",
        "voice_loaded": voice is not None
    }

@app.get("/synthesize")
async def synthesize_text(text: str = Query(..., min_length=1, max_length=1000)):
    if not voice:
        raise HTTPException(status_code=503, detail="Голос не загружен")
    
    try:
        wav_io = io.BytesIO()
        with wave.open(wav_io, 'wb') as wav_file:
            voice.synthesize_wav(text, wav_file)
        
        wav_io.seek(0)
        audio_data = wav_io.getvalue()
        
        return StreamingResponse(
            io.BytesIO(audio_data),
            media_type="audio/wav",
            headers={"Content-Disposition": 'inline; filename="speech.wav"'}
        )
        
    except Exception as e:
        logger.error(f"Ошибка синтеза: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка синтеза: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
