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

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Глобальная переменная для голоса
voice = None

def download_voice_model():
    """Скачивает голосовую модель если её нет"""
    voice_path = "ru_RU-irina-medium.onnx"
    
    if not os.path.exists(voice_path):
        logger.info("Скачиваем голосовую модель...")
        try:
            # Скачиваем через piper
            os.system("python -m piper.download_voices en_US-lessac-medium")
            logger.info("✓ Голосовая модель скачана")
        except Exception as e:
            logger.error(f"Ошибка скачивания голоса: {e}")
            return False
    return True

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global voice
    try:
        voice_path = "ru_RU-irina-medium.onnx"
        
        # Скачиваем голос если нужно
        if not download_voice_model():
            raise Exception("Не удалось скачать голосовую модель")
        
        logger.info("Загружаем голос...")
        voice = PiperVoice.load(voice_path, use_cuda=False)
        logger.info("✓ Голос успешно загружен")
        
    except Exception as e:
        logger.error(f"✗ Ошибка загрузки голоса: {e}")
        # Не прерываем запуск, но логируем ошибку
    
    yield
    # Shutdown
    # Cleanup code here

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
@app.post("/synthesize")
async def synthesize_text(
    text: str = Query(..., description="Текст для синтеза"),
    speed: float = Query(1.0, description="Скорость речи (0.5-2.0)"),
    volume: float = Query(1.0, description="Громкость (0.0-2.0)")
):
    """
    Синтез речи из текста - поддерживает GET и POST
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
                "X-Text-Length": str(len(text)),
                "Access-Control-Expose-Headers": "*"
            }
        )
        
    except Exception as e:
        logger.error(f"Ошибка синтеза: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка синтеза: {str(e)}")

@app.get("/synthesize-base64")
async def synthesize_base64(
    text: str = Query(..., description="Текст для синтеза")
):
    """Возвращает аудио в base64"""
    if not voice:
        raise HTTPException(status_code=503, detail="Голос не загружен")
    
    try:
        if not text:
            raise HTTPException(status_code=400, detail="Текст не может быть пустым")
        
        # Создаем аудио в памяти
        wav_io = io.BytesIO()
        with wave.open(wav_io, 'wb') as wav_file:
            voice.synthesize_wav(text, wav_file)
        
        wav_io.seek(0)
        audio_data = wav_io.getvalue()
        
        import base64
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        
        return {
            "audio": f"data:audio/wav;base64,{audio_base64}",
            "text": text,
            "length": len(audio_data)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Для Railway важно использовать порт из переменной окружения
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
