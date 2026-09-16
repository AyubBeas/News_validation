import os
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Import agent and logic functions from main.py
import main

app = FastAPI(title="Audio Validator Web Suite")

# API: Read Knowledge Base
@app.get("/api/kb")
def get_kb():
    kb_content = main.get_kb_text()
    return {"kb": kb_content}

class KbUpdate(BaseModel):
    kb: str

# API: Update Knowledge Base
@app.post("/api/kb")
def post_kb(update: KbUpdate):
    main.update_kb_text(update.kb)
    return {"status": "success"}

# API: Read Masked Keys Configuration
@app.get("/api/keys")
def get_keys():
    return main.get_api_keys()

class KeysUpdate(BaseModel):
    active_provider: str
    gemini_key: str = None
    groq_key: str = None
    openai_key: str = None
    tavily_key: str = None

# API: Update API Configuration
@app.post("/api/keys")
def post_keys(update: KeysUpdate):
    main.update_api_keys({
        "active_provider": update.active_provider,
        "gemini_key": update.gemini_key,
        "groq_key": update.groq_key,
        "openai_key": update.openai_key,
        "tavily_key": update.tavily_key
    })
    return {"status": "success"}

# API: History Logs
@app.get("/api/history")
def get_history():
    return main.get_history_logs()

# API: Serve Audio File
@app.get("/api/audio/{filename}")
def get_audio(filename: str):
    audio_path = main.get_audio_filepath(filename)
    if not audio_path or not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="Audio segment not found")
        
    mime_type = "audio/webm"
    if filename.endswith(".ogg"):
        mime_type = "audio/ogg"
    elif filename.endswith(".wav"):
        mime_type = "audio/wav"
        
    return FileResponse(audio_path, media_type=mime_type)

# API: Validate Segment
@app.post("/api/validate")
async def validate_audio(audio: UploadFile = File(...)):
    try:
        # Pass the uploaded file stream and filename to the core agent logic
        result = await main.process_and_validate_uploaded_audio(audio)
        return result
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Serve static files at root paths specifically to avoid WebSocket routing conflicts
@app.get("/")
def read_root():
    return FileResponse("static/index.html")

@app.get("/style.css")
def get_css():
    return FileResponse("static/style.css", media_type="text/css")

@app.get("/app.js")
def get_js():
    return FileResponse("static/app.js", media_type="application/javascript")
