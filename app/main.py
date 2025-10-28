import os
import io
import wave
import librosa
import soundfile as sf
import asyncio
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types
import nest_asyncio
from dotenv import load_dotenv
from pathlib import Path
import numpy as np
import tempfile
import time
import uuid
import math


from app.models import AudioResponse

# Apply nest_asyncio for async operations
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Gemini Audio Processing API",
    description="API for processing audio files using Google Gemini",
    version="1.0.0"
)

# Create static directory for serving audio files
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize Gemini client
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
print(f"GEMINI_API_KEY: {GEMINI_API_KEY}")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is required")

client = genai.Client(api_key=GEMINI_API_KEY)

# Gemini model configuration
MODEL_NAME = "gemini-2.5-flash-native-audio-preview-09-2025"

config =  types.LiveConnectConfig(
    response_modalities=[
        "AUDIO",
    ],
    media_resolution="MEDIA_RESOLUTION_MEDIUM",
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Puck")
        )
    ),
    context_window_compression=types.ContextWindowCompressionConfig(
        trigger_tokens=25600,
        sliding_window=types.SlidingWindow(target_tokens=12800),
    ),
)

def cleanup_file(file_path: str):
    """Clean up temporary files"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"Error cleaning up file {file_path}: {e}")

@app.get("/")
async def root():
    return {"message": "Gemini Audio Processing API", "status": "running"}


@app.post("/process-audio", response_model=AudioResponse)
async def process_audio(
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(..., description="Audio file to process"),
    sample_rate: int = Form(16000),
    output_sample_rate: int = Form(24000),
):
    """
    Process an audio file using Gemini AI and return the processed audio
    """
    
    # return AudioResponse(
    #         message="Audio processed successfully",
    #         audio_url=f"/static/processed_audio_record_20251028_115826_24000_1761638320_3446674e-5226-4dfc-a21c-aad67dbc3020.wav",
    #         success=True,
    #         file_size="126"  # Add this to your AudioResponse model if needed
    #     )
    


    import tempfile
    
    
    temp_file_path = None
    try:
        print(f"Processing audio: {audio_file.filename} -> {output_sample_rate} Hz")
        print(f"Audio size: {audio_file.size}")
        
        # Validate file type
        allowed_content_types = ['audio/wav', 'audio/x-wav', 'audio/3gpp', 'audio/mpeg', 'audio/mp4', 'audio/aac', 'audio/m4a']
        if audio_file.content_type not in allowed_content_types:
            print(f"Warning: Unexpected content type: {audio_file.content_type}")

        # Read and validate the uploaded file
        file_contents = await audio_file.read()
        if len(file_contents) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        
        # Generate output filename
        input_filename = audio_file.filename or "uploaded_audio"
        output_filename = f"processed_{os.path.splitext(input_filename)[0]}_{output_sample_rate}_{math.floor(time.time())}_{uuid.uuid4()}.wav"
        output_path = Path(f"static/{output_filename}")
        
        # Ensure the directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Output file path: {output_path}")

        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_file.write(file_contents)
            temp_file_path = temp_file.name
        
        original_filename = f"original_{os.path.splitext(input_filename)[0]}_{math.floor(time.time())}_{uuid.uuid4()}.mp4"
        project_root = Path(__file__).parent.parent  # Adjust this based on your structure
        original_path = project_root / "static" / "originals" / original_filename
       
        
        # Ensure directories exist
        original_path.parent.mkdir(parents=True, exist_ok=True)
        with open(original_path, 'wb') as orig_file:
            orig_file.write(file_contents)
        print(f"Original audio saved to: {original_path}")
        
            
        
       
        
        print(f"Temporary file created at: {temp_file_path}")

        # Process audio with Gemini
        async with client.aio.live.connect(model=MODEL_NAME, config=config) as session:
            try:
                # Load audio from temporary file
                y, sr = librosa.load(temp_file_path, sr=sample_rate, mono=True)
                print("sr", sr)
                print("y", y.shape)
                # Convert to bytes in required format
                converted_buffer = io.BytesIO()
                sf.write(converted_buffer, y, sr, format="RAW", subtype="PCM_16")
                converted_buffer.seek(0)
                audio_bytes = converted_buffer.read()
                
            except Exception as e:
                print(f"Error processing audio file: {str(e)}")
                raise HTTPException(
                    status_code=400, 
                    detail=f"Error processing audio file: {str(e)}"
                )

            # Send audio to the model
            await session.send_realtime_input(
                audio=types.Blob(
                    data=audio_bytes, 
                    mime_type=f"audio/pcm;rate={sample_rate}"
                )
            )

            # Collect all audio data first
            audio_data_chunks = []
            
            # Receive streamed audio
            async for response in session.receive():
                if response.data is not None:
                    audio_data_chunks.append(response.data)
                    print(f"Received chunk of {len(response.data)} bytes")

            # Check if we received any audio data
            if not audio_data_chunks:
                raise HTTPException(
                    status_code=500,
                    detail="No audio data received from Gemini"
                )

            # Combine all chunks
            combined_audio_data = b''.join(audio_data_chunks)
            print(f"Total audio data received: {len(combined_audio_data)} bytes")

            # Write to WAV file
            with wave.open(str(output_path), "wb") as wf:
                wf.setnchannels(1)  # mono
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(output_sample_rate)  # output sample rate
                wf.writeframes(combined_audio_data)
            
            # Verify file was created and has content
            if not output_path.exists():
                raise HTTPException(
                    status_code=500,
                    detail="Failed to create output audio file"
                )
            
            file_size = output_path.stat().st_size
            print(f"Output file created: {output_path}, size: {file_size} bytes")
            
            if file_size == 0:
                raise HTTPException(
                    status_code=500,
                    detail="Output audio file is empty"
                )
                
        
        return FileResponse(
            path=str(output_path),
            media_type='audio/wav',
            filename=output_filename,
            headers={
                "Content-Disposition": f"attachment; filename={output_filename}",
                "X-File-Size": str(file_size),
                "X-Audio-Duration": str(file_size / (output_sample_rate * 2))  # Approximate duration
            }
        )
        
        return AudioResponse(
            message="Audio processed successfully",
            audio_url=f"/static/{output_filename}",
            success=True,
            file_size=file_size  # Add this to your AudioResponse model if needed
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in process_audio: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing audio: {str(e)}"
        )
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                # os.remove(temp_file_path)
                print(f"Cleaned up temporary file: {temp_file_path}")
            except Exception as e:
                print(f"Error cleaning up temp file: {e}")

        
@app.get("/download-audio/{filename}")
async def download_audio(filename: str):
    """
    Download processed audio file
    """
    file_path = Path(f"static/{filename}")
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    # Check file size
    file_size = file_path.stat().st_size
    if file_size == 0:
        raise HTTPException(status_code=500, detail="Audio file is empty")
    
    return FileResponse(
        path=str(file_path),
        media_type='audio/wav',
        filename=filename
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Gemini Audio API"}

@app.get("/list-audio-files")
async def list_audio_files():
    """List all available audio files in the static directory"""
    try:
        files = []
        static_dir = Path("static")
        
        if static_dir.exists():
            for file_path in static_dir.glob("*.wav"):
                files.append({
                    "filename": file_path.name,
                    "size": file_path.stat().st_size,
                    "url": f"/static/{file_path.name}"
                })
        
        return {"files": files, "count": len(files)}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error listing files: {str(e)}"
        )