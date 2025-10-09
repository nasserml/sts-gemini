from pydantic import BaseModel
from typing import Optional

class AudioResponse(BaseModel):
    message: str
    audio_url: Optional[str] = None
    success: bool

class ProcessRequest(BaseModel):
    sample_rate: int = 16000
    output_sample_rate: int = 24000