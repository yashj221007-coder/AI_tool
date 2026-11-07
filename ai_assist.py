import os
import time
from fastapi import FastAPI, HTTPException # pyright: ignore[reportMissingImports]
from pydantic import BaseModel # pyright: ignore[reportMissingImports]
from google import genai
from google.genai import types # pyright: ignore[reportMissingImports]

app = FastAPI()

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

class ReviewInput(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    sentiment: str
    confidence_score: float
    summary_sentence: str

@app.get("/")
def home():
    return {"message": "AI Sentiment Analyzer is running!"}

@app.post("/analyze-sentiment")
async def analyze_sentiment(data: ReviewInput):
    if not data.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    
    # Try up to 3 times if the server is busy
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model='gemini-3.8-flash',
                contents=f"Analyze the sentiment of the following customer review: {data.text}",
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=SentimentResponse,
                ),
            )
            return response.text
        except Exception as e:
            # If it's a server overload error, wait 2 seconds and try again
            if "503" in str(e) and attempt < 2:
                time.sleep(2)
                continue
            raise HTTPException(status_code=500, detail=str(e))
