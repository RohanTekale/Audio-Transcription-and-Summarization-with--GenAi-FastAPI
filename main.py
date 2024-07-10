# Necessary libraries to be imported
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import os
import uvicorn
import aiofiles
import whisper
from transformers import pipeline
import librosa
import ssl

# created instance for Fastapi
app = FastAPI()

# ensured necessary Directories Created
os.makedirs('uploads', exist_ok=True)
os.makedirs('transcriptions', exist_ok=True)
os.makedirs('summaries', exist_ok=True)
os.makedirs('timestamps', exist_ok=True)

# Create an unverified SSL context
ssl._create_default_https_context = ssl._create_unverified_context

#  Here I have Load models as Large-V3
def load_whisper_model():
    try:
        return whisper.load_model("large-v3")
    except Exception as e:
        raise RuntimeError(f"Failed to load Whisper model: {e}")
# taken proper summarizer
whisper_model = load_whisper_model()
summarizer = pipeline("summarization")

# Extracted Timestamp of audio
def extract_timestamps(audio_path):
    y, sr = librosa.load(audio_path)
    intervals = librosa.effects.split(y, top_db=20)
    timestamps = [(start / sr, end / sr) for start, end in intervals]
    return timestamps

# Created Fastapi endpoint for uploading Audio File
@app.post("/upload-audio/")
async def upload_audio(file: UploadFile = File(...)):
    try:
        file_location = f"uploads/{file.filename}"
        async with aiofiles.open(file_location, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
        return {"info": f"file '{file.filename}' saved at '{file_location}'"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Creted Fastapi endpoints for transcribe audio
@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        file_location = f"uploads/{file.filename}"
        async with aiofiles.open(file_location, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)

        # Transcribe the audio file
        transcription = whisper_model.transcribe(file_location)

        # Save the transcription
        transcription_text = transcription['text']
        transcription_file = f"transcriptions/{file.filename}.txt"
        async with aiofiles.open(transcription_file, 'w') as out_file:
            await out_file.write(transcription_text)

        return {"transcription": transcription_text}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Created fastapi endpoints for summarize the end point
@app.post("/summarize/")
async def summarize_transcription(file: UploadFile = File(...)):
    try:
        file_location = f"uploads/{file.filename}"
        async with aiofiles.open(file_location, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)

        # Transcribe the audio file
        transcription = whisper_model.transcribe(file_location)
        transcription_text = transcription['text']

        # Generate summary
        summary = summarizer(transcription_text, max_length=150, min_length=40, do_sample=False)
        summary_text = summary[0]['summary_text']

        # Save the summary
        summary_file = f"summaries/{file.filename}.txt"
        async with aiofiles.open(summary_file, 'w') as out_file:
            await out_file.write(summary_text)

        return {"summary": summary_text}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Creted Fastapi endpoint for Timestamps
@app.post("/extract-timestamps/")
async def extract_timestamps_endpoint(file: UploadFile = File(...)):
    try:
        file_location = f"uploads/{file.filename}"
        async with aiofiles.open(file_location, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)

        # Extract timestamps
        timestamps = extract_timestamps(file_location)

        # Save the timestamps
        timestamp_file = f"timestamps/{file.filename}.txt"
        async with aiofiles.open(timestamp_file, 'w') as out_file:
            for start, end in timestamps:
                await out_file.write(f"{start:.2f} - {end:.2f}\n")

        return {"timestamps": timestamps}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)


# Created by:
    # Rohan Tekale