# Audio Transcription and Summarization with FastAPI
 
This is a FastAPI application that accepts audio files, transcribes them using the `whisper-large-v3` model, and generates summaries using a summarization model. The transcription, summary, and timestamps are saved locally on the machine.

## Setup Instructions

1. **Clone the Repository**

    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2. **Create a Virtual Environment**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install Dependencies**

    ```bash
    pip install fastapi uvicorn whisper transformers
    ```

4. **Run the Application**

    ```bash
    uvicorn main:app --reload
    ```

5. **API Endpoint**

    - **POST /transcribe/**

        Upload an audio file to transcribe and summarize.

        **Request:**
        - `file`: The audio file to be transcribed (multipart/form-data)

        **Response:**
        ```json
        {
            "transcription": "Transcribed text here...",
            "summary": "Summary of the transcription...",
            "timestamps": [list of timestamps]
        }
        ```

## File Structure

- `main.py`: The main FastAPI application file.
- `transcriptions/`: Directory where transcriptions, summaries, and timestamps are saved.

## Error Handling

The application includes `try-except` blocks to handle potential errors such as:
- Model loading failures
- File format issues
- Unexpected API responses

## Notes

- Ensure that the `whisper-large-v3` model and the summarization model are correctly downloaded and available.
- The transcriptions, summaries, and timestamps are saved in the `transcriptions/` directory with unique filenames.
