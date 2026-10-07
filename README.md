# Image Translation API

FastAPI service that accepts an image, extracts text from it, translates the text into a requested language, and returns both source and translated text.

## Setup

python -m venv .venv

Install: pip install -r requirements.txt

Copy .env.example to .env and set OPENAI_API_KEY.

Run: uvicorn app.main:app --reload

Swagger: http://127.0.0.1:8000/docs

## API

POST /translate-image

Form fields: image, target_language

Example response:

{
  "success": true,
  "filename": "sample.png",
  "target_language": "Hindi",
  "source_text": "Hello, how are you?",
  "translated_text": "नमस्ते, आप कैसे हैं?"
}

This version returns translated text. It does not draw the translation back onto the image.