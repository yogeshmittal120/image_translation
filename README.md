# Image Translation API

FastAPI service that accepts an image, detects visible text regions, translates them into a requested language, renders the translations back into the original locations, and returns the translated PNG image.

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set `OPENAI_API_KEY`.

```bash
uvicorn app.main:app --reload
```

Swagger: http://127.0.0.1:8000/docs

## API

### POST /translate-image

Multipart form fields:
- `image`: JPEG, PNG, or WEBP
- `target_language`: e.g. Hindi, French, Spanish

The API returns a PNG image with translated text drawn into the detected text regions.

### cURL

```bash
curl -X POST "http://127.0.0.1:8000/translate-image" ^
  -F "image=@sample.png" ^
  -F "target_language=Hindi" ^
  --output translated.png
```

## How it works

1. FastAPI receives the image.
2. The vision model detects readable text and its pixel bounding boxes.
3. The model translates each text region.
4. Pillow covers the original region and renders the translated text.
5. The generated PNG is returned as the API response.

## Notes

The current renderer uses a white rectangle over each detected text region. This works well for documents, screenshots, signs, and simple backgrounds. For complex photographs/backgrounds, the next improvement should be background-aware inpainting plus font/style matching.

For languages that require special fonts, set `FONT_PATH` in `.env` to a compatible TTF font.

## Environment

- `OPENAI_API_KEY`: required
- `OPENAI_MODEL`: vision-capable model to use
- `MAX_IMAGE_SIZE_MB`: maximum upload size, default 10 MB
- `FONT_PATH`: optional TTF font path
