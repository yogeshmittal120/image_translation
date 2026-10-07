# Image Translation API

FastAPI service using **AWS Bedrock Claude Sonnet** to detect text in an image, translate it into a requested language, render the translation into the detected regions, and return the translated PNG image.

## Setup

Configure AWS credentials using the normal AWS credential chain (environment variables, AWS profile, IAM role, etc.).

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

Copy `.env.example` to `.env` and configure your Bedrock region and model ID.

```bash
uvicorn app.main:app --reload
```

Swagger: http://127.0.0.1:8000/docs

## AWS Bedrock

The application uses the Bedrock Runtime `converse` API with an image content block and a text prompt.

Required AWS permission:
- `bedrock:InvokeModel` for the selected Claude Sonnet model.

Make sure the selected Claude Sonnet model is enabled/available in your AWS Bedrock account and region.

## API

### POST /translate-image

Multipart form fields:
- `image`: JPEG, PNG, WEBP, or GIF
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
2. Claude Sonnet analyzes the image and identifies readable text regions.
3. Claude returns source text, translated text, and pixel bounding boxes.
4. Pillow covers the original text region and renders the translation.
5. The generated PNG is returned to the client.

## Environment

- `AWS_REGION`: Bedrock region, e.g. `us-east-1`
- `BEDROCK_MODEL_ID`: your Claude Sonnet model ID
- `MAX_IMAGE_SIZE_MB`: maximum upload size, default 3 MB
- `FONT_PATH`: optional TTF font path for translated text

## Production note

The current renderer uses a white rectangle over each detected text region. This works well for documents and simple backgrounds. For photographs or complex backgrounds, the next step should be background-aware inpainting and font/style matching.
