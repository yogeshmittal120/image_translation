# Image Translation API

FastAPI service using **AWS Textract + AWS Bedrock Claude Sonnet** to detect text, translate it, remove the original text while preserving the background, and render the translated text back into the original locations.

## Architecture

Image -> FastAPI -> AWS Textract OCR -> Claude Sonnet Translation -> OpenCV Inpainting -> Pillow Rendering -> Translated PNG

## Setup

Configure AWS credentials using the normal AWS credential chain (AWS CLI profile, environment variables, EC2/ECS IAM role, etc.).

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

Copy `.env.example` to `.env` and configure the AWS region and Claude Sonnet model ID.

```bash
uvicorn app.main:app --reload
```

Swagger: http://127.0.0.1:8000/docs

## AWS permissions

The AWS identity needs permissions for both services:

- `textract:DetectDocumentText`
- `bedrock:InvokeModel`

Example policy:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "textract:DetectDocumentText",
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": "bedrock:InvokeModel",
      "Resource": "*"
    }
  ]
}
```

Make sure the selected Claude Sonnet model is enabled/available in your Bedrock account and region.

## API

### POST /translate-image

Multipart form fields:

- `image`: JPEG, PNG, or WEBP
- `target_language`: e.g. Hindi, French, Spanish

The API returns a PNG image with translated text rendered in the original text locations.

### cURL

```bash
curl -X POST "http://127.0.0.1:8000/translate-image" ^
  -F "image=@sample.png" ^
  -F "target_language=English" ^
  --output translated.png
```

## How it works

1. FastAPI receives the image.
2. AWS Textract extracts text **lines and their exact bounding boxes**.
3. Claude Sonnet translates the extracted text. Claude does not determine coordinates.
4. OpenCV inpaints the detected text areas to preserve the surrounding background.
5. Pillow renders each translated line inside its original bounding box.
6. The translated PNG is returned.

## Environment

- `AWS_REGION`: AWS region, e.g. `us-east-1`
- `BEDROCK_MODEL_ID`: exact Claude Sonnet model ID enabled in your Bedrock account
- `MAX_IMAGE_SIZE_MB`: maximum upload size, default 5 MB
- `FONT_PATH`: optional TTF font path for translated text

## Notes

For course screenshots, slides, and documents this pipeline is more reliable than asking the LLM to detect coordinates.

The renderer currently uses OpenCV inpainting to preserve the original background. Font size is automatically reduced when a translation is longer than the source region.
