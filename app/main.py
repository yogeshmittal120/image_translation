import base64
import io
import json
import os

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from openai import OpenAI
from PIL import Image

load_dotenv()

app = FastAPI(title='Image Translation API', version='1.0.0')

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-6-luna')
MAX_IMAGE_SIZE_MB = int(os.getenv('MAX_IMAGE_SIZE_MB', '10'))

if not OPENAI_API_KEY:
    raise RuntimeError('OPENAI_API_KEY is not configured.')

client = OpenAI(api_key=OPENAI_API_KEY)

ALLOWED_CONTENT_TYPES = {'image/jpeg', 'image/png', 'image/webp', 'image/gif'}

@app.get('/health')
async def health():
    return {'status': 'ok'}

@app.post('/translate-image')
async def translate_image(
    image: UploadFile = File(...),
    target_language: str = Form(...),
):
    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail='Unsupported image type. Use JPEG, PNG, WEBP, or GIF.')

    target_language = target_language.strip()
    if not target_language:
        raise HTTPException(status_code=400, detail='target_language is required.')

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail='The uploaded image is empty.')

    if len(image_bytes) > MAX_IMAGE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f'Image size must not exceed {MAX_IMAGE_SIZE_MB} MB.')

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img.verify()
    except Exception as exc:
        raise HTTPException(status_code=400, detail='Invalid image file.') from exc

    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    prompt = f'''Read all human-readable text in this image and translate it into {target_language}.
Preserve meaning, line breaks, names, URLs, and email addresses where appropriate.
If there is no readable text, return empty strings.
Return ONLY valid JSON with keys source_text and translated_text.'''

    try:
        response = client.responses.create(
            model=OPENAI_MODEL,
            input=[{'role': 'user', 'content': [
                {'type': 'input_text', 'text': prompt},
                {'type': 'input_image', 'image_url': f'data:{image.content_type};base64,{image_base64}'},
            ]}],
        )
        result = response.output_text.strip()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f'Translation service failed: {exc}') from exc

    if result.startswith('```'):
        result = result.replace('```json', '', 1).replace('```', '', 1).strip()

    try:
        parsed = json.loads(result)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail='Translation service returned invalid JSON.') from exc

    return {
        'success': True,
        'filename': image.filename,
        'target_language': target_language,
        'source_text': parsed.get('source_text', ''),
        'translated_text': parsed.get('translated_text', ''),
    }