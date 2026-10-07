import base64
import io
import json
import os
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image, ImageDraw, ImageFont

load_dotenv()

app = FastAPI(title='Image Translation API', version='2.1.0')

AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
BEDROCK_MODEL_ID = os.getenv('BEDROCK_MODEL_ID', 'global.anthropic.claude-sonnet-5')
MAX_IMAGE_SIZE_MB = int(os.getenv('MAX_IMAGE_SIZE_MB', '3'))
FONT_PATH = os.getenv('FONT_PATH', '')

bedrock = boto3.client('bedrock-runtime', region_name=AWS_REGION)
ALLOWED_CONTENT_TYPES = {'image/jpeg', 'image/png', 'image/webp', 'image/gif'}

def get_font(size: int):
    candidates = []
    if FONT_PATH: candidates.append(FONT_PATH)
    candidates += [str(Path(__file__).parent / 'fonts' / 'NotoSansDevanagari-Regular.ttf'),'C:/Windows/Fonts/arial.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']
    for path in candidates:
        if Path(path).exists():
            try: return ImageFont.truetype(path, size=size)
            except OSError: pass
    return ImageFont.load_default()

def fit_text(draw, text, box_width, box_height):
    text=text.strip()
    for size in range(max(12,min(64,box_height)),7,-1):
        font=get_font(size); words=text.split(); lines=[]; current=''
        for word in words:
            candidate=word if not current else current+' '+word
            if draw.textbbox((0,0),candidate,font=font)[2] <= box_width: current=candidate
            else:
                if current: lines.append(current)
                current=word
        if current: lines.append(current)
        rendered='\n'.join(lines); bbox=draw.multiline_textbbox((0,0),rendered,font=font,spacing=4)
        if bbox[2]-bbox[0] <= box_width and bbox[3]-bbox[1] <= box_height: return font,rendered
    return get_font(10),text

def render_translations(image_bytes, regions):
    image=Image.open(io.BytesIO(image_bytes)).convert('RGB'); draw=ImageDraw.Draw(image)
    for region in regions:
        bbox=region.get('bbox'); translated=str(region.get('translated_text','')).strip()
        if not bbox or not translated or len(bbox)!=4: continue
        x1,y1,x2,y2=[max(0,int(v)) for v in bbox]; x1=min(x1,image.width); y1=min(y1,image.height); x2=min(max(x2,x1+1),image.width); y2=min(max(y2,y1+1),image.height)
        draw.rectangle((x1,y1,x2,y2),fill='white')
        font,text=fit_text(draw,translated,x2-x1-8,y2-y1-8); tb=draw.multiline_textbbox((0,0),text,font=font,spacing=4); tw,th=tb[2]-tb[0],tb[3]-tb[1]
        draw.multiline_text((x1+max(4,((x2-x1)-tw)//2),y1+max(4,((y2-y1)-th)//2)),text,font=font,fill='black',spacing=4,align='center')
    output=io.BytesIO(); image.save(output,format='PNG'); output.seek(0); return output

def translate_with_bedrock(image_bytes, content_type, target_language):
    image_format=content_type.split('/')[-1]
    prompt=f'''Analyze this image for visible human-readable text. Translate every text region into {target_language}. Return ONLY valid JSON in this exact shape: {{"regions":[{{"source_text":"...","translated_text":"...","bbox":[x1,y1,x2,y2]}}]}}. Coordinates must be pixel coordinates relative to the original image, origin top-left. Include one region for each distinct text block. Do not invent text. Preserve names, URLs and email addresses unless translation is appropriate.'''
    response=bedrock.converse(modelId=BEDROCK_MODEL_ID,messages=[{'role':'user','content':[{'image':{'format':image_format,'source':{'bytes':image_bytes}}},{'text':prompt}]}],inferenceConfig={'maxTokens':4096,'temperature':0})
    parts=response.get('output',{}).get('message',{}).get('content',[])
    return ''.join(p.get('text','') for p in parts if 'text' in p).strip()

@app.get('/health')
async def health(): return {'status':'ok','provider':'aws-bedrock','model':BEDROCK_MODEL_ID}

@app.post('/translate-image')
async def translate_image(image: UploadFile=File(...),target_language: str=Form(...)):
    if image.content_type not in ALLOWED_CONTENT_TYPES: raise HTTPException(400,'Unsupported image type. Use JPEG, PNG, WEBP, or GIF.')
    target_language=target_language.strip()
    if not target_language: raise HTTPException(400,'target_language is required.')
    image_bytes=await image.read()
    if not image_bytes: raise HTTPException(400,'The uploaded image is empty.')
    if len(image_bytes)>MAX_IMAGE_SIZE_MB*1024*1024: raise HTTPException(413,f'Image size must not exceed {MAX_IMAGE_SIZE_MB} MB.')
    try:
        with Image.open(io.BytesIO(image_bytes)) as img: img.verify()
    except Exception as exc: raise HTTPException(400,'Invalid image file.') from exc
    try: result=translate_with_bedrock(image_bytes,image.content_type,target_language)
    except (ClientError,BotoCoreError) as exc: raise HTTPException(502,f'AWS Bedrock request failed: {exc}') from exc
    except Exception as exc: raise HTTPException(502,f'Translation service failed: {exc}') from exc
    if result.startswith('```'): result=result.replace('```json','',1).replace('```','',1).strip()
    try: parsed=json.loads(result)
    except json.JSONDecodeError as exc: raise HTTPException(502,'Bedrock returned invalid JSON.') from exc
    regions=parsed.get('regions',[])
    if not isinstance(regions,list): raise HTTPException(502,'Bedrock returned invalid regions.')
    try: output=render_translations(image_bytes,regions)
    except Exception as exc: raise HTTPException(500,f'Unable to render translated image: {exc}') from exc
    filename=Path(image.filename or 'translated_image').stem+'_translated.png'
    return StreamingResponse(output,media_type='image/png',headers={'Content-Disposition':f'attachment; filename="{filename}"'})