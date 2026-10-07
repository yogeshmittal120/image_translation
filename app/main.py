import io
import json
import os
from pathlib import Path

import boto3
import cv2
import numpy as np
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image, ImageDraw, ImageFont

load_dotenv()

app = FastAPI(title="Image Translation API", version="3.1.0")

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "")
MAX_IMAGE_SIZE_MB = int(os.getenv("MAX_IMAGE_SIZE_MB", "5"))
FONT_PATH = os.getenv("FONT_PATH", "")

textract = boto3.client("textract", region_name=AWS_REGION)
bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


def normalize_image(image_bytes: bytes) -> bytes:
    with Image.open(io.BytesIO(image_bytes)) as img:
        img = img.convert("RGB")
        output = io.BytesIO()
        img.save(output, format="PNG")
        return output.getvalue()


def box_to_pixels(box, width, height):
    left = max(0, int(box.get("Left", 0) * width))
    top = max(0, int(box.get("Top", 0) * height))
    right = min(width, int((box.get("Left", 0) + box.get("Width", 0)) * width))
    bottom = min(height, int((box.get("Top", 0) + box.get("Height", 0)) * height))
    return [left, top, right, bottom]


def extract_text_regions(image_bytes: bytes):
    """Extract LINE text plus WORD boxes for precise background removal."""
    png_bytes = normalize_image(image_bytes)
    response = textract.detect_document_text(Document={"Bytes": png_bytes})

    with Image.open(io.BytesIO(png_bytes)) as img:
        width, height = img.size

    blocks = response.get("Blocks", [])
    words_by_id = {
        block.get("Id"): block
        for block in blocks
        if block.get("BlockType") == "WORD"
    }

    regions = []

    for block in blocks:
        if block.get("BlockType") != "LINE":
            continue

        text = block.get("Text", "").strip()
        if not text:
            continue

        line_box = block.get("Geometry", {}).get("BoundingBox", {})
        if not line_box:
            continue

        word_boxes = []
        for relationship in block.get("Relationships", []):
            if relationship.get("Type") != "CHILD":
                continue

            for word_id in relationship.get("Ids", []):
                word = words_by_id.get(word_id)
                if not word:
                    continue

                word_box = word.get("Geometry", {}).get("BoundingBox")
                if word_box:
                    word_boxes.append(box_to_pixels(word_box, width, height))

        regions.append({
            "id": len(regions),
            "source_text": text,
            "bbox": box_to_pixels(line_box, width, height),
            "mask_boxes": word_boxes or [box_to_pixels(line_box, width, height)],
        })

    return regions


def translate_regions(regions, target_language):
    if not regions:
        return []

    if not BEDROCK_MODEL_ID:
        raise RuntimeError("BEDROCK_MODEL_ID is not configured.")

    items = "\n".join(
        f'{r["id"]}: {json.dumps(r["source_text"], ensure_ascii=False)}'
        for r in regions
    )

    prompt = f"""Translate the following OCR text lines into {target_language}.

Rules:
- Keep every numeric ID unchanged.
- Return exactly one translation for every input ID.
- Do not add explanations.
- Preserve URLs, email addresses, product names and proper nouns when appropriate.
- Keep the meaning natural in {target_language}.
- Return ONLY valid JSON in this format:
{{"translations":[{{"id":0,"translated_text":"..."}}]}}

OCR lines:
{items}
"""

    response = bedrock.converse(
        modelId=BEDROCK_MODEL_ID,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 4096, "temperature": 0},
    )

    parts = response.get("output", {}).get("message", {}).get("content", [])
    result = "".join(
        part.get("text", "") for part in parts if "text" in part
    ).strip()

    fence = chr(96) * 3
    if result.startswith(fence):
        result = result.replace(fence + "json", "", 1).replace(fence, "", 1).strip()

    parsed = json.loads(result)
    translations = parsed.get("translations", [])

    if not isinstance(translations, list):
        raise ValueError("Bedrock returned invalid translations.")

    translation_map = {
        int(item["id"]): str(item["translated_text"]).strip()
        for item in translations
        if "id" in item and "translated_text" in item
    }

    return [
        {**region, "translated_text": translation_map[region["id"]]}
        for region in regions
        if region["id"] in translation_map
    ]


def estimate_text_color(rgb_image, bbox):
    """Estimate original foreground color from contrast against local background."""
    x1, y1, x2, y2 = bbox
    height, width = rgb_image.shape[:2]

    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))

    crop = rgb_image[y1:y2, x1:x2]
    if crop.size == 0:
        return (0, 0, 0)

    # Sample a border around the text box as the local background.
    pad = max(3, int((y2 - y1) * 0.7))
    bx1 = max(0, x1 - pad)
    by1 = max(0, y1 - pad)
    bx2 = min(width, x2 + pad)
    by2 = min(height, y2 + pad)

    surrounding = rgb_image[by1:by2, bx1:bx2]
    gray_crop = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    gray_surrounding = cv2.cvtColor(surrounding, cv2.COLOR_RGB2GRAY)

    background_luma = float(np.median(gray_surrounding))
    inside_luma = gray_crop.reshape(-1)

    # Text is usually the pixels farthest from the local background.
    distance = np.abs(inside_luma.astype(np.float32) - background_luma)
    threshold = max(20.0, float(np.percentile(distance, 75)))
    candidate_mask = distance >= threshold

    pixels = crop.reshape(-1, 3)[candidate_mask]

    if len(pixels) < 5:
        return (255, 255, 255) if background_luma < 128 else (0, 0, 0)

    color = np.median(pixels, axis=0).astype(int)
    return tuple(int(v) for v in color)


def get_font(size: int):
    candidates = []

    if FONT_PATH:
        candidates.append(FONT_PATH)

    candidates.extend([
        str(Path(__file__).parent / "fonts" / "NotoSansDevanagari-Regular.ttf"),
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ])

    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                pass

    return ImageFont.load_default()


def wrap_text(draw, text, font, max_width):
    words = text.split()
    lines = []
    current = ""

    for word in words:
        candidate = word if not current else f"{current} {word}"
        bbox = draw.textbbox((0, 0), candidate, font=font)

        if bbox[2] - bbox[0] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word

    if current:
        lines.append(current)

    return "\n".join(lines)


def fit_text(draw, text, box_width, box_height, preferred_size):
    """Start near the source font size and shrink only when required."""
    start = max(8, min(96, int(preferred_size)))

    for size in range(start, 7, -1):
        font = get_font(size)
        rendered = wrap_text(draw, text.strip(), font, box_width)

        bbox = draw.multiline_textbbox(
            (0, 0),
            rendered,
            font=font,
            spacing=max(1, int(size * 0.12)),
            align="center",
        )

        if (
            bbox[2] - bbox[0] <= box_width
            and bbox[3] - bbox[1] <= box_height
        ):
            return font, rendered, max(1, int(size * 0.12))

    font = get_font(8)
    return font, text.strip(), 1


def remove_text_background(image_bytes, regions):
    png_bytes = normalize_image(image_bytes)

    image = cv2.imdecode(
        np.frombuffer(png_bytes, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise ValueError("Unable to decode image.")

    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    for region in regions:
        for box in region.get("mask_boxes", [region["bbox"]]):
            x1, y1, x2, y2 = box
            pad = max(2, int((y2 - y1) * 0.18))

            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(image.shape[1], x2 + pad)
            y2 = min(image.shape[0], y2 + pad)

            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    if np.any(mask):
        image = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

    return image


def render_translations(image_bytes, regions):
    original_png = normalize_image(image_bytes)
    original_rgb = cv2.cvtColor(
        cv2.imdecode(
            np.frombuffer(original_png, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        ),
        cv2.COLOR_BGR2RGB,
    )

    cleaned = remove_text_background(image_bytes, regions)
    cleaned_rgb = cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB)

    pil_image = Image.fromarray(cleaned_rgb)
    draw = ImageDraw.Draw(pil_image)

    for region in regions:
        x1, y1, x2, y2 = region["bbox"]
        translated = region["translated_text"]

        source_height = max(8, y2 - y1)
        # Approximate the source font size from the original OCR line height.
        preferred_size = max(10, int(source_height * 0.82))

        padding = max(2, int(source_height * 0.08))
        box_width = max(10, x2 - x1 - padding * 2)
        box_height = max(10, y2 - y1 - padding * 2)

        font, text, spacing = fit_text(
            draw,
            translated,
            box_width,
            box_height,
            preferred_size,
        )

        text_bbox = draw.multiline_textbbox(
            (0, 0),
            text,
            font=font,
            spacing=spacing,
            align="center",
        )

        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        tx = x1 + max(1, ((x2 - x1) - text_width) // 2)
        ty = y1 + max(1, ((y2 - y1) - text_height) // 2)

        text_color = estimate_text_color(original_rgb, region["bbox"])

        draw.multiline_text(
            (tx, ty),
            text,
            font=font,
            fill=text_color,
            spacing=spacing,
            align="center",
        )

    output = io.BytesIO()
    pil_image.save(output, format="PNG")
    output.seek(0)
    return output


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "provider": "aws-bedrock",
        "ocr": "aws-textract",
        "model": BEDROCK_MODEL_ID,
        "style_matching": "auto",
    }


@app.post("/translate-image")
async def translate_image(
    image: UploadFile = File(...),
    target_language: str = Form(...),
):
    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Unsupported image type. Use JPEG, PNG, or WEBP.",
        )

    target_language = target_language.strip()
    if not target_language:
        raise HTTPException(status_code=400, detail="target_language is required.")

    image_bytes = await image.read()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")

    if len(image_bytes) > MAX_IMAGE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"Image size must not exceed {MAX_IMAGE_SIZE_MB} MB.",
        )

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img.verify()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image file.") from exc

    try:
        regions = extract_text_regions(image_bytes)

        if not regions:
            raise HTTPException(
                status_code=422,
                detail="No readable text was detected in the image.",
            )

        translated_regions = translate_regions(regions, target_language)
        output = render_translations(image_bytes, translated_regions)

    except HTTPException:
        raise
    except (ClientError, BotoCoreError) as exc:
        raise HTTPException(
            status_code=502,
            detail=f"AWS service request failed: {exc}",
        ) from exc
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=502,
            detail="Claude returned invalid translation JSON.",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Image translation failed: {exc}",
        ) from exc

    filename = Path(image.filename or "translated_image").stem + "_translated.png"

    return StreamingResponse(
        output,
        media_type="image/png",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        },
    )
