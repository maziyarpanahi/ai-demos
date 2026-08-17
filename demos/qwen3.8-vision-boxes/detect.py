#!/usr/bin/env python3
"""Ask an OpenAI-compatible vision model for boxes and draw them on a photo."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
from pathlib import Path
from typing import Any

from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont, ImageOps


def required_setting(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"Set {name} before running this demo.")
    return value


def load_image(path: Path) -> tuple[Image.Image, str]:
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return image, f"data:image/png;base64,{encoded}"


def prompt_for(labels: list[str]) -> str:
    allowed = ", ".join(json.dumps(label) for label in labels)
    return (
        "Detect every clearly visible instance of these objects in the image: "
        f"{', '.join(labels)}. Return only one JSON object with this schema: "
        '{"detections":[{"label":"object label","box_2d":'
        "[x_min,y_min,x_max,y_max]}]}. Coordinates must be integers normalized "
        "from 0 to 1000 against the full image. Use tight boxes, one box per "
        f"object, and only these labels: {allowed}."
    )


def extract_detections(text: str, allowed_labels: set[str]) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    payload: dict[str, Any] | None = None
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and isinstance(candidate.get("detections"), list):
            payload = candidate
            break
    if payload is None:
        raise ValueError("The model response did not contain a detections JSON object.")

    detections: list[dict[str, Any]] = []
    for item in payload["detections"]:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip().lower()
        box = item.get("box_2d")
        if label not in allowed_labels or not isinstance(box, list) or len(box) != 4:
            continue
        try:
            x1, y1, x2, y2 = [max(0.0, min(1000.0, float(value))) for value in box]
        except (TypeError, ValueError):
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append({"label": label, "box_2d": [x1, y1, x2, y2]})
    return detections


def choose_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def draw_boxes(image: Image.Image, detections: list[dict[str, Any]]) -> Image.Image:
    result = image.copy()
    draw = ImageDraw.Draw(result)
    width, height = result.size
    line_width = max(2, min(width, height) // 300)
    font = choose_font(max(14, min(width, height) // 45))
    color = "#ff2d55"

    for detection in detections:
        x1, y1, x2, y2 = detection["box_2d"]
        box = (
            round(x1 * width / 1000),
            round(y1 * height / 1000),
            round(x2 * width / 1000),
            round(y2 * height / 1000),
        )
        draw.rectangle(box, outline=color, width=line_width)
        label = detection["label"]
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        label_width = right - left
        label_height = bottom - top
        label_x = box[0]
        label_y = max(0, box[1] - label_height - 10)
        draw.rectangle(
            (label_x, label_y, label_x + label_width + 12, label_y + label_height + 8),
            fill=color,
        )
        draw.text((label_x + 6, label_y + 3), label, fill="white", font=font)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="photo to detect objects in")
    parser.add_argument("--labels", required=True, help="comma-separated object labels")
    parser.add_argument("--output", type=Path, help="output path; defaults to IMAGE-boxed.png")
    parser.add_argument("--no-open", action="store_true", help="save without opening the result")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image.is_file():
        raise SystemExit(f"Image not found: {args.image}")

    labels = [label.strip().lower() for label in args.labels.split(",") if label.strip()]
    if not labels:
        raise SystemExit("Give at least one label with --labels.")

    image, image_url = load_image(args.image)
    client = OpenAI(
        base_url=required_setting("OPENAI_BASE_URL"),
        api_key=os.environ.get("OPENAI_API_KEY", "local"),
    )
    completion = client.chat.completions.create(
        model=required_setting("MODEL"),
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_for(labels)},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
    )
    content = completion.choices[0].message.content
    if not isinstance(content, str):
        raise RuntimeError("The model returned no text response.")

    detections = extract_detections(content, set(labels))
    if not detections:
        raise RuntimeError("The model returned no valid boxes for the requested labels.")

    result = draw_boxes(image, detections)
    output = args.output or args.image.with_name(f"{args.image.stem}-boxed.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.save(output)
    print(f"Saved {len(detections)} boxes to {output.resolve()}")
    if not args.no_open:
        result.show(title=output.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
