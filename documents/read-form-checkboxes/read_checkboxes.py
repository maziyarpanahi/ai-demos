#!/usr/bin/env python3
"""Detect and label form checkboxes with an OpenAI-compatible vision model."""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
from pathlib import Path
from typing import Any

from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont, ImageOps


DEMO_DIR = Path(__file__).resolve().parent
SAMPLE_IMAGE = DEMO_DIR / "assets" / "sample-form.png"
ALLOWED_STATES = {"checked", "unchecked", "unknown"}
STATE_COLORS = {
    "checked": "#15803d",
    "unchecked": "#c2410c",
    "unknown": "#4b5563",
}

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "checkbox_detections",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "checkboxes": {
                    "type": "array",
                    "maxItems": 100,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "label": {"type": "string"},
                            "state": {
                                "type": "string",
                                "enum": ["checked", "unchecked", "unknown"],
                            },
                            "box_2d": {
                                "type": "array",
                                "minItems": 4,
                                "maxItems": 4,
                                "items": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 1000,
                                },
                            },
                        },
                        "required": ["label", "state", "box_2d"],
                    },
                }
            },
            "required": ["checkboxes"],
        },
    },
}

PROMPT = """Find every visible square checkbox in this image.

For each checkbox:
- read the nearest text that labels that checkbox;
- classify the state as checked, unchecked, or unknown;
- return a tight box around the checkbox square only, not its text label;
- use [x_min, y_min, x_max, y_max] coordinates normalized from 0 to 1000
  against the full image.

Ignore page borders, section dividers, decorative shapes, radio buttons, and
checkmarks that are not inside a checkbox square. Do not invent missing boxes
or labels. Return only the JSON required by the schema."""


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


def first_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("The model response did not contain a JSON object.")


def validate_checkboxes(payload: dict[str, Any]) -> list[dict[str, Any]]:
    items = payload.get("checkboxes")
    if not isinstance(items, list):
        raise ValueError("The JSON object must contain a checkboxes array.")
    if len(items) > 100:
        raise ValueError("The model returned more than 100 checkboxes.")

    checkboxes: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"Checkbox {index + 1} is not an object.")

        label = item.get("label")
        state = item.get("state")
        box = item.get("box_2d")
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"Checkbox {index + 1} has no readable label.")
        if not isinstance(state, str) or state.lower() not in ALLOWED_STATES:
            raise ValueError(f"Checkbox {index + 1} has an invalid state.")
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError(f"Checkbox {index + 1} needs four box coordinates.")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in box
        ):
            raise ValueError(f"Checkbox {index + 1} has non-numeric coordinates.")

        coordinates = [float(value) for value in box]
        if any(not math.isfinite(value) or not 0 <= value <= 1000 for value in coordinates):
            raise ValueError(f"Checkbox {index + 1} has coordinates outside 0–1000.")
        x1, y1, x2, y2 = coordinates
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Checkbox {index + 1} has an unordered box.")

        checkboxes.append(
            {
                "label": label.strip(),
                "state": state.lower(),
                "box_2d": coordinates,
            }
        )
    return checkboxes


def choose_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def draw_checkboxes(image: Image.Image, checkboxes: list[dict[str, Any]]) -> Image.Image:
    result = image.copy()
    draw = ImageDraw.Draw(result)
    width, height = result.size
    line_width = max(3, min(width, height) // 250)
    font = choose_font(max(14, min(width, height) // 45))

    for checkbox in checkboxes:
        x1, y1, x2, y2 = checkbox["box_2d"]
        box = (
            round(x1 * width / 1000),
            round(y1 * height / 1000),
            round(x2 * width / 1000),
            round(y2 * height / 1000),
        )
        color = STATE_COLORS[checkbox["state"]]
        draw.rectangle(box, outline="white", width=line_width + 4)
        draw.rectangle(box, outline=color, width=line_width)

        text = f"{checkbox['state'].upper()} · {checkbox['label']}"
        if len(text) > 72:
            text = text[:69].rstrip() + "…"
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        text_width = right - left
        text_height = bottom - top
        padding_x, padding_y = 8, 6
        tag_width = text_width + padding_x * 2
        tag_height = text_height + padding_y * 2
        tag_x = max(0, min(box[0], width - tag_width))
        above_y = box[1] - tag_height - 5
        tag_y = above_y if above_y >= 0 else min(height - tag_height, box[3] + 5)
        draw.rounded_rectangle(
            (tag_x, tag_y, tag_x + tag_width, tag_y + tag_height),
            radius=max(3, tag_height // 5),
            fill=color,
        )
        draw.text(
            (tag_x + padding_x, tag_y + padding_y - top),
            text,
            font=font,
            fill="white",
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        default=SAMPLE_IMAGE,
        help="form image; defaults to the included synthetic sample",
    )
    parser.add_argument("--output", type=Path, help="output path; defaults beside the input")
    parser.add_argument("--no-open", action="store_true", help="save without opening the result")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image.is_file():
        raise SystemExit(f"Image not found: {args.image}")

    image, image_url = load_image(args.image)
    client = OpenAI(
        base_url=required_setting("OPENAI_BASE_URL"),
        api_key=required_setting("OPENAI_API_KEY"),
    )
    completion = client.chat.completions.create(
        model=required_setting("OPENAI_MODEL"),
        messages=[
            {
                "role": "system",
                "content": "You are a precise visual form reader. Follow the JSON schema exactly.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ],
        response_format=RESPONSE_FORMAT,
        temperature=0.0,
        max_tokens=2048,
    )
    content = completion.choices[0].message.content
    if not isinstance(content, str):
        raise RuntimeError("The model returned no text response.")

    checkboxes = validate_checkboxes(first_json_object(content))
    result = draw_checkboxes(image, checkboxes)
    output = args.output or args.image.with_name(f"{args.image.stem}-checkboxes.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.save(output)
    print(f"Saved {len(checkboxes)} checkboxes to {output.resolve()}")
    for checkbox in checkboxes:
        print(f"- {checkbox['state']}: {checkbox['label']}")
    if not args.no_open:
        result.show(title=output.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
