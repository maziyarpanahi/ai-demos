# Read form checkboxes

Give a vision model any form image. The script reads each checkbox label,
classifies its state, draws the returned boxes, saves the annotated image, and
opens it.

```text
form image → label + checked/unchecked state + annotated image
```

**Reference model:** Muse Glimmer 30B. The implementation is endpoint-neutral
and can use another compatible multimodal model.

## See it in action

[Watch the Muse Glimmer 30B checkbox-reading demo on X](https://x.com/MaziyarPanahi/status/2086877542233899406).

## Set up

Python 3.10 or newer and an OpenAI-compatible endpoint with image input and
JSON Schema output are required.

```bash
cd documents/read-form-checkboxes
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Configure only these three environment variables:

```bash
export OPENAI_BASE_URL="https://your-compatible-endpoint.example/v1"
export OPENAI_API_KEY="your-key-or-local-placeholder"
export OPENAI_MODEL="your-multimodal-model-id"
```

For a local endpoint that ignores authentication, use any non-empty placeholder
for `OPENAI_API_KEY`. A remote endpoint receives the complete image.

## Run it

Run the included synthetic sample with one command:

```bash
python read_checkboxes.py
```

Or pass any Pillow-supported image:

```bash
python read_checkboxes.py path/to/form.png
```

The result is saved beside the input as `<name>-checkboxes.png` and opens in
your default image viewer. Use `--output result.png` to choose another path or
`--no-open` in a headless environment.

## What the model returns

The request uses a strict schema with one compact object per checkbox:

```json
{
  "checkboxes": [
    {
      "label": "Morning session",
      "state": "checked",
      "box_2d": [101, 302, 132, 327]
    }
  ]
}
```

Coordinates are `[x_min, y_min, x_max, y_max]` normalized from `0` to `1000`
against the full image. The script validates every label, state, and coordinate
before drawing anything.

## How it works

1. Pillow rotates the image correctly, converts it to RGB, and re-encodes it as
   a PNG data URL without carrying over image metadata.
2. The model receives the image, the checkbox-reading prompt, and the strict
   JSON Schema.
3. The script validates the JSON and maps normalized coordinates to pixels.
4. Pillow draws a colored box and the associated checked, unchecked, or
   unknown label.

Model outputs can miss boxes, associate the wrong nearby text, or misread a
faint mark. Review results before using them in a real workflow. Do not send
sensitive forms to a remote endpoint unless its data policy is appropriate.

Method inspiration: [@ariG23498](https://x.com/ariG23498). This implementation
is original and does not copy third-party source code.

Return to the [document demos](../) or browse the [full catalog](../../CATALOG.md).
