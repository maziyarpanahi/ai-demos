# Draw object boxes with a vision model

Turn a photo and a list of labels into an annotated image with model-generated
bounding boxes.

```text
photo.jpg + “person, car, bicycle” → photo-boxed.png
```

The reference idea uses Qwen3.8-27B, but the code works with any compatible
vision model that accepts image input through an OpenAI-compatible chat
completions endpoint.

## What you will learn

- Send an image as a base64 data URL alongside a text prompt.
- Ask a vision model for a small, constrained JSON shape.
- Validate imperfect model output before using it.
- Convert normalized `0–1000` coordinates into image pixels.
- Draw readable labels and boxes without a computer-vision framework.

## Requirements

- Python 3.10 or newer.
- An OpenAI-compatible `/v1/chat/completions` endpoint.
- A model on that endpoint that supports `image_url` content parts.
- A JPEG, PNG, or other image format supported by Pillow.

The two direct Python dependencies are listed in `requirements.txt`.

## 1. Set up

From a fresh clone of the repository:

```bash
cd ai-demos/vision/draw-object-boxes
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 2. Choose a model endpoint

Set the endpoint, API key, and exact model identifier exposed by that endpoint:

```bash
export OPENAI_BASE_URL="http://localhost:11434/v1"
export OPENAI_API_KEY="local"
export MODEL="your-vision-model-id"
```

Common OpenAI-compatible base URLs:

| Endpoint | `OPENAI_BASE_URL` |
| --- | --- |
| [Ollama](https://docs.ollama.com/api/openai-compatibility) | `http://localhost:11434/v1` |
| [LM Studio](https://lmstudio.ai/docs/developer/openai-compat) | `http://localhost:1234/v1` |
| [llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) | `http://localhost:8080/v1` |
| [OpenRouter](https://openrouter.ai/docs/quickstart) | `https://openrouter.ai/api/v1` |

For a local server that ignores authentication, any non-empty API key such as
`local` is enough. A hosted endpoint requires its real key. Never put the key
in this repository or commit an `.env` file.

## 3. Run it

```bash
python detect.py photo.jpg --labels "person, car, bicycle"
```

The script saves the result beside the input as `photo-boxed.png` and opens it
in your default image viewer.

Choose another output path or skip opening the viewer:

```bash
python detect.py photo.jpg \
  --labels "person, car" \
  --output result.png \
  --no-open
```

Successful output looks like this:

```text
Saved 3 boxes to /absolute/path/to/photo-boxed.png
```

The detection count will vary with the image, labels, model, and sampling
behavior.

## How it works

1. Pillow loads the image, applies its orientation metadata, and converts it
   to RGB.
2. The script encodes the image as a PNG data URL.
3. The prompt asks for only the requested labels and a compact detections JSON
   object.
4. The parser finds the first valid detections object, rejects unknown labels,
   clamps coordinates, and drops malformed boxes.
5. Pillow maps the normalized coordinates back to pixels and renders the final
   labels and rectangles.

The entire flow lives in [`detect.py`](detect.py). Start with `prompt_for()` to
change the requested schema and `draw_boxes()` to change the visual style.

## Make it yours

Try one of these extensions:

- Give every label its own color.
- Save the validated detections as JSON beside the image.
- Add confidence values when your model supports them reliably.
- Turn the script into a drag-and-drop web interface.
- Compare two vision models against the same labeled image.

## Privacy, cost, and limitations

- A remote endpoint receives the complete image. Use a local endpoint for
  images that must stay on your machine.
- Hosted endpoints may charge for image and output tokens.
- Model-generated boxes are variable and can be missing, duplicated, or
  inaccurate.
- The parser validates shape and ranges; it does not prove that a box is
  visually correct.
- This is a learning demo, not a benchmark or a safety-critical detector.

## What was simplified

A social version can include live capture, a custom interface, tuned inputs,
and editing. This open-source version keeps the transferable core: image input,
structured coordinates, defensive parsing, and rendering in one readable
script.

Return to the [vision demos](../) or browse the [full catalog](../../CATALOG.md).
