# Draw object boxes with Qwen3.8

Give a vision-capable model a photo and a list of object labels. The script
asks the model for normalized coordinates, draws the returned boxes on the
original photo, saves the result, and opens it.

The code uses the standard OpenAI Python client. It is not tied to a provider.

## 1. Choose an OpenAI-compatible vision endpoint

Set these three environment variables:

```bash
export OPENAI_BASE_URL="http://localhost:11434/v1"
export OPENAI_API_KEY="local"
export MODEL="your-vision-model-id"
```

`MODEL` is whatever model identifier your server or provider exposes. For
Qwen3.8-27B, use the Qwen3.8-27B identifier shown by that endpoint.

Common base URLs:

| Endpoint | `OPENAI_BASE_URL` |
| --- | --- |
| [Ollama](https://docs.ollama.com/api/openai-compatibility) | `http://localhost:11434/v1` |
| [LM Studio](https://lmstudio.ai/docs/developer/openai-compat) | `http://localhost:1234/v1` |
| [llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) | `http://localhost:8080/v1` |
| [OpenRouter](https://openrouter.ai/docs/quickstart) | `https://openrouter.ai/api/v1` |

For a local server that does not check credentials, any non-empty value such
as `local` is fine. For a hosted endpoint, set its real API key in
`OPENAI_API_KEY`. Never put a key in the source code or commit an `.env` file.

The endpoint and selected model must support image input through
`/v1/chat/completions` using an `image_url` content part.

## 2. Install

```bash
cd demos/qwen3.8-vision-boxes
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Run

```bash
python detect.py photo.jpg --labels "person, car, bicycle"
```

The result is saved beside the input as `photo-boxed.png` and opened in the
default image viewer.

Choose another output path or skip opening the viewer when needed:

```bash
python detect.py photo.jpg \
  --labels "person, car" \
  --output result.png \
  --no-open
```

If the endpoint is remote, the photo is sent to that endpoint. Use a local
server for images that must stay on your machine.
