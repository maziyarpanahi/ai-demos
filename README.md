<div align="center">

# ⚡ AI Demos

**Open-source AI demos you can understand, run, and remix.**

The shortest path from *“that demo looks impossible”* to *“I built it.”*

[Explore the demos](#explore-the-demos) · [Run one now](#run-one-now) · [Browse the catalog](CATALOG.md) · [Contribute](CONTRIBUTING.md)

</div>

---

The demos you see on X and LinkedIn are a whole production: live models,
carefully chosen inputs, polished interfaces, recording, editing, and timing.
This repository strips each one down to the useful core so you can learn the
idea, run it yourself, and turn it into something new.

No mystery framework. No provider lock-in. No private production machinery.
Just small, readable builds with one clear result.

## Explore the demos

| Demo | Model(s) | What you build | Input → output | Run it |
| --- | --- | --- | --- | --- |
| **[Draw object boxes with a vision model](vision/draw-object-boxes/)** | Qwen3.8-27B | Ask a vision model for normalized coordinates and render the boxes | Photo + labels → annotated image | Local or hosted |
| **[Read form checkboxes](documents/read-form-checkboxes/)** | Muse Glimmer 30B | Read checkbox labels and states, then annotate the form | Form image → annotated image | Local or hosted |

There is one row per demo. When another model is verified with the same code,
its name joins the existing **Model(s)** cell instead of creating a duplicate
row.

More demos will land across vision, audio, documents, agents, video, and
text-and-code. The complete, growing index lives in the **[demo
catalog](CATALOG.md)**.

## Run one now

Clone the repository and launch the first demo:

```bash
git clone https://github.com/maziyarpanahi/ai-demos.git
cd ai-demos/vision/draw-object-boxes

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

export OPENAI_BASE_URL="http://localhost:11434/v1"
export OPENAI_API_KEY="local"
export MODEL="your-vision-model-id"

python detect.py your-photo.jpg --labels "person, car, bicycle"
```

Use a local OpenAI-compatible server or point the same demo at a hosted
endpoint. Each demo tells you exactly what it needs and what leaves your
machine.

## Every demo follows the same contract

- **Small:** one focused idea, not an application framework.
- **Readable:** one obvious entry point and code you can understand quickly.
- **Reproducible:** copy-paste setup, explicit requirements, and expected
  output.
- **Portable:** standard interfaces over provider-specific SDKs whenever
  practical.
- **Honest:** clear local-versus-hosted boundaries, privacy notes, and known
  limitations.
- **Hackable:** the important prompt, schema, and rendering logic stay easy to
  find and change.

## How the repository is organized

```text
<category>/<outcome>/
├── README.md          # what it does, how to run it, how to adapt it
├── <entry-point>      # the smallest readable implementation
├── requirements.txt  # direct dependencies
└── assets/            # optional, licensed example inputs and outputs
```

Categories live directly at the repository root, so the collection stays
visible without turning the root into hundreds of folders. Demo paths are
named after outcomes rather than providers, which keeps links useful as models
change.

## From social demo to open-source build

These are not frame-for-frame replicas of the original posts. Social versions
may include custom interfaces, capture tooling, larger pipelines, and private
operational details. The repository version preserves the technique and the
learning value while removing everything you do not need to reproduce the
core result.

## Build something with it

Start with a demo, swap the input, change the prompt, try another compatible
model, or combine the idea with your own product. If you improve one, add a
new runtime, or build a clean variation, read **[CONTRIBUTING.md](CONTRIBUTING.md)**.

If a demo helps you make something, star the repository so you can find it
again—and share your remix. That is how this collection gets better.

## License

Code in this repository is released under the [MIT License](LICENSE).
