# Demo catalog

Every demo is self-contained, readable, and designed to teach one transferable
idea. Pick an outcome, bring a compatible model endpoint, and build from there.

## Vision

| Demo | Model(s) | What you learn | Model interface | Runtime | Level |
| --- | --- | --- | --- | --- | --- |
| [Draw object boxes](vision/draw-object-boxes/) | Qwen3.8-27B | Image input, constrained JSON, normalized coordinates, and image annotation | OpenAI-compatible vision chat completions | Local or hosted | Beginner |

## Documents

| Demo | Model(s) | What you learn | Model interface | Runtime | Level |
| --- | --- | --- | --- | --- | --- |
| [Read form checkboxes](documents/read-form-checkboxes/) | Muse Glimmer 30B | Checkbox detection, label reading, strict JSON, normalized coordinates, and document annotation | OpenAI-compatible vision chat completions | Local or hosted | Beginner |

## Omniverse

| Demo | Model(s) | What you learn | Model interface | Runtime | Level |
| --- | --- | --- | --- | --- | --- |
| [Simulate robots in an ICU](omniverse/simulate-robots-in-an-icu/) | Microduck velstand (Pollen Robotics) | SimReady OpenUSD physics variants, Newton world replication, batched RL policy inference, and record-then-render | ONNX Runtime policy in Newton (MuJoCo Warp) | Local (CPU or NVIDIA GPU) | Intermediate |

## How to read the catalog

- **Local or hosted** means you choose where inference runs by changing the
  endpoint and model environment variables.
- **Local (CPU or NVIDIA GPU)** means everything runs on your machine; the
  device only changes how many copies of a scene you can simulate at once.
- A model named in a demo is the reference model for the idea, not a required
  provider.
- Multiple verified models for one implementation belong in the same
  **Model(s)** cell, not in duplicate demo rows.
- Every remote endpoint receives the inputs sent to it. Check the demo’s
  privacy note before using sensitive data.

The catalog will grow by stable learning goal: vision, audio, documents,
agents, video, text-and-code, and Omniverse for physical AI. Secondary
details such as model family and domain belong in each demo’s README so a
demo has one permanent home.

Want to add one? Read the [contribution guide](CONTRIBUTING.md).
