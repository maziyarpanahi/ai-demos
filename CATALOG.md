# Demo catalog

Every demo is self-contained, readable, and designed to teach one transferable
idea. Pick an outcome, bring a compatible model endpoint, and build from there.

## Vision

| Demo | What you learn | Model interface | Runtime | Level |
| --- | --- | --- | --- | --- |
| [Draw object boxes](vision/draw-object-boxes/) | Image input, constrained JSON, normalized coordinates, and image annotation | OpenAI-compatible vision chat completions | Local or hosted | Beginner |

## How to read the catalog

- **Local or hosted** means you choose where inference runs by changing the
  endpoint and model environment variables.
- A model named in a demo is the reference model for the idea, not a required
  provider.
- Every remote endpoint receives the inputs sent to it. Check the demo’s
  privacy note before using sensitive data.

The catalog will grow by stable learning goal: vision, audio, documents,
agents, video, and text-and-code. Secondary details such as model family and
domain belong in each demo’s README so a demo has one permanent home.

Want to add one? Read the [contribution guide](CONTRIBUTING.md).
