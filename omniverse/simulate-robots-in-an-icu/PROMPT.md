# Build your own version with a coding agent

This demo was first built in one evening by a coding agent with a shell, the
Hugging Face Hub and a cloud GPU. The prompt below is the cleaned-up version of
what worked. Paste it into any agent that can run commands, from an empty
folder, and change the bracketed parts.

```text
Build a physics simulation of a [hospital ICU bay] that scales to hundreds of
parallel copies on one NVIDIA GPU. Nothing gets trained; every number you show
me must come from a run.

Scene
- Compose it from NVIDIA's SimReady assets on Hugging Face
  (nvidia/simready-assets, OpenUSD, CC BY 4.0). List the asset folders first,
  then pick [two beds, bed head units, a ventilator, patient monitors, IV
  poles, a privacy curtain, a supply cart, a drug cabinet, a wheelchair].
- Each asset ships physics for several engines behind a USD variant set.
  Select the Newton variant.
- Furniture is static scenery: remove RigidBodyAPI from its prims and put the
  room and furniture in one negative collision group, so they collide only
  with robots.

Robot
- Use [Pollen Robotics' Microduck]: the MJCF from pollen-robotics/microduck_rl
  and the walking policy velstand.onnx from pollen-robotics/microduck-policies.
- Read the ONNX metadata and the training repository to rebuild the exact
  observation, action scale and control rate.
- Before touching the scene, prove the policy walks: run it in plain MuJoCo on
  flat ground, then the same robot in Newton, and print height and distance
  travelled for both.

Scale
- Build one ward in a Newton ModelBuilder, then replicate() it N times into one
  model and step it with SolverMuJoCo.
- Batch the policy: one inference call for every robot in every world.
- A simple waypoint controller picks where each robot walks; the policy does
  the walking.

Check
- Log how many robots are upright every simulated second, and save the robot
  poses to an .npz file plus a JSON summary.
- Run 1 world on CPU first, then [256] worlds on the GPU.

Render
- Render separately from the .npz with Newton's ViewerGL (headless) into an MP4.
- Show me a contact sheet of frames before you call it done.
```

## Lessons that saved the most time

- **Validate the policy alone first.** A walking policy that fails inside a
  full scene can fail for a dozen reasons. Ten lines of plain MuJoCo on flat
  ground tell you whether the observation is right before anything else exists.
- **Read the policy's metadata.** Joint order and default pose ship inside the
  ONNX file. Guessing them is the most common way to get a robot that falls
  over on step one.
- **Let the asset choose the physics.** SimReady variant sets mean the same
  USD file works in Newton, MuJoCo or PhysX. You pick a variant instead of
  authoring colliders.
- **Static scenery scales; welded bodies do not.** Hundreds of copies of
  furniture as rigid bodies overflow the contact buffers. Static colliders in
  one collision group keep the solver's work on the robots.
- **Simulate and render in separate steps.** Record poses wherever the GPU is,
  render wherever you have OpenGL. Cloud GPU containers often expose compute
  without graphics, so this split also keeps you unblocked.
- **Count, don't eyeball.** An "upright at end" number per run catches a
  broken change faster than watching a video.
