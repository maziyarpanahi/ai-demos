# Download everything the ICU needs into ./downloads (about 1.4 GB, mostly SimReady albedo textures):
#   1. 13 SimReady hospital assets from NVIDIA's Hugging Face dataset (CC BY 4.0)
#   2. the Microduck robot description from Pollen Robotics' microduck_rl repository
#   3. Pollen's velstand walking policy from Hugging Face, re-saved with a dynamic batch size
# Revisions are pinned to the ones the published run used. Set them to "main" to try the latest.
import io
import os
import shutil
import tarfile

import httpx
import numpy as np
import onnx
import onnxruntime as ort
from huggingface_hub import hf_hub_download, snapshot_download

from ward import ASSET_FOLDERS, SIMREADY_DIR

HERE = os.path.dirname(os.path.abspath(__file__))
DOWNLOADS = os.path.join(HERE, "downloads")

SIMREADY_REPO, SIMREADY_REV = "nvidia/simready-assets", "c72b4dea0c5738abb1fb42c04b644b753cb975e7"
DUCK_REPO, DUCK_REV = "pollen-robotics/microduck_rl", "bae39f5900b318f7e87fc485c30a31213cfa14eb"
DUCK_PATH = "src/mjlab_microduck/robot/microduck/"
POLICY_REPO, POLICY_REV = "pollen-robotics/microduck-policies", "665ae4eb20bfe13bca1ef4ea6d4bb6b4a096e448"


def fetch_simready():
    # Only what the scene needs: the root USD, its physics payloads, albedo textures and the license.
    patterns = []
    for a in ASSET_FOLDERS:
        patterns += [f"{a}/simready_usd/*.usd", f"{a}/simready_usd/payloads/**", f"{a}/simready_usd/Textures/*_alb.png",
                     f"{a}/.metadata/LICENSE.txt"]
    snapshot_download(SIMREADY_REPO, repo_type="dataset", revision=SIMREADY_REV, local_dir=SIMREADY_DIR,
                      allow_patterns=patterns, max_workers=16)
    print(f"[simready] {len(ASSET_FOLDERS)} assets in {SIMREADY_DIR}")


def fetch_microduck():
    out = os.path.join(DOWNLOADS, "microduck")
    url = f"https://github.com/{DUCK_REPO}/archive/{DUCK_REV}.tar.gz"
    data = httpx.get(url, follow_redirects=True, timeout=180).raise_for_status().content
    root = f"microduck_rl-{DUCK_REV}/"
    keep = {root + DUCK_PATH: "", root + "LICENSE": "LICENSE", root + "LICENSE-MODELS.md": "LICENSE-MODELS.md"}
    shutil.rmtree(out, ignore_errors=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        members = []
        for m in tar.getmembers():
            for src, dst in keep.items():
                if m.name == src.rstrip("/") or (src.endswith("/") and m.name.startswith(src)):
                    m.name = dst + m.name[len(src):] if src.endswith("/") else dst
                    members.append(m)
                    break
        tar.extractall(out, members=[m for m in members if m.name], filter="data")
    print(f"[microduck] robot description from {DUCK_REPO}@{DUCK_REV[:7]} in {out}")


def fetch_policy():
    out = os.path.join(DOWNLOADS, "policy")
    src = hf_hub_download(POLICY_REPO, "velstand.onnx", revision=POLICY_REV, local_dir=out)
    hf_hub_download(POLICY_REPO, "manifest.json", revision=POLICY_REV, local_dir=out)
    # The exported policy has a fixed batch of 1. Make the batch dimension dynamic so one call serves every duck in
    # every ward, then check that the outputs match the original exactly.
    dst = os.path.join(out, "velstand_batched.onnx")
    model = onnx.load(src)
    for t in list(model.graph.input) + list(model.graph.output):
        dim = t.type.tensor_type.shape.dim[0]
        dim.ClearField("dim_value")
        dim.dim_param = "N"
    onnx.save(model, dst)
    one, many = ort.InferenceSession(src), ort.InferenceSession(dst)
    x = np.random.default_rng(0).normal(scale=0.3, size=(8, 61)).astype(np.float32)
    a = np.stack([one.run(None, {"obs": x[i:i + 1]})[0][0] for i in range(len(x))])
    b = many.run(None, {"obs": x})[0]
    diff = float(np.abs(a - b).max())
    assert diff < 1e-5, f"batched policy differs from the original by {diff}"
    print(f"[policy] velstand from {POLICY_REPO}@{POLICY_REV[:7]}, batched copy max diff {diff:.1e}")


if __name__ == "__main__":
    fetch_microduck()
    fetch_policy()
    fetch_simready()
    print("done")
