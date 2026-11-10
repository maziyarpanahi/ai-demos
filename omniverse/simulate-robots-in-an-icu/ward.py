# The ICU bay: NVIDIA SimReady hospital assets (OpenUSD with built-in physics) placed in a simple room.
# Every SimReady asset ships physics payloads for several engines behind a USD variant set; we select Newton's.
import math
import os

import numpy as np
import warp as wp
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

import newton

HERE = os.path.dirname(os.path.abspath(__file__))
SIMREADY_DIR = os.path.join(HERE, "downloads", "simready")

# name, SimReady asset folder, (x, y, z) in metres, yaw in degrees. The room is 7 m wide (x) and 6 m deep (y). The
# back wall is at y = +3 and the -y side stays open for the camera. Beds run along y with the head at the back wall.
LAYOUT = [
    ("bed_left", "bed_hospital_standard_a01", (-1.8, 1.3, 0.0), 90),
    ("bed_right", "bed_hospital_standard_a02", (1.8, 1.3, 0.0), 90),
    ("head_left", "unit_head_bed_a01", (-1.8, 3.0, 0.0), -90),
    ("head_right", "unit_head_bed_a01", (1.8, 3.0, 0.0), -90),
    ("curtain_right", "screen_hospital_b01", (1.85, -0.24, 0.0), 90),
    ("vent_left", "lung_ventilation_artificial_a01", (-3.05, 2.0, 0.0), 0),
    ("monitor_left", "monitor_patient_a01", (-3.05, 0.9, 0.0), 0),
    ("iv_left", "IV_pole_a01", (-0.8, 2.2, 0.0), 0),
    ("iv_right", "IV_pole_b01", (0.8, 2.3, 0.0), 0),
    ("monitor_right", "monitor_patient_a01", (3.05, 2.0, 0.0), 180),
    ("uvc_robot", "lamp_uvc_robot_a01", (0.0, 2.65, 0.0), 0),
    ("cart", "cart_medical_utility_a01", (-2.95, -1.4, 0.0), 0),
    ("cabinet", "cabinet_for_drugs_a01", (-2.95, -2.45, 0.0), 0),
    ("chair", "chair_hospital_a01", (2.95, -0.5, 0.0), 200),
    ("wheelchair", "wheelChair_c01", (2.6, -2.0, 0.0), 120),
]
ASSET_FOLDERS = sorted({folder for _, folder, _, _ in LAYOUT})

ROOM = dict(half_x=3.5, half_y=3.0, wall_h=3.0)
FLOOR_COLOR = (0.66, 0.71, 0.73)
WALL_COLOR = (0.90, 0.93, 0.92)
TRIM_COLOR = (0.47, 0.67, 0.66)


def root_usd(folder):
    d = os.path.join(SIMREADY_DIR, folder, "simready_usd")
    if not os.path.isdir(d):
        raise FileNotFoundError(f"{d} is missing. Run `python fetch_assets.py` first.")
    return os.path.join(d, next(f for f in sorted(os.listdir(d)) if f.endswith(".usd")))


def open_newton_stage(folder):
    """Open one SimReady asset with its Newton physics variant enabled and every other engine's variant disabled."""
    stage = Usd.Stage.Open(root_usd(folder))
    variant_sets = stage.GetDefaultPrim().GetVariantSets()
    for name in variant_sets.GetNames():
        variant_sets.GetVariantSet(name).SetVariantSelection("Enabled" if name == "Newton" else "Disabled")
    # Furniture is scenery: make it static colliders on the world instead of thousands of welded bodies, which
    # would overflow the solver's contact buffers once the ward is copied hundreds of times.
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
    return stage


def _albedo_mean(path, cache={}):
    if path not in cache:
        from PIL import Image

        Image.MAX_IMAGE_PIXELS = None  # some SimReady albedo maps are very large
        im = Image.open(path).convert("RGB")
        im.thumbnail((64, 64))
        cache[path] = tuple(float(c) / 255.0 for c in np.asarray(im).reshape(-1, 3).mean(0))
    return cache[path]


def _material_color(material, base_dir):
    for child in material.GetPrim().GetChildren():
        shader = UsdShade.Shader(child)
        if not shader:
            continue
        if shader.GetIdAttr().Get() == "UsdUVTexture" and child.GetName().lower().startswith("albedo"):
            f = shader.GetInput("file").Get()
            if f is not None:
                p = f.resolvedPath or os.path.join(base_dir, f.path)
                if os.path.exists(p):
                    return _albedo_mean(p)
        if shader.GetIdAttr().Get() == "UsdPreviewSurface":
            inp = shader.GetInput("diffuseColor")
            if inp and not inp.HasConnectedSource() and inp.Get() is not None:
                return tuple(inp.Get())
    return None


def recolor_shapes(builder, stage, s0, s1):
    """Give each shape one flat color: the face-weighted average albedo of its USD materials. The OpenGL viewer
    otherwise paints a multi-material mesh (a curtain with its rail) with the first material's texture everywhere."""
    base_dir = os.path.dirname(stage.GetRootLayer().realPath)
    for i in range(s0, s1):
        prim = stage.GetPrimAtPath(builder.shape_label[i]) if builder.shape_label[i] else None
        if not prim or not prim.IsValid():
            continue
        parts = []
        for subset in UsdGeom.Subset.GetAllGeomSubsets(UsdGeom.Imageable(prim)):
            mat = UsdShade.MaterialBindingAPI(subset.GetPrim()).ComputeBoundMaterial()[0]
            n = len(subset.GetIndicesAttr().Get() or [])
            if mat and n:
                parts.append((n, mat))
        if not parts:
            mat = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
            if mat:
                parts.append((1, mat))
        acc, total = np.zeros(3), 0
        for n, mat in parts:
            c = _material_color(mat, base_dir)
            if c is not None:
                acc += n * np.asarray(c[:3])
                total += n
        if total:
            builder.shape_color[i] = tuple(float(x) for x in acc / total)
            src = builder.shape_source[i]
            if isinstance(src, newton.Mesh):
                src._texture = None  # use the flat color, not the first material's texture


def yaw_quat(deg):
    return wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), math.radians(deg))


def add_room(builder):
    """Floor, three walls and a guard rail, all static boxes on the world body (-1)."""
    hx, hy, h = ROOM["half_x"], ROOM["half_y"], ROOM["wall_h"]
    t = 0.05
    box = builder.add_shape_box
    box(-1, xform=wp.transform((0.0, 0.0, -t), wp.quat_identity()), hx=hx, hy=hy, hz=t, color=FLOOR_COLOR,
        label="floor")
    box(-1, xform=wp.transform((0.0, hy + t, h / 2), wp.quat_identity()), hx=hx + 2 * t, hy=t, hz=h / 2,
        color=WALL_COLOR, label="wall_back")
    box(-1, xform=wp.transform((0.0, hy - 0.01, 0.95), wp.quat_identity()), hx=hx, hy=0.02, hz=0.06,
        color=TRIM_COLOR, label="rail_back")
    for sx in (-1.0, 1.0):
        box(-1, xform=wp.transform((sx * (hx + t), 0.0, h / 2), wp.quat_identity()), hx=t, hy=hy, hz=h / 2,
            color=WALL_COLOR, label=f"wall_side_{int(sx)}")
        box(-1, xform=wp.transform((sx * (hx - 0.01), 0.0, 0.95), wp.quat_identity()), hx=0.02, hy=hy, hz=0.06,
            color=TRIM_COLOR, label=f"rail_side_{int(sx)}")


def add_furniture(builder):
    """Import every LAYOUT entry from its SimReady USD (Newton variant) as static scenery."""
    placed = []
    for name, folder, pos, yaw in LAYOUT:
        s0 = builder.shape_count
        stage = open_newton_stage(folder)
        builder.add_usd(stage, xform=wp.transform(pos, yaw_quat(yaw)), floating=False,
                        ignore_paths=[".*/Joints/.*"], enable_self_collisions=False)
        recolor_shapes(builder, stage, s0, builder.shape_count)
        placed.append({"name": name, "asset": folder, "shapes": builder.shape_count - s0, "pos": pos, "yaw": yaw})
    return placed
