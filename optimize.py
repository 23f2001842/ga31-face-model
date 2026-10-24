import os
import trimesh
from PIL import Image

INPUT = "fabio.glb"
OUTPUT = "fabio-small.glb"

scene = trimesh.load(INPUT, force="scene")

for mesh in scene.geometry.values():
    visual = mesh.visual
    material = getattr(visual, "material", None)

    if material is None:
        continue

    # Trimesh PBRMaterial normally exposes its base-color texture
    # through baseColorTexture.
    image = getattr(material, "baseColorTexture", None)

    if image is None:
        print("No base-color texture found on this mesh.")
        continue

    image = image.convert("RGB")
    image.thumbnail((256, 256), Image.Resampling.LANCZOS)
    material.baseColorTexture = image

scene.export(OUTPUT, file_type="glb")

size = os.path.getsize(OUTPUT)
print(f"Output: {OUTPUT}")
print(f"Size: {size:,} bytes")
print("PASS" if size < 2_000_000 else "Still over 2 MB")
