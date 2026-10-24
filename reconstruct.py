import os
import sys

import cv2
import numpy as np
import mediapipe as mp
import trimesh

from PIL import Image
from scipy.spatial import Delaunay

INPUT = "fabio.jpg"
OUTPUT = "fabio.glb"

if not os.path.exists(INPUT):
    sys.exit(f"Missing {INPUT}. Put the assigned photograph in this directory.")

# Load the photograph.
image_bgr = cv2.imread(INPUT)
if image_bgr is None:
    sys.exit("Could not read the photograph.")

image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
height, width = image_rgb.shape[:2]

# Detect the face and its 3D landmarks.
with mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
) as detector:
    result = detector.process(image_rgb)

if not result.multi_face_landmarks:
    sys.exit("No face detected. Try a clear, front-facing photograph.")

landmarks = result.multi_face_landmarks[0].landmark

# Convert image landmarks to model coordinates.
# X and Y follow the image plane; Z estimates relative facial depth.
points_2d = np.array(
    [[lm.x * width, lm.y * height] for lm in landmarks],
    dtype=np.float64,
)

vertices = np.array(
    [
        [
            (lm.x - 0.5) * width,
            (0.5 - lm.y) * height,
            -lm.z * width,
        ]
        for lm in landmarks
    ],
    dtype=np.float64,
)

# Triangulate the face landmarks using their image-plane positions.
triangulation = Delaunay(points_2d)
faces = triangulation.simplices.copy()

# Remove triangles spanning large gaps between landmarks.
# This reduces some unwanted triangles near the outer boundary.
kept_faces = []
for face in faces:
    triangle = points_2d[face]
    edges = [
        np.linalg.norm(triangle[0] - triangle[1]),
        np.linalg.norm(triangle[1] - triangle[2]),
        np.linalg.norm(triangle[2] - triangle[0]),
    ]
    if max(edges) < 0.12 * min(width, height):
        kept_faces.append(face)

faces = np.asarray(kept_faces, dtype=np.int64)

if len(faces) == 0:
    sys.exit("Triangulation failed.")

# Texture coordinates point back to the original photograph.
uv = np.array(
    [[lm.x, lm.y] for lm in landmarks],
    dtype=np.float64,
)

texture = Image.fromarray(image_rgb)

# Build the textured face surface.
mesh = trimesh.Trimesh(
    vertices=vertices,
    faces=faces,
    process=False,
)

mesh.visual = trimesh.visual.texture.TextureVisuals(
    uv=uv,
    image=texture,
)

# Export a glTF 2.0 binary file.
mesh.export(OUTPUT, file_type="glb")

size = os.path.getsize(OUTPUT)
print(f"Vertices: {len(vertices)}")
print(f"Triangles: {len(faces)}")
print(f"Output: {os.path.abspath(OUTPUT)}")
print(f"Size: {size:,} bytes")

if size >= 2_000_000:
    print("WARNING: GLB exceeds the 2,000,000-byte limit.")
else:
    print("PASS: GLB is below the 2,000,000-byte limit.")
