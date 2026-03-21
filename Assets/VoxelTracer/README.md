# VoxelTracer

Real-time GPU voxelizer and ray marcher for Unity. Converts scene geometry (meshes, skinned meshes, terrains) into a 3D voxel grid on the GPU, then visualises it with a DDA ray march composited over the scene camera.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Scene Setup](#scene-setup)
3. [Components Reference](#components-reference)
4. [Algorithms](#algorithms)
5. [Static / Dynamic Split](#static--dynamic-split)
6. [GPU Pipeline](#gpu-pipeline)
7. [Ray Marching & Rendering](#ray-marching--rendering)
8. [Inspector Settings](#inspector-settings)
9. [Debug Tools](#debug-tools)
10. [Performance Notes](#performance-notes)
11. [File Reference](#file-reference)

---

## Architecture Overview

```
Scene Geometry (Meshes, Terrains, Skinned Meshes)
        │
        ▼  CPU: triangle extraction
  ┌─────────────────┐
  │ VoxelTracerSystem│  ← main controller
  └────────┬────────┘
           │  GPU dispatches
           ▼
  ┌──────────────────────────────────────────────┐
  │           VoxelTracerCore.compute             │
  │                                               │
  │  Clear → Surface SAT → Sweep Fill → Build Tex │
  │                  → Blur Fill → Compute Normals │
  └────────────────────┬─────────────────────────┘
                       │
            _FillTex (3D)   _NormalTex (3D)
                       │
                       ▼
  ┌──────────────────────────────────────┐
  │    VoxelTracerRayMarch.compute       │
  │    DDA ray march per screen pixel    │
  └──────────────────┬───────────────────┘
                     │
                     ▼
  ┌──────────────────────────────────────┐
  │    VoxelTracerComposite.shader       │
  │    Depth-aware blend over scene      │
  └──────────────────────────────────────┘
```

**Data flow at a glance:**
- CPU extracts triangles from scene renderers and terrains.
- GPU voxelises triangles into a `StructuredBuffer<uint>`, then flood-fills interior, then writes to `RenderTexture3D` (`_FillTex`).
- A separate blur pass smooths the fill for normal computation.
- The camera does full-screen DDA ray marching against the 3D texture and composites the result with depth awareness.

---

## Scene Setup

### Minimal Setup (3 steps)

1. **Create an empty GameObject** and add the **`VoxelTracerSystem`** component.
   - Assign the **`VoxelTracerCore`** compute shader to the `Core CS` slot.

2. **On your Main Camera**, add the **`VoxelTracerCamera`** component.
   - Assign the same `VoxelTracerSystem` to the `Voxel System` slot.
   - Assign the **`VoxelTracerRayMarch`** compute shader to the `Ray March CS` slot.

3. **Press Play.** All active `MeshRenderer`, `SkinnedMeshRenderer`, and `Terrain` objects in the scene will be voxelised automatically.

### Making Objects Dynamic

By default, all mesh renderers are **static** (voxelised once and cached). To make an object update every frame:

- Add the **`VoxelDynamic`** component to its GameObject.

Skinned mesh renderers are always treated as dynamic. Terrains are always static.

### Mesh Read/Write

Imported meshes (FBX, OBJ, glTF) must have **Read/Write Enabled** in their import settings:

> Select mesh asset → Inspector → Model tab → **Read/Write** checkbox → Apply

Unity built-in primitives (Cube, Sphere, Plane, etc.) are readable by default.

### Bounds

- **AutoFitScene** (default): bounds auto-computed from all geometry each static rebuild, with configurable padding.
- **Manual**: set `gridMin` and `gridMax` in world space for a fixed volume.

---

## Components Reference

| Component | Purpose | Attach To |
|-----------|---------|-----------|
| `VoxelTracerSystem` | Main voxelizer. Extracts triangles, dispatches GPU pipeline. | Any GameObject |
| `VoxelTracerCamera` | Ray march visualisation. Composites voxels over scene. | Camera |
| `VoxelDynamic` | Marker: flags a MeshRenderer as dynamic (re-voxelised per frame). | Dynamic GameObjects |
| `VoxelSliceViewer` | Debug: full-screen 2D slice viewer of the voxel volume. | Any GameObject |
| `VoxelNormalGizmos` | Debug: draws normal lines on surface voxels in Scene view. | Any GameObject |

---

## Algorithms

### Surface Voxelisation (SAT Triangle-AABB)

Each triangle is tested against every voxel in its axis-aligned bounding box using the **Separating Axis Theorem** (Tomas Akenine-Möller method). This is the gold-standard for conservative voxelisation — it catches all voxels a triangle touches, with no false negatives.

**13 separating axes tested:**
- 3 face normals of the AABB (X, Y, Z)
- 1 triangle face normal
- 9 cross products of AABB edges × triangle edges

Voxels that intersect are marked via `InterlockedOr` (atomic write), allowing all triangles to run in parallel without race conditions.

**Dispatch:** one thread per triangle, each thread tests its local AABB region.

### Axis-Sweep Flood Fill

The flood fill determines which voxels are **inside** closed geometry vs **outside**. It works by propagating an "outside" flag from the grid boundaries inward, stopping at surface voxels.

**Per axis (X, Y, Z), each thread sweeps one line:**
1. **Forward pass** (+direction): start from boundary face, propagate "outside" until hitting a surface voxel.
2. **Backward pass** (−direction): same from the opposite face.
3. Previously-marked outside voxels restart propagation (allows flowing around corners).

After all 3 axes (6 directions), unmarked non-surface voxels are **interior** (filled).

Multiple rounds handle concave geometry where a single round can't reach all cavities.

**Result:** `_FillTex[voxel] = 1.0` for surface + interior voxels, `0.0` for outside.

### Normal Computation (Blurred Gradient)

1. **Blur pass**: 3×3×3 box blur on the binary `_FillTex`, writing to `_BlurredFillTex`. This converts the hard 0/1 boundaries to smooth fractional values (0/27 through 27/27).

2. **Gradient pass**: central-difference gradient on the blurred texture, then normalize.

Without the blur, normals are limited to ~26 directions (the 3×3×3 integer gradient neighbourhood). With the blur, gradient components take many fractional values, yielding **thousands of unique normal directions** — smooth enough to eliminate visible faceting.

The blur does **not** affect `_FillTex` itself — ray march hit detection still uses the sharp binary volume.

---

## Static / Dynamic Split

The most expensive GPU step is SAT surface voxelisation (testing every triangle against its AABB voxels). The static/dynamic architecture avoids repeating this for geometry that doesn't move.

```
                    ┌─────────────────────────┐
  Static rebuild    │  Terrain + unmarked      │  ← runs once
  (on startup or    │  meshes → SAT → cache    │
   MarkStaticDirty) │  in _staticVoxelBuf      │
                    └────────────┬────────────┘
                                 │
                    ┌────────────┴────────────┐
  Per frame         │  Copy static cache       │  ← fast linear O(N)
                    │  + SAT dynamic tris only │  ← few triangles
                    │  + sweep fill + build    │
                    │  + blur + normals        │
                    └─────────────────────────┘
```

| Category | What's included | When voxelised |
|----------|----------------|----------------|
| **Static** | `MeshRenderer` without `VoxelDynamic`, all `Terrain` | Once at startup (or when `MarkStaticDirty()` called) |
| **Dynamic** | `MeshRenderer` with `VoxelDynamic`, all `SkinnedMeshRenderer` | Every frame |

### Triggering a Static Rebuild

- **Inspector:** right-click VoxelTracerSystem → "Rebuild Static"
- **Script:** `voxelSystem.MarkStaticDirty()` (rebuilds on next frame)
- **Full rebuild:** `voxelSystem.ForceRebuild()` (immediate)

---

## GPU Pipeline

### Per-Frame Dispatch Order

| Step | Kernel | Thread Group | Purpose |
|------|--------|-------------|---------|
| 1 | `CopyStaticToWorking` | (256,1,1) linear | Copy cached static voxels → working buffer |
| 2 | `ClearFlood` | (256,1,1) linear | Zero the flood buffer |
| 3 | `Surface` | (64,1,1) linear | SAT voxelise dynamic triangles only |
| 4 | `SweepFill` × 3 axes | (8,8,1) per sweep plane | Bidirectional flood fill |
| 5 | `BuildTexture` | (8,8,8) | Write `_FillTex` from surface + interior |
| 6 | `BlurFill` | (8,8,8) | 3×3×3 box blur → `_BlurredFillTex` |
| 7 | `ComputeNormals` | (8,8,8) | Gradient normals from blurred fill |

Steps 1–2 are lightweight linear passes (~0.1ms each).
Step 3 cost scales with **dynamic triangle count only**.
Steps 4–7 cost scales with **grid volume** (width × height × depth).

### GPU Buffers

| Buffer | Type | Purpose |
|--------|------|---------|
| `_VoxelBuffer` | `RWStructuredBuffer<uint>` | Working surface voxels (combined static + dynamic) |
| `_FloodBuffer` | `RWStructuredBuffer<uint>` | Flood fill outside markers |
| `_StaticVoxelBuffer` | `StructuredBuffer<uint>` | Cached static surface (read-only per frame) |
| `_FillTex` | `RWTexture3D<float>` | Binary fill volume (ray march reads this) |
| `_BlurredFillTex` | `RWTexture3D<float>` | Blurred fill (normal computation reads this) |
| `_NormalTex` | `RWTexture3D<float4>` | Gradient-based normals |

---

## Ray Marching & Rendering

### DDA Traversal

The ray march uses the **3D-DDA** (Digital Differential Analyser) algorithm — the same approach used in Wolfenstein-style raycasters, extended to 3D.

For each screen pixel:
1. Build a camera ray from NDC coordinates.
2. **Ray-AABB intersection** against the voxel volume bounds. If miss, output transparent.
3. Step into the grid. At each step, advance to the **nearest voxel face crossing** along whichever axis has the smallest `tMax`.
4. Sample `_FillTex` at the current voxel. If filled (`≥ 0.5`), it's a hit.
5. **Normal** is derived from the DDA step axis — whichever face the ray entered through. This gives clean axis-aligned normals per voxel face (6 directions).

### Compositing

Two compositor shaders are available:

- **`VoxelComposite.shader`** — simple alpha-over blend.
- **`VoxelTracerComposite.shader`** — depth-aware: compares voxel hit depth against scene depth buffer. Scene geometry in front of voxels correctly occludes them.

### Visualisation Modes

| Mode | Description |
|------|-------------|
| **Lit** | Lambertian shading with configurable light direction, colour, and ambient |
| **Normals** | Raw normal visualisation (RGB = XYZ mapped to 0–1) |

---

## Inspector Settings

### VoxelTracerSystem

| Section | Property | Default | Description |
|---------|----------|---------|-------------|
| **Grid** | `boundsMode` | AutoFitScene | Manual or auto-computed bounds |
| | `gridMin` / `gridMax` | (-10,-2,-10) / (10,10,10) | Manual bounds (when Manual mode) |
| | `voxelSize` | 0.25 | World-space size of one voxel |
| | `autoFitPadding` | 1.0 | Extra padding around auto-fit bounds |
| **Volume Fill** | `fillVolume` | true | Enable interior flood fill |
| | `fillSweepRounds` | 1 | Sweep rounds (1 = most cases, 2 = complex concavities) |
| **Scene Input** | `includeMeshRenderers` | true | Include MeshFilter/MeshRenderer objects |
| | `includeSkinnedMeshRenderers` | true | Include SkinnedMeshRenderer objects |
| | `includeTerrains` | true | Include Unity Terrain objects |
| | `terrainSampleStep` | 4 | Terrain heightmap sampling stride (higher = faster, coarser) |
| **Safety** | `maxVoxelsPerAxis` | 256 | Hard cap per grid dimension |
| | `maxVoxelCountMillions` | 32 | Total voxel budget |

### VoxelTracerCamera

| Property | Default | Description |
|----------|---------|-------------|
| `visMode` | Lit | Lit shading or raw normal view |
| `surfaceColor` | (0.85, 0.85, 0.85) | Base albedo for lit mode |
| `lightDirection` | (0.5, 1, 0.3) | Directional light vector |
| `lightColor` | White | Light colour |
| `ambientColor` | (0.12, 0.12, 0.18) | Ambient fill light |
| `maxSteps` | 1024 | DDA steps per ray (256–4096) |

---

## Debug Tools

### VoxelSliceViewer

Full-screen overlay that shows a 2D cross-section of the voxel volume.

| Control | Action |
|---------|--------|
| **F2** (default) | Toggle slice viewer on/off |
| **Scroll wheel** | Move slice position along current axis |
| **1 / 2 / 3** | Switch slice axis to X / Y / Z |

Surface voxels (those with at least one empty 6-connected neighbour) are highlighted differently from interior voxels.

### VoxelNormalGizmos

Draws coloured lines in the Scene view showing computed surface normals. Useful for verifying that the blur pass produces smooth normal distributions.

- **Surface-only mode**: only draws normals on voxels adjacent to empty space.
- **Configurable**: max line count, refresh interval, line length.

---

## Performance Notes

### Cost Breakdown (typical 128³ grid)

| Stage | Approximate Cost |
|-------|-----------------|
| CopyStatic + ClearFlood | ~0.2ms |
| Surface (dynamic tris only) | ~0.1–2ms depending on triangle count |
| SweepFill (3 axes × 1 round) | ~0.3ms |
| BuildTexture | ~0.1ms |
| BlurFill | ~0.2ms |
| ComputeNormals | ~0.1ms |
| Ray March (1080p, 1024 steps) | ~1–3ms |

### Tips for Higher Framerates

- **Increase `voxelSize`** — halving resolution (e.g., 0.25 → 0.5) reduces voxel count by 8×.
- **Reduce `maxSteps`** on the camera if rays don't need to traverse the full volume.
- **Mark static objects correctly** — only `VoxelDynamic`-tagged and skinned meshes pay the per-frame SAT cost.
- **Increase `terrainSampleStep`** — value of 8 or 16 for distant/large terrains.
- **`fillSweepRounds = 1`** is sufficient for most convex and simple concave geometry.
- **Set `fillVolume = false`** for open surfaces (planes, terrain-only scenes) where interior detection isn't needed.

### Open Surfaces

Unity Plane, single-sided meshes, and terrain are **open surfaces** — they have no enclosed interior. With `fillVolume` enabled, the flood fill correctly marks both sides as outside, producing a **1-voxel-thick surface layer**. This is correct behaviour, not a bug.

---

## File Reference

| File | Description |
|------|-------------|
| `VoxelTracerSystem.cs` | Main controller: CPU triangle extraction, GPU dispatch orchestration, static/dynamic cache |
| `VoxelTracerCore.compute` | GPU kernels: Clear, Surface SAT, SweepFill, BuildTexture, BlurFill, ComputeNormals, CopyStatic, ClearFlood |
| `VoxelTracerCamera.cs` | Camera component: dispatches ray march, handles compositing |
| `VoxelTracerRayMarch.compute` | GPU kernel: full-screen DDA ray march with lighting |
| `VoxelTracerComposite.shader` | Depth-aware compositor (voxels + scene) |
| `VoxelComposite.shader` | Simple alpha-over compositor |
| `VoxelDynamic.cs` | Marker component for per-frame voxelisation |
| `VoxelSliceViewer.cs` | Debug: 2D slice cross-section viewer |
| `VoxelNormalGizmos.cs` | Debug: surface normal line gizmos |
| `RandomRotator.cs` | Utility: random rotation for test objects |
