// SceneVoxelizer.cs
// CPU voxelizer for whole Unity scene (Meshes + SkinnedMeshes + Terrains)
// - Surface voxelization via Triangle–AABB intersection (SAT)
// - Solid fill via outside flood fill (optional)
// - Hollow objects via Shell mode (surface only) using layer mask or per-object override
// - Terrain voxelization via height sampling (fast)
// - Debug visualization: voxel wire cubes + surface normal lines (Gizmos)
//
// Usage:
// 1) Put this file under Assets/Scripts/SceneVoxelizer.cs
// 2) Add SceneVoxelizer component to an empty GameObject
// 3) Set voxelSize + boundsMode
// 4) Press Play (auto Build on Start) or use context menu "Build Voxel Grid"
// 5) Toggle drawGizmos / drawNormals to visualize
//
// Practical note:
// - Full-scene voxelization is heavy. Start with voxelSize 0.25 or 0.5.
// - If you hit "Voxel grid too large", increase voxelSize or use CustomBounds.

using System;
using System.Collections.Generic;
using UnityEngine;

[DisallowMultipleComponent]
public class SceneVoxelizer : MonoBehaviour
{
    public enum BoundsMode
    {
        WholeSceneRenderersAndTerrains,
        CustomBounds
    }

    public enum ObjectVoxelMode
    {
        Solid,
        Shell
    }

    [Header("Grid")]
    [Min(0.001f)] public float voxelSize = 0.25f;
    public BoundsMode boundsMode = BoundsMode.WholeSceneRenderersAndTerrains;
    public Bounds customBounds = new Bounds(Vector3.zero, Vector3.one * 10f);

    [Header("Object Mode Rules")]
    [Tooltip("Objects on these layers are treated as Shell (hollow). Others default to Solid (unless overridden by component).")]
    public LayerMask shellLayerMask = 0;

    [Tooltip("Default mode if no per-object override exists.")]
    public ObjectVoxelMode defaultMode = ObjectVoxelMode.Solid;

    [Header("Surface Thickness")]
    [Tooltip("Adds thickness around surface voxels (in voxel units). 0 = single voxel surface.")]
    [Range(0, 4)] public int surfaceThickness = 1;

    [Header("Solid Fill")]
    [Tooltip("If true, performs outside flood fill to compute inside solids for Solid meshes.")]
    public bool enableSolidFill = true;

    [Tooltip("Max total voxels allowed for the grid (safety).")]
    public int maxVoxels = 50_000_000;

    [Header("Terrain")]
    public bool voxelizeTerrains = true;
    public bool terrainAsSolid = true;

    [Header("Debug Draw")]
    public bool drawGizmos = true;
    public bool drawOnlySlice = true;
    [Range(0, 1)] public float sliceT = 0.5f;
    [Range(1, 200_000)] public int gizmoDrawLimit = 20_000;

    [Header("Normal Debug")]
    public bool drawNormals = true;
    [Range(0.01f, 5f)] public float normalLength = 0.5f;

    public VoxelGrid Grid { get; private set; }

    private HashSet<int> _solidVoxels = new HashSet<int>(1024);
    private HashSet<int> _fillSurface = new HashSet<int>(1024);
    private HashSet<int> _shellSurface = new HashSet<int>(1024);
    private Dictionary<int, Vector3> _cachedNormals;

    private void Start()
    {
        Build();
    }

    [ContextMenu("Build Voxel Grid")]
    public void Build()
    {
        ValidateConfigOrThrow();

        Bounds worldBounds = (boundsMode == BoundsMode.CustomBounds) ? customBounds : ComputeSceneBounds();
        if (worldBounds.size.x <= 0 || worldBounds.size.y <= 0 || worldBounds.size.z <= 0)
            throw new InvalidOperationException("SceneVoxelizer: computed bounds are invalid/empty.");

        Grid = VoxelGrid.Create(worldBounds, voxelSize, maxVoxels);

        _solidVoxels.Clear();
        _fillSurface.Clear();
        _shellSurface.Clear();
        _cachedNormals = null;

        if (voxelizeTerrains)
            VoxelizeTerrainsIntoSolid(Grid, _solidVoxels);

        VoxelizeSceneMeshesToSurfaces(Grid, _fillSurface, _shellSurface);

        if (_shellSurface.Count > 0)
        {
            var shellThick = (surfaceThickness > 0)
                ? Dilate(Grid, _shellSurface, surfaceThickness)
                : new HashSet<int>(_shellSurface);
            UnionInto(_solidVoxels, shellThick);
        }

        if (_fillSurface.Count > 0)
        {
            var walls = (surfaceThickness > 0)
                ? Dilate(Grid, _fillSurface, surfaceThickness)
                : new HashSet<int>(_fillSurface);

            UnionInto(_solidVoxels, walls);

            if (enableSolidFill)
            {
                var inside = ComputeInsideByOutsideFloodFill(Grid, blocked: _solidVoxels);
                UnionInto(_solidVoxels, inside);
            }
        }

        _cachedNormals = ComputeSurfaceNormalsFromOccupancy();

        Debug.Log($"SceneVoxelizer: Build done. Grid {Grid.Nx}x{Grid.Ny}x{Grid.Nz} = {Grid.VoxelCount} voxels. Solid count: {_solidVoxels.Count}. Normals: {_cachedNormals?.Count ?? 0}.");
    }

    public bool IsSolid(int x, int y, int z)
    {
        if (Grid == null) return false;
        if (!Grid.InBounds(x, y, z)) return false;
        return _solidVoxels.Contains(Grid.ToIndex(x, y, z));
    }

    public IEnumerable<int> EnumerateSolidIndices()
    {
        return _solidVoxels;
    }

    public IReadOnlyDictionary<int, Vector3> GetCachedSurfaceNormals()
    {
        return _cachedNormals;
    }

    private Bounds ComputeSceneBounds()
    {
        bool hasAny = false;
        Bounds b = new Bounds(Vector3.zero, Vector3.zero);

        var renderers = FindObjectsByType<Renderer>(FindObjectsSortMode.None);
        foreach (var r in renderers)
        {
            if (r == null || !r.enabled) continue;
            if (!hasAny) { b = r.bounds; hasAny = true; }
            else b.Encapsulate(r.bounds);
        }

        if (voxelizeTerrains)
        {
            var terrains = FindObjectsByType<Terrain>(FindObjectsSortMode.None);
            foreach (var t in terrains)
            {
                if (t == null || t.terrainData == null) continue;
                Bounds tb = TerrainWorldBounds(t);
                if (!hasAny) { b = tb; hasAny = true; }
                else b.Encapsulate(tb);
            }
        }

        if (!hasAny)
            b = new Bounds(Vector3.zero, Vector3.one);

        b.Expand(voxelSize * 2f);
        return b;
    }

    private static Bounds TerrainWorldBounds(Terrain t)
    {
        Vector3 size = t.terrainData.size;
        Vector3 center = t.transform.position + size * 0.5f;
        return new Bounds(center, size);
    }

    private void VoxelizeTerrainsIntoSolid(VoxelGrid grid, HashSet<int> solid)
    {
        if (grid == null) throw new ArgumentNullException(nameof(grid));
        if (solid == null) throw new ArgumentNullException(nameof(solid));

        var terrains = FindObjectsByType<Terrain>(FindObjectsSortMode.None);
        foreach (var t in terrains)
        {
            if (t == null || t.terrainData == null) continue;
            if (!t.gameObject.activeInHierarchy) continue;
            if (!terrainAsSolid) continue;

            VoxelizeSingleTerrainAsSolid(grid, t, solid);
        }
    }

    private void VoxelizeSingleTerrainAsSolid(VoxelGrid grid, Terrain t, HashSet<int> solid)
    {
        TerrainData td = t.terrainData;
        Vector3 tPos = t.transform.position;
        Vector3 tSize = td.size;

        Bounds tb = TerrainWorldBounds(t);
        Bounds overlap = IntersectBounds(grid.WorldBounds, tb);
        if (overlap.size.x <= 0 || overlap.size.y <= 0 || overlap.size.z <= 0) return;

        grid.WorldToVoxelBounds(overlap, out int x0, out int y0, out int z0, out int x1, out int y1, out int z1);

        x0 = Mathf.Clamp(x0, 0, grid.Nx - 1);
        x1 = Mathf.Clamp(x1, 0, grid.Nx - 1);
        z0 = Mathf.Clamp(z0, 0, grid.Nz - 1);
        z1 = Mathf.Clamp(z1, 0, grid.Nz - 1);

        for (int z = z0; z <= z1; z++)
        {
            for (int x = x0; x <= x1; x++)
            {
                Vector3 wp = grid.VoxelCenterWorld(x, 0, z);
                float u = (wp.x - tPos.x) / tSize.x;
                float v = (wp.z - tPos.z) / tSize.z;
                if (u < 0f || u > 1f || v < 0f || v > 1f) continue;

                float h = td.GetInterpolatedHeight(u, v) + tPos.y;

                int yMax = grid.WorldYToVoxelY(h);
                yMax = Mathf.Clamp(yMax, 0, grid.Ny - 1);

                for (int y = 0; y <= yMax; y++)
                    solid.Add(grid.ToIndex(x, y, z));
            }
        }
    }

    private static Bounds IntersectBounds(Bounds a, Bounds b)
    {
        Vector3 min = Vector3.Max(a.min, b.min);
        Vector3 max = Vector3.Min(a.max, b.max);
        Vector3 size = max - min;
        if (size.x <= 0 || size.y <= 0 || size.z <= 0) return new Bounds(Vector3.zero, Vector3.zero);
        return new Bounds((min + max) * 0.5f, size);
    }

    private void VoxelizeSceneMeshesToSurfaces(VoxelGrid grid, HashSet<int> fillSurface, HashSet<int> shellSurface)
    {
        if (grid == null) throw new ArgumentNullException(nameof(grid));
        if (fillSurface == null) throw new ArgumentNullException(nameof(fillSurface));
        if (shellSurface == null) throw new ArgumentNullException(nameof(shellSurface));

        var meshFilters = FindObjectsByType<MeshFilter>(FindObjectsSortMode.None);
        foreach (var mf in meshFilters)
        {
            if (mf == null) continue;
            var go = mf.gameObject;
            if (!go.activeInHierarchy) continue;

            var mr = go.GetComponent<MeshRenderer>();
            if (mr == null || !mr.enabled) continue;

            var mesh = mf.sharedMesh;
            if (mesh == null) continue;

            ObjectVoxelMode mode = GetModeForObject(go);
            var targetSet = (mode == ObjectVoxelMode.Solid) ? fillSurface : shellSurface;

            VoxelizeMeshSurface(grid, mesh, mf.transform.localToWorldMatrix, targetSet);
        }

        var skinned = FindObjectsByType<SkinnedMeshRenderer>(FindObjectsSortMode.None);
        Mesh baked = new Mesh();
        foreach (var smr in skinned)
        {
            if (smr == null || !smr.enabled) continue;
            if (!smr.gameObject.activeInHierarchy) continue;

            baked.Clear();
            try { smr.BakeMesh(baked); }
            catch { continue; }

            ObjectVoxelMode mode = GetModeForObject(smr.gameObject);
            var targetSet = (mode == ObjectVoxelMode.Solid) ? fillSurface : shellSurface;

            VoxelizeMeshSurface(grid, baked, smr.transform.localToWorldMatrix, targetSet);
        }
    }

    private ObjectVoxelMode GetModeForObject(GameObject go)
    {
        if (go == null) return defaultMode;
        var overrideComp = go.GetComponent<VoxelModeOverride>();
        if (overrideComp != null) return overrideComp.mode;

        bool isShellLayer = (shellLayerMask.value & (1 << go.layer)) != 0;
        return isShellLayer ? ObjectVoxelMode.Shell : defaultMode;
    }

    private void VoxelizeMeshSurface(VoxelGrid grid, Mesh mesh, Matrix4x4 localToWorld, HashSet<int> surfaceOut)
    {
        if (grid == null) throw new ArgumentNullException(nameof(grid));
        if (mesh == null) throw new ArgumentNullException(nameof(mesh));
        if (surfaceOut == null) throw new ArgumentNullException(nameof(surfaceOut));

        var verts = mesh.vertices;
        var tris = mesh.triangles;
        if (verts == null || verts.Length == 0) return;
        if (tris == null || tris.Length < 3) return;

        Vector3[] wv = new Vector3[verts.Length];
        for (int i = 0; i < verts.Length; i++)
            wv[i] = localToWorld.MultiplyPoint3x4(verts[i]);

        for (int i = 0; i < tris.Length; i += 3)
        {
            int i0 = tris[i];
            int i1 = tris[i + 1];
            int i2 = tris[i + 2];

            if ((uint)i0 >= (uint)wv.Length || (uint)i1 >= (uint)wv.Length || (uint)i2 >= (uint)wv.Length)
                continue;

            Vector3 a = wv[i0];
            Vector3 b = wv[i1];
            Vector3 c = wv[i2];

            Vector3 tMin = Vector3.Min(a, Vector3.Min(b, c));
            Vector3 tMax = Vector3.Max(a, Vector3.Max(b, c));
            Bounds triBounds = new Bounds((tMin + tMax) * 0.5f, tMax - tMin);

            if (!grid.WorldBounds.Intersects(triBounds)) continue;

            grid.WorldToVoxelBounds(triBounds, out int x0, out int y0, out int z0, out int x1, out int y1, out int z1);

            x0 = Mathf.Clamp(x0, 0, grid.Nx - 1);
            y0 = Mathf.Clamp(y0, 0, grid.Ny - 1);
            z0 = Mathf.Clamp(z0, 0, grid.Nz - 1);
            x1 = Mathf.Clamp(x1, 0, grid.Nx - 1);
            y1 = Mathf.Clamp(y1, 0, grid.Ny - 1);
            z1 = Mathf.Clamp(z1, 0, grid.Nz - 1);

            for (int z = z0; z <= z1; z++)
                for (int y = y0; y <= y1; y++)
                    for (int x = x0; x <= x1; x++)
                    {
                        AABB box = grid.VoxelAABB(x, y, z);
                        if (TriAabbIntersect(a, b, c, box))
                            surfaceOut.Add(grid.ToIndex(x, y, z));
                    }
        }
    }

    private static HashSet<int> Dilate(VoxelGrid grid, HashSet<int> src, int radius)
    {
        if (grid == null) throw new ArgumentNullException(nameof(grid));
        if (src == null) throw new ArgumentNullException(nameof(src));
        if (radius <= 0) return new HashSet<int>(src);

        var outSet = new HashSet<int>(src.Count * 2);
        foreach (int idx in src)
        {
            if (!grid.TryFromIndex(idx, out int x, out int y, out int z)) continue;

            for (int dz = -radius; dz <= radius; dz++)
                for (int dy = -radius; dy <= radius; dy++)
                    for (int dx = -radius; dx <= radius; dx++)
                    {
                        int nx = x + dx, ny = y + dy, nz = z + dz;
                        if (!grid.InBounds(nx, ny, nz)) continue;
                        outSet.Add(grid.ToIndex(nx, ny, nz));
                    }
        }
        return outSet;
    }

    private static void UnionInto(HashSet<int> dst, HashSet<int> src)
    {
        if (dst == null) throw new ArgumentNullException(nameof(dst));
        if (src == null) return;
        foreach (var v in src) dst.Add(v);
    }

    private static HashSet<int> ComputeInsideByOutsideFloodFill(VoxelGrid grid, HashSet<int> blocked)
    {
        if (grid == null) throw new ArgumentNullException(nameof(grid));
        if (blocked == null) throw new ArgumentNullException(nameof(blocked));

        var outsideVisited = new HashSet<int>(1024);
        var q = new Queue<int>(1024);

        void EnqueueIfEmpty(int x, int y, int z)
        {
            if (!grid.InBounds(x, y, z)) return;
            int idx = grid.ToIndex(x, y, z);
            if (blocked.Contains(idx)) return;
            if (outsideVisited.Add(idx)) q.Enqueue(idx);
        }

        for (int x = 0; x < grid.Nx; x++)
            for (int z = 0; z < grid.Nz; z++)
            {
                EnqueueIfEmpty(x, 0, z);
                EnqueueIfEmpty(x, grid.Ny - 1, z);
            }

        for (int y = 0; y < grid.Ny; y++)
            for (int z = 0; z < grid.Nz; z++)
            {
                EnqueueIfEmpty(0, y, z);
                EnqueueIfEmpty(grid.Nx - 1, y, z);
            }

        for (int x = 0; x < grid.Nx; x++)
            for (int y = 0; y < grid.Ny; y++)
            {
                EnqueueIfEmpty(x, y, 0);
                EnqueueIfEmpty(x, y, grid.Nz - 1);
            }

        while (q.Count > 0)
        {
            int idx = q.Dequeue();
            if (!grid.TryFromIndex(idx, out int x, out int y, out int z)) continue;

            TryVisit(x + 1, y, z);
            TryVisit(x - 1, y, z);
            TryVisit(x, y + 1, z);
            TryVisit(x, y - 1, z);
            TryVisit(x, y, z + 1);
            TryVisit(x, y, z - 1);

            void TryVisit(int nx, int ny, int nz)
            {
                if (!grid.InBounds(nx, ny, nz)) return;
                int nidx = grid.ToIndex(nx, ny, nz);
                if (outsideVisited.Contains(nidx)) return;
                if (blocked.Contains(nidx)) return;
                outsideVisited.Add(nidx);
                q.Enqueue(nidx);
            }
        }

        var inside = new HashSet<int>(1024);
        int total = grid.VoxelCount;
        for (int idx = 0; idx < total; idx++)
        {
            if (blocked.Contains(idx)) continue;
            if (outsideVisited.Contains(idx)) continue;
            inside.Add(idx);
        }

        return inside;
    }

    public Dictionary<int, Vector3> ComputeSurfaceNormalsFromOccupancy()
    {
        if (Grid == null) throw new InvalidOperationException("Grid is null. Call Build() first.");

        var normals = new Dictionary<int, Vector3>(_solidVoxels.Count);

        foreach (int idx in _solidVoxels)
        {
            if (!Grid.TryFromIndex(idx, out int x, out int y, out int z)) continue;
            if (!IsSurfaceVoxel(x, y, z)) continue;

            float xm = SampleOcc(x - 1, y, z);
            float xp = SampleOcc(x + 1, y, z);
            float ym = SampleOcc(x, y - 1, z);
            float yp = SampleOcc(x, y + 1, z);
            float zm = SampleOcc(x, y, z - 1);
            float zp = SampleOcc(x, y, z + 1);

            Vector3 g = new Vector3(xm - xp, ym - yp, zm - zp);
            if (g.sqrMagnitude < 1e-12f) continue;

            normals[idx] = g.normalized;
        }

        return normals;

        float SampleOcc(int ix, int iy, int iz)
        {
            if (!Grid.InBounds(ix, iy, iz)) return 0f;
            return _solidVoxels.Contains(Grid.ToIndex(ix, iy, iz)) ? 1f : 0f;
        }

        bool IsSurfaceVoxel(int ix, int iy, int iz)
        {
            if (!Grid.InBounds(ix, iy, iz)) return false;
            return
                !IsSolidInternal(ix + 1, iy, iz) ||
                !IsSolidInternal(ix - 1, iy, iz) ||
                !IsSolidInternal(ix, iy + 1, iz) ||
                !IsSolidInternal(ix, iy - 1, iz) ||
                !IsSolidInternal(ix, iy, iz + 1) ||
                !IsSolidInternal(ix, iy, iz - 1);
        }

        bool IsSolidInternal(int ix, int iy, int iz)
        {
            if (!Grid.InBounds(ix, iy, iz)) return false;
            return _solidVoxels.Contains(Grid.ToIndex(ix, iy, iz));
        }
    }

    private void ValidateConfigOrThrow()
    {
        if (voxelSize <= 0f)
            throw new ArgumentOutOfRangeException(nameof(voxelSize), "voxelSize must be > 0.");

        if (maxVoxels < 1_000)
            throw new ArgumentOutOfRangeException(nameof(maxVoxels), "maxVoxels is too small.");

        if (boundsMode == BoundsMode.CustomBounds)
        {
            if (customBounds.size.x <= 0 || customBounds.size.y <= 0 || customBounds.size.z <= 0)
                throw new InvalidOperationException("customBounds must have positive size.");
        }
    }

    private void OnDrawGizmos()
    {
        if (!drawGizmos) return;
        if (Grid == null) return;
        if (_solidVoxels == null) return;

        Gizmos.matrix = Matrix4x4.identity;

        int ySlice = Mathf.Clamp(Mathf.RoundToInt(sliceT * (Grid.Ny - 1)), 0, Grid.Ny - 1);

        Gizmos.color = Color.cyan;

        int drawn = 0;
        foreach (var idx in _solidVoxels)
        {
            if (!Grid.TryFromIndex(idx, out int x, out int y, out int z)) continue;
            if (drawOnlySlice && y != ySlice) continue;

            Vector3 c = Grid.VoxelCenterWorld(x, y, z);
            Gizmos.DrawWireCube(c, Vector3.one * Grid.VoxelSize);

            drawn++;
            if (drawn >= gizmoDrawLimit) break;
        }

        if (drawNormals && _cachedNormals != null)
        {
            Gizmos.color = Color.red;
            int ndrawn = 0;

            foreach (var pair in _cachedNormals)
            {
                int idx = pair.Key;
                Vector3 n = pair.Value;

                if (!Grid.TryFromIndex(idx, out int x, out int y, out int z)) continue;
                if (drawOnlySlice && y != ySlice) continue;

                Vector3 c = Grid.VoxelCenterWorld(x, y, z);
                Gizmos.DrawLine(c, c + n * normalLength);

                ndrawn++;
                if (ndrawn >= gizmoDrawLimit) break;
            }
        }

        Gizmos.color = Color.yellow;
        Gizmos.DrawWireCube(Grid.WorldBounds.center, Grid.WorldBounds.size);
    }

    public sealed class VoxelGrid
    {
        public Bounds WorldBounds { get; private set; }
        public float VoxelSize { get; private set; }
        public int Nx { get; private set; }
        public int Ny { get; private set; }
        public int Nz { get; private set; }
        public int VoxelCount { get; private set; }

        private Vector3 _min;

        private VoxelGrid() { }

        public static VoxelGrid Create(Bounds worldBounds, float voxelSize, int maxVoxels)
        {
            if (voxelSize <= 0f) throw new ArgumentOutOfRangeException(nameof(voxelSize));

            Vector3 size = worldBounds.size;
            int nx = Mathf.Max(1, Mathf.CeilToInt(size.x / voxelSize));
            int ny = Mathf.Max(1, Mathf.CeilToInt(size.y / voxelSize));
            int nz = Mathf.Max(1, Mathf.CeilToInt(size.z / voxelSize));

            long total = (long)nx * ny * nz;
            if (total > maxVoxels)
                throw new InvalidOperationException($"Voxel grid too large: {nx}x{ny}x{nz} = {total} > maxVoxels({maxVoxels}). Increase voxelSize or reduce bounds.");

            Vector3 snappedSize = new Vector3(nx, ny, nz) * voxelSize;
            Bounds snapped = new Bounds(worldBounds.min + snappedSize * 0.5f, snappedSize);

            return new VoxelGrid
            {
                WorldBounds = snapped,
                VoxelSize = voxelSize,
                Nx = nx,
                Ny = ny,
                Nz = nz,
                VoxelCount = (int)total,
                _min = snapped.min
            };
        }

        public bool InBounds(int x, int y, int z)
        {
            return (uint)x < (uint)Nx && (uint)y < (uint)Ny && (uint)z < (uint)Nz;
        }

        public int ToIndex(int x, int y, int z)
        {
            return x + Nx * (y + Ny * z);
        }

        public bool TryFromIndex(int idx, out int x, out int y, out int z)
        {
            x = y = z = 0;
            if ((uint)idx >= (uint)VoxelCount) return false;

            int xy = Nx * Ny;
            z = idx / xy;
            int rem = idx - z * xy;
            y = rem / Nx;
            x = rem - y * Nx;
            return true;
        }

        public Vector3 VoxelCenterWorld(int x, int y, int z)
        {
            return _min + new Vector3((x + 0.5f) * VoxelSize, (y + 0.5f) * VoxelSize, (z + 0.5f) * VoxelSize);
        }

        public AABB VoxelAABB(int x, int y, int z)
        {
            Vector3 min = _min + new Vector3(x * VoxelSize, y * VoxelSize, z * VoxelSize);
            Vector3 max = min + Vector3.one * VoxelSize;
            return new AABB(min, max);
        }

        public int WorldYToVoxelY(float worldY)
        {
            float fy = (worldY - _min.y) / VoxelSize - 0.5f;
            return Mathf.FloorToInt(fy);
        }

        public void WorldToVoxelBounds(Bounds b, out int x0, out int y0, out int z0, out int x1, out int y1, out int z1)
        {
            Vector3 min = b.min;
            Vector3 max = b.max;

            x0 = Mathf.FloorToInt((min.x - _min.x) / VoxelSize);
            y0 = Mathf.FloorToInt((min.y - _min.y) / VoxelSize);
            z0 = Mathf.FloorToInt((min.z - _min.z) / VoxelSize);

            x1 = Mathf.FloorToInt((max.x - _min.x) / VoxelSize);
            y1 = Mathf.FloorToInt((max.y - _min.y) / VoxelSize);
            z1 = Mathf.FloorToInt((max.z - _min.z) / VoxelSize);
        }
    }

    public readonly struct AABB
    {
        public readonly Vector3 Min;
        public readonly Vector3 Max;
        public Vector3 Center => (Min + Max) * 0.5f;
        public Vector3 Extents => (Max - Min) * 0.5f;

        public AABB(Vector3 min, Vector3 max)
        {
            Min = min;
            Max = max;
        }
    }

    private static bool TriAabbIntersect(Vector3 v0, Vector3 v1, Vector3 v2, AABB box)
    {
        Vector3 c = box.Center;
        Vector3 e = box.Extents;

        Vector3 tv0 = v0 - c;
        Vector3 tv1 = v1 - c;
        Vector3 tv2 = v2 - c;

        Vector3 f0 = tv1 - tv0;
        Vector3 f1 = tv2 - tv1;
        Vector3 f2 = tv0 - tv2;

        if (!AxisTestCross(f0, tv0, tv1, tv2, e)) return false;
        if (!AxisTestCross(f1, tv0, tv1, tv2, e)) return false;
        if (!AxisTestCross(f2, tv0, tv1, tv2, e)) return false;

        if (!OverlapOnAxis(tv0.x, tv1.x, tv2.x, e.x)) return false;
        if (!OverlapOnAxis(tv0.y, tv1.y, tv2.y, e.y)) return false;
        if (!OverlapOnAxis(tv0.z, tv1.z, tv2.z, e.z)) return false;

        Vector3 n = Vector3.Cross(f0, f1);
        if (!PlaneBoxOverlap(n, tv0, e)) return false;

        return true;
    }

    private static bool OverlapOnAxis(float a, float b, float c, float r)
    {
        float min = Mathf.Min(a, Mathf.Min(b, c));
        float max = Mathf.Max(a, Mathf.Max(b, c));
        return !(min > r || max < -r);
    }

    private static bool AxisTestCross(Vector3 edge, Vector3 v0, Vector3 v1, Vector3 v2, Vector3 ext)
    {
        if (!AxisTest(new Vector3(0f, edge.z, -edge.y), v0, v1, v2, ext)) return false;
        if (!AxisTest(new Vector3(-edge.z, 0f, edge.x), v0, v1, v2, ext)) return false;
        if (!AxisTest(new Vector3(edge.y, -edge.x, 0f), v0, v1, v2, ext)) return false;
        return true;
    }

    private static bool AxisTest(Vector3 axis, Vector3 v0, Vector3 v1, Vector3 v2, Vector3 ext)
    {
        float ax2 = axis.x * axis.x + axis.y * axis.y + axis.z * axis.z;
        if (ax2 < 1e-20f) return true;

        float p0 = Vector3.Dot(v0, axis);
        float p1 = Vector3.Dot(v1, axis);
        float p2 = Vector3.Dot(v2, axis);

        float min = Mathf.Min(p0, Mathf.Min(p1, p2));
        float max = Mathf.Max(p0, Mathf.Max(p1, p2));

        float r =
            ext.x * Mathf.Abs(axis.x) +
            ext.y * Mathf.Abs(axis.y) +
            ext.z * Mathf.Abs(axis.z);

        return !(min > r || max < -r);
    }

    private static bool PlaneBoxOverlap(Vector3 normal, Vector3 vert, Vector3 maxBox)
    {
        float r =
            maxBox.x * Mathf.Abs(normal.x) +
            maxBox.y * Mathf.Abs(normal.y) +
            maxBox.z * Mathf.Abs(normal.z);

        float s = Vector3.Dot(normal, vert);
        return Mathf.Abs(s) <= r;
    }
}

public sealed class VoxelModeOverride : MonoBehaviour
{
    public SceneVoxelizer.ObjectVoxelMode mode = SceneVoxelizer.ObjectVoxelMode.Solid;
}