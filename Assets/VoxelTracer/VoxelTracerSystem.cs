using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;

/// <summary>
/// GPU voxelizer with static/dynamic split:
/// - Static geometry (terrain, unmarked meshes) is voxelized once and cached on GPU.
/// - Dynamic geometry (VoxelDynamic-tagged, skinned meshes) is re-voxelized every frame.
/// - Per-frame cost: 1 buffer copy + dynamic SAT + sweep fill + build + normals.
/// </summary>
public sealed class VoxelTracerSystem : MonoBehaviour
{
    // ================================================================
    // Inspector
    // ================================================================

    [Header("Compute Shaders")]
    public ComputeShader coreCS;

    [Header("Grid")]
    public BoundsMode boundsMode = BoundsMode.AutoFitScene;
    [Tooltip("Only used when Bounds Mode = Manual")]
    public Vector3 gridMin = new Vector3(-10, -2, -10);
    [Tooltip("Only used when Bounds Mode = Manual")]
    public Vector3 gridMax = new Vector3(10, 10, 10);
    [Min(0.01f)] public float voxelSize = 0.25f;
    [Tooltip("Padding (in world units) added around auto-fit bounds")]
    [Min(0)] public float autoFitPadding = 1f;

    [Header("Volume Fill")]
    [Tooltip("Fill interior volume between front and back surfaces")]
    public bool fillVolume = true;
    [Tooltip("Number of sweep rounds for flood fill (1 handles most geometry, 2 for complex concavities)")]
    [Range(1, 4)] public int fillSweepRounds = 1;

    [Header("Scene Input")]
    public bool includeMeshRenderers = true;
    public bool includeSkinnedMeshRenderers = true;
    public bool includeTerrains = true;
    [Range(1, 32)] public int terrainSampleStep = 4;

    [Header("Safety")]
    [Range(32, 512)] public int maxVoxelsPerAxis = 256;
    [Min(1)] public float maxVoxelCountMillions = 32f;

    // ================================================================
    // Enums / Structs
    // ================================================================

    public enum BoundsMode { Manual, AutoFitScene }

    [StructLayout(LayoutKind.Sequential)]
    struct Tri
    {
        public Vector3 a, b, c, n;
    }

    // ================================================================
    // Public accessors for the camera
    // ================================================================

    public RenderTexture FillTexture => _fillTex;
    public RenderTexture NormalsTexture => _normalsTex;
    public int Nx => _nx;
    public int Ny => _ny;
    public int Nz => _nz;
    public Vector3 ActiveGridMin => _activeMin;
    public float ActiveVoxelSize => voxelSize;
    public bool IsReady => _fillTex != null && _nx > 0;

    // ================================================================
    // Private state
    // ================================================================

    // Kernel indices
    int KClear, KSurface, KSweepFill, KBuildTexture, KComputeNormals;
    int KCopyStaticToWorking, KClearFlood, KBlurFill;
    bool _kernelsCached;

    // GPU buffers
    ComputeBuffer _voxelBuffer;      // working buffer (combined static + dynamic)
    ComputeBuffer _floodBuffer;      // flood fill outside markers
    ComputeBuffer _staticVoxelBuf;   // cached static surface voxels
    ComputeBuffer _staticTriBuffer;  // static triangles (uploaded once)
    ComputeBuffer _dynamicTriBuffer; // dynamic triangles (uploaded every frame)
    int _staticTriCount;
    int _dynamicTriCount;

    // Textures
    RenderTexture _fillTex;
    RenderTexture _blurredFillTex;
    RenderTexture _normalsTex;

    // Grid state
    int _nx, _ny, _nz;
    int _totalVoxels;
    Vector3 _activeMin, _activeMax;

    // Triangle lists (CPU)
    readonly List<Tri> _staticTriList = new List<Tri>(128 * 1024);
    readonly List<Tri> _dynamicTriList = new List<Tri>(16 * 1024);
    Mesh _bakedMesh;

    // Dirty flags
    bool _staticDirty = true;      // rebuild static tris + re-voxelize static layer
    bool _hasDynamicObjects;       // any dynamic objects exist in scene

    // ================================================================
    // Lifecycle
    // ================================================================

    void OnEnable()
    {
        if (coreCS == null) return;
        CacheKernels();
        RebuildStatic();
        VoxelizeFrame();
    }

    void OnDisable()
    {
        ReleaseAll();
    }

    void LateUpdate()
    {
        if (coreCS == null) return;

        if (_staticDirty)
            RebuildStatic();

        VoxelizeFrame();
    }

    // ================================================================
    // Public API
    // ================================================================

    /// <summary>Call when static geometry changes (e.g. terrain edited, static objects added/removed).</summary>
    [ContextMenu("Rebuild Static")]
    public void MarkStaticDirty() => _staticDirty = true;

    /// <summary>Full rebuild of everything.</summary>
    [ContextMenu("Force Voxelize")]
    public void ForceRebuild()
    {
        _staticDirty = true;
        RebuildStatic();
        VoxelizeFrame();
    }

    // ================================================================
    // Static rebuild (runs once, or when MarkStaticDirty called)
    // ================================================================

    void RebuildStatic()
    {
        if (!_kernelsCached) CacheKernels();
        _staticDirty = false;

        // Gather ALL triangles (static + dynamic) to compute bounds
        BuildTriangleLists();

        int totalTris = _staticTriList.Count + _dynamicTriList.Count;
        if (totalTris == 0) { _nx = _ny = _nz = 0; return; }

        // Compute bounds from all geometry (static + dynamic)
        ComputeBoundsFromBothLists(out Vector3 mn, out Vector3 mx);
        _activeMin = mn;
        _activeMax = mx;

        ComputeGridSize(mn, mx, out int gx, out int gy, out int gz);
        _nx = gx; _ny = gy; _nz = gz;
        _totalVoxels = gx * gy * gz;

        AllocateResources(gx, gy, gz);

        // Upload static triangles
        UploadStaticTriangles();

        if (_staticTriCount == 0)
        {
            // No static geometry — just clear the static voxel cache
            ClearStaticVoxelCache();
            return;
        }

        // Voxelize static geometry once → store in _staticVoxelBuf
        SetGridUniforms(gx, gy, gz);
        BindClearBuffers();
        Dispatch3D(KClear, gx, gy, gz);

        coreCS.SetBuffer(KSurface, "_VoxelBuffer", _voxelBuffer);
        coreCS.SetBuffer(KSurface, "_Tris", _staticTriBuffer);
        coreCS.SetInt("_TriCount", _staticTriCount);
        DispatchLinear(KSurface, _staticTriCount);

        // Copy result to static cache
        CopyWorkingToStaticCache();
    }

    void ClearStaticVoxelCache()
    {
        // Fill static cache with zeros
        var zeros = new uint[_totalVoxels];
        _staticVoxelBuf.SetData(zeros);
    }

    void CopyWorkingToStaticCache()
    {
        // GPU→CPU→GPU copy via GetData/SetData for the one-time static bake
        var data = new uint[_totalVoxels];
        _voxelBuffer.GetData(data);
        _staticVoxelBuf.SetData(data);
    }

    // ================================================================
    // Per-frame voxelization (fast path)
    // ================================================================

    void VoxelizeFrame()
    {
        if (_nx == 0 || _fillTex == null) return;

        int gx = _nx, gy = _ny, gz = _nz;

        // Rebuild dynamic triangle list every frame
        BuildDynamicTriangleList();
        UploadDynamicTriangles();

        SetGridUniforms(gx, gy, gz);

        // 1) Copy cached static voxels → working buffer (fast linear kernel)
        coreCS.SetInt("_TotalVoxels", _totalVoxels);
        coreCS.SetBuffer(KCopyStaticToWorking, "_VoxelBuffer", _voxelBuffer);
        coreCS.SetBuffer(KCopyStaticToWorking, "_StaticVoxelBuffer", _staticVoxelBuf);
        DispatchLinear(KCopyStaticToWorking, _totalVoxels);

        // 2) Clear flood buffer only (not voxels — we just populated them)
        coreCS.SetBuffer(KClearFlood, "_FloodBuffer", _floodBuffer);
        coreCS.SetInt("_TotalVoxels", _totalVoxels);
        DispatchLinear(KClearFlood, _totalVoxels);

        // 3) Surface voxelization — dynamic triangles only (atomic OR on top of static)
        if (_dynamicTriCount > 0)
        {
            coreCS.SetBuffer(KSurface, "_VoxelBuffer", _voxelBuffer);
            coreCS.SetBuffer(KSurface, "_Tris", _dynamicTriBuffer);
            coreCS.SetInt("_TriCount", _dynamicTriCount);
            DispatchLinear(KSurface, _dynamicTriCount);
        }

        // 4) Sweep flood fill
        if (fillVolume)
        {
            coreCS.SetBuffer(KSweepFill, "_VoxelBuffer", _voxelBuffer);
            coreCS.SetBuffer(KSweepFill, "_FloodBuffer", _floodBuffer);
            for (int round = 0; round < fillSweepRounds; round++)
            {
                DispatchSweep(0, gy, gz);
                DispatchSweep(1, gx, gz);
                DispatchSweep(2, gx, gy);
            }
        }

        // 5) Build fill texture
        coreCS.SetInt("_FillVolume", fillVolume ? 1 : 0);
        coreCS.SetBuffer(KBuildTexture, "_VoxelBuffer", _voxelBuffer);
        coreCS.SetBuffer(KBuildTexture, "_FloodBuffer", _floodBuffer);
        coreCS.SetTexture(KBuildTexture, "_FillTex", _fillTex);
        Dispatch3D(KBuildTexture, gx, gy, gz);

        // 6) Blur fill texture for smooth normals
        coreCS.SetTexture(KBlurFill, "_FillTex", _fillTex);
        coreCS.SetTexture(KBlurFill, "_BlurredFillTex", _blurredFillTex);
        Dispatch3D(KBlurFill, gx, gy, gz);

        // 7) Compute normals from blurred fill
        coreCS.SetTexture(KComputeNormals, "_FillTex", _fillTex);
        coreCS.SetTexture(KComputeNormals, "_BlurredFillTex", _blurredFillTex);
        coreCS.SetTexture(KComputeNormals, "_NormalTex", _normalsTex);
        Dispatch3D(KComputeNormals, gx, gy, gz);
    }

    void SetGridUniforms(int gx, int gy, int gz)
    {
        coreCS.SetInt("_Width", gx);
        coreCS.SetInt("_Height", gy);
        coreCS.SetInt("_Depth", gz);
        coreCS.SetVector("_Start", _activeMin);
        coreCS.SetFloat("_Unit", voxelSize);
        coreCS.SetFloat("_HalfUnit", voxelSize * 0.5f);
    }

    void BindClearBuffers()
    {
        coreCS.SetBuffer(KClear, "_VoxelBuffer", _voxelBuffer);
        coreCS.SetBuffer(KClear, "_FloodBuffer", _floodBuffer);
        coreCS.SetTexture(KClear, "_FillTex", _fillTex);
        coreCS.SetTexture(KClear, "_NormalTex", _normalsTex);
    }

    // ================================================================
    // Triangle extraction (CPU)
    // ================================================================

    void BuildTriangleLists()
    {
        _staticTriList.Clear();
        _dynamicTriList.Clear();
        _hasDynamicObjects = false;

        if (includeMeshRenderers)
        {
            var filters = FindObjectsByType<MeshFilter>(FindObjectsSortMode.None);
            foreach (var mf in filters)
            {
                if (mf == null || !mf.gameObject.activeInHierarchy) continue;
                var mr = mf.GetComponent<MeshRenderer>();
                if (mr == null || !mr.enabled) continue;
                if (mf.sharedMesh == null) continue;

                bool isDynamic = mf.GetComponent<VoxelDynamic>() != null;
                if (isDynamic)
                {
                    AppendMesh(mf.sharedMesh, mf.transform.localToWorldMatrix, _dynamicTriList);
                    _hasDynamicObjects = true;
                }
                else
                {
                    AppendMesh(mf.sharedMesh, mf.transform.localToWorldMatrix, _staticTriList);
                }
            }
        }

        if (includeSkinnedMeshRenderers)
        {
            if (_bakedMesh == null) _bakedMesh = new Mesh();
            var skins = FindObjectsByType<SkinnedMeshRenderer>(FindObjectsSortMode.None);
            foreach (var smr in skins)
            {
                if (smr == null || !smr.enabled || !smr.gameObject.activeInHierarchy) continue;
                _bakedMesh.Clear();
                try { smr.BakeMesh(_bakedMesh); } catch { continue; }
                AppendMesh(_bakedMesh, smr.transform.localToWorldMatrix, _dynamicTriList);
                _hasDynamicObjects = true;
            }
        }

        if (includeTerrains)
        {
            var terrains = FindObjectsByType<Terrain>(FindObjectsSortMode.None);
            foreach (var t in terrains)
            {
                if (t == null || !t.isActiveAndEnabled) continue;
                AppendTerrain(t, terrainSampleStep, _staticTriList);
            }
        }
    }

    /// <summary>Lightweight per-frame rebuild of dynamic triangles only.</summary>
    void BuildDynamicTriangleList()
    {
        _dynamicTriList.Clear();

        if (includeMeshRenderers)
        {
            var dynamics = FindObjectsByType<VoxelDynamic>(FindObjectsSortMode.None);
            foreach (var vd in dynamics)
            {
                if (vd == null || !vd.gameObject.activeInHierarchy) continue;
                var mf = vd.GetComponent<MeshFilter>();
                if (mf == null || mf.sharedMesh == null) continue;
                var mr = vd.GetComponent<MeshRenderer>();
                if (mr == null || !mr.enabled) continue;
                AppendMesh(mf.sharedMesh, mf.transform.localToWorldMatrix, _dynamicTriList);
            }
        }

        if (includeSkinnedMeshRenderers)
        {
            if (_bakedMesh == null) _bakedMesh = new Mesh();
            var skins = FindObjectsByType<SkinnedMeshRenderer>(FindObjectsSortMode.None);
            foreach (var smr in skins)
            {
                if (smr == null || !smr.enabled || !smr.gameObject.activeInHierarchy) continue;
                _bakedMesh.Clear();
                try { smr.BakeMesh(_bakedMesh); } catch { continue; }
                AppendMesh(_bakedMesh, smr.transform.localToWorldMatrix, _dynamicTriList);
            }
        }
    }

    void AppendMesh(Mesh mesh, Matrix4x4 l2w, List<Tri> target)
    {
        var verts = mesh.vertices;
        var tris = mesh.triangles;
        if (verts == null || tris == null || tris.Length < 3) return;

        for (int i = 0; i < tris.Length; i += 3)
        {
            int i0 = tris[i], i1 = tris[i + 1], i2 = tris[i + 2];
            if ((uint)i0 >= (uint)verts.Length ||
                (uint)i1 >= (uint)verts.Length ||
                (uint)i2 >= (uint)verts.Length) continue;

            Vector3 a = l2w.MultiplyPoint3x4(verts[i0]);
            Vector3 b = l2w.MultiplyPoint3x4(verts[i1]);
            Vector3 c = l2w.MultiplyPoint3x4(verts[i2]);

            Vector3 n = Vector3.Cross(b - a, c - a);
            float len2 = n.sqrMagnitude;
            if (len2 < 1e-20f) continue;
            n /= Mathf.Sqrt(len2);

            target.Add(new Tri { a = a, b = b, c = c, n = n });
        }
    }

    void AppendTerrain(Terrain terrain, int step, List<Tri> target)
    {
        var td = terrain.terrainData;
        if (td == null) return;

        Vector3 tPos = terrain.transform.position;
        Vector3 tSize = td.size;
        int hRes = td.heightmapResolution;
        step = Mathf.Max(1, step);

        float[,] heights = td.GetHeights(0, 0, hRes, hRes);

        int xSteps = (hRes - 1) / step;
        int zSteps = (hRes - 1) / step;

        float bottomY = tPos.y; // bottom of terrain volume

        for (int iz = 0; iz < zSteps; iz++)
            for (int ix = 0; ix < xSteps; ix++)
            {
                int x0 = ix * step, x1 = Mathf.Min(x0 + step, hRes - 1);
                int z0 = iz * step, z1 = Mathf.Min(z0 + step, hRes - 1);

                Vector3 v00 = HToW(x0, z0, heights[z0, x0], hRes, tPos, tSize);
                Vector3 v10 = HToW(x1, z0, heights[z0, x1], hRes, tPos, tSize);
                Vector3 v01 = HToW(x0, z1, heights[z1, x0], hRes, tPos, tSize);
                Vector3 v11 = HToW(x1, z1, heights[z1, x1], hRes, tPos, tSize);

                // Top surface (heightmap)
                AddTri(v00, v11, v10, target);
                AddTri(v00, v01, v11, target);

                // Bottom surface (flat at terrain base Y, reversed winding)
                Vector3 b00 = new Vector3(v00.x, bottomY, v00.z);
                Vector3 b10 = new Vector3(v10.x, bottomY, v10.z);
                Vector3 b01 = new Vector3(v01.x, bottomY, v01.z);
                Vector3 b11 = new Vector3(v11.x, bottomY, v11.z);
                AddTri(b00, b10, b11, target);
                AddTri(b00, b11, b01, target);
            }

        // Side-wall skirts to seal the mesh when edges are raised.
        // Without these, the sweep fill leaks through the gap between
        // the top heightmap edge and the flat bottom surface.

        // Z-min edge (iz == 0, front face, winding faces outward -Z)
        for (int ix = 0; ix < xSteps; ix++)
        {
            int x0 = ix * step, x1 = Mathf.Min(x0 + step, hRes - 1);
            Vector3 top0 = HToW(x0, 0, heights[0, x0], hRes, tPos, tSize);
            Vector3 top1 = HToW(x1, 0, heights[0, x1], hRes, tPos, tSize);
            Vector3 bot0 = new Vector3(top0.x, bottomY, top0.z);
            Vector3 bot1 = new Vector3(top1.x, bottomY, top1.z);
            AddTri(top0, top1, bot1, target);
            AddTri(top0, bot1, bot0, target);
        }

        // Z-max edge (iz == zSteps, back face, winding faces outward +Z)
        for (int ix = 0; ix < xSteps; ix++)
        {
            int x0 = ix * step, x1 = Mathf.Min(x0 + step, hRes - 1);
            int z = Mathf.Min(zSteps * step, hRes - 1);
            Vector3 top0 = HToW(x0, z, heights[z, x0], hRes, tPos, tSize);
            Vector3 top1 = HToW(x1, z, heights[z, x1], hRes, tPos, tSize);
            Vector3 bot0 = new Vector3(top0.x, bottomY, top0.z);
            Vector3 bot1 = new Vector3(top1.x, bottomY, top1.z);
            AddTri(top0, bot1, top1, target);
            AddTri(top0, bot0, bot1, target);
        }

        // X-min edge (ix == 0, left face, winding faces outward -X)
        for (int iz = 0; iz < zSteps; iz++)
        {
            int z0 = iz * step, z1 = Mathf.Min(z0 + step, hRes - 1);
            Vector3 top0 = HToW(0, z0, heights[z0, 0], hRes, tPos, tSize);
            Vector3 top1 = HToW(0, z1, heights[z1, 0], hRes, tPos, tSize);
            Vector3 bot0 = new Vector3(top0.x, bottomY, top0.z);
            Vector3 bot1 = new Vector3(top1.x, bottomY, top1.z);
            AddTri(top0, bot0, bot1, target);
            AddTri(top0, bot1, top1, target);
        }

        // X-max edge (ix == xSteps, right face, winding faces outward +X)
        for (int iz = 0; iz < zSteps; iz++)
        {
            int z0 = iz * step, z1 = Mathf.Min(z0 + step, hRes - 1);
            int x = Mathf.Min(xSteps * step, hRes - 1);
            Vector3 top0 = HToW(x, z0, heights[z0, x], hRes, tPos, tSize);
            Vector3 top1 = HToW(x, z1, heights[z1, x], hRes, tPos, tSize);
            Vector3 bot0 = new Vector3(top0.x, bottomY, top0.z);
            Vector3 bot1 = new Vector3(top1.x, bottomY, top1.z);
            AddTri(top0, bot1, bot0, target);
            AddTri(top0, top1, bot1, target);
        }
    }

    static Vector3 HToW(int x, int z, float h, int hRes, Vector3 tPos, Vector3 tSize)
    {
        float fx = (float)x / (hRes - 1);
        float fz = (float)z / (hRes - 1);
        return new Vector3(tPos.x + fx * tSize.x,
                           tPos.y + h * tSize.y,
                           tPos.z + fz * tSize.z);
    }

    void AddTri(Vector3 a, Vector3 b, Vector3 c, List<Tri> target)
    {
        Vector3 n = Vector3.Cross(b - a, c - a);
        float len2 = n.sqrMagnitude;
        if (len2 < 1e-20f) return;
        n /= Mathf.Sqrt(len2);
        target.Add(new Tri { a = a, b = b, c = c, n = n });
    }

    // ================================================================
    // Bounds
    // ================================================================

    void ComputeBoundsFromBothLists(out Vector3 mn, out Vector3 mx)
    {
        if (boundsMode == BoundsMode.Manual)
        {
            mn = gridMin;
            mx = gridMax;
            return;
        }

        mn = new Vector3(float.MaxValue, float.MaxValue, float.MaxValue);
        mx = new Vector3(float.MinValue, float.MinValue, float.MinValue);

        ExpandBounds(_staticTriList, ref mn, ref mx);
        ExpandBounds(_dynamicTriList, ref mn, ref mx);

        if (mn.x > mx.x) { mn = gridMin; mx = gridMax; return; }

        Vector3 pad = Vector3.one * autoFitPadding;
        mn -= pad;
        mx += pad;
    }

    static void ExpandBounds(List<Tri> list, ref Vector3 mn, ref Vector3 mx)
    {
        for (int i = 0; i < list.Count; i++)
        {
            var t = list[i];
            mn = Vector3.Min(mn, Vector3.Min(t.a, Vector3.Min(t.b, t.c)));
            mx = Vector3.Max(mx, Vector3.Max(t.a, Vector3.Max(t.b, t.c)));
        }
    }

    // ================================================================
    // Grid sizing
    // ================================================================

    void ComputeGridSize(Vector3 mn, Vector3 mx, out int gx, out int gy, out int gz)
    {
        Vector3 size = mx - mn;
        gx = Mathf.Max(1, Mathf.CeilToInt(size.x / voxelSize));
        gy = Mathf.Max(1, Mathf.CeilToInt(size.y / voxelSize));
        gz = Mathf.Max(1, Mathf.CeilToInt(size.z / voxelSize));

        int cap = Mathf.Max(32, maxVoxelsPerAxis);
        gx = Mathf.Min(gx, cap);
        gy = Mathf.Min(gy, cap);
        gz = Mathf.Min(gz, cap);

        long total = (long)gx * gy * gz;
        long budget = (long)(maxVoxelCountMillions * 1_000_000);
        if (total > budget)
        {
            float scale = Mathf.Pow((float)budget / total, 1f / 3f);
            gx = Mathf.Max(1, Mathf.FloorToInt(gx * scale));
            gy = Mathf.Max(1, Mathf.FloorToInt(gy * scale));
            gz = Mathf.Max(1, Mathf.FloorToInt(gz * scale));
        }
    }

    // ================================================================
    // GPU resource management
    // ================================================================

    void CacheKernels()
    {
        KClear = coreCS.FindKernel("Clear");
        KSurface = coreCS.FindKernel("Surface");
        KSweepFill = coreCS.FindKernel("SweepFill");
        KBuildTexture = coreCS.FindKernel("BuildTexture");
        KComputeNormals = coreCS.FindKernel("ComputeNormals");
        KCopyStaticToWorking = coreCS.FindKernel("CopyStaticToWorking");
        KClearFlood = coreCS.FindKernel("ClearFlood");
        KBlurFill = coreCS.FindKernel("BlurFill");
        _kernelsCached = true;
    }

    void AllocateResources(int gx, int gy, int gz)
    {
        ReleaseBuffers();
        ReleaseTextures();

        int totalVoxels = gx * gy * gz;
        _totalVoxels = totalVoxels;
        _voxelBuffer = new ComputeBuffer(totalVoxels, sizeof(uint));
        _floodBuffer = new ComputeBuffer(totalVoxels, sizeof(uint));
        _staticVoxelBuf = new ComputeBuffer(totalVoxels, sizeof(uint));

        _fillTex = new RenderTexture(gx, gy, 0, RenderTextureFormat.RFloat)
        {
            dimension = UnityEngine.Rendering.TextureDimension.Tex3D,
            volumeDepth = gz,
            enableRandomWrite = true,
            useMipMap = false,
            autoGenerateMips = false,
            wrapMode = TextureWrapMode.Clamp,
            filterMode = FilterMode.Point
        };
        _fillTex.Create();

        _blurredFillTex = new RenderTexture(gx, gy, 0, RenderTextureFormat.RFloat)
        {
            dimension = UnityEngine.Rendering.TextureDimension.Tex3D,
            volumeDepth = gz,
            enableRandomWrite = true,
            useMipMap = false,
            autoGenerateMips = false,
            wrapMode = TextureWrapMode.Clamp,
            filterMode = FilterMode.Point
        };
        _blurredFillTex.Create();

        _normalsTex = new RenderTexture(gx, gy, 0, RenderTextureFormat.ARGBFloat)
        {
            dimension = UnityEngine.Rendering.TextureDimension.Tex3D,
            volumeDepth = gz,
            enableRandomWrite = true,
            useMipMap = false,
            autoGenerateMips = false,
            wrapMode = TextureWrapMode.Clamp,
            filterMode = FilterMode.Bilinear
        };
        _normalsTex.Create();
    }

    void UploadStaticTriangles()
    {
        _staticTriCount = _staticTriList.Count;
        if (_staticTriCount == 0) return;

        if (_staticTriBuffer != null && _staticTriBuffer.count != _staticTriCount)
        { _staticTriBuffer.Release(); _staticTriBuffer = null; }
        if (_staticTriBuffer == null)
            _staticTriBuffer = new ComputeBuffer(_staticTriCount, Marshal.SizeOf(typeof(Tri)));

        _staticTriBuffer.SetData(_staticTriList);
    }

    void UploadDynamicTriangles()
    {
        _dynamicTriCount = _dynamicTriList.Count;
        if (_dynamicTriCount == 0)
        {
            // Release to save memory when no dynamic objects
            if (_dynamicTriBuffer != null) { _dynamicTriBuffer.Release(); _dynamicTriBuffer = null; }
            return;
        }

        if (_dynamicTriBuffer != null && _dynamicTriBuffer.count < _dynamicTriCount)
        { _dynamicTriBuffer.Release(); _dynamicTriBuffer = null; }
        if (_dynamicTriBuffer == null)
            _dynamicTriBuffer = new ComputeBuffer(_dynamicTriCount, Marshal.SizeOf(typeof(Tri)));

        _dynamicTriBuffer.SetData(_dynamicTriList);
    }

    void ReleaseBuffers()
    {
        if (_voxelBuffer != null) { _voxelBuffer.Release(); _voxelBuffer = null; }
        if (_floodBuffer != null) { _floodBuffer.Release(); _floodBuffer = null; }
        if (_staticVoxelBuf != null) { _staticVoxelBuf.Release(); _staticVoxelBuf = null; }
    }

    void ReleaseTextures()
    {
        if (_fillTex != null) { _fillTex.Release(); Destroy(_fillTex); _fillTex = null; }
        if (_blurredFillTex != null) { _blurredFillTex.Release(); Destroy(_blurredFillTex); _blurredFillTex = null; }
        if (_normalsTex != null) { _normalsTex.Release(); Destroy(_normalsTex); _normalsTex = null; }
    }

    void ReleaseTriBuffers()
    {
        if (_staticTriBuffer != null) { _staticTriBuffer.Release(); _staticTriBuffer = null; }
        if (_dynamicTriBuffer != null) { _dynamicTriBuffer.Release(); _dynamicTriBuffer = null; }
        _staticTriCount = 0;
        _dynamicTriCount = 0;
    }

    void ReleaseAll()
    {
        ReleaseBuffers();
        ReleaseTextures();
        ReleaseTriBuffers();
        if (_bakedMesh != null) { Destroy(_bakedMesh); _bakedMesh = null; }
    }

    // ================================================================
    // Dispatch helpers
    // ================================================================

    void Dispatch3D(int kernel, int gx, int gy, int gz)
    {
        coreCS.GetKernelThreadGroupSizes(kernel, out uint tx, out uint ty, out uint tz);
        coreCS.Dispatch(kernel,
            Mathf.CeilToInt(gx / (float)tx),
            Mathf.CeilToInt(gy / (float)ty),
            Mathf.CeilToInt(gz / (float)tz));
    }

    void Dispatch2D(int kernel, int gx, int gy)
    {
        coreCS.GetKernelThreadGroupSizes(kernel, out uint tx, out uint ty, out _);
        coreCS.Dispatch(kernel,
            Mathf.CeilToInt(gx / (float)tx),
            Mathf.CeilToInt(gy / (float)ty),
            1);
    }

    void DispatchSweep(int axis, int planeA, int planeB)
    {
        coreCS.SetInt("_SweepAxis", axis);
        Dispatch2D(KSweepFill, planeA, planeB);
    }

    void DispatchLinear(int kernel, int count)
    {
        coreCS.GetKernelThreadGroupSizes(kernel, out uint tx, out _, out _);
        coreCS.Dispatch(kernel, Mathf.CeilToInt(count / (float)tx), 1, 1);
    }

    // ================================================================
    // Editor gizmos
    // ================================================================

    void OnDrawGizmosSelected()
    {
        if (boundsMode == BoundsMode.Manual)
        {
            Gizmos.color = new Color(1, 1, 0, 0.3f);
            Vector3 s = gridMax - gridMin;
            Gizmos.DrawWireCube(gridMin + s * 0.5f, s);
        }
        else if (_nx > 0 && Application.isPlaying)
        {
            Gizmos.color = new Color(0, 1, 1, 0.3f);
            Vector3 s = _activeMax - _activeMin;
            Gizmos.DrawWireCube(_activeMin + s * 0.5f, s);
        }
    }
}
