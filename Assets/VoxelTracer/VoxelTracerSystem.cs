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

    [Header("Normals")]
    [Tooltip("Compute gradient normals each frame. Only enable if a consumer reads NormalsTexture.")]
    public bool computeNormals = false;

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
        public Vector3 a, b, c;
    }

    /// <summary>Per-object dirty region in grid coordinates.</summary>
    struct DirtyRegion
    {
        public Vector3Int min, max;
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
    int KBlurFill;
    int KCopyAndClearFlood, KCopyAndClearFloodLinear;
    int KRestoreStaticFull, KRestoreStaticFullLinear;
    int KClearVoxelBuffer, KCopyWorkingToStatic;
    bool _kernelsCached;

    // GPU buffers
    ComputeBuffer _voxelBuffer;      // working buffer: packed (bit 0=surface, bit 1=outside)
    ComputeBuffer _staticVoxelBuf;   // cached static packed (surface + flood)
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

    // Reusable mesh data lists (zero GC per frame)
    readonly List<Vector3> _tmpVerts = new List<Vector3>(4096);
    readonly List<int> _tmpIndices = new List<int>(12288);

    // Registration-based object tracking (avoids FindObjectsByType per frame)
    static readonly HashSet<VoxelDynamic> _registeredDynamics = new HashSet<VoxelDynamic>();
    static readonly HashSet<SkinnedMeshRenderer> _registeredSkins = new HashSet<SkinnedMeshRenderer>();

    public static void RegisterDynamic(VoxelDynamic vd) { if (vd != null) _registeredDynamics.Add(vd); }
    public static void UnregisterDynamic(VoxelDynamic vd) { _registeredDynamics.Remove(vd); }
    public static void RegisterSkin(SkinnedMeshRenderer smr) { if (smr != null) _registeredSkins.Add(smr); }
    public static void UnregisterSkin(SkinnedMeshRenderer smr) { _registeredSkins.Remove(smr); }

    // Dirty flags
    bool _staticDirty = true;      // rebuild static tris + re-voxelize static layer
    bool _hasDynamicObjects;       // any dynamic objects exist in scene

    // Per-object dirty region tracking
    readonly List<DirtyRegion> _curDirtyRegions = new List<DirtyRegion>(16);
    readonly List<DirtyRegion> _prevDirtyRegions = new List<DirtyRegion>(16);
    readonly List<DirtyRegion> _mergedDirtyRegions = new List<DirtyRegion>(32);
    readonly List<DirtyRegion> _consolidatedRegions = new List<DirtyRegion>(16);

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
            SetGridUniforms(gx, gy, gz);
            SetRegionMin(0, 0, 0);
            ClearStaticVoxelCache(gx, gy, gz);
            return;
        }

        // Voxelize static geometry once → store in _staticVoxelBuf
        SetGridUniforms(gx, gy, gz);
        SetRegionMin(0, 0, 0);
        BindClearBuffers();
        Dispatch3D(KClear, gx, gy, gz);

        coreCS.SetBuffer(KSurface, "_VoxelBuffer", _voxelBuffer);
        coreCS.SetBuffer(KSurface, "_Tris", _staticTriBuffer);
        coreCS.SetInt("_TriCount", _staticTriCount);
        DispatchLinear(KSurface, _staticTriCount);

        // Run the full fill pipeline so output textures are pre-populated
        // with the static-only result. This makes per-frame work ZERO
        // when no dynamic objects exist.
        var fullMin = Vector3Int.zero;
        var fullMax = new Vector3Int(gx - 1, gy - 1, gz - 1);
        RunFillPipeline(gx, gy, gz, fullMin, fullMax);

        // Copy AFTER fill pipeline: static cache now includes surface (bit 0) + flood (bit 1).
        // This enables restoration frames to skip sweep entirely.
        CopyWorkingToStaticCache(gx, gy, gz);

        // Reset dirty-region tracking after full bake
        _prevDirtyRegions.Clear();
    }

    void ClearStaticVoxelCache(int gx, int gy, int gz)
    {
        // GPU-side clear via the ClearVoxelBuffer kernel (3D dispatch, safe for any grid size)
        coreCS.SetBuffer(KClearVoxelBuffer, "_VoxelBuffer", _staticVoxelBuf);
        Dispatch3D(KClearVoxelBuffer, gx, gy, gz);
    }

    void CopyWorkingToStaticCache(int gx, int gy, int gz)
    {
        // GPU-side copy: working → static cache (no readback stall)
        coreCS.SetBuffer(KCopyWorkingToStatic, "_VoxelBuffer", _voxelBuffer);
        coreCS.SetBuffer(KCopyWorkingToStatic, "_DstBuffer", _staticVoxelBuf);
        Dispatch3D(KCopyWorkingToStatic, gx, gy, gz);
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

        bool hasDynamics = _dynamicTriList.Count > 0;

        // Fast path: no dynamic objects and no previous dirty regions to restore
        if (!hasDynamics && _prevDirtyRegions.Count == 0)
            return;

        // Collect all raw regions (current + previous, un-padded)
        _mergedDirtyRegions.Clear();
        for (int i = 0; i < _curDirtyRegions.Count; i++)
            _mergedDirtyRegions.Add(_curDirtyRegions[i]);
        for (int i = 0; i < _prevDirtyRegions.Count; i++)
            _mergedDirtyRegions.Add(_prevDirtyRegions[i]);

        // Pad all regions, then consolidate overlapping/nearby ones
        const int pad = 3;
        var gridMax = new Vector3Int(gx - 1, gy - 1, gz - 1);
        for (int i = 0; i < _mergedDirtyRegions.Count; i++)
        {
            var r = _mergedDirtyRegions[i];
            r.min = Vector3Int.Max(r.min - new Vector3Int(pad, pad, pad), Vector3Int.zero);
            r.max = Vector3Int.Min(r.max + new Vector3Int(pad, pad, pad), gridMax);
            _mergedDirtyRegions[i] = r;
        }
        ConsolidateRegions(_mergedDirtyRegions, _consolidatedRegions);

        // Update tracking: current becomes previous for next frame
        _prevDirtyRegions.Clear();
        for (int i = 0; i < _curDirtyRegions.Count; i++)
            _prevDirtyRegions.Add(_curDirtyRegions[i]); // store UN-padded

        SetGridUniforms(gx, gy, gz);

        // Decide: full-grid fast path OR per-region path.
        bool useFullGrid = false;
        if (_consolidatedRegions.Count == 1)
        {
            var r = _consolidatedRegions[0];
            long dirtyVol = (long)(r.max.x - r.min.x + 1)
                          * (r.max.y - r.min.y + 1)
                          * (r.max.z - r.min.z + 1);
            if (dirtyVol * 2 >= _totalVoxels)
                useFullGrid = true;
        }

        // Non-dynamic restoration: static cache includes flood, skip sweep entirely.
        if (!hasDynamics)
        {
            VoxelizeFrameRestore(gx, gy, gz, useFullGrid);
            return;
        }

        UploadDynamicTriangles();

        if (useFullGrid)
        {
            VoxelizeFrameFullGrid(gx, gy, gz,
                _consolidatedRegions[0]);
        }
        else
        {
            VoxelizeFrameRegions(gx, gy, gz);
        }
    }

    /// <summary>Restoration path: dynamics have disappeared, restore static state.
    /// Static cache includes pre-computed flood marks, so sweep is skipped entirely.
    /// Saves 3 sweep dispatches + surface dispatch on transition frames.</summary>
    void VoxelizeFrameRestore(int gx, int gy, int gz, bool useFullGrid)
    {
        if (useFullGrid)
        {
            // Restore full static surface + flood with linear coalescing
            coreCS.SetInt("_TotalVoxels", _totalVoxels);
            coreCS.SetBuffer(KRestoreStaticFullLinear, "_VoxelBuffer", _voxelBuffer);
            coreCS.SetBuffer(KRestoreStaticFullLinear, "_StaticVoxelBuffer", _staticVoxelBuf);
            DispatchLinear(KRestoreStaticFullLinear, _totalVoxels);

            // BuildTexture scoped to dirty region (skip sweep — flood is correct from cache)
            var r = _consolidatedRegions[0];
            RunBuildOnly(gx, gy, gz, r.min, r.max);
        }
        else
        {
            // Per-region restore: copy full static (surface + flood) into dirty regions
            coreCS.SetBuffer(KRestoreStaticFull, "_VoxelBuffer", _voxelBuffer);
            coreCS.SetBuffer(KRestoreStaticFull, "_StaticVoxelBuffer", _staticVoxelBuf);
            for (int i = 0; i < _consolidatedRegions.Count; i++)
            {
                var r = _consolidatedRegions[i];
                Vector3Int sz = r.max - r.min + Vector3Int.one;
                SetRegionMin(r.min.x, r.min.y, r.min.z);
                Dispatch3D(KRestoreStaticFull, sz.x, sz.y, sz.z);
            }

            // BuildTexture per region (no sweep needed)
            for (int i = 0; i < _consolidatedRegions.Count; i++)
            {
                var r = _consolidatedRegions[i];
                RunBuildOnly(gx, gy, gz, r.min, r.max);
            }
        }
    }

    /// <summary>Full-grid path: linear kernel for buffer copy (perfect coalescing),
    /// fill pipeline scoped to dirty region. Used when dirty volume is large.</summary>
    void VoxelizeFrameFullGrid(int gx, int gy, int gz, DirtyRegion dirtyRegion)
    {
        // 1) Copy static surface, clear flood — single linear dispatch
        coreCS.SetInt("_TotalVoxels", _totalVoxels);
        coreCS.SetBuffer(KCopyAndClearFloodLinear, "_VoxelBuffer", _voxelBuffer);
        coreCS.SetBuffer(KCopyAndClearFloodLinear, "_StaticVoxelBuffer", _staticVoxelBuf);
        DispatchLinear(KCopyAndClearFloodLinear, _totalVoxels);

        // 2) Surface voxelization — dynamic triangles
        coreCS.SetBuffer(KSurface, "_VoxelBuffer", _voxelBuffer);
        coreCS.SetBuffer(KSurface, "_Tris", _dynamicTriBuffer);
        coreCS.SetInt("_TriCount", _dynamicTriCount);
        DispatchLinear(KSurface, _dynamicTriCount);

        // 3) Fill pipeline — scoped to dirty region
        RunFillPipeline(gx, gy, gz, dirtyRegion.min, dirtyRegion.max);
    }

    /// <summary>Per-region path: only processes dirty sub-volumes.
    /// Used when dirty volume is small relative to total grid.</summary>
    void VoxelizeFrameRegions(int gx, int gy, int gz)
    {
        // 1) Restore static surface + clear flood per dirty region
        coreCS.SetBuffer(KCopyAndClearFlood, "_VoxelBuffer", _voxelBuffer);
        coreCS.SetBuffer(KCopyAndClearFlood, "_StaticVoxelBuffer", _staticVoxelBuf);
        for (int i = 0; i < _consolidatedRegions.Count; i++)
        {
            var r = _consolidatedRegions[i];
            Vector3Int sz = r.max - r.min + Vector3Int.one;
            SetRegionMin(r.min.x, r.min.y, r.min.z);
            Dispatch3D(KCopyAndClearFlood, sz.x, sz.y, sz.z);
        }

        // 2) Surface voxelization — all dynamic triangles at once
        coreCS.SetBuffer(KSurface, "_VoxelBuffer", _voxelBuffer);
        coreCS.SetBuffer(KSurface, "_Tris", _dynamicTriBuffer);
        coreCS.SetInt("_TriCount", _dynamicTriCount);
        DispatchLinear(KSurface, _dynamicTriCount);

        // 3) Fill pipeline per consolidated region
        for (int i = 0; i < _consolidatedRegions.Count; i++)
        {
            var r = _consolidatedRegions[i];
            RunFillPipeline(gx, gy, gz, r.min, r.max);
        }
    }

    /// <summary>Merge overlapping or nearby regions to minimize dispatch count.
    /// Uses greedy iterative merging: any two regions whose AABBs overlap are
    /// unioned into one. Repeats until stable. O(N^2) but N is tiny (< 20).</summary>
    static void ConsolidateRegions(List<DirtyRegion> input, List<DirtyRegion> output)
    {
        output.Clear();
        for (int i = 0; i < input.Count; i++)
            output.Add(input[i]);

        bool merged = true;
        while (merged)
        {
            merged = false;
            for (int i = 0; i < output.Count; i++)
            {
                for (int j = i + 1; j < output.Count; j++)
                {
                    var a = output[i];
                    var b = output[j];

                    // Check AABB overlap (regions already padded, so touching = overlapping)
                    if (a.min.x <= b.max.x && a.max.x >= b.min.x &&
                        a.min.y <= b.max.y && a.max.y >= b.min.y &&
                        a.min.z <= b.max.z && a.max.z >= b.min.z)
                    {
                        // Union them
                        output[i] = new DirtyRegion
                        {
                            min = Vector3Int.Min(a.min, b.min),
                            max = Vector3Int.Max(a.max, b.max)
                        };
                        output.RemoveAt(j);
                        merged = true;
                        break;
                    }
                }
                if (merged) break;
            }
        }
    }

    /// <summary>Shared fill pipeline: sweep fill → build texture → blur → normals.
    /// Region parameters control which voxels are processed.</summary>
    void RunFillPipeline(int gx, int gy, int gz, Vector3Int regMin, Vector3Int regMax)
    {
        Vector3Int regSize = regMax - regMin + Vector3Int.one;

        // Sweep flood fill — only lines that cross the dirty region,
        // but each line sweeps full axis length for correctness
        if (fillVolume)
        {
            coreCS.SetBuffer(KSweepFill, "_VoxelBuffer", _voxelBuffer);
            for (int round = 0; round < fillSweepRounds; round++)
            {
                DispatchSweepRegion(0, regMin, regMax);
                DispatchSweepRegion(1, regMin, regMax);
                DispatchSweepRegion(2, regMin, regMax);
            }
        }

        RunBuildOnly(gx, gy, gz, regMin, regMax);
    }

    /// <summary>Build fill texture (+ optional blur/normals) without sweep.
    /// Used both after sweep and for restoration frames where flood is pre-computed.</summary>
    void RunBuildOnly(int gx, int gy, int gz, Vector3Int regMin, Vector3Int regMax)
    {
        Vector3Int regSize = regMax - regMin + Vector3Int.one;

        // Build fill texture (dirty region only)
        SetRegionMin(regMin.x, regMin.y, regMin.z);
        coreCS.SetInt("_FillVolume", fillVolume ? 1 : 0);
        coreCS.SetBuffer(KBuildTexture, "_VoxelBuffer", _voxelBuffer);
        coreCS.SetTexture(KBuildTexture, "_FillTex", _fillTex);
        Dispatch3D(KBuildTexture, regSize.x, regSize.y, regSize.z);

        // Blur fill + compute normals (padded by 1 for neighbor reads)
        if (computeNormals && _blurredFillTex != null && _normalsTex != null)
        {
            Vector3Int blurMin = Vector3Int.Max(regMin - Vector3Int.one, Vector3Int.zero);
            Vector3Int blurMax = Vector3Int.Min(regMax + Vector3Int.one,
                new Vector3Int(gx - 1, gy - 1, gz - 1));
            Vector3Int blurSize = blurMax - blurMin + Vector3Int.one;

            SetRegionMin(blurMin.x, blurMin.y, blurMin.z);
            coreCS.SetTexture(KBlurFill, "_FillTex", _fillTex);
            coreCS.SetTexture(KBlurFill, "_BlurredFillTex", _blurredFillTex);
            Dispatch3D(KBlurFill, blurSize.x, blurSize.y, blurSize.z);

            coreCS.SetTexture(KComputeNormals, "_FillTex", _fillTex);
            coreCS.SetTexture(KComputeNormals, "_BlurredFillTex", _blurredFillTex);
            coreCS.SetTexture(KComputeNormals, "_NormalTex", _normalsTex);
            Dispatch3D(KComputeNormals, blurSize.x, blurSize.y, blurSize.z);
        }
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

    void SetRegionMin(int x, int y, int z)
    {
        coreCS.SetInt("_RegionMinX", x);
        coreCS.SetInt("_RegionMinY", y);
        coreCS.SetInt("_RegionMinZ", z);
    }

    void BindClearBuffers()
    {
        coreCS.SetBuffer(KClear, "_VoxelBuffer", _voxelBuffer);
        coreCS.SetTexture(KClear, "_FillTex", _fillTex);
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

    /// <summary>Lightweight per-frame rebuild of dynamic triangles only.
    /// Uses registration-based tracking (O(1) list access) instead of FindObjectsByType (O(N) scene scan).
    /// Also computes per-object grid AABBs for dirty-region tracking.</summary>
    void BuildDynamicTriangleList()
    {
        _dynamicTriList.Clear();
        _curDirtyRegions.Clear();

        float inv = 1f / voxelSize;
        var gridClampMax = new Vector3Int(_nx - 1, _ny - 1, _nz - 1);

        if (includeMeshRenderers)
        {
            foreach (var vd in _registeredDynamics)
            {
                if (vd == null || !vd.gameObject.activeInHierarchy) continue;
                var mf = vd.GetComponent<MeshFilter>();
                if (mf == null || mf.sharedMesh == null) continue;
                var mr = vd.GetComponent<MeshRenderer>();
                if (mr == null || !mr.enabled) continue;

                AppendMesh(mf.sharedMesh, mf.transform.localToWorldMatrix, _dynamicTriList);
                AddDirtyRegionFromBounds(mr.bounds, inv, gridClampMax);
            }
        }

        if (includeSkinnedMeshRenderers)
        {
            if (_bakedMesh == null) _bakedMesh = new Mesh();
            foreach (var smr in _registeredSkins)
            {
                if (smr == null || !smr.enabled || !smr.gameObject.activeInHierarchy) continue;
                _bakedMesh.Clear();
                try { smr.BakeMesh(_bakedMesh); } catch { continue; }

                AppendMesh(_bakedMesh, smr.transform.localToWorldMatrix, _dynamicTriList);
                AddDirtyRegionFromBounds(smr.bounds, inv, gridClampMax);
            }
        }
    }

    void AddDirtyRegionFromBounds(Bounds bounds, float inv, Vector3Int gridClampMax)
    {
        Vector3 mn = bounds.min;
        Vector3 mx = bounds.max;

        var gMin = new Vector3Int(
            Mathf.FloorToInt((mn.x - _activeMin.x) * inv),
            Mathf.FloorToInt((mn.y - _activeMin.y) * inv),
            Mathf.FloorToInt((mn.z - _activeMin.z) * inv)
        );
        var gMax = new Vector3Int(
            Mathf.CeilToInt((mx.x - _activeMin.x) * inv),
            Mathf.CeilToInt((mx.y - _activeMin.y) * inv),
            Mathf.CeilToInt((mx.z - _activeMin.z) * inv)
        );

        gMin = Vector3Int.Max(gMin, Vector3Int.zero);
        gMax = Vector3Int.Min(gMax, gridClampMax);

        _curDirtyRegions.Add(new DirtyRegion { min = gMin, max = gMax });
    }

    /// <summary>Zero-GC mesh triangle extraction. Uses Mesh.GetVertices/GetIndices
    /// which reuse pre-allocated Lists instead of allocating new arrays.
    /// Iterates all submeshes to match the old mesh.triangles behavior.</summary>
    void AppendMesh(Mesh mesh, Matrix4x4 l2w, List<Tri> target)
    {
        mesh.GetVertices(_tmpVerts);
        int vertCount = _tmpVerts.Count;
        if (vertCount == 0) return;

        int subMeshCount = mesh.subMeshCount;
        for (int sub = 0; sub < subMeshCount; sub++)
        {
            if (mesh.GetTopology(sub) != MeshTopology.Triangles) continue;

            mesh.GetIndices(_tmpIndices, sub);
            int idxCount = _tmpIndices.Count;
            if (idxCount < 3) continue;

            for (int i = 0; i < idxCount; i += 3)
            {
                int i0 = _tmpIndices[i], i1 = _tmpIndices[i + 1], i2 = _tmpIndices[i + 2];
                if ((uint)i0 >= (uint)vertCount ||
                    (uint)i1 >= (uint)vertCount ||
                    (uint)i2 >= (uint)vertCount) continue;

                Vector3 a = l2w.MultiplyPoint3x4(_tmpVerts[i0]);
                Vector3 b = l2w.MultiplyPoint3x4(_tmpVerts[i1]);
                Vector3 c = l2w.MultiplyPoint3x4(_tmpVerts[i2]);

                // Degenerate triangle check (no sqrt needed — just check cross product magnitude)
                Vector3 cross = Vector3.Cross(b - a, c - a);
                if (cross.sqrMagnitude < 1e-20f) continue;

                target.Add(new Tri { a = a, b = b, c = c });
            }
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
        Vector3 cross = Vector3.Cross(b - a, c - a);
        if (cross.sqrMagnitude < 1e-20f) return;
        target.Add(new Tri { a = a, b = b, c = c });
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
        KBlurFill = coreCS.FindKernel("BlurFill");
        KCopyAndClearFlood = coreCS.FindKernel("CopyAndClearFlood");
        KCopyAndClearFloodLinear = coreCS.FindKernel("CopyAndClearFloodLinear");
        KRestoreStaticFull = coreCS.FindKernel("RestoreStaticFull");
        KRestoreStaticFullLinear = coreCS.FindKernel("RestoreStaticFullLinear");
        KClearVoxelBuffer = coreCS.FindKernel("ClearVoxelBuffer");
        KCopyWorkingToStatic = coreCS.FindKernel("CopyWorkingToStatic");
        _kernelsCached = true;
    }

    void AllocateResources(int gx, int gy, int gz)
    {
        ReleaseBuffers();
        ReleaseTextures();

        int totalVoxels = gx * gy * gz;
        _totalVoxels = totalVoxels;
        _voxelBuffer = new ComputeBuffer(totalVoxels, sizeof(uint));
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

        // Normals textures: only allocate when computeNormals is enabled
        if (computeNormals)
        {
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

    void DispatchSweepRegion(int axis, Vector3Int regMin, Vector3Int regMax)
    {
        coreCS.SetInt("_SweepAxis", axis);
        SetRegionMin(regMin.x, regMin.y, regMin.z);
        int planeA, planeB;
        if (axis == 0) { planeA = regMax.y - regMin.y + 1; planeB = regMax.z - regMin.z + 1; }
        else if (axis == 1) { planeA = regMax.x - regMin.x + 1; planeB = regMax.z - regMin.z + 1; }
        else { planeA = regMax.x - regMin.x + 1; planeB = regMax.y - regMin.y + 1; }
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
