using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

public sealed class GPUSdfVoxelizer : MonoBehaviour
{
    [Header("Compute")]
    public ComputeShader voxelCS;

    [Header("Grid")]
    public Vector3 gridMinWorld = new Vector3(-5, -5, -5);
    public Vector3 gridMaxWorld = new Vector3(5, 5, 5);
    [Min(0.001f)] public float voxelSize = 0.1f;

    [Header("Scene Input")]
    public bool includeMeshRenderers = true;
    public bool includeSkinnedMeshRenderers = true;

    [Header("JFA")]
    public bool runJFA = true;

    [Header("Outputs")]
    public bool keepOutputsAlive = true;

    // 3D outputs
    private RenderTexture _seedPosA;
    private RenderTexture _seedPosB;
    private RenderTexture _seedNrm;
    private RenderTexture _sdf;
    private RenderTexture _normals;

    public RenderTexture SeedPosA => _seedPosA;
    public RenderTexture SeedPosB => _seedPosB;
    public RenderTexture SeedNrm => _seedNrm;
    public RenderTexture SDF => _sdf;
    public RenderTexture Normals => _normals;

    // 2D debug slice RTs
    private RenderTexture _sdfSlice2D;
    private RenderTexture _normalsSlice2D;

    // CPU slice caches
    private float[] _sdfSliceCPU;
    private Vector4[] _normalsSliceCPU;

    // Readback state
    private bool _sdfSliceReady;
    private bool _sdfSlicePending;
    private int _sdfSliceYLast = -1;

    private bool _normSliceReady;
    private bool _normSlicePending;
    private int _normSliceYLast = -1;

    // Triangle buffer
    private ComputeBuffer _triBuffer;
    private int _triCount;

    // Kernels
    private int KClear;
    private int KVoxelize;
    private int KJFA;
    private int KSdf;
    private int KNormals;
    private int KExtractSdfSlice;
    private int KExtractNormalsSlice;

    // Grid dims
    private int nx, ny, nz;

    [Header("Debug Gizmos")]
    public bool drawGridBounds = true;

    public bool drawFilledInteriorSlice = true;
    [Range(0, 1)] public float sliceT = 0.5f;
    [Range(1, 200000)] public int voxelDrawLimit = 30000;

    [Header("Normals Gizmos")]
    public bool drawNormals = true;
    [Range(1, 16)] public int normalStride = 3;
    [Min(0.01f)] public float normalLengthInVoxels = 1.5f;

    public bool normalsOnlyNearSurface = true;
    [Min(0.0001f)] public float normalsSurfaceBand = 0.25f; // in world units (|SDF| < band)

    [Serializable]
    private struct Tri
    {
        public Vector3 a, b, c;
        public Vector3 n;
    }

    private void OnEnable()
    {
        if (voxelCS == null) return;
        CacheKernels();
        Build();
    }

    private void OnDisable()
    {
        ReleaseAll();
    }

    [ContextMenu("Build (GPU SDF Voxelizer)")]
    public void Build()
    {
        ValidateOrThrow();

        ComputeGridSize(out nx, out ny, out nz);
        AllocateTextures(nx, ny, nz);

        BuildTriangleBuffer(out _triCount);
        if (_triCount == 0)
        {
            Debug.LogWarning("GPUSdfVoxelizer: No triangles found. Nothing to voxelize.");
            return;
        }

        voxelCS.SetInts("_GridSize", nx, ny, nz);
        voxelCS.SetVector("_GridMinWorld", gridMinWorld);
        voxelCS.SetFloat("_VoxelSize", voxelSize);
        voxelCS.SetInt("_TriCount", _triCount);

        BindResources();

        Dispatch3D(KClear, nx, ny, nz);

        voxelCS.SetBuffer(KVoxelize, "_Tris", _triBuffer);
        DispatchTriangles(KVoxelize, _triCount);

        int ping = 0;
        if (runJFA)
        {
            int maxDim = Mathf.Max(nx, Mathf.Max(ny, nz));
            int step = HighestPowerOfTwoAtOrBelow(maxDim) / 2;
            if (step < 1) step = 1;

            while (step >= 1)
            {
                voxelCS.SetInt("_Step", step);
                voxelCS.SetInt("_PingPong", ping);
                Dispatch3D(KJFA, nx, ny, nz);
                ping ^= 1;
                step >>= 1;
            }
        }

        voxelCS.SetInt("_PingPong", ping);
        Dispatch3D(KSdf, nx, ny, nz);

        Dispatch3D(KNormals, nx, ny, nz);

        Debug.Log($"GPUSdfVoxelizer: done. Grid={nx}x{ny}x{nz}, tris={_triCount}.");

        InvalidateSlices();

        if (!keepOutputsAlive)
            ReleaseAll(keepShader: true);
    }

    private void CacheKernels()
    {
        KClear = voxelCS.FindKernel("Clear");
        KVoxelize = voxelCS.FindKernel("VoxelizeTriangles");
        KJFA = voxelCS.FindKernel("JFA3D");
        KSdf = voxelCS.FindKernel("ComputeSDF");
        KNormals = voxelCS.FindKernel("ComputeNormals");
        KExtractSdfSlice = voxelCS.FindKernel("ExtractSdfSlice");
        KExtractNormalsSlice = voxelCS.FindKernel("ExtractNormalsSlice");
    }

    private void ValidateOrThrow()
    {
        if (voxelCS == null) throw new InvalidOperationException("Assign voxelCS.");
        if (voxelSize <= 0f) throw new ArgumentOutOfRangeException(nameof(voxelSize));
        if (gridMaxWorld.x <= gridMinWorld.x ||
            gridMaxWorld.y <= gridMinWorld.y ||
            gridMaxWorld.z <= gridMinWorld.z)
            throw new InvalidOperationException("gridMaxWorld must be > gridMinWorld on all axes.");
    }

    private void ComputeGridSize(out int gx, out int gy, out int gz)
    {
        Vector3 size = gridMaxWorld - gridMinWorld;
        gx = Mathf.Max(1, Mathf.CeilToInt(size.x / voxelSize));
        gy = Mathf.Max(1, Mathf.CeilToInt(size.y / voxelSize));
        gz = Mathf.Max(1, Mathf.CeilToInt(size.z / voxelSize));
    }

    private void AllocateTextures(int gx, int gy, int gz)
    {
        ReleaseTexturesOnly();

        _seedPosA = NewRT3D(gx, gy, gz, RenderTextureFormat.ARGBInt);
        _seedPosB = NewRT3D(gx, gy, gz, RenderTextureFormat.ARGBInt);
        _seedNrm = NewRT3D(gx, gy, gz, RenderTextureFormat.ARGBFloat);
        _sdf = NewRT3D(gx, gy, gz, RenderTextureFormat.RFloat);
        _normals = NewRT3D(gx, gy, gz, RenderTextureFormat.ARGBFloat);

        _sdfSlice2D = NewRT2D(gx, gz, RenderTextureFormat.RFloat);
        _normalsSlice2D = NewRT2D(gx, gz, RenderTextureFormat.ARGBFloat);

        _sdfSliceCPU = new float[gx * gz];
        _normalsSliceCPU = new Vector4[gx * gz];

        InvalidateSlices();
    }

    private static RenderTexture NewRT3D(int w, int h, int d, RenderTextureFormat fmt)
    {
        var rt = new RenderTexture(w, h, 0, fmt)
        {
            dimension = TextureDimension.Tex3D,
            volumeDepth = d,
            enableRandomWrite = true,
            useMipMap = false,
            autoGenerateMips = false,
            wrapMode = TextureWrapMode.Clamp,
            filterMode = FilterMode.Point
        };
        rt.Create();
        return rt;
    }

    private static RenderTexture NewRT2D(int w, int h, RenderTextureFormat fmt)
    {
        var rt = new RenderTexture(w, h, 0, fmt)
        {
            dimension = TextureDimension.Tex2D,
            enableRandomWrite = true,
            useMipMap = false,
            autoGenerateMips = false,
            wrapMode = TextureWrapMode.Clamp,
            filterMode = FilterMode.Point
        };
        rt.Create();
        return rt;
    }

    private void BindResources()
    {
        // Clear
        voxelCS.SetTexture(KClear, "_SeedPosA", _seedPosA);
        voxelCS.SetTexture(KClear, "_SeedPosB", _seedPosB);
        voxelCS.SetTexture(KClear, "_SeedNrm", _seedNrm);
        voxelCS.SetTexture(KClear, "_SDF", _sdf);
        voxelCS.SetTexture(KClear, "_Normals", _normals);

        // Voxelize
        voxelCS.SetTexture(KVoxelize, "_SeedPosA", _seedPosA);
        voxelCS.SetTexture(KVoxelize, "_SeedNrm", _seedNrm);

        // JFA
        voxelCS.SetTexture(KJFA, "_SeedPosA", _seedPosA);
        voxelCS.SetTexture(KJFA, "_SeedPosB", _seedPosB);

        // SDF
        voxelCS.SetTexture(KSdf, "_SeedPosA", _seedPosA);
        voxelCS.SetTexture(KSdf, "_SeedPosB", _seedPosB);
        voxelCS.SetTexture(KSdf, "_SeedNrm", _seedNrm);
        voxelCS.SetTexture(KSdf, "_SDF", _sdf);

        // Normals
        voxelCS.SetTexture(KNormals, "_SDF", _sdf);
        voxelCS.SetTexture(KNormals, "_Normals", _normals);

        // Slice extraction
        voxelCS.SetTexture(KExtractSdfSlice, "_SDF", _sdf);
        voxelCS.SetTexture(KExtractSdfSlice, "_SdfSlice2D", _sdfSlice2D);

        voxelCS.SetTexture(KExtractNormalsSlice, "_Normals", _normals);
        voxelCS.SetTexture(KExtractNormalsSlice, "_NormalsSlice2D", _normalsSlice2D);
    }

    private void BuildTriangleBuffer(out int triCount)
    {
        ReleaseTriBuffer();

        var tris = new List<Tri>(64 * 1024);

        if (includeMeshRenderers)
        {
            var mfs = FindObjectsByType<MeshFilter>(FindObjectsSortMode.None);
            foreach (var mf in mfs)
            {
                if (mf == null || !mf.gameObject.activeInHierarchy) continue;
                var mr = mf.GetComponent<MeshRenderer>();
                if (mr == null || !mr.enabled) continue;
                var mesh = mf.sharedMesh;
                if (mesh == null) continue;
                AppendMeshTriangles(mesh, mf.transform.localToWorldMatrix, tris);
            }
        }

        if (includeSkinnedMeshRenderers)
        {
            var sks = FindObjectsByType<SkinnedMeshRenderer>(FindObjectsSortMode.None);
            var baked = new Mesh();
            foreach (var smr in sks)
            {
                if (smr == null || !smr.enabled || !smr.gameObject.activeInHierarchy) continue;
                baked.Clear();
                try { smr.BakeMesh(baked); }
                catch { continue; }
                AppendMeshTriangles(baked, smr.transform.localToWorldMatrix, tris);
            }
            Destroy(baked);
        }

        triCount = tris.Count;
        if (triCount == 0) return;

        _triBuffer = new ComputeBuffer(triCount, sizeof(float) * 12, ComputeBufferType.Structured);
        _triBuffer.SetData(tris);
    }

    private static void AppendMeshTriangles(Mesh mesh, Matrix4x4 l2w, List<Tri> outTris)
    {
        var v = mesh.vertices;
        var t = mesh.triangles;
        if (v == null || t == null || t.Length < 3) return;

        for (int i = 0; i < t.Length; i += 3)
        {
            int i0 = t[i], i1 = t[i + 1], i2 = t[i + 2];
            if ((uint)i0 >= (uint)v.Length || (uint)i1 >= (uint)v.Length || (uint)i2 >= (uint)v.Length) continue;

            Vector3 a = l2w.MultiplyPoint3x4(v[i0]);
            Vector3 b = l2w.MultiplyPoint3x4(v[i1]);
            Vector3 c = l2w.MultiplyPoint3x4(v[i2]);

            Vector3 n = Vector3.Cross(b - a, c - a);
            float len2 = n.sqrMagnitude;
            if (len2 < 1e-20f) continue;
            n /= Mathf.Sqrt(len2);

            outTris.Add(new Tri { a = a, b = b, c = c, n = n });
        }
    }

    private void DispatchTriangles(int kernel, int triCount)
    {
        voxelCS.GetKernelThreadGroupSizes(kernel, out uint tx, out _, out _);
        int groupsX = Mathf.CeilToInt(triCount / (float)tx);
        voxelCS.Dispatch(kernel, groupsX, 1, 1);
    }

    private void Dispatch3D(int kernel, int gx, int gy, int gz)
    {
        voxelCS.GetKernelThreadGroupSizes(kernel, out uint tx, out uint ty, out uint tz);
        int groupsX = Mathf.CeilToInt(gx / (float)tx);
        int groupsY = Mathf.CeilToInt(gy / (float)ty);
        int groupsZ = Mathf.CeilToInt(gz / (float)tz);
        voxelCS.Dispatch(kernel, groupsX, groupsY, groupsZ);
    }

    private static int HighestPowerOfTwoAtOrBelow(int x)
    {
        x = Mathf.Max(1, x);
        int p = 1;
        while (p <= x / 2) p <<= 1;
        return p;
    }

    private void ReleaseTriBuffer()
    {
        if (_triBuffer != null)
        {
            _triBuffer.Release();
            _triBuffer = null;
        }
        _triCount = 0;
    }

    private void ReleaseTexturesOnly()
    {
        static void Rel(GPUSdfVoxelizer owner, ref RenderTexture rt)
        {
            if (rt == null) return;
            rt.Release();
            Destroy(rt);
            rt = null;
        }

        Rel(this, ref _seedPosA);
        Rel(this, ref _seedPosB);
        Rel(this, ref _seedNrm);
        Rel(this, ref _sdf);
        Rel(this, ref _normals);
        Rel(this, ref _sdfSlice2D);
        Rel(this, ref _normalsSlice2D);
    }

    private void ReleaseAll(bool keepShader = false)
    {
        ReleaseTriBuffer();
        ReleaseTexturesOnly();
        if (!keepShader) voxelCS = null;
    }

    private void InvalidateSlices()
    {
        _sdfSliceReady = false;
        _sdfSlicePending = false;
        _sdfSliceYLast = -1;

        _normSliceReady = false;
        _normSlicePending = false;
        _normSliceYLast = -1;
    }

    private int CurrentSliceY()
    {
        if (ny <= 0) return 0;
        return Mathf.Clamp(Mathf.RoundToInt(sliceT * (ny - 1)), 0, ny - 1);
    }

    private void RequestSdfSliceIfNeeded()
    {
        if (!Application.isPlaying) return;
        if (_sdf == null || _sdfSlice2D == null) return;
        if (_sdfSlicePending) return;

        int y = CurrentSliceY();
        if (_sdfSliceReady && y == _sdfSliceYLast) return;

        voxelCS.SetInts("_GridSize", nx, ny, nz);
        voxelCS.SetInt("_SliceY", y);

        voxelCS.GetKernelThreadGroupSizes(KExtractSdfSlice, out uint tx, out uint ty, out _);
        int gx = Mathf.CeilToInt(nx / (float)tx);
        int gy = Mathf.CeilToInt(nz / (float)ty);
        voxelCS.Dispatch(KExtractSdfSlice, gx, gy, 1);

        _sdfSlicePending = true;
        _sdfSliceReady = false;

        AsyncGPUReadback.Request(_sdfSlice2D, 0, req =>
        {
            _sdfSlicePending = false;
            if (req.hasError) { _sdfSliceReady = false; return; }

            var data = req.GetData<float>();
            if (_sdfSliceCPU == null || _sdfSliceCPU.Length != data.Length)
                _sdfSliceCPU = new float[data.Length];

            data.CopyTo(_sdfSliceCPU);
            _sdfSliceYLast = y;
            _sdfSliceReady = true;

            // Changing slice implies normals slice is stale too
            _normSliceReady = false;
            _normSliceYLast = -1;
        });
    }

    private void RequestNormalsSliceIfNeeded()
    {
        if (!Application.isPlaying) return;
        if (_normals == null || _normalsSlice2D == null) return;
        if (_normSlicePending) return;

        int y = CurrentSliceY();
        if (_normSliceReady && y == _normSliceYLast) return;
        if (!_sdfSliceReady || _sdfSliceYLast != y) return; // keep in sync with SDF slice

        voxelCS.SetInts("_GridSize", nx, ny, nz);
        voxelCS.SetInt("_SliceY", y);

        voxelCS.GetKernelThreadGroupSizes(KExtractNormalsSlice, out uint tx, out uint ty, out _);
        int gx = Mathf.CeilToInt(nx / (float)tx);
        int gy = Mathf.CeilToInt(nz / (float)ty);
        voxelCS.Dispatch(KExtractNormalsSlice, gx, gy, 1);

        _normSlicePending = true;
        _normSliceReady = false;

        AsyncGPUReadback.Request(_normalsSlice2D, 0, req =>
        {
            _normSlicePending = false;
            if (req.hasError) { _normSliceReady = false; return; }

            var data = req.GetData<Vector4>();
            if (_normalsSliceCPU == null || _normalsSliceCPU.Length != data.Length)
                _normalsSliceCPU = new Vector4[data.Length];

            data.CopyTo(_normalsSliceCPU);
            _normSliceYLast = y;
            _normSliceReady = true;
        });
    }

    private void OnDrawGizmos()
    {
        if (!Application.isPlaying) return;

        if (drawGridBounds)
        {
            Vector3 size = gridMaxWorld - gridMinWorld;
            Gizmos.color = Color.yellow;
            Gizmos.DrawWireCube(gridMinWorld + size * 0.5f, size);
        }

        if (_sdf == null || nx <= 0 || ny <= 0 || nz <= 0) return;

        RequestSdfSliceIfNeeded();
        if (!_sdfSliceReady || _sdfSliceCPU == null) return;

        int y = _sdfSliceYLast;

        // Draw interior fill on slice (d < 0)
        if (drawFilledInteriorSlice)
        {
            Gizmos.color = Color.cyan;
            int drawn = 0;

            for (int z = 0; z < nz; z++)
            {
                for (int x = 0; x < nx; x++)
                {
                    float d = _sdfSliceCPU[x + nx * z];
                    if (d >= 0f) continue;

                    Vector3 center = gridMinWorld + new Vector3(
                        (x + 0.5f) * voxelSize,
                        (y + 0.5f) * voxelSize,
                        (z + 0.5f) * voxelSize
                    );

                    Gizmos.DrawWireCube(center, Vector3.one * voxelSize);

                    drawn++;
                    if (drawn >= voxelDrawLimit) break;
                }
                if (drawn >= voxelDrawLimit) break;
            }
        }

        if (!drawNormals) return;

        RequestNormalsSliceIfNeeded();
        if (!_normSliceReady || _normalsSliceCPU == null) return;

        Gizmos.color = Color.magenta;

        float rayLen = voxelSize * normalLengthInVoxels;

        for (int z = 0; z < nz; z += Mathf.Max(1, normalStride))
        {
            for (int x = 0; x < nx; x += Mathf.Max(1, normalStride))
            {
                float d = _sdfSliceCPU[x + nx * z];
                if (normalsOnlyNearSurface && Mathf.Abs(d) > normalsSurfaceBand) continue;

                Vector4 nv = _normalsSliceCPU[x + nx * z];
                Vector3 n = new Vector3(nv.x, nv.y, nv.z);
                if (n.sqrMagnitude < 1e-10f) continue;
                n.Normalize();

                Vector3 p = gridMinWorld + new Vector3(
                    (x + 0.5f) * voxelSize,
                    (y + 0.5f) * voxelSize,
                    (z + 0.5f) * voxelSize
                );

                Gizmos.DrawRay(p, n * rayLen);
            }
        }
    }
}