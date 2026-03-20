using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;

/// <summary>
/// GPU voxelizer following the proven mattatz/unity-voxel architecture:
/// StructuredBuffer voxelization, z-scan volume fill, Texture3D for ray march.
/// Reliably detects inside/outside surface via fill flag.
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

    [Header("Scene Input")]
    public bool includeMeshRenderers = true;
    public bool includeSkinnedMeshRenderers = true;
    public bool includeTerrains = true;
    [Range(1, 32)] public int terrainSampleStep = 4;

    [Header("Dynamic")]
    [Tooltip("When true, the full pipeline runs every frame.")]
    public bool voxelizeEveryFrame = true;

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

    [StructLayout(LayoutKind.Sequential)]
    struct Voxel_t
    {
        public uint fill;
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

    int KClear, KSurface, KSeedOutside, KFloodStep, KBuildTexture, KComputeNormals;
    bool _kernelsCached;

    ComputeBuffer _voxelBuffer;
    ComputeBuffer _floodBuffer;
    RenderTexture _fillTex;
    RenderTexture _normalsTex;
    ComputeBuffer _triBuffer;
    int _triCount;

    int _nx, _ny, _nz;
    Vector3 _activeMin, _activeMax;

    readonly List<Tri> _triList = new List<Tri>(128 * 1024);
    Mesh _bakedMesh;

    // ================================================================
    // Lifecycle
    // ================================================================

    void OnEnable()
    {
        if (coreCS == null) return;
        CacheKernels();
        Voxelize();
    }

    void OnDisable()
    {
        ReleaseAll();
    }

    void LateUpdate()
    {
        if (voxelizeEveryFrame && coreCS != null)
            Voxelize();
    }

    // ================================================================
    // Main pipeline
    // ================================================================

    [ContextMenu("Force Voxelize")]
    public void Voxelize()
    {
        if (coreCS == null) return;
        if (!_kernelsCached) CacheKernels();

        BuildTriangleList();
        if (_triList.Count == 0) { _nx = _ny = _nz = 0; return; }

        ComputeBounds(out Vector3 mn, out Vector3 mx);
        _activeMin = mn;
        _activeMax = mx;

        ComputeGridSize(mn, mx, out int gx, out int gy, out int gz);
        bool sizeChanged = gx != _nx || gy != _ny || gz != _nz;
        _nx = gx; _ny = gy; _nz = gz;

        if (sizeChanged || _fillTex == null)
            AllocateResources(gx, gy, gz);

        UploadTriangles();

        float unit = voxelSize;
        float hunit = unit * 0.5f;

        coreCS.SetInt("_Width", gx);
        coreCS.SetInt("_Height", gy);
        coreCS.SetInt("_Depth", gz);
        coreCS.SetVector("_Start", _activeMin);
        coreCS.SetFloat("_Unit", unit);
        coreCS.SetFloat("_HalfUnit", hunit);
        coreCS.SetInt("_TriCount", _triCount);

        BindResources();

        // 1) Clear voxel buffer and fill texture
        Dispatch3D(KClear, gx, gy, gz);

        // 2) Surface voxelization — all triangles (atomic)
        coreCS.SetBuffer(KSurface, "_Tris", _triBuffer);
        DispatchLinear(KSurface, _triCount);

        // 3) Flood-fill volume (optional)
        if (fillVolume)
        {
            // Seed boundary voxels as "outside"
            Dispatch3D(KSeedOutside, gx, gy, gz);

            // Fixed iteration flood: max dimension guarantees convergence
            // Each pass propagates at least 1 layer outward from boundary
            int maxDim = Mathf.Max(gx, Mathf.Max(gy, gz));
            for (int iter = 0; iter < maxDim; iter++)
                Dispatch3D(KFloodStep, gx, gy, gz);
        }

        // 4) Build fill texture from voxel buffer + flood buffer
        coreCS.SetInt("_FillVolume", fillVolume ? 1 : 0);
        Dispatch3D(KBuildTexture, gx, gy, gz);

        // 5) Compute gradient normals from filled volume
        coreCS.SetTexture(KComputeNormals, "_FillTex", _fillTex);
        coreCS.SetTexture(KComputeNormals, "_NormalTex", _normalsTex);
        Dispatch3D(KComputeNormals, gx, gy, gz);
    }

    // ================================================================
    // Triangle extraction (CPU)
    // ================================================================

    void BuildTriangleList()
    {
        _triList.Clear();

        if (includeMeshRenderers)
        {
            var filters = FindObjectsByType<MeshFilter>(FindObjectsSortMode.None);
            foreach (var mf in filters)
            {
                if (mf == null || !mf.gameObject.activeInHierarchy) continue;
                var mr = mf.GetComponent<MeshRenderer>();
                if (mr == null || !mr.enabled) continue;
                if (mf.sharedMesh == null) continue;
                AppendMesh(mf.sharedMesh, mf.transform.localToWorldMatrix);
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
                AppendMesh(_bakedMesh, smr.transform.localToWorldMatrix);
            }
        }

        if (includeTerrains)
        {
            var terrains = FindObjectsByType<Terrain>(FindObjectsSortMode.None);
            foreach (var t in terrains)
            {
                if (t == null || !t.isActiveAndEnabled) continue;
                AppendTerrain(t, terrainSampleStep);
            }
        }
    }

    void AppendMesh(Mesh mesh, Matrix4x4 l2w)
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

            _triList.Add(new Tri { a = a, b = b, c = c, n = n });
        }
    }

    void AppendTerrain(Terrain terrain, int step)
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
                AddTri(v00, v11, v10);
                AddTri(v00, v01, v11);

                // Bottom surface (flat at terrain base Y, reversed winding)
                Vector3 b00 = new Vector3(v00.x, bottomY, v00.z);
                Vector3 b10 = new Vector3(v10.x, bottomY, v10.z);
                Vector3 b01 = new Vector3(v01.x, bottomY, v01.z);
                Vector3 b11 = new Vector3(v11.x, bottomY, v11.z);
                AddTri(b00, b10, b11);
                AddTri(b00, b11, b01);

                // Side walls along terrain edges (connect top heightmap to bottom)
                // Left edge (ix == 0)
                if (ix == 0)
                {
                    AddTri(v00, b00, b01);
                    AddTri(v00, b01, v01);
                }
                // Right edge (ix == xSteps - 1)
                if (ix == xSteps - 1)
                {
                    AddTri(v10, v11, b11);
                    AddTri(v10, b11, b10);
                }
                // Front edge (iz == 0)
                if (iz == 0)
                {
                    AddTri(v00, v10, b10);
                    AddTri(v00, b10, b00);
                }
                // Back edge (iz == zSteps - 1)
                if (iz == zSteps - 1)
                {
                    AddTri(v01, b01, b11);
                    AddTri(v01, b11, v11);
                }
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

    void AddTri(Vector3 a, Vector3 b, Vector3 c)
    {
        Vector3 n = Vector3.Cross(b - a, c - a);
        float len2 = n.sqrMagnitude;
        if (len2 < 1e-20f) return;
        n /= Mathf.Sqrt(len2);
        _triList.Add(new Tri { a = a, b = b, c = c, n = n });
    }

    // ================================================================
    // Bounds
    // ================================================================

    void ComputeBounds(out Vector3 mn, out Vector3 mx)
    {
        if (boundsMode == BoundsMode.Manual)
        {
            mn = gridMin;
            mx = gridMax;
            return;
        }

        mn = new Vector3(float.MaxValue, float.MaxValue, float.MaxValue);
        mx = new Vector3(float.MinValue, float.MinValue, float.MinValue);

        for (int i = 0; i < _triList.Count; i++)
        {
            var t = _triList[i];
            mn = Vector3.Min(mn, Vector3.Min(t.a, Vector3.Min(t.b, t.c)));
            mx = Vector3.Max(mx, Vector3.Max(t.a, Vector3.Max(t.b, t.c)));
        }

        if (mn.x > mx.x) { mn = gridMin; mx = gridMax; return; }

        Vector3 pad = Vector3.one * autoFitPadding;
        mn -= pad;
        mx += pad;
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
        KSeedOutside = coreCS.FindKernel("SeedOutside");
        KFloodStep = coreCS.FindKernel("FloodStep");
        KBuildTexture = coreCS.FindKernel("BuildTexture");
        KComputeNormals = coreCS.FindKernel("ComputeNormals");
        _kernelsCached = true;
    }

    void AllocateResources(int gx, int gy, int gz)
    {
        ReleaseVoxelBuffer();
        ReleaseTextures();

        int totalVoxels = gx * gy * gz;
        _voxelBuffer = new ComputeBuffer(totalVoxels, sizeof(uint));
        _floodBuffer = new ComputeBuffer(totalVoxels, sizeof(uint));

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

    void BindResources()
    {
        coreCS.SetBuffer(KClear, "_VoxelBuffer", _voxelBuffer);
        coreCS.SetBuffer(KClear, "_FloodBuffer", _floodBuffer);
        coreCS.SetBuffer(KSurface, "_VoxelBuffer", _voxelBuffer);
        coreCS.SetBuffer(KSeedOutside, "_VoxelBuffer", _voxelBuffer);
        coreCS.SetBuffer(KSeedOutside, "_FloodBuffer", _floodBuffer);
        coreCS.SetBuffer(KFloodStep, "_VoxelBuffer", _voxelBuffer);
        coreCS.SetBuffer(KFloodStep, "_FloodBuffer", _floodBuffer);
        coreCS.SetBuffer(KBuildTexture, "_VoxelBuffer", _voxelBuffer);
        coreCS.SetBuffer(KBuildTexture, "_FloodBuffer", _floodBuffer);

        coreCS.SetTexture(KClear, "_FillTex", _fillTex);
        coreCS.SetTexture(KClear, "_NormalTex", _normalsTex);
        coreCS.SetTexture(KBuildTexture, "_FillTex", _fillTex);
    }

    void UploadTriangles()
    {
        _triCount = _triList.Count;
        if (_triCount == 0) return;

        if (_triBuffer != null && _triBuffer.count != _triCount)
        {
            _triBuffer.Release();
            _triBuffer = null;
        }
        if (_triBuffer == null)
            _triBuffer = new ComputeBuffer(_triCount, Marshal.SizeOf(typeof(Tri)));

        _triBuffer.SetData(_triList);
    }

    void ReleaseVoxelBuffer()
    {
        if (_voxelBuffer != null) { _voxelBuffer.Release(); _voxelBuffer = null; }
        if (_floodBuffer != null) { _floodBuffer.Release(); _floodBuffer = null; }
    }

    void ReleaseTextures()
    {
        if (_fillTex != null) { _fillTex.Release(); Destroy(_fillTex); _fillTex = null; }
        if (_normalsTex != null) { _normalsTex.Release(); Destroy(_normalsTex); _normalsTex = null; }
    }

    void ReleaseTriBuffer()
    {
        if (_triBuffer != null) { _triBuffer.Release(); _triBuffer = null; }
        _triCount = 0;
    }

    void ReleaseAll()
    {
        ReleaseVoxelBuffer();
        ReleaseTextures();
        ReleaseTriBuffer();
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
