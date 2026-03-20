using UnityEngine;

/// <summary>
/// Draws normal lines on filled surface voxels using GL immediate-mode rendering.
/// Attach to the same Camera that has VoxelTracerCamera.
/// Uses OnRenderImage to draw lines AFTER the voxel composite.
/// </summary>
[RequireComponent(typeof(Camera))]
[DefaultExecutionOrder(100)] // run after VoxelTracerCamera
public class VoxelNormalGizmos : MonoBehaviour
{
    [Header("References")]
    public VoxelTracerSystem voxelSystem;

    [Header("Display")]
    [Tooltip("Length of normal lines in world units")]
    public float normalLength = 0.5f;

    [Tooltip("Color of normal lines")]
    public Color normalColor = Color.green;

    [Tooltip("Only show normals for surface voxels (those with at least one empty neighbor)")]
    public bool surfaceOnly = true;

    [Tooltip("Max voxels to draw normals for (performance limit)")]
    [Range(100, 50000)]
    public int maxLines = 10000;

    [Tooltip("How often to refresh the voxel data (seconds). 0 = every frame.")]
    [Min(0)] public float refreshInterval = 0.2f;

    Material _glMat;
    Vector3[] _lineStarts;
    Vector3[] _lineEnds;
    int _lineCount;
    float _lastRefresh = -999f;

    // Cached readback data
    float[] _fillData;
    int _cachedNx, _cachedNy, _cachedNz;

    void OnEnable()
    {
        // Unlit colored line material
        var shader = Shader.Find("Hidden/Internal-Colored");
        if (shader == null) return;
        _glMat = new Material(shader);
        _glMat.hideFlags = HideFlags.HideAndDontSave;
        _glMat.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
        _glMat.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
        _glMat.SetInt("_Cull", (int)UnityEngine.Rendering.CullMode.Off);
        _glMat.SetInt("_ZWrite", 0);
        _glMat.SetInt("_ZTest", (int)UnityEngine.Rendering.CompareFunction.Always);
    }

    void OnDisable()
    {
        if (_glMat != null) { DestroyImmediate(_glMat); _glMat = null; }
        _lineStarts = null;
        _lineEnds = null;
        _fillData = null;
    }

    void Update()
    {
        if (voxelSystem == null || !voxelSystem.IsReady) return;
        if (Time.time - _lastRefresh < refreshInterval) return;

        _lastRefresh = Time.time;
        RebuildLines();
    }

    void RebuildLines()
    {
        int nx = voxelSystem.Nx;
        int ny = voxelSystem.Ny;
        int nz = voxelSystem.Nz;
        int total = nx * ny * nz;

        // GPU readback of fill texture into CPU array
        var fillRT = voxelSystem.FillTexture;
        if (fillRT == null) return;

        // Create a temporary Texture3D readback via compute buffer copy
        // Since direct Texture3D readback is complex, use AsyncGPUReadback
        // For simplicity, use a RenderTexture.active trick per-slice
        if (_fillData == null || _fillData.Length != total ||
            _cachedNx != nx || _cachedNy != ny || _cachedNz != nz)
        {
            _fillData = new float[total];
            _cachedNx = nx;
            _cachedNy = ny;
            _cachedNz = nz;
        }

        // Read each Z slice
        var tempRT = RenderTexture.GetTemporary(nx, ny, 0, RenderTextureFormat.RFloat);
        var tempTex = new Texture2D(nx, ny, TextureFormat.RFloat, false);

        for (int z = 0; z < nz; z++)
        {
            Graphics.CopyTexture(fillRT, z, 0, tempRT, 0, 0);
            var prev = RenderTexture.active;
            RenderTexture.active = tempRT;
            tempTex.ReadPixels(new Rect(0, 0, nx, ny), 0, 0, false);
            tempTex.Apply(false);
            RenderTexture.active = prev;

            var raw = tempTex.GetRawTextureData<float>();
            for (int i = 0; i < nx * ny; i++)
                _fillData[z * (nx * ny) + i] = raw[i];
        }

        RenderTexture.ReleaseTemporary(tempRT);
        Destroy(tempTex);

        // Build normal lines from surface voxels
        float unit = voxelSystem.ActiveVoxelSize;
        float halfUnit = unit * 0.5f;
        Vector3 start = voxelSystem.ActiveGridMin;

        if (_lineStarts == null || _lineStarts.Length != maxLines)
        {
            _lineStarts = new Vector3[maxLines];
            _lineEnds = new Vector3[maxLines];
        }

        _lineCount = 0;

        for (int z = 0; z < nz && _lineCount < maxLines; z++)
            for (int y = 0; y < ny && _lineCount < maxLines; y++)
                for (int x = 0; x < nx && _lineCount < maxLines; x++)
                {
                    int idx = z * (nx * ny) + y * nx + x;
                    if (_fillData[idx] < 0.5f) continue;

                    if (surfaceOnly)
                    {
                        // Check if it has at least one empty neighbor (6-connected)
                        bool isSurface = false;
                        if (x == 0 || GetFill(x - 1, y, z, nx, ny, nz) < 0.5f) isSurface = true;
                        else if (x == nx - 1 || GetFill(x + 1, y, z, nx, ny, nz) < 0.5f) isSurface = true;
                        else if (y == 0 || GetFill(x, y - 1, z, nx, ny, nz) < 0.5f) isSurface = true;
                        else if (y == ny - 1 || GetFill(x, y + 1, z, nx, ny, nz) < 0.5f) isSurface = true;
                        else if (z == 0 || GetFill(x, y, z - 1, nx, ny, nz) < 0.5f) isSurface = true;
                        else if (z == nz - 1 || GetFill(x, y, z + 1, nx, ny, nz) < 0.5f) isSurface = true;

                        if (!isSurface) continue;
                    }

                    // Compute gradient normal from fill field (central differences)
                    float gx = GetFill(Mathf.Min(x + 1, nx - 1), y, z, nx, ny, nz)
                              - GetFill(Mathf.Max(x - 1, 0), y, z, nx, ny, nz);
                    float gy = GetFill(x, Mathf.Min(y + 1, ny - 1), z, nx, ny, nz)
                              - GetFill(x, Mathf.Max(y - 1, 0), z, nx, ny, nz);
                    float gz = GetFill(x, y, Mathf.Min(z + 1, nz - 1), nx, ny, nz)
                              - GetFill(x, y, Mathf.Max(z - 1, 0), nx, ny, nz);

                    Vector3 grad = new Vector3(gx, gy, gz);
                    float len2 = grad.sqrMagnitude;
                    if (len2 < 1e-8f) continue;

                    // Outward normal = -gradient (gradient points from filled to empty)
                    Vector3 normal = -grad / Mathf.Sqrt(len2);

                    Vector3 center = new Vector3(
                        start.x + unit * x + halfUnit,
                        start.y + unit * y + halfUnit,
                        start.z + unit * z + halfUnit
                    );

                    _lineStarts[_lineCount] = center;
                    _lineEnds[_lineCount] = center + normal * normalLength;
                    _lineCount++;
                }
    }

    float GetFill(int x, int y, int z, int nx, int ny, int nz)
    {
        if (x < 0 || x >= nx || y < 0 || y >= ny || z < 0 || z >= nz) return 0;
        return _fillData[z * (nx * ny) + y * nx + x];
    }

    void OnRenderImage(RenderTexture src, RenderTexture dest)
    {
        // Pass-through blit first so we don't lose the composited scene + voxels
        Graphics.Blit(src, dest);

        if (_glMat == null || _lineCount == 0) return;

        // Explicitly set render target and viewport after the blit
        bool toTexture = (dest != null);
        Graphics.SetRenderTarget(dest);
        int vpW = toTexture ? dest.width : Screen.width;
        int vpH = toTexture ? dest.height : Screen.height;
        GL.Viewport(new Rect(0, 0, vpW, vpH));

        var cam = GetComponent<Camera>();
        GL.PushMatrix();
        GL.LoadProjectionMatrix(GL.GetGPUProjectionMatrix(cam.projectionMatrix, toTexture));
        GL.modelview = cam.worldToCameraMatrix;

        _glMat.SetPass(0);
        GL.Begin(GL.LINES);
        GL.Color(normalColor);

        for (int i = 0; i < _lineCount; i++)
        {
            GL.Vertex(_lineStarts[i]);
            GL.Vertex(_lineEnds[i]);
        }

        GL.End();
        GL.PopMatrix();
    }
}
