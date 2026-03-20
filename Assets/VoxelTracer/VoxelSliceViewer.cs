using UnityEngine;

/// <summary>
/// Full-screen slice viewer of the voxel fill volume.
/// Press a key to toggle the overlay on/off.
/// Scrub slices with scroll wheel while overlay is visible.
/// </summary>
[RequireComponent(typeof(Camera))]
[DefaultExecutionOrder(200)]
public class VoxelSliceViewer : MonoBehaviour
{
    [Header("References")]
    public VoxelTracerSystem voxelSystem;

    [Header("Slice")]
    public SliceAxis axis = SliceAxis.Y;
    [Range(0f, 1f)]
    public float slicePosition = 0.5f;

    [Header("Colors")]
    public Color filledColor = Color.white;
    public Color emptyColor = new Color(0, 0, 0, 0.85f);
    public Color surfaceColor = new Color(0.2f, 0.8f, 0.2f, 1f);
    public bool highlightSurface = true;

    [Header("Controls")]
    [Tooltip("Key to toggle the slice overlay on/off")]
    public KeyCode toggleKey = KeyCode.F2;

    public enum SliceAxis { X, Y, Z }

    bool _visible;
    Texture2D _sliceTex;
    float[] _fillData;
    int _cachedNx, _cachedNy, _cachedNz;
    int _texW, _texH;
    float _lastRefresh = -999f;
    GUIStyle _labelStyle;
    GUIStyle _boxStyle;

    void OnDisable()
    {
        if (_sliceTex != null) { Destroy(_sliceTex); _sliceTex = null; }
        _fillData = null;
    }

    void Update()
    {
        if (Input.GetKeyDown(toggleKey))
            _visible = !_visible;

        if (!_visible) return;
        if (voxelSystem == null || !voxelSystem.IsReady) return;

        // Scroll wheel scrubs slice position when overlay is visible
        float scroll = Input.GetAxis("Mouse ScrollWheel");
        if (Mathf.Abs(scroll) > 0.001f)
        {
            int maxSlice = GetSliceCount() - 1;
            if (maxSlice > 0)
            {
                float step = 1f / maxSlice;
                slicePosition = Mathf.Clamp01(slicePosition + scroll * step * 5f);
            }
        }

        // Axis switching: 1/2/3 keys while overlay is open
        if (Input.GetKeyDown(KeyCode.Alpha1)) axis = SliceAxis.X;
        if (Input.GetKeyDown(KeyCode.Alpha2)) axis = SliceAxis.Y;
        if (Input.GetKeyDown(KeyCode.Alpha3)) axis = SliceAxis.Z;

        if (Time.time - _lastRefresh < 0.1f) return;
        _lastRefresh = Time.time;
        BuildSliceTexture();
    }

    int GetSliceCount()
    {
        if (voxelSystem == null) return 1;
        return axis switch
        {
            SliceAxis.X => voxelSystem.Nx,
            SliceAxis.Y => voxelSystem.Ny,
            SliceAxis.Z => voxelSystem.Nz,
            _ => 1
        };
    }

    void ReadFillData()
    {
        int nx = voxelSystem.Nx;
        int ny = voxelSystem.Ny;
        int nz = voxelSystem.Nz;
        int total = nx * ny * nz;

        var fillRT = voxelSystem.FillTexture;
        if (fillRT == null) return;

        if (_fillData == null || _fillData.Length != total ||
            _cachedNx != nx || _cachedNy != ny || _cachedNz != nz)
        {
            _fillData = new float[total];
            _cachedNx = nx;
            _cachedNy = ny;
            _cachedNz = nz;
        }

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
    }

    float GetFill(int x, int y, int z)
    {
        if (x < 0 || x >= _cachedNx || y < 0 || y >= _cachedNy || z < 0 || z >= _cachedNz) return 0;
        return _fillData[z * (_cachedNx * _cachedNy) + y * _cachedNx + x];
    }

    bool IsSurface(int x, int y, int z)
    {
        if (GetFill(x, y, z) < 0.5f) return false;
        return GetFill(x - 1, y, z) < 0.5f || GetFill(x + 1, y, z) < 0.5f ||
               GetFill(x, y - 1, z) < 0.5f || GetFill(x, y + 1, z) < 0.5f ||
               GetFill(x, y, z - 1) < 0.5f || GetFill(x, y, z + 1) < 0.5f;
    }

    void BuildSliceTexture()
    {
        ReadFillData();
        if (_fillData == null) return;

        int nx = _cachedNx, ny = _cachedNy, nz = _cachedNz;

        // Determine slice dimensions based on axis
        int sliceW, sliceH, sliceIdx;
        switch (axis)
        {
            case SliceAxis.X:
                sliceW = nz; sliceH = ny;
                sliceIdx = Mathf.Clamp(Mathf.RoundToInt(slicePosition * (nx - 1)), 0, nx - 1);
                break;
            case SliceAxis.Y:
                sliceW = nx; sliceH = nz;
                sliceIdx = Mathf.Clamp(Mathf.RoundToInt(slicePosition * (ny - 1)), 0, ny - 1);
                break;
            default: // Z
                sliceW = nx; sliceH = ny;
                sliceIdx = Mathf.Clamp(Mathf.RoundToInt(slicePosition * (nz - 1)), 0, nz - 1);
                break;
        }

        if (sliceW <= 0 || sliceH <= 0) return;

        if (_sliceTex == null || _texW != sliceW || _texH != sliceH)
        {
            if (_sliceTex != null) Destroy(_sliceTex);
            _sliceTex = new Texture2D(sliceW, sliceH, TextureFormat.RGBA32, false)
            {
                filterMode = FilterMode.Point,
                wrapMode = TextureWrapMode.Clamp
            };
            _texW = sliceW;
            _texH = sliceH;
        }

        var pixels = _sliceTex.GetPixels32();

        for (int v = 0; v < sliceH; v++)
        {
            for (int u = 0; u < sliceW; u++)
            {
                int x, y, z;
                switch (axis)
                {
                    case SliceAxis.X: x = sliceIdx; y = v; z = u; break;
                    case SliceAxis.Y: x = u; y = sliceIdx; z = v; break;
                    default: x = u; y = v; z = sliceIdx; break;
                }

                float fill = GetFill(x, y, z);
                Color c;
                if (fill > 0.5f)
                {
                    if (highlightSurface && IsSurface(x, y, z))
                        c = surfaceColor;
                    else
                        c = filledColor;
                }
                else
                {
                    c = emptyColor;
                }

                pixels[v * sliceW + u] = c;
            }
        }

        _sliceTex.SetPixels32(pixels);
        _sliceTex.Apply(false);
    }

    void OnGUI()
    {
        if (!_visible || _sliceTex == null || voxelSystem == null || !voxelSystem.IsReady) return;

        // Lazy-init styles
        if (_labelStyle == null)
        {
            _labelStyle = new GUIStyle(GUI.skin.label)
            {
                fontSize = 18,
                fontStyle = FontStyle.Bold,
                alignment = TextAnchor.MiddleCenter
            };
            _labelStyle.normal.textColor = Color.white;

            _boxStyle = new GUIStyle(GUI.skin.box);
        }

        float sw = Screen.width;
        float sh = Screen.height;

        // Dark semi-transparent background covering the whole screen
        GUI.color = new Color(0, 0, 0, 0.7f);
        GUI.DrawTexture(new Rect(0, 0, sw, sh), Texture2D.whiteTexture);
        GUI.color = Color.white;

        // Fit slice texture into screen with correct aspect ratio + padding
        float padding = 60f;
        float barH = 40f; // top info bar
        float availW = sw - padding * 2f;
        float availH = sh - padding * 2f - barH;
        float aspect = (float)_texW / Mathf.Max(_texH, 1);

        float imgW, imgH;
        if (availW / availH > aspect)
        {
            imgH = availH;
            imgW = imgH * aspect;
        }
        else
        {
            imgW = availW;
            imgH = imgW / Mathf.Max(aspect, 0.001f);
        }

        float imgX = (sw - imgW) * 0.5f;
        float imgY = barH + (availH - imgH) * 0.5f + padding;

        // Draw the slice
        GUI.DrawTexture(new Rect(imgX, imgY, imgW, imgH), _sliceTex, ScaleMode.StretchToFill, true);

        // Thin border
        GUI.color = Color.green;
        float b = 2f;
        GUI.DrawTexture(new Rect(imgX - b, imgY - b, imgW + b * 2, b), Texture2D.whiteTexture); // top
        GUI.DrawTexture(new Rect(imgX - b, imgY + imgH, imgW + b * 2, b), Texture2D.whiteTexture); // bottom
        GUI.DrawTexture(new Rect(imgX - b, imgY, b, imgH), Texture2D.whiteTexture); // left
        GUI.DrawTexture(new Rect(imgX + imgW, imgY, b, imgH), Texture2D.whiteTexture); // right
        GUI.color = Color.white;

        // Info bar
        int maxIdx = GetSliceCount() - 1;
        int sliceIdx = Mathf.RoundToInt(slicePosition * Mathf.Max(maxIdx, 1));

        string info = $"{axis} Slice {sliceIdx}/{maxIdx}   |   [Scroll] change slice   [1/2/3] change axis   [{toggleKey}] close";
        GUI.Label(new Rect(0, 10, sw, barH), info, _labelStyle);
    }
}
