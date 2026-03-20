using UnityEngine;

/// <summary>
/// Attach to a Camera to visualise the voxel volume produced by
/// <see cref="VoxelTracerSystem"/> via DDA ray marching.
/// Composites voxels on top of the normal scene rendering.
/// Normals are derived from DDA face-crossing direction.
/// </summary>
[RequireComponent(typeof(Camera))]
public sealed class VoxelTracerCamera : MonoBehaviour
{
    // ================================================================
    // Inspector
    // ================================================================

    [Header("References")]
    public VoxelTracerSystem voxelSystem;
    public ComputeShader rayMarchCS;

    [Header("Visualisation")]
    public VisMode visMode = VisMode.Lit;
    public Color surfaceColor = new Color(0.85f, 0.85f, 0.85f);

    [Header("Lighting")]
    public Vector3 lightDirection = new Vector3(0.5f, 1f, 0.3f);
    public Color lightColor = Color.white;
    public Color ambientColor = new Color(0.12f, 0.12f, 0.18f);

    [Header("Quality")]
    [Range(256, 4096)] public int maxSteps = 1024;

    public enum VisMode { Lit = 0, Normals = 1 }

    // ================================================================
    // Private state
    // ================================================================

    Camera _cam;
    int _kernel;
    RenderTexture _colorRT;
    Material _compositeMat;

    static readonly int _SrcTex = Shader.PropertyToID("_SrcTex");
    static readonly int _VoxTex = Shader.PropertyToID("_VoxTex");

    // ================================================================
    // Lifecycle
    // ================================================================

    void OnEnable()
    {
        _cam = GetComponent<Camera>();

        if (rayMarchCS == null)
        {
            Debug.LogError("VoxelTracerCamera: assign rayMarchCS.");
            enabled = false;
            return;
        }

        _kernel = rayMarchCS.FindKernel("RayMarch");

        // Let the camera render the scene normally
        // (don't touch cullingMask or clearFlags)

        // Create composite shader at runtime
        _compositeMat = new Material(Shader.Find("Hidden/VoxelComposite"));
        if (_compositeMat == null || _compositeMat.shader == null || !_compositeMat.shader.isSupported)
        {
            Debug.LogError("VoxelTracerCamera: could not find Hidden/VoxelComposite shader.");
        }
    }

    void OnDisable()
    {
        ReleaseRTs();
        if (_compositeMat != null) { Destroy(_compositeMat); _compositeMat = null; }
    }

    // ================================================================
    // Render
    // ================================================================

    void OnRenderImage(RenderTexture src, RenderTexture dest)
    {
        if (voxelSystem == null || !voxelSystem.IsReady || rayMarchCS == null)
        {
            Graphics.Blit(src, dest);
            return;
        }

        int w = src.width;
        int h = src.height;
        EnsureRTs(w, h);

        // Camera
        rayMarchCS.SetMatrix("_CamToWorld", _cam.cameraToWorldMatrix);
        rayMarchCS.SetMatrix("_InvProj", _cam.projectionMatrix.inverse);
        rayMarchCS.SetVector("_CamPos", _cam.transform.position);
        rayMarchCS.SetVector("_ScreenSize", new Vector4(w, h, 0, 0));

        // Volume
        rayMarchCS.SetInt("_Width", voxelSystem.Nx);
        rayMarchCS.SetInt("_Height", voxelSystem.Ny);
        rayMarchCS.SetInt("_Depth", voxelSystem.Nz);
        rayMarchCS.SetVector("_Start", voxelSystem.ActiveGridMin);
        rayMarchCS.SetFloat("_Unit", voxelSystem.ActiveVoxelSize);

        // Fill texture
        rayMarchCS.SetTexture(_kernel, "_FillTex", voxelSystem.FillTexture);

        // Shading
        rayMarchCS.SetInt("_VisMode", (int)visMode);
        rayMarchCS.SetVector("_SurfaceColor", (Vector4)surfaceColor);
        rayMarchCS.SetVector("_BackgroundColor", new Vector4(0, 0, 0, 0));

        Vector3 ld = lightDirection.normalized;
        rayMarchCS.SetVector("_LightDir", new Vector4(ld.x, ld.y, ld.z, 0));
        rayMarchCS.SetVector("_LightColor", (Vector4)lightColor);
        rayMarchCS.SetVector("_AmbientColor", (Vector4)ambientColor);

        rayMarchCS.SetInt("_MaxSteps", maxSteps);

        // Output
        rayMarchCS.SetTexture(_kernel, "_ColorOut", _colorRT);

        // Dispatch
        rayMarchCS.Dispatch(_kernel, Mathf.CeilToInt(w / 8f), Mathf.CeilToInt(h / 8f), 1);

        // Composite: scene (src) as background, voxels (_colorRT) on top using alpha
        if (_compositeMat != null)
        {
            _compositeMat.SetTexture(_VoxTex, _colorRT);
            Graphics.Blit(src, dest, _compositeMat);
        }
        else
        {
            Graphics.Blit(_colorRT, dest);
        }
    }

    // ================================================================
    // RT management
    // ================================================================

    void EnsureRTs(int w, int h)
    {
        if (_colorRT != null && _colorRT.width == w && _colorRT.height == h)
            return;

        ReleaseRTs();

        _colorRT = new RenderTexture(w, h, 0, RenderTextureFormat.ARGBFloat)
        {
            enableRandomWrite = true,
            filterMode = FilterMode.Point,
            useMipMap = false
        };
        _colorRT.Create();
    }

    void ReleaseRTs()
    {
        if (_colorRT != null) { _colorRT.Release(); Destroy(_colorRT); _colorRT = null; }
    }

}
