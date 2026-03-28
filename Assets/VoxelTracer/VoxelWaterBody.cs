using UnityEngine;

/// <summary>
/// Defines an axis-aligned water body volume in world space.
/// Voxels inside this volume are marked as fluid phase with the specified temperature.
/// The volume is drawn in the scene view as a wireframe box filled with voxel dots.
/// Self-registers with VoxelTracerSystem.
/// </summary>
public sealed class VoxelWaterBody : MonoBehaviour
{
    [Tooltip("Size of the water body volume in world units")]
    public Vector3 size = new Vector3(4f, 2f, 4f);

    [Tooltip("Initial temperature of the water body")]
    public float initialTemperature = 25f;

    /// <summary>World-space AABB min corner.</summary>
    public Vector3 WorldMin => transform.position - size * 0.5f;

    /// <summary>World-space AABB max corner.</summary>
    public Vector3 WorldMax => transform.position + size * 0.5f;

    void OnEnable() => VoxelTracerSystem.RegisterWaterBody(this);
    void OnDisable() => VoxelTracerSystem.UnregisterWaterBody(this);

    void OnDrawGizmos()
    {
        // Semi-transparent blue wireframe
        Gizmos.color = new Color(0f, 0.4f, 1f, 0.3f);
        Gizmos.DrawWireCube(transform.position, size);

        // Solid fill for visibility
        Gizmos.color = new Color(0f, 0.3f, 0.8f, 0.08f);
        Gizmos.DrawCube(transform.position, size);
    }

    void OnDrawGizmosSelected()
    {
        // Brighter when selected
        Gizmos.color = new Color(0f, 0.6f, 1f, 0.5f);
        Gizmos.DrawWireCube(transform.position, size);

        // Draw voxel grid preview when a VoxelTracerSystem is available
        var sys = FindAnyObjectByType<VoxelTracerSystem>();
        if (sys == null || !sys.IsReady) return;

        float vs = sys.ActiveVoxelSize;
        Vector3 mn = WorldMin;
        Vector3 mx = WorldMax;

        // Clamp to grid bounds
        Vector3 gridMn = sys.ActiveGridMin;
        Vector3 gridMx = gridMn + new Vector3(sys.Nx, sys.Ny, sys.Nz) * vs;
        mn = Vector3.Max(mn, gridMn);
        mx = Vector3.Min(mx, gridMx);

        int countX = Mathf.CeilToInt((mx.x - mn.x) / vs);
        int countY = Mathf.CeilToInt((mx.y - mn.y) / vs);
        int countZ = Mathf.CeilToInt((mx.z - mn.z) / vs);

        // Limit gizmo dots to avoid editor stalls
        int totalPreview = countX * countY * countZ;
        if (totalPreview <= 0 || totalPreview > 5000) return;

        Gizmos.color = new Color(0f, 0.5f, 1f, 0.25f);
        float half = vs * 0.5f;
        for (int z = 0; z < countZ; z++)
            for (int y = 0; y < countY; y++)
                for (int x = 0; x < countX; x++)
                {
                    Vector3 center = mn + new Vector3(x * vs + half, y * vs + half, z * vs + half);
                    Gizmos.DrawCube(center, Vector3.one * vs * 0.3f);
                }
    }
}
