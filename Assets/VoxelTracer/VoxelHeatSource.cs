using UnityEngine;

/// <summary>
/// Marker component: attach to any GameObject to mark it as a heat source.
/// The voxelizer writes temperature values into voxels overlapping this object's bounds.
/// Self-registers with VoxelTracerSystem.
/// </summary>
public sealed class VoxelHeatSource : MonoBehaviour
{
    [Tooltip("Temperature value written into overlapping voxels (Kelvin or arbitrary units)")]
    public float temperature = 500f;

    [Tooltip("Radius of effect in world units. 0 = use renderer bounds.")]
    [Min(0)] public float radius = 0f;

    void OnEnable() => VoxelTracerSystem.RegisterHeatSource(this);
    void OnDisable() => VoxelTracerSystem.UnregisterHeatSource(this);

    void OnDrawGizmosSelected()
    {
        Gizmos.color = new Color(1f, 0.3f, 0f, 0.4f);
        if (radius > 0f)
        {
            Gizmos.DrawWireSphere(transform.position, radius);
        }
        else
        {
            var r = GetComponent<Renderer>();
            if (r != null)
            {
                Gizmos.DrawWireCube(r.bounds.center, r.bounds.size);
            }
            else
            {
                Gizmos.DrawWireSphere(transform.position, 0.5f);
            }
        }
    }
}
