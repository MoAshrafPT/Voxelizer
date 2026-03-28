using UnityEngine;

/// <summary>
/// Marker component: attach to any GameObject to mark it as a fluid particle source.
/// The external sim module reads registered sources to spawn SPH particles.
/// The voxelizer writes fluid phase into voxels overlapping this object's effect radius.
/// Self-registers with VoxelTracerSystem.
/// </summary>
public sealed class VoxelFluidSource : MonoBehaviour
{
    [Tooltip("Initial temperature of emitted fluid particles")]
    public float initialTemperature = 25f;

    [Tooltip("Emission radius in world units around the object's position")]
    [Min(0.01f)] public float emissionRadius = 1f;

    [Tooltip("Particle emission rate (particles per second). Used by external sim module.")]
    [Min(0)] public float emissionRate = 100f;

    [Tooltip("Initial velocity direction of emitted particles (world space)")]
    public Vector3 emissionDirection = Vector3.down;

    [Tooltip("Initial speed of emitted particles")]
    [Min(0)] public float emissionSpeed = 2f;

    void OnEnable() => VoxelTracerSystem.RegisterFluidSource(this);
    void OnDisable() => VoxelTracerSystem.UnregisterFluidSource(this);

    void OnDrawGizmosSelected()
    {
        Gizmos.color = new Color(0f, 0.5f, 1f, 0.4f);
        Gizmos.DrawWireSphere(transform.position, emissionRadius);

        // Draw emission direction
        Gizmos.color = new Color(0f, 0.8f, 1f, 0.8f);
        Gizmos.DrawRay(transform.position, emissionDirection.normalized * emissionRadius * 2f);
    }
}
