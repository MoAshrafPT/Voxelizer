using UnityEngine;

/// <summary>
/// Attachable material properties for solid objects.
/// Defines thermal and physical properties that the external sim module
/// reads when computing heat transfer between solids and fluids.
/// Matches relevant fields from the SimParams/Particle structs in the sim module.
///
/// Attach to any GameObject with a Renderer (static or VoxelDynamic).
/// The voxelizer stamps temperature into the TemperatureTexture for overlapping voxels.
/// The external sim module reads these properties for boundary condition computation.
/// Self-registers with VoxelTracerSystem.
/// </summary>
public sealed class VoxelSolidMaterial : MonoBehaviour
{
    [Header("Thermal")]
    [Tooltip("Temperature of this solid (Kelvin or sim units)")]
    public float temperature = 25f;

    [Tooltip("How fast heat spreads through this solid (m²/s). " +
             "Maps to thermalDiffusivity in SimParams.")]
    [Min(0)] public float thermalDiffusivity = 0.1f;

    [Tooltip("Rate at which this solid exchanges heat with adjacent fluid. " +
             "Maps to coolingRate in SimParams when used as a boundary.")]
    [Range(0f, 1f)] public float coolingRate = 0.01f;

    [Header("Physical")]
    [Tooltip("Density of this solid (kg/m³). Used by sim for buoyancy / boundary forces.")]
    [Min(0.01f)] public float density = 2500f;

    [Tooltip("Mass per voxel. 0 = auto-compute from density × voxel volume.")]
    [Min(0)] public float mass = 0f;

    [Tooltip("Phase tag: 0 = generic solid, negative values for user-defined sub-types. " +
             "Maps to the 'phase' field in the Particle struct.")]
    public int phase = 0;

    void OnEnable()  => VoxelTracerSystem.RegisterSolidMaterial(this);
    void OnDisable() => VoxelTracerSystem.UnregisterSolidMaterial(this);

    void OnDrawGizmosSelected()
    {
        // Visualize thermal intensity: hotter = more red
        float t = Mathf.InverseLerp(0f, 1000f, temperature);
        Gizmos.color = new Color(t, 0.2f * (1f - t), 1f - t, 0.3f);

        var r = GetComponent<Renderer>();
        if (r != null)
            Gizmos.DrawWireCube(r.bounds.center, r.bounds.size);
        else
            Gizmos.DrawWireSphere(transform.position, 0.5f);
    }
}
