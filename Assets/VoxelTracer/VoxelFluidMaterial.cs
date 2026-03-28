using UnityEngine;

/// <summary>
/// Attachable material properties for fluid volumes and sources.
/// Exposes all SPH simulation parameters from the SimParams struct so each
/// fluid body/source can define its own physical behaviour.
///
/// Attach to a VoxelFluidSource (particle emitter) or VoxelWaterBody (static volume).
/// The external sim module reads these properties when initialising particles
/// and building the per-frame SimParams buffer.
/// Self-registers with VoxelTracerSystem.
/// </summary>
public sealed class VoxelFluidMaterial : MonoBehaviour
{
    [Header("SPH Parameters")]
    [Tooltip("Target rest density — should match natural packing density (kg/m³). " +
             "Maps to restDensity in SimParams.")]
    public float restDensity = 60f;

    [Tooltip("Gas stiffness constant. Higher = more incompressible. " +
             "Maps to gasStiffness in SimParams.")]
    [Range(0.1f, 500f)] public float gasStiffness = 10f;

    [Tooltip("Viscosity coefficient. Higher = thicker fluid. " +
             "Maps to viscosity in SimParams.")]
    [Range(0.001f, 50f)] public float viscosity = 15f;

    [Tooltip("SPH smoothing kernel radius (world units). " +
             "Maps to smoothingRadius in SimParams.")]
    [Range(0.01f, 5f)] public float smoothingRadius = 0.5f;

    [Tooltip("Mass per particle. Maps to particleMass in SimParams.")]
    [Range(0.001f, 100f)] public float particleMass = 1f;

    [Tooltip("Per-frame velocity damping (0.9–1.0). 1.0 = no drag, lower = more drag. " +
             "Maps to damping in SimParams.")]
    [Range(0.9f, 1f)] public float damping = 0.998f;

    [Header("Thermal")]
    [Tooltip("Initial temperature of this fluid (Kelvin or sim units). " +
             "Maps to temperature in the Particle struct.")]
    public float temperature = 25f;

    [Tooltip("How fast heat spreads between particles (m²/s). " +
             "Maps to thermalDiffusivity in SimParams.")]
    [Range(0f, 1f)] public float thermalDiffusivity = 0.1f;

    [Tooltip("Ambient temperature the fluid cools toward. " +
             "Maps to ambientTemperature in SimParams.")]
    [Range(0f, 1000f)] public float ambientTemperature = 25f;

    [Tooltip("How fast particles lose heat to the environment. " +
             "Maps to coolingRate in SimParams.")]
    [Range(0.001f, 1f)] public float coolingRate = 0.01f;

    [Header("Physical")]
    [Tooltip("Gravity acceleration (m/s²). Maps to gravity in SimParams.")]
    public Vector3 gravity = new Vector3(0f, -9.81f, 0f);

    [Tooltip("Phase tag for this fluid (positive int). " +
             "Maps to the 'phase' field in the Particle struct. " +
             "Use different values to distinguish fluid types (water=1, oil=2, etc).")]
    [Min(1)] public int phase = 1;

    void OnEnable() => VoxelTracerSystem.RegisterFluidMaterial(this);
    void OnDisable() => VoxelTracerSystem.UnregisterFluidMaterial(this);

    void OnDrawGizmosSelected()
    {
        // Visualize viscosity: thicker = more opaque blue
        float v = Mathf.InverseLerp(0f, 50f, viscosity);
        Gizmos.color = new Color(0f, 0.3f, 1f, 0.15f + 0.35f * v);

        var wb = GetComponent<VoxelWaterBody>();
        if (wb != null)
        {
            Gizmos.DrawWireCube(transform.position, wb.size);
        }
        else
        {
            var fs = GetComponent<VoxelFluidSource>();
            if (fs != null)
                Gizmos.DrawWireSphere(transform.position, fs.emissionRadius);
            else
                Gizmos.DrawWireSphere(transform.position, 0.5f);
        }
    }
}
