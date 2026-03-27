using UnityEngine;

/// <summary>
/// Auto-registers SkinnedMeshRenderers with VoxelTracerSystem.
/// Attach to any GameObject with a SkinnedMeshRenderer, or add
/// RequireComponent in your character setup. Avoids per-frame
/// FindObjectsByType scene scans.
/// </summary>
[RequireComponent(typeof(SkinnedMeshRenderer))]
public sealed class VoxelSkinRegistrar : MonoBehaviour
{
    SkinnedMeshRenderer _smr;

    void OnEnable()
    {
        _smr = GetComponent<SkinnedMeshRenderer>();
        VoxelTracerSystem.RegisterSkin(_smr);
    }

    void OnDisable()
    {
        VoxelTracerSystem.UnregisterSkin(_smr);
    }
}
