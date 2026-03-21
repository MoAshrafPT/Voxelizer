using UnityEngine;

/// <summary>
/// Marker component: attach to any GameObject to flag its MeshRenderer
/// as dynamic for the voxelizer. Dynamic objects are re-voxelized every
/// frame while static objects are voxelized once and cached on the GPU.
/// </summary>
public sealed class VoxelDynamic : MonoBehaviour { }
