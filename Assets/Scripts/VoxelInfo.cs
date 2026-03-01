using UnityEngine;

/// <summary>
/// Attached to each voxel group GameObject to identify which source scene object it came from.
/// Use this component to query voxel origin at runtime.
/// </summary>
public class VoxelInfo : MonoBehaviour
{
    [Tooltip("Name of the original scene object this voxel group was generated from")]
    public string sourceObjectName;

    [Tooltip("Total number of voxels in this group")]
    public int voxelCount;
}
