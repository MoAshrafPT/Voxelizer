using UnityEngine;

public class Voxelizer : MonoBehaviour
{
    [Header("Input")]
    [SerializeField] private Mesh inputMesh;
    [SerializeField] private int gridResolution = 64;

    [Header("Compute")]
    [SerializeField] private ComputeShader voxelizeShader;
    [SerializeField] private ComputeShader sortShader;

    [Header("Visualization")]
    [SerializeField] private Material sliceMaterial;
    [SerializeField] private int sliceAxis = 1;      // 0=X, 1=Y, 2=Z
    [SerializeField][Range(0, 128)] private int sliceIndex = 32;
    [SerializeField] private bool showSlice = true;

    // Buffers
    private GraphicsBuffer voxelGridBuffer;
    private GraphicsBuffer triangleBuffer;
    private GraphicsBuffer cellTriPairsBuffer;
    private GraphicsBuffer cellOffsetsBuffer;
    private GraphicsBuffer cellCountsBuffer;
    private GraphicsBuffer floodChangedBuffer;

    // Kernels
    private int clearGridKernel;
    private int assignTriKernel;
    private int buildOffsetsKernel;
    private int surfaceVoxelizeKernel;
    private int floodFillKernel;
    private int finalizeKernel;
    private int clearCellDataKernel;

    private GPUBitonicSort gpuSort;
    private int triangleCount;
    private int totalVoxels;
    private int totalCells;
    private int spatialGridDim;
    private Vector3 boundsMin;
    private Vector3 boundsMax;
    private Vector3 voxelSize;
    private Bounds renderBounds;

    private int sliceInstanceCount;

    void Start()
    {
        Voxelize();
    }

    void Voxelize()
    {
        // ========== 1. Extract triangles from mesh ==========
        Vector3[] vertices = inputMesh.vertices;
        int[] indices = inputMesh.triangles;
        triangleCount = indices.Length / 3;

        Triangle[] tris = new Triangle[triangleCount];
        for (int i = 0; i < triangleCount; i++)
        {
            tris[i].v0 = vertices[indices[i * 3 + 0]];
            tris[i].v1 = vertices[indices[i * 3 + 1]];
            tris[i].v2 = vertices[indices[i * 3 + 2]];
        }

        // ========== 2. Compute bounds ==========
        Bounds meshBounds = inputMesh.bounds;
        // Add padding so surface voxels aren't on grid edge (flood fill needs clear boundary)
        float padding = meshBounds.size.magnitude * 0.1f;
        boundsMin = meshBounds.min - Vector3.one * padding;
        boundsMax = meshBounds.max + Vector3.one * padding;
        Vector3 size = boundsMax - boundsMin;
        voxelSize = size / gridResolution;

        totalVoxels = gridResolution * gridResolution * gridResolution;

        // Spatial hash grid — cell size should match voxel size or be slightly larger
        spatialGridDim = gridResolution;
        totalCells = spatialGridDim * spatialGridDim * spatialGridDim;
        float cellSize = Mathf.Max(voxelSize.x, Mathf.Max(voxelSize.y, voxelSize.z));

        Debug.Log($"Voxelizer: {triangleCount} triangles → {gridResolution}³ grid ({totalVoxels} voxels)");
        Debug.Log($"Bounds: {boundsMin} to {boundsMax}, voxelSize: {voxelSize}");

        // ========== 3. Find kernels ==========
        clearGridKernel = voxelizeShader.FindKernel("ClearGrid");
        assignTriKernel = voxelizeShader.FindKernel("AssignTrianglesToCells");
        buildOffsetsKernel = voxelizeShader.FindKernel("BuildOffsets");
        surfaceVoxelizeKernel = voxelizeShader.FindKernel("SurfaceVoxelize");
        floodFillKernel = voxelizeShader.FindKernel("FloodFillStep");
        finalizeKernel = voxelizeShader.FindKernel("FinalizeInterior");
        clearCellDataKernel = voxelizeShader.FindKernel("ClearCellData");
        gpuSort = new GPUBitonicSort(sortShader);

        // ========== 4. Create buffers ==========
        voxelGridBuffer = new GraphicsBuffer(
            GraphicsBuffer.Target.Structured, totalVoxels, sizeof(uint));

        triangleBuffer = new GraphicsBuffer(
            GraphicsBuffer.Target.Structured, triangleCount, Triangle.Size);
        triangleBuffer.SetData(tris);

        cellTriPairsBuffer = new GraphicsBuffer(
            GraphicsBuffer.Target.Structured, triangleCount, sizeof(uint) * 2);

        cellOffsetsBuffer = new GraphicsBuffer(
            GraphicsBuffer.Target.Structured, totalCells, sizeof(uint));

        cellCountsBuffer = new GraphicsBuffer(
            GraphicsBuffer.Target.Structured, totalCells, sizeof(uint));

        floodChangedBuffer = new GraphicsBuffer(
            GraphicsBuffer.Target.Structured, 1, sizeof(uint));

        // ========== 5. Bind buffers to all kernels ==========
        int[] allKernels = { clearGridKernel, assignTriKernel, buildOffsetsKernel,
                             surfaceVoxelizeKernel, floodFillKernel, finalizeKernel,
                             clearCellDataKernel };

        foreach (int k in allKernels)
        {
            voxelizeShader.SetBuffer(k, "voxelGrid", voxelGridBuffer);
            voxelizeShader.SetBuffer(k, "triangles", triangleBuffer);
            voxelizeShader.SetBuffer(k, "cellTriPairs", cellTriPairsBuffer);
            voxelizeShader.SetBuffer(k, "cellOffsets", cellOffsetsBuffer);
            voxelizeShader.SetBuffer(k, "cellCounts", cellCountsBuffer);
            voxelizeShader.SetBuffer(k, "floodChanged", floodChangedBuffer);
        }

        // ========== 6. Set uniforms ==========
        voxelizeShader.SetInt("gridResX", gridResolution);
        voxelizeShader.SetInt("gridResY", gridResolution);
        voxelizeShader.SetInt("gridResZ", gridResolution);
        voxelizeShader.SetVector("boundsMin", new Vector4(boundsMin.x, boundsMin.y, boundsMin.z, 0));
        voxelizeShader.SetVector("boundsMax", new Vector4(boundsMax.x, boundsMax.y, boundsMax.z, 0));
        voxelizeShader.SetVector("voxelSize", new Vector4(voxelSize.x, voxelSize.y, voxelSize.z, 0));
        voxelizeShader.SetInt("triangleCount", triangleCount);
        voxelizeShader.SetInt("totalCells", totalCells);
        voxelizeShader.SetInt("gridDimension", spatialGridDim);
        voxelizeShader.SetFloat("cellSize", cellSize);

        // ========== 7. Dispatch ==========
        int voxelThreadGroups = (totalVoxels + 255) / 256;
        int triThreadGroups = (triangleCount + 255) / 256;
        int cellThreadGroups = (totalCells + 255) / 256;
        int voxelThreadGroups3D = (gridResolution + 3) / 4;

        // Clear
        voxelizeShader.Dispatch(clearGridKernel, voxelThreadGroups, 1, 1);
        voxelizeShader.Dispatch(clearCellDataKernel, cellThreadGroups, 1, 1);

        // Spatial hash
        voxelizeShader.Dispatch(assignTriKernel, triThreadGroups, 1, 1);
        gpuSort.Sort(cellTriPairsBuffer, triangleCount);
        voxelizeShader.Dispatch(buildOffsetsKernel, triThreadGroups, 1, 1);

        // Surface voxelize
        voxelizeShader.Dispatch(surfaceVoxelizeKernel,
            voxelThreadGroups3D, voxelThreadGroups3D, voxelThreadGroups3D);

        Debug.Log("Surface voxelization complete. Starting flood fill...");

        // Flood fill — iterate until no changes
        uint[] changedData = new uint[1];
        int maxIterations = gridResolution * 3; // safety limit
        int iteration = 0;

        do
        {
            changedData[0] = 0;
            floodChangedBuffer.SetData(changedData);

            voxelizeShader.Dispatch(floodFillKernel,
                voxelThreadGroups3D, voxelThreadGroups3D, voxelThreadGroups3D);

            floodChangedBuffer.GetData(changedData);
            iteration++;

        } while (changedData[0] > 0 && iteration < maxIterations);

        Debug.Log($"Flood fill complete in {iteration} iterations");

        // Finalize — unknown → solid, exterior → empty
        voxelizeShader.Dispatch(finalizeKernel, voxelThreadGroups, 1, 1);

        Debug.Log("Voxelization complete!");

        // ========== 8. Setup visualization ==========
        SetupSliceVisualization();
    }

    void SetupSliceVisualization()
    {
        sliceMaterial.SetBuffer("voxelGrid", voxelGridBuffer);
        sliceMaterial.SetVector("gridResolution", new Vector4(gridResolution, gridResolution, gridResolution, 0));
        sliceMaterial.SetVector("boundsMin", new Vector4(boundsMin.x, boundsMin.y, boundsMin.z, 0));
        sliceMaterial.SetVector("voxelSize", new Vector4(voxelSize.x, voxelSize.y, voxelSize.z, 0));

        renderBounds = new Bounds(
            (boundsMin + boundsMax) * 0.5f,
            boundsMax - boundsMin + Vector3.one * 2f);
    }

    void Update()
    {
        if (!showSlice || sliceMaterial == null) return;

        sliceIndex = Mathf.Clamp(sliceIndex, 0, gridResolution - 1);
        sliceMaterial.SetInt("_SliceAxis", sliceAxis);
        sliceMaterial.SetInt("_SliceIndex", sliceIndex);

        // How many cells in this slice
        int2 sliceDims;
        if (sliceAxis == 0)
            sliceDims = new int2(gridResolution, gridResolution); // Y × Z
        else if (sliceAxis == 1)
            sliceDims = new int2(gridResolution, gridResolution); // X × Z
        else
            sliceDims = new int2(gridResolution, gridResolution); // X × Y

        sliceInstanceCount = sliceDims.x * sliceDims.y;

        Graphics.DrawProcedural(
            sliceMaterial,
            renderBounds,
            MeshTopology.Triangles,
            6,
            sliceInstanceCount);
    }

    // Public API for simulation team
    public GraphicsBuffer GetVoxelGrid() => voxelGridBuffer;
    public int GetGridResolution() => gridResolution;
    public Vector3 GetBoundsMin() => boundsMin;
    public Vector3 GetVoxelSize() => voxelSize;

    void OnDestroy()
    {
        voxelGridBuffer?.Dispose();
        triangleBuffer?.Dispose();
        cellTriPairsBuffer?.Dispose();
        cellOffsetsBuffer?.Dispose();
        cellCountsBuffer?.Dispose();
        floodChangedBuffer?.Dispose();
    }

    // Helper struct for int2 since Unity doesn't have it outside Mathematics
    private struct int2
    {
        public int x, y;
        public int2(int x, int y) { this.x = x; this.y = y; }
    }
}