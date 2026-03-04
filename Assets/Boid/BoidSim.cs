using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.PlayerLoop;

public struct BoidData
{
    public Vector3 position;
    public Vector3 velocity;
    public Vector3 acceleration;
    public int neighborCount;

    public static int Size => sizeof(float) * 9 + sizeof(int);

}



public class BoidSim : MonoBehaviour
{
    [Header("References")]
    [SerializeField] private ComputeShader computeShader;
    [SerializeField] private ComputeShader sortShader;
    private GPUBitonicSort gpuSort;
    [SerializeField] private Material renderMaterial;

    [Header("Settings")]
    [SerializeField] private int boidCount = 5000;
    [SerializeField] private float neighborRadius = 3f;
    [SerializeField] private int gridDimension = 20;  // 20×20×20 cells

    private GraphicsBuffer boidBuffer;
    private GraphicsBuffer cellBoidPairsBuffer;
    private GraphicsBuffer cellOffsetsBuffer;
    private GraphicsBuffer cellCountsBuffer;

    private int assignCellsKernel;
    private int buildOffsetsKernel;
    private int findNeighborsKernel;
    private int updateKernel;
    private int clearOffsetsKernel;

    private int boidThreadGroups;
    private int cellThreadGroups;
    private int totalCells;
    private Bounds bounds;

    void Start()
    {
        totalCells = gridDimension * gridDimension * gridDimension;
        float cellSize = neighborRadius;

        // Find kernels
        assignCellsKernel = computeShader.FindKernel("AssignCells");
        buildOffsetsKernel = computeShader.FindKernel("BuildOffsets");
        findNeighborsKernel = computeShader.FindKernel("FindNeighbors");
        updateKernel = computeShader.FindKernel("UpdateBoids");
        clearOffsetsKernel = computeShader.FindKernel("ClearOffsets");
        gpuSort = new GPUBitonicSort(sortShader);
        // Create buffers
        boidBuffer = new GraphicsBuffer(
            GraphicsBuffer.Target.Structured, boidCount, BoidData.Size);

        cellBoidPairsBuffer = new GraphicsBuffer(
            GraphicsBuffer.Target.Structured, boidCount, sizeof(uint) * 2);

        cellOffsetsBuffer = new GraphicsBuffer(
            GraphicsBuffer.Target.Structured, totalCells, sizeof(uint));

        cellCountsBuffer = new GraphicsBuffer(
            GraphicsBuffer.Target.Structured, totalCells, sizeof(uint));

        // Initialize boids
        BoidData[] initial = new BoidData[boidCount];
        for (int i = 0; i < boidCount; i++)
        {
            initial[i].position = Random.insideUnitSphere * 20f;
            initial[i].velocity = Random.insideUnitSphere * 2f;
        }
        boidBuffer.SetData(initial);

        // Clear cell data
        cellOffsetsBuffer.SetData(new uint[totalCells]);
        cellCountsBuffer.SetData(new uint[totalCells]);

        // Bind to ALL kernels that use each buffer
        int[] allKernels = { assignCellsKernel, buildOffsetsKernel, findNeighborsKernel, updateKernel, clearOffsetsKernel };
        foreach (int k in allKernels)
        {
            computeShader.SetBuffer(k, "boids", boidBuffer);
            computeShader.SetBuffer(k, "cellBoidPairs", cellBoidPairsBuffer);
            computeShader.SetBuffer(k, "cellOffsets", cellOffsetsBuffer);
            computeShader.SetBuffer(k, "cellCounts", cellCountsBuffer);
        }

        // Bind to render material
        renderMaterial.SetBuffer("boids", boidBuffer);

        // Set constants
        computeShader.SetInt("boidCount", boidCount);
        computeShader.SetFloat("neighborRadius", neighborRadius);
        computeShader.SetFloat("cellSize", cellSize);
        computeShader.SetInt("gridDimension", gridDimension);

        boidThreadGroups = (boidCount + 255) / 256;
        cellThreadGroups = (totalCells + 255) / 256;
        bounds = new Bounds(Vector3.zero, Vector3.one * 100f);
    }

    void Update()
    {
        computeShader.SetFloat("deltaTime", Time.deltaTime);

        // 1. Clear cell data from last frame
        computeShader.Dispatch(clearOffsetsKernel, cellThreadGroups, 1, 1);

        // 2. Assign each boid to a cell
        computeShader.Dispatch(assignCellsKernel, boidThreadGroups, 1, 1);

        // 3. Sort cellBoidPairs by cellID
        gpuSort.Sort(cellBoidPairsBuffer, boidCount);

        // 4. Build offset table
        computeShader.Dispatch(buildOffsetsKernel, boidThreadGroups, 1, 1);

        // 5. Find neighbors using spatial hash
        computeShader.Dispatch(findNeighborsKernel, boidThreadGroups, 1, 1);

        // 6. Update positions
        computeShader.Dispatch(updateKernel, boidThreadGroups, 1, 1);

        // Render
        Graphics.DrawProcedural(
            renderMaterial, bounds, MeshTopology.Triangles, 6, boidCount);
    }

    // TEMPORARY — CPU sort until we implement GPU bitonic sort
    // void SortOnCPU()
    // {
    //     uint[] pairData = new uint[boidCount * 2];
    //     cellBoidPairsBuffer.GetData(pairData);

    //     // Convert to sortable array
    //     uint2Pair[] pairs = new uint2Pair[boidCount];
    //     for (int i = 0; i < boidCount; i++)
    //     {
    //         pairs[i].cellID = pairData[i * 2];
    //         pairs[i].boidIndex = pairData[i * 2 + 1];
    //     }

    //     // Sort by cellID
    //     System.Array.Sort(pairs, (a, b) => a.cellID.CompareTo(b.cellID));

    //     // Write back
    //     for (int i = 0; i < boidCount; i++)
    //     {
    //         pairData[i * 2] = pairs[i].cellID;
    //         pairData[i * 2 + 1] = pairs[i].boidIndex;
    //     }
    //     cellBoidPairsBuffer.SetData(pairData);
    // }

    struct uint2Pair
    {
        public uint cellID;
        public uint boidIndex;
    }

    void OnDestroy()
    {
        boidBuffer?.Dispose();
        cellBoidPairsBuffer?.Dispose();
        cellOffsetsBuffer?.Dispose();
        cellCountsBuffer?.Dispose();
    }
}