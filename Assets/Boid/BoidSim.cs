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
    [SerializeField] private Material renderMaterial;

    [Header("Settings")]
    [SerializeField] private int boidCount = 1000;
    [SerializeField] private float neighborRadius = 3f;

    private GraphicsBuffer boidBuffer;
    private int findNeighborsKernel;
    private int updateKernel;
    private int threadGroups;
    private Bounds bounds;

    void Start()
    {
        findNeighborsKernel = computeShader.FindKernel("FindNeighbors");
        updateKernel = computeShader.FindKernel("UpdateBoids");

        // Create buffer
        boidBuffer = new GraphicsBuffer(
            GraphicsBuffer.Target.Structured,
            boidCount,
            BoidData.Size
        );

        // Initialize with random positions and velocities
        BoidData[] initial = new BoidData[boidCount];
        for (int i = 0; i < boidCount; i++)
        {
            initial[i].position = Random.insideUnitSphere * 20f;
            initial[i].velocity = Random.insideUnitSphere * 2f;
            initial[i].acceleration = Vector3.zero;
            initial[i].neighborCount = 0;
        }
        boidBuffer.SetData(initial);

        // Bind to both kernels
        computeShader.SetBuffer(findNeighborsKernel, "boids", boidBuffer);
        computeShader.SetBuffer(updateKernel, "boids", boidBuffer);

        // Bind to render material
        renderMaterial.SetBuffer("boids", boidBuffer);

        // Constants
        computeShader.SetInt("boidCount", boidCount);
        computeShader.SetFloat("neighborRadius", neighborRadius);

        threadGroups = (boidCount + 255) / 256;
        bounds = new Bounds(Vector3.zero, Vector3.one * 50f);
    }

    void Update()
    {
        computeShader.SetFloat("deltaTime", Time.deltaTime);

        // Pass 1 — find neighbors, compute forces
        computeShader.Dispatch(findNeighborsKernel, threadGroups, 1, 1);

        // Pass 2 — integrate velocity and position
        computeShader.Dispatch(updateKernel, threadGroups, 1, 1);

        // Render
        Graphics.DrawProcedural(
            renderMaterial,
            bounds,
            MeshTopology.Triangles,
            6,
            boidCount
        );
    }

    void OnDestroy()
    {
        boidBuffer?.Dispose();
    }
}
