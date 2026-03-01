using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GPUParticleSystem : MonoBehaviour
{
    // Start is called before the first frame update
    [Header("References")]
    [SerializeField] private ComputeShader computeShader;
    [SerializeField] private Material renderMaterial;
    [Header("Settings")]
    [SerializeField] private int particleCount = 10000;

    private GraphicsBuffer particleBuffer;
    private int emitKernel;
    private int updateKernel;
    private int threadGroups;

    private Bounds bounds;

    void Start()
    {
        emitKernel = computeShader.FindKernel("Emit");
        updateKernel = computeShader.FindKernel("Update");

        Particle[] initial = new Particle[particleCount];

        particleBuffer = new GraphicsBuffer(
            GraphicsBuffer.Target.Structured,
            particleCount,
            Particle.Size
        );

        particleBuffer.SetData(initial);
        //bind buffers to both kernels
        computeShader.SetBuffer(emitKernel, "particles", particleBuffer);
        computeShader.SetBuffer(updateKernel, "particles", particleBuffer);
        renderMaterial.SetBuffer("particles", particleBuffer);
        renderMaterial.SetFloat("_Glossiness", 0.8f);
        computeShader.SetInt("particleCount", particleCount);
        threadGroups = (particleCount + 255) / 256;

        bounds = new Bounds(Vector3.zero, Vector3.one * 100f);


    }

    // Update is called once per frame
    void Update()
    {
        computeShader.SetFloat("deltaTime", Time.deltaTime);
        computeShader.SetFloat("time", Time.time);
        computeShader.SetVector("emitterPosition", transform.position);

        //respawn
        computeShader.Dispatch(emitKernel, threadGroups, 1, 1);

        //update physics
        computeShader.Dispatch(updateKernel, threadGroups, 1, 1);

        //6 vertices per particle
        Graphics.DrawProcedural(
            renderMaterial,
            bounds,
            MeshTopology.Triangles,
            6,
            particleCount
        );

    }

    void OnDestroy()
    {
        particleBuffer?.Dispose();

    }
}
