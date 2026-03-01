using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ComputeTest : MonoBehaviour
{
    [SerializeField] private ComputeShader ComputeShader;
    public Mesh InputMesh;
    private Mesh OutputMesh;

    public MeshFilter Filter;
    private GraphicsBuffer InBuffer;
    private GraphicsBuffer OutBuffer;
    // Start is called before the first frame update
    void Start()
    {
        OutputMesh = Instantiate(InputMesh);
        Filter.mesh = OutputMesh;

        InputMesh.vertexBufferTarget = GraphicsBuffer.Target.Raw;
        OutputMesh.vertexBufferTarget = GraphicsBuffer.Target.Raw;
        InBuffer = InputMesh.GetVertexBuffer(0);
        OutBuffer = OutputMesh.GetVertexBuffer(0);
        ComputeShader.SetBuffer(0, "sourceVerts", InBuffer);
        ComputeShader.SetBuffer(0, "outVerts", OutBuffer);
        ComputeShader.SetInt("MeshVertexCount", InputMesh.vertexCount);
        ComputeShader.SetInt("SourceVertsBufferStride", InBuffer.stride);

    }

    // Update is called once per frame
    void Update()
    {
        ComputeShader.Dispatch(0, InputMesh.vertexCount / 128 + 1, 1, 1);
    }

    void OnDestroy()
    {
        InBuffer?.Dispose();
        OutBuffer?.Dispose();
        if (OutputMesh != null)
        {
            Destroy(OutputMesh);
        }
    }
}
