using UnityEngine;

public class GPUBitonicSort
{
    private ComputeShader sortShader;
    private int kernel;
    private int threadGroups;

    public GPUBitonicSort(ComputeShader shader)
    {
        sortShader = shader;
        kernel = sortShader.FindKernel("BitonicSort");
    }

    public void Sort(GraphicsBuffer buffer, int count)
    {
        sortShader.SetBuffer(kernel, "data", buffer);
        sortShader.SetInt("count", count);
        threadGroups = (count + 255) / 256;

        // Bitonic sort requires power-of-2 passes
        // Outer loop: block size doubles each time (2, 4, 8, 16, ...)
        for (int block = 2; block <= NextPowerOf2(count); block <<= 1)
        {
            // Inner loop: dimension halves within each block (block/2, block/4, ..., 1)
            for (int dim = block >> 1; dim > 0; dim >>= 1)
            {
                sortShader.SetInt("block", block);
                sortShader.SetInt("dimension", dim);
                sortShader.Dispatch(kernel, threadGroups, 1, 1);
            }
        }
    }

    private int NextPowerOf2(int n)
    {
        n--;
        n |= n >> 1;
        n |= n >> 2;
        n |= n >> 4;
        n |= n >> 8;
        n |= n >> 16;
        return n + 1;
    }
}