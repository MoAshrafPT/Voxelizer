Shader "Custom/VoxelSlice"
{
    Properties
    {
        _SliceAxis ("Slice Axis (0=X, 1=Y, 2=Z)", Int) = 1
        _SliceIndex ("Slice Index", Int) = 32
        _FilledColor ("Filled Color", Color) = (1, 0.3, 0, 1)
        _EmptyColor ("Empty Color", Color) = (0.1, 0.1, 0.1, 0.3)
    }
    SubShader
    {
        Tags { "Queue"="Transparent" "RenderType"="Transparent" }
        Blend SrcAlpha OneMinusSrcAlpha
        ZWrite Off
        Cull Off

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            StructuredBuffer<uint> voxelGrid;
            float4 gridResolution;  // xyz components used, set via SetVector
            float4 boundsMin;
            float4 voxelSize;
            int _SliceAxis;
            int _SliceIndex;
            float4 _FilledColor;
            float4 _EmptyColor;

            struct v2f
            {
                float4 pos : SV_POSITION;
                float4 color : COLOR;
                float2 uv : TEXCOORD0;
            };

            uint getVoxel(int3 coord)
            {
                int resX = (int)gridResolution.x;
                int resY = (int)gridResolution.y;
                return voxelGrid[coord.x + coord.y * resX + coord.z * resX * resY];
            }

            v2f vert(uint vertexID : SV_VertexID, uint instanceID : SV_InstanceID)
            {
                v2f o;

                int resX = (int)gridResolution.x;
                int resY = (int)gridResolution.y;
                int resZ = (int)gridResolution.z;

                // Determine which 2D grid axes based on slice axis
                int2 gridSize2D;
                int3 coord3D;

                // instanceID is a cell in the 2D slice
                int ax, ay;
                if (_SliceAxis == 0) // X slice
                {
                    gridSize2D = int2(resY, resZ);
                    ax = instanceID % gridSize2D.x;
                    ay = instanceID / gridSize2D.x;
                    coord3D = int3(_SliceIndex, ax, ay);
                }
                else if (_SliceAxis == 1) // Y slice
                {
                    gridSize2D = int2(resX, resZ);
                    ax = instanceID % gridSize2D.x;
                    ay = instanceID / gridSize2D.x;
                    coord3D = int3(ax, _SliceIndex, ay);
                }
                else // Z slice
                {
                    gridSize2D = int2(resX, resY);
                    ax = instanceID % gridSize2D.x;
                    ay = instanceID / gridSize2D.x;
                    coord3D = int3(ax, ay, _SliceIndex);
                }

                // Check if out of bounds
                if (ax >= gridSize2D.x || ay >= gridSize2D.y)
                {
                    o.pos = float4(0,0,0,0);
                    o.color = float4(0,0,0,0);
                    o.uv = float2(0,0);
                    return o;
                }

                // Get voxel value
                uint val = getVoxel(coord3D);
                o.color = (val == 1) ? _FilledColor : _EmptyColor;

                // Build quad in world space
                float2 corners[4] = {
                    float2(0, 0),
                    float2(0, 1),
                    float2(1, 1),
                    float2(1, 0)
                };
                int indices[6] = { 0, 1, 2, 2, 3, 0 };
                float2 corner = corners[indices[vertexID]];

                // Position the quad in world space
                float3 worldPos;
                if (_SliceAxis == 0)
                {
                    worldPos = boundsMin.xyz + float3(_SliceIndex, ax, ay) * voxelSize.xyz
                             + float3(0, corner.x, corner.y) * voxelSize.xyz;
                }
                else if (_SliceAxis == 1)
                {
                    worldPos = boundsMin.xyz + float3(ax, _SliceIndex, ay) * voxelSize.xyz
                             + float3(corner.x, 0, corner.y) * voxelSize.xyz;
                }
                else
                {
                    worldPos = boundsMin.xyz + float3(ax, ay, _SliceIndex) * voxelSize.xyz
                             + float3(corner.x, corner.y, 0) * voxelSize.xyz;
                }

                o.pos = UnityWorldToClipPos(worldPos);
                o.uv = corner;
                return o;
            }

            float4 frag(v2f i) : SV_Target
            {
                // Grid lines
                float2 grid = abs(frac(i.uv) - 0.5);
                float edge = 1.0 - smoothstep(0.4, 0.5, max(grid.x, grid.y));
                return float4(i.color.rgb * (1.0 - edge * 0.3), i.color.a);
            }
            ENDCG
        }
    }
}