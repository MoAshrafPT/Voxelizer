Shader "Custom/BoidSim"
{
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

            struct BoidData
            {
                float3 position;
                float3 velocity;
                float3 acceleration;
                int neighborCount;
            };

            StructuredBuffer<BoidData> boids;

            struct v2f
            {
                float4 pos : SV_POSITION;
                float4 color : COLOR;
                float2 uv : TEXCOORD0;
            };

            v2f vert(uint vertexID : SV_VertexID, uint instanceID : SV_InstanceID)
            {
                v2f o;
                BoidData b = boids[instanceID];

                float2 corners[4] = {
                    float2(-1, -1),
                    float2(-1,  1),
                    float2( 1,  1),
                    float2( 1, -1)
                };
                int indices[6] = { 0, 1, 2, 2, 3, 0 };
                float2 corner = corners[indices[vertexID]];

                float size = 0.15;
                float3 worldPos = b.position
                    + unity_CameraToWorld._m00_m10_m20 * corner.x * size
                    + unity_CameraToWorld._m01_m11_m21 * corner.y * size;

                o.pos = UnityWorldToClipPos(worldPos);
                o.uv = corner * 0.5 + 0.5;

                // Color based on neighbor count
                float t = saturate(b.neighborCount / 20.0);
                o.color = float4(t, 0.3, 1.0 - t, 1.0);  // blue=lonely, red=crowded

                return o;
            }

            float4 frag(v2f i) : SV_Target
            {
                float dist = length(i.uv - 0.5) * 2.0;
                float alpha = saturate(1.0 - dist);
                return float4(i.color.rgb, i.color.a * alpha);
            }
            ENDCG
        }
    }
}