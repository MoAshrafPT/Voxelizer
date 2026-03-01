Shader "Custom/ParticleRender"
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

            struct Particle
            {
                float3 position;
                float3 velocity;
                float4 color;
                float life;
                float size;
            };

            StructuredBuffer<Particle> particles;

            struct v2f
            {
                float4 pos : SV_POSITION;
                float4 color : COLOR;
                float2 uv : TEXCOORD0;
            };

            v2f vert(uint vertexID : SV_VertexID, uint instanceID : SV_InstanceID)
            {
                v2f o;

                Particle p = particles[instanceID];

                // Quad corners: 2 triangles, 6 vertices per particle
                //   1---2        vertices: 0,1,2  and  2,3,0
                //   | / |
                //   0---3
                float2 corners[4] = {
                    float2(-1, -1),
                    float2(-1,  1),
                    float2( 1,  1),
                    float2( 1, -1)
                };
                int indices[6] = { 0, 1, 2, 2, 3, 0 };
                float2 corner = corners[indices[vertexID]];

                // Billboard — face camera
                float3 worldPos = p.position
                    + unity_CameraToWorld._m00_m10_m20 * corner.x * p.size
                    + unity_CameraToWorld._m01_m11_m21 * corner.y * p.size;

                o.pos = UnityWorldToClipPos(worldPos);
                o.color = p.color;
                o.uv = corner * 0.5 + 0.5;

                // Hide dead particles
                if (p.life <= 0.0)
                    o.pos = float4(0, 0, 0, 0);

                return o;
            }

            float4 frag(v2f i) : SV_Target
            {
                // Soft circle
                float dist = length(i.uv - 0.5) * 2.0;
                float alpha = saturate(1.0 - dist);
                return float4(i.color.rgb, i.color.a * alpha);
            }
            ENDCG
        }
    }
}