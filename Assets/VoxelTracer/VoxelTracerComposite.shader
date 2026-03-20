Shader "Hidden/VoxelTracer/Composite"
{
    Properties
    {
        _MainTex   ("Scene",       2D) = "black" {}
        _VoxelTex  ("Voxel Color", 2D) = "black" {}
        _VoxelDepth("Voxel Depth", 2D) = "white" {}
    }

    SubShader
    {
        Cull Off ZWrite Off ZTest Always

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            sampler2D _MainTex;
            sampler2D _VoxelTex;
            sampler2D _VoxelDepth;
            sampler2D _CameraDepthTexture;

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv     : TEXCOORD0;
            };

            struct v2f
            {
                float4 pos : SV_POSITION;
                float2 uv  : TEXCOORD0;
            };

            v2f vert(appdata v)
            {
                v2f o;
                o.pos = UnityObjectToClipPos(v.vertex);
                o.uv  = v.uv;
                return o;
            }

            fixed4 frag(v2f i) : SV_Target
            {
                float4 scene  = tex2D(_MainTex, i.uv);
                float4 voxel  = tex2D(_VoxelTex, i.uv);
                float  vDepth = tex2D(_VoxelDepth, i.uv).r;

                // Scene depth (linear eye-space)
                float rawDepth  = SAMPLE_DEPTH_TEXTURE(_CameraDepthTexture, i.uv);
                float sceneDepth = LinearEyeDepth(rawDepth);

                // If voxel hit (depth < 1e9) and closer than scene geometry
                if (vDepth > 0.0 && vDepth < 1e9 && vDepth < sceneDepth)
                    return voxel;

                return scene;
            }
            ENDCG
        }
    }
}
