Shader "Hidden/VoxelComposite"
{
    Properties
    {
        _MainTex ("Scene", 2D) = "white" {}
        _VoxTex  ("Voxels", 2D) = "black" {}
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

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
            };

            sampler2D _MainTex;
            sampler2D _VoxTex;

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
                return o;
            }

            fixed4 frag (v2f i) : SV_Target
            {
                fixed4 scene = tex2D(_MainTex, i.uv);
                fixed4 voxel = tex2D(_VoxTex, i.uv);
                // Alpha-over: where voxel hit (a=1), show voxel; where miss (a=0), show scene
                return lerp(scene, fixed4(voxel.rgb, 1.0), voxel.a);
            }
            ENDCG
        }
    }
}
