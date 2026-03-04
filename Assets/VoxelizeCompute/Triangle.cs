using UnityEngine;

public struct Triangle
{
    public Vector3 v0;
    public Vector3 v1;
    public Vector3 v2;

    public static int Size => sizeof(float) * 9;
}