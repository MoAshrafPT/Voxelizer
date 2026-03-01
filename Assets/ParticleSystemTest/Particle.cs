using UnityEngine;

public struct Particle
{
    public Vector3 position;
    public Vector3 velocity;
    // public Vector3 density;
    // public Vector3 temperature;
    public Color color;
    public float life;
    public float size;

    public static int Size => sizeof(float) * 12;
}