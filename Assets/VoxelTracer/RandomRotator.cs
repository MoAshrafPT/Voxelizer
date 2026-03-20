using UnityEngine;

public class RandomRotator : MonoBehaviour
{
    [Tooltip("Max rotation speed in degrees per second on each axis.")]
    public float maxSpeed = 90f;

    Vector3 _rotSpeed;

    void OnEnable()
    {
        _rotSpeed = new Vector3(
            Random.Range(-maxSpeed, maxSpeed),
            Random.Range(-maxSpeed, maxSpeed),
            Random.Range(-maxSpeed, maxSpeed)
        );
    }

    void Update()
    {
        transform.Rotate(_rotSpeed * Time.deltaTime, Space.Self);
    }
}
