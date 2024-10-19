using DeZero.NET.Transforms;
using System.Diagnostics;

namespace DeZero.NET.Datasets
{
    public class Dataset
    {
        public bool Train { get; }
        public Transform Transform { get; }
        public Transform TargetTransform { get; }
        public NDarray Data { get; protected set; }
        public NDarray Label { get; protected set; }

        public Dataset(bool train = true, Transform transform = null, Transform target_transform = null)
        {
            Train = train;
            Transform = transform;
            TargetTransform = target_transform;
            if (Transform is null)
            {
                Transform = new Transform();
            }

            if (TargetTransform is null)
            {
                TargetTransform = new Transform();
            }

            Data = null;
            Label = null;
            Prepare();
        }

        public virtual (NDarray, NDarray) this[int index]
        {
            get
            {
                Debug.Assert(xp.isscalar(index));
                if (Label is null)
                {
                    return (Transform.Call<NDarray>(Data[index]), null);
                }
                else
                {
                    return (Transform.Call<NDarray>(Data[index]), TargetTransform.Call<NDarray>(Label[index]));
                }
            }
        }

        public virtual void Prepare()
        {
        }
    }
}
