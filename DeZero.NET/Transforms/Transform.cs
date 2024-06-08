using DeZero.NET.Core;
using DeZero.NET.PIL;

namespace DeZero.NET.Transforms
{
    public class Transform
    {
        public virtual T Call<T>(IDeZeroObject obj) where T : class, IDeZeroObject
        {
            return obj as T;
        }

        protected T InternalCall<T>(IDeZeroObject obj)
        {
            if (typeof(T) == typeof(Image))
            {
                if (obj is Image image)
                {
                    return (T)(object)ToImage(image);
                }
                else if (obj is NDarray array)
                {
                    return (T)(object)ToImage(array);
                }
            }
            else if (typeof(T) == typeof(NDarray))
            {
                if (obj is Image image)
                {
                    return (T)(object)ToNDarray(image);
                }
                else if (obj is NDarray array)
                {
                    return (T)(object)ToNDarray(array);
                }
            }

            throw new NotSupportedException();
        }

        public virtual Image ToImage(Image image)
        {
            throw new NotSupportedException();
        }

        public virtual Image ToImage(NDarray array)
        {
            throw new NotSupportedException();
        }

        public virtual NDarray ToNDarray(IDeZeroObject obj)
        {
            throw new NotSupportedException();
        }

        public virtual NDarray ToNDarray(NDarray array)
        {
            throw new NotSupportedException();
        }
    }
}
