using DeZero.NET.Core;
using DeZero.NET.PIL;

namespace DeZero.NET.Transforms
{
    public class Flatten : Transform
    {
        public override T Call<T>(IDeZeroObject obj)
        {
            return InternalCall<T>(obj);
        }

        public override Image ToImage(Image image)
        {
            throw new NotSupportedException();
        }

        public override Image ToImage(NDarray array)
        {
            throw new NotSupportedException();
        }

        public override NDarray ToNDarray(IDeZeroObject obj)
        {
            throw new NotSupportedException();
        }

        public override NDarray ToNDarray(NDarray array)
        {
            return array.flatten();
        }
    }
}
