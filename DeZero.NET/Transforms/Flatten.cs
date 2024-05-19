using DeZero.NET.PIL;

namespace DeZero.NET.Transforms
{
    public class Flatten : Transform
    {
        public override T Call<T>(PythonObject obj)
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

        public override NDarray ToNDarray(PythonObject obj)
        {
            throw new NotSupportedException();
        }

        public override NDarray ToNDarray(NDarray array)
        {
            return array.flatten();
        }
    }
}
