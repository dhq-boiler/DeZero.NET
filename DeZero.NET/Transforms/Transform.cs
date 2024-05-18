using DeZero.NET.PIL;

namespace DeZero.NET.Transforms
{
    public abstract class Transform
    {
        public abstract Image ToImage(Image image);
        public abstract Image ToImage(NDarray array);
        public abstract NDarray ToNDarray(PythonObject obj);

        public abstract NDarray ToNDarray(NDarray array);
    }
}
