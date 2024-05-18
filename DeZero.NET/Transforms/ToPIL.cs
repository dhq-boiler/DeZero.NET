using DeZero.NET.PIL;

namespace DeZero.NET.Transforms
{
    public class ToPIL : Transform
    {
        public override Image ToImage(Image image)
        {
            throw new NotSupportedException();
        }


        public override NDarray ToNDarray(PythonObject obj)
        {
            throw new NotSupportedException();
        }

        public override NDarray ToNDarray(NDarray array)
        {
            throw new NotSupportedException();
        }

        public override Image ToImage(NDarray array)
        {
            var data = array.transpose(1, 2, 0);
            return Image.fromarray(data);
        }
    }
}
