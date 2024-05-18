using DeZero.NET.PIL;

namespace DeZero.NET.Transforms
{
    public class Compose : Transform
    {
        public Transform[] Transforms { get; set; }

        public Compose(params Transform[] transforms)
        {
            this.Transforms = transforms;
        }

        public override Image ToImage(Image image)
        {
            if (!Transforms.Any())
                return image;

            foreach (var t in Transforms)
            {
                image = t.ToImage(image);
            }

            return image;
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
            throw new NotSupportedException();
        }
    }
}
