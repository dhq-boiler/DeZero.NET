using DeZero.NET.PIL;

namespace DeZero.NET.Transforms
{
    public class Convert : Transform
    {
        public string Mode { get; }

        public Convert(string mode = "RGB")
        {
            Mode = mode;
        }

        public override T Call<T>(PythonObject obj)
        {
            return InternalCall<T>(obj);
        }

        public override Image ToImage(Image image)
        {
            if (Mode == "BGR")
            {
                image = image.convert("RGB");
                var split = image.split();
                Image r = split[0], g = split[1], b = split[2];
                image = Image.merge("RGB", (b, g, r));
                return image;
            }
            else
            {
                return image.convert(Mode);
            }
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
