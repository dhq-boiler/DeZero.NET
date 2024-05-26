using DeZero.NET.PIL;

namespace DeZero.NET.Transforms
{
    public class CenterCrop : Transform
    {
        public (int, int) Size { get; private set; }

        public CenterCrop(int size)
        {
            Size = (size, size);
        }

        public CenterCrop((int, int) size)
        {
            Size = size;
        }

        public override T Call<T>(PythonObject obj)
        {
            return InternalCall<T>(obj);
        }

        public override Image ToImage(Image image)
        {
            var (W, H) = image.size;
            var (OW, OH) = Size;
            var left = (int)((W - OW) / 2);
            var right = W - ((int)((W - OW) / 2) + (H - OH) % 2);
            var up = (int)((H - OH) / 2);
            var bottom = H - ((int)((H - OH) / 2) + (H - OH) % 2);
            return image.crop((left, up, right, bottom));
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
