using DeZero.NET.PIL;

namespace DeZero.NET.Transforms
{
    public class ToArray : Transform
    {
        public Dtype Dtype { get; private set; }

        public ToArray(Dtype dtype)
        {
            if (dtype is null)
            {
                Dtype = xp.float32;
            }
            else
            {
                Dtype = dtype;
            }
        }

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
            if (obj is NDarray arr)
            {
                return arr;
            }
            else if (obj is Image img)
            {
                var ret = xp.array(img);
                ret = ret.transpose(2, 0, 1);
                ret = ret.astype(Dtype);
                return ret;
            }
            else
            {
                throw new ArgumentException();
            }
        }

        public override NDarray ToNDarray(NDarray array)
        {
            throw new NotSupportedException();
        }
    }
}
