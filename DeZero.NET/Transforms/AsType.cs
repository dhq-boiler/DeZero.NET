using DeZero.NET.PIL;

namespace DeZero.NET.Transforms
{
    public class AsType : Transform
    {
        public virtual Dtype DefaultDtype => xp.float32;

        public Dtype Dtype { get; private set; }

        public AsType(Dtype dtype)
        {
            if (dtype is null)
            {
                Dtype = DefaultDtype;
            }
            else
            {
                Dtype = dtype;
            }
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
            return array.astype(Dtype);
        }
    }

    public class ToFloat : AsType
    {
        public override Dtype DefaultDtype => xp.float32;

        public ToFloat() : base(null)
        {
        }

        public ToFloat(Dtype dtype) : base(dtype)
        {
        }
    }

    public class ToInt : AsType
    {
        public override Dtype DefaultDtype => xp.int32;

        public ToInt() : base(null)
        {
        }

        public ToInt(Dtype dtype) : base(dtype)
        {
        }
    }
}
