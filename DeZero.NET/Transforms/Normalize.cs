using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DeZero.NET.PIL;

namespace DeZero.NET.Transforms
{
    public class Normalize : Transform
    {
        public NDarray Mean { get; set; }
        public NDarray Std { get; set; }

        public Normalize(NDarray mean, NDarray std)
        {
            if (mean is null)
            {
                Mean = new NDarray(0);
            }
            else
            {
                Mean = mean;
            }

            if (std is null)
            {
                Std = new NDarray(1);
            }
            else
            {
                Std = std;
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
            throw new NotSupportedException();
        }

        public override NDarray ToNDarray(NDarray array)
        {
            var mean = Mean;
            var std = Std;

            if (!xp.isscalar(mean))
            {
                var mshape = new Shape(1) * array.ndim;
                var first = Mean.len == 1 ? array.len : Mean.len;
                mshape = new Shape([first, ..mshape.Dimensions.Skip(1).ToArray()]);
                mean = xp.array(Mean, dtype: array.dtype).reshape(mshape);
            }

            if (!xp.isscalar(std))
            {
                var sshape = new Shape(1) * array.ndim;
                var first = Std.len == 1 ? array.len : Std.len;
                sshape = new Shape([first, ..sshape.Dimensions.Skip(1).ToArray()]);
                std = xp.array(Std, dtype: array.dtype).reshape(sshape);
            }

            return (array - mean) / std;
        }
    }
}
