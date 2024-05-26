using Cupy;
using Numpy;
using Python.Runtime;


namespace DeZero.NET
{
    public static partial class xp
    {
        public static NDarray frombuffer(byte[] buffer, Dtype dtype = null, int count = -1, int offset = 0)
        {
            if ((Gpu.Available && Gpu.Use))
            {
                //return new NDarray(cp.frombuffer(buffer, dtype?.CupyDtype, count, offset));
                var __self__ = Py.Import("numpy");
                var pyargs = ToTuple(new object[]
                {
                    buffer.ToPython()

                });
                var kwargs = new PyDict();
                if (dtype != null) kwargs["dtype"] = dtype.CupyDtype.PyObject;
                kwargs["count"] = ToPython(count);
                kwargs["offset"] = ToPython(offset);
                dynamic py = __self__.InvokeMethod("frombuffer", pyargs, kwargs);
                return new DeZero.NET.NDarray(cpExtensions.asarray(ToCsharpNumpy<Numpy.NDarray>(py)));
            }
            else
            {
                //throw new NotSupportedException();
                var __self__ = Py.Import("numpy");
                var pyargs = ToTuple(new object[]
                {
                    buffer.ToPython()

                });
                var kwargs = new PyDict();
                if (dtype != null) kwargs["dtype"] = ToPython(dtype.NumpyDtype);
                kwargs["count"] = ToPython(count);
                kwargs["offset"] = ToPython(offset);
                dynamic py = __self__.InvokeMethod("frombuffer", pyargs, kwargs);
                return new DeZero.NET.NDarray(ToCsharp<Numpy.NDarray>(py));
            }
        }
    }
}
