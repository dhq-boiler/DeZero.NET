using Numpy;
using Python.Runtime;

namespace DeZero.NET
{
    public static class Extensions
    {
        public static Variable ToVariable(this NDarray array, bool autoSwitch = false, bool useCupy = true)
        {
            var ret = new Variable(array);
            
            if (autoSwitch)
            {
                switch (useCupy)
                {
                    case true when ret.Data.CupyNDarray is null:
                    case false when ret.Data.NumpyNDarray is null:
                        ret.Data.Switch(deleteOriginal: false);
                        break;
                }
            }

            return ret;
        }

        public static Shape ToShape(this NDarray array, bool useCupy = true)
        {
            switch (useCupy)
            {
                case true:
                    try
                    {
                        array.Push(ArrayMode.cp);
                        return new Shape(array.CupyNDarray.shape);
                    }
                    finally
                    {
                        array.Pop();
                    }
                case false:
                    try
                    {
                        array.Push(ArrayMode.np);
                        return new Shape(array.NumpyNDarray.shape);
                    }
                    finally
                    {
                        array.Pop();
                    }
            }
        }

        public static Dtype dtype(Dtype dtype, bool? align = false, bool? copy = false,
            params (string, string)[] metadata)
        {
            PyObject self = Py.Import("cupy");
            PyTuple tuple = NDarray.ToTuple(new Object[] { dtype.CupyDtype.PyObject }.ToArray());
            PyDict kw = new PyDict();
            if (align.HasValue)
                kw[nameof(align)] = NDarray.ToPython((object)align);
            if (copy.HasValue)
                kw[nameof(copy)] = NDarray.ToPython((object)copy);
            if (metadata.Any())
                kw[nameof(metadata)] = NDarray.ToPython((object)metadata);
            dynamic ret = self.InvokeMethod(nameof(dtype), tuple, kw);
            var cpDtype = NDarray.ToCsharp<Cupy.Dtype>(ret);

            self = Py.Import("np");
            tuple = NDarray.ToTuple(new Object[] { dtype.NumpyDtype.PyObject }.ToArray());
            kw = new PyDict();
            if (align.HasValue)
                kw[nameof(align)] = NDarray.ToPython((object)align);
            if (copy.HasValue)
                kw[nameof(copy)] = NDarray.ToPython((object)copy);
            if (metadata.Any())
                kw[nameof(metadata)] = NDarray.ToPython((object)metadata);
            ret = self.InvokeMethod(nameof(dtype), tuple, kw);
            var npDtype = NDarray.ToCsharp<Numpy.Dtype>(ret);

            return new Dtype(npDtype, cpDtype);
        }

        public static Dtype dtype(string dtype, bool? align = false, bool? copy = false,
            params (string, string)[] metadata)
        {
            PyObject self = Py.Import("cupy");
            PyTuple tuple = NDarray.ToTuple(new Object[] { NDarray.ToPython(dtype) }.ToArray());
            PyDict kw = new PyDict();
            if (align.HasValue)
                kw[nameof(align)] = NDarray.ToPython((object)align);
            if (copy.HasValue)
                kw[nameof(copy)] = NDarray.ToPython((object)copy);
            if (metadata.Any())
                kw[nameof(metadata)] = NDarray.ToPython((object)metadata);
            dynamic ret = self.InvokeMethod(nameof(dtype), tuple, kw);
            var cpDtype = NDarray.ToCsharp<Cupy.Dtype>(ret);

            self = Py.Import("numpy");
            tuple = NDarray.ToTuple(new Object[] { NDarray.ToPython(dtype) }.ToArray());
            kw = new PyDict();
            if (align.HasValue)
                kw[nameof(align)] = NDarray.ToPython((object)align);
            if (copy.HasValue)
                kw[nameof(copy)] = NDarray.ToPython((object)copy);
            if (metadata.Any())
                kw[nameof(metadata)] = NDarray.ToPython((object)metadata);
            ret = self.InvokeMethod(nameof(dtype), tuple, kw);
            var npDtype = NDarray.ToCsharp<Numpy.Dtype>(ret);

            return new Dtype(npDtype, cpDtype);
        }
    }
}
