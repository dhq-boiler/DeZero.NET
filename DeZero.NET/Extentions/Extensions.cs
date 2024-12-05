using Cupy;
using DocumentFormat.OpenXml.Bibliography;
using Python.Runtime;
using System.Diagnostics;

namespace DeZero.NET.Extensions
{
    public static class Extensions
    {
        [DebuggerStepThrough]
        public static Variable ToVariable(this NDarray array, bool autoSwitch = false, bool useCupy = true)
        {
            var ret = new Variable(array);

            if (autoSwitch)
            {
                switch (useCupy)
                {
                    case true when ret.Data.Value.CupyNDarray is null:
                    case false when ret.Data.Value.NumpyNDarray is null:
                        ret.Data.Value.Switch(deleteOriginal: false);
                        break;
                }
            }

            return ret;
        }

        [DebuggerStepThrough]
        public static Variable ToVariable(this NDarray array, Variable source, Function f = null, bool autoSwitch = false, bool useCupy = true)
        {
            var ret = new Variable(array)
            {
                Creator = source.Creator,
                CreatorList = f is not null ? source.CreatorList.Union([f]).ToList() : source.CreatorList,
                Generation = source.Generation,
                Grad =
                {
                    Value = source.Grad.Value
                },
                Origins = [source.Creator],
            };

            if (autoSwitch)
            {
                switch (useCupy)
                {
                    case true when ret.Data.Value.CupyNDarray is null:
                    case false when ret.Data.Value.NumpyNDarray is null:
                        ret.Data.Value.Switch(deleteOriginal: false);
                        break;
                }
            }

            return ret;
        }

        public static Variable Relay(this Variable v, Function f = null, params Variable[] origins)
        {
            return new Variable(v.Data.Value)
            {
                Creator = v.Creator,
                CreatorList = f is not null ? v.CreatorList.Union([f]).ToList() : v.CreatorList,
                Generation = v.Generation,
                Grad =
                {
                    Value = v.Grad.Value
                },
                Origins = origins is not null && v.Origins is not null ? origins.Where(x => x is not null).Select(x => x.Creator).Union(v.Origins).ToArray() :
                    origins is not null ? origins.Where(x => x is not null).Select(x => x.Creator).ToArray() :
                            v.Origins is not null ? v.Origins : null
            };
        }

        public static Variable Relay(this NDarray arr, Function f = null, params Variable[] origins)
        {
            return new Variable(arr)
            {
                Creator = f,
                CreatorList = f is null ? null : [f],
                Generation = 0,
                Grad =
                {
                    Value = null
                },
                Origins = origins.Where(x => x is not null).Select(x => x.Creator).ToArray()
            };
        }

        [DebuggerStepThrough]
        public static Variable ToVariable(this NDarray array, Function function, bool autoSwitch = false, bool useCupy = true)
        {
            var ret = new Variable(array) { Creator = function };

            if (autoSwitch)
            {
                switch (useCupy)
                {
                    case true when ret.Data.Value.CupyNDarray is null:
                    case false when ret.Data.Value.NumpyNDarray is null:
                        ret.Data.Value.Switch(deleteOriginal: false);
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
                    if (cp.isscalar(array.ToCupyNDarray))
                    {
                        return new Shape(cp.array(array.ToCupyNDarray).shape);
                    }
                    else
                    {
                        return new Shape(array.ToCupyNDarray.shape);
                    }
                case false:
                    return new Shape(array.ToNumpyNDarray.shape);
            }
        }

        public static Dtype dtype(Dtype dtype, bool? align = false, bool? copy = false,
            params (string, string)[] metadata)
        {
            using PyObject self = Py.Import("cupy");
            using PyTuple tuple = NDarray.ToTuple(new Object[] { dtype.CupyDtype.PyObject }.ToArray());
            using PyDict kw = new PyDict();
            if (align.HasValue)
                kw[nameof(align)] = NDarray.ToPython((object)align);
            if (copy.HasValue)
                kw[nameof(copy)] = NDarray.ToPython((object)copy);
            if (metadata.Any())
                kw[nameof(metadata)] = NDarray.ToPython((object)metadata);
            dynamic ret = self.InvokeMethod(nameof(dtype), tuple, kw);
            var cpDtype = NDarray.ToCsharp<Cupy.Dtype>(ret);

            using PyObject self2 = Py.Import("np");
            using PyTuple tuple2 = NDarray.ToTuple(new Object[] { dtype.NumpyDtype.PyObject }.ToArray());
            using PyDict kw2 = new PyDict();
            if (align.HasValue)
                kw2[nameof(align)] = NDarray.ToPython((object)align);
            if (copy.HasValue)
                kw2[nameof(copy)] = NDarray.ToPython((object)copy);
            if (metadata.Any())
                kw2[nameof(metadata)] = NDarray.ToPython((object)metadata);
            ret = self2.InvokeMethod(nameof(dtype), tuple2, kw2);
            var npDtype = NDarray.ToCsharp<Numpy.Dtype>(ret);

            return new Dtype(npDtype, cpDtype);
        }

        public static Dtype dtype(string dtype, bool? align = false, bool? copy = false,
            params (string, string)[] metadata)
        {
            using PyObject self = Py.Import("numpy");
            using PyTuple tuple = NDarray.ToTuple(new Object[] { Dtype.ToPython(dtype) }.ToArray());
            using PyDict kw = new PyDict();
            if (align.HasValue)
                kw[nameof(align)] = NDarray.ToPython((object)align);
            if (copy.HasValue)
                kw[nameof(copy)] = NDarray.ToPython((object)copy);
            if (metadata.Any())
                kw[nameof(metadata)] = NDarray.ToPython((object)metadata);
            dynamic ret = self.InvokeMethod(nameof(dtype), tuple, kw);
            var cpDtype = NDarray.ToCsharp<Cupy.Dtype>(ret);

            using PyTuple tuple2 = NDarray.ToTuple(new Object[] { Dtype.ToPython(dtype) }.ToArray());
            using PyDict kw2 = new PyDict();
            if (align.HasValue)
                kw2[nameof(align)] = NDarray.ToPython((object)align);
            if (copy.HasValue)
                kw2[nameof(copy)] = NDarray.ToPython((object)copy);
            if (metadata.Any())
                kw2[nameof(metadata)] = NDarray.ToPython((object)metadata);
            ret = self.InvokeMethod(nameof(dtype), tuple2, kw2);
            var npDtype = NDarray.ToCsharp<Numpy.Dtype>(ret);

            return new Dtype(npDtype, cpDtype);
        }

        public static bool isscalar(this NDarray array)
        {
            return cp.isscalar(array.ToCupyNDarray);
        }

        public static bool isarray(this Cupy.NDarray array)
        {
            return !(array.PyObject is PyInt
                   || array.PyObject is PyFloat);
        }

        public static bool isarray(this Numpy.NDarray array)
        {
            return !(array.PyObject is PyInt
                     || array.PyObject is PyFloat);
        }
    }
}
