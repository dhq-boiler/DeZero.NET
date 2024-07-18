using Cupy;
using Python.Runtime;

namespace DeZero.NET.Tests.Chainer
{
    internal class BatchNormalization : PythonObject
    {
        public Func<NDarray, NDarray> F => x => __call__(x);

        public BatchNormalization(int C)
        {
            dynamic chainerLinks = Py.Import("chainer.links");
            this.self = chainerLinks.BatchNormalization(C);
        }

        private NDarray __call__(NDarray x)
        {
            if (Gpu.Available && Gpu.Use)
            {
                var __self__ = self;
                var pyargs = ToTuple(new object[]
                {
                    x.ToNumpyNDarray.PyObject,
                });
                dynamic py = __self__.InvokeMethod("__call__", pyargs);
                return new NDarray(ToCsharp<Numpy.NDarray>(py).data);
            }
            else
            {
                var __self__ = self;
                var pyargs = ToTuple(new object[]
                {
                    x.NumpyNDarray.PyObject,
                });
                dynamic py = __self__.InvokeMethod("__call__", pyargs);
                return new NDarray(ToCsharp<Numpy.NDarray>(py).data);
            }
        }

        public NDarray avg_mean
        {
            get
            {
                dynamic __self__ = self;
                dynamic py = __self__.avg_mean;
                return new NDarray(ToCsharp<Numpy.NDarray>(py).data);
            }
        }
        public NDarray avg_var
        {
            get
            {
                dynamic __self__ = self;
                dynamic py = __self__.avg_var;
                return new NDarray(ToCsharp<Numpy.NDarray>(py).data);
            }
        }

        private static PyTuple ToTuple(Array input)
        {
            var array = new PyObject[input.Length];
            for (var i = 0; i < input.Length; i++) array[i] = ToPython(input.GetValue(i));
            return new PyTuple(array);
        }

        //auto-generated
        internal static PyObject ToPython(object obj)
        {
            if (obj == null) return Runtime.None;
            switch (obj)
            {
                // basic types
                case int o: return new PyInt(o);
                case long o: return new PyInt(o);
                case float o: return new PyFloat(o);
                case double o: return new PyFloat(o);
                case string o: return new PyString(o);
                case bool o: return o.ToPython();
                case Cupy.NDarray o: return o.data.ToPython();
                case Numpy.NDarray o: return o.data.ToPython();
                case PyObject o: return o;
                // sequence types
                case Array o: return ToTuple(o);
                // special types from 'ToPythonConversions'
                case Axis o: return o.Axes == null ? null : ToTuple(o.Axes);
                case Shape o: return ToTuple(o.Dimensions);
                case Slice o: return o.ToPython();
                case PythonObject o: return o.PyObject;
                case Dictionary<string, NDarray> o: return ToDict(o);
                default:
                    throw new NotImplementedException(
                        $"Type is not yet supported: {obj.GetType().Name}. Add it to 'ToPythonConversions'");
            }
        }

        //auto-generated
        internal static T ToCsharp<T>(dynamic pyobj)
        {
            switch (typeof(T).Name)
            {
                // types from 'ToCsharpConversions'
                case "Dtype": return (T)(object)new Numpy.Dtype(pyobj);
                case "NDarray": return (T)(object)new Numpy.NDarray(pyobj);
                case "NDarray`1":
                    switch (typeof(T).GenericTypeArguments[0].Name)
                    {
                        case "Byte": return (T)(object)new Numpy.NDarray<byte>(pyobj);
                        case "Short": return (T)(object)new Numpy.NDarray<short>(pyobj);
                        case "Boolean": return (T)(object)new Numpy.NDarray<bool>(pyobj);
                        case "Int32": return (T)(object)new Numpy.NDarray<int>(pyobj);
                        case "Int64": return (T)(object)new Numpy.NDarray<long>(pyobj);
                        case "Single": return (T)(object)new Numpy.NDarray<float>(pyobj);
                        case "Double": return (T)(object)new Numpy.NDarray<double>(pyobj);
                        default:
                            throw new NotImplementedException(
                                $"Type NDarray<{typeof(T).GenericTypeArguments[0].Name}> missing. Add it to 'ToCsharpConversions'");
                    }

                    break;
                case "NDarray[]":
                    var po = pyobj as PyObject;
                    var len = po.Length();
                    var rv = new Numpy.NDarray[len];
                    for (var i = 0; i < len; i++)
                        rv[i] = ToCsharp<Numpy.NDarray>(po[i]);
                    return (T)(object)rv;
                case "Matrix": return (T)(object)new Matrix(pyobj);
                default:
                    var pyClass = $"{pyobj.__class__}";
                    if (pyClass == "<class 'str'>") return (T)(object)pyobj.ToString();
                    if (pyClass.StartsWith("<class 'Cupy")) return (pyobj.item() as PyObject).As<T>();
                    if (pyClass.StartsWith("<class 'cupy")) return (pyobj.item() as PyObject).As<T>();
                    try
                    {
                        return pyobj.As<T>();
                    }
                    catch (Exception e)
                    {
                        throw new NotImplementedException(
                            $"conversion from {pyobj.__class__} to {typeof(T).Name} not implemented", e);
                        return default;
                    }
            }
        }

        //auto-generated: SpecialConversions
        private static PyDict ToDict(Dictionary<string, NDarray> d)
        {
            var dict = new PyDict();
            foreach (var pair in d)
            {
                using var key = new PyString(pair.Key);
                dict[key] = pair.Value.self;
            }
            return dict;
        }
    }
}
