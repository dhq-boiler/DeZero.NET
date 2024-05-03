using DeZero.NET;
using Python.Runtime;

namespace Cupy
{

    public static partial class cpExtensions
    {
        public static Numpy.NDarray asnumpy(this Cupy.NDarray array)
        {
            var self = Py.Import("cupy");
            var args = ToTuple(new object[]
            {
                array.PyObject
            });
            var py = self.InvokeMethod("asnumpy", args);
            args.Dispose();
            return ToCsharpNp<Numpy.NDarray>(py);
        }

        public static Numpy.NDarray get(this Cupy.NDarray array)
        {
            var self = Py.Import("cupy");
            var args = ToTuple(new object[]
            {
                array.PyObject
            });
            var py = self.InvokeMethod("get", args);
            args.Dispose();
            return ToCsharpNp<Numpy.NDarray>(py);
        }

        public static NDarray asarray(this Numpy.NDarray a, Dtype dtype = null)
        {
            var __self__ = Py.Import("cupy");
            var pyargs = ToTuple(new object[]
            {
                a.PyObject
            });
            var kwargs = new PyDict();
            if (dtype != null) kwargs["dtype"] = ToPython(dtype);
            dynamic py = __self__.InvokeMethod("asarray", pyargs, kwargs);
            return ToCsharpCp<NDarray>(py);
        }

        private static PyTuple ToTuple(Array input)
        {
            var array = new PyObject[input.Length];
            for (var i = 0; i < input.Length; i++) array[i] = ToPython(input.GetValue(i));
            return new PyTuple(array);
        }

        private static PyObject ToPython(object obj)
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
                case PyObject o: return o;
                // sequence types
                case Array o: return ToTuple(o);
                // special types from 'ToPythonConversions'
                case Cupy.Models.Axis o: return o.Axes == null ? null : ToTuple(o.Axes);
                case Numpy.Models.Axis o: return o.Axes == null ? null : ToTuple(o.Axes);
                case Cupy.Models.Shape o: return ToTuple(o.Dimensions);
                case Numpy.Models.Shape o: return ToTuple(o.Dimensions);
                case Cupy.Models.Slice o: return o.ToPython();
                case Numpy.Models.Slice o: return o.ToPython();
                case Cupy.PythonObject o: return o.PyObject;
                case Numpy.PythonObject o: return o.PyObject;
                case Dictionary<string, NDarray> o: return ToDict(o);
                default:
                    throw new NotImplementedException(
                        $"Type is not yet supported: {obj.GetType().Name}. Add it to 'ToPythonConversions'");
            }
        }
        private static PyDict ToDict(Dictionary<string, NDarray> d)
        {
            var dict = new PyDict();
            foreach (var pair in d)
                dict[new PyString(pair.Key)] = pair.Value.self;
            return dict;
        }

        private static T ToCsharpNp<T>(dynamic pyobj)
        {
            switch (typeof(T).Name)
            {
                // types from 'ToCsharpConversions'
                case "Dtype": return (T)(object)new Dtype(pyobj);
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
                        rv[i] = ToCsharpNp<Numpy.NDarray>(po[i]);
                    return (T)(object)rv;
                case "Matrix": return (T)(object)new Matrix(pyobj);
                default:
                    var pyClass = $"{pyobj.__class__}";
                    if (pyClass == "<class 'str'>") return (T)(object)pyobj.ToString();
                    if (pyClass.StartsWith("<class 'Pillow")) return (pyobj.item() as PyObject).As<T>();
                    if (pyClass.StartsWith("<class 'PIL")) return (pyobj.item() as PyObject).As<T>();
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

        private static T ToCsharpCp<T>(dynamic pyobj)
        {
            switch (typeof(T).Name)
            {
                // types from 'ToCsharpConversions'
                case "Dtype": return (T)(object)new Dtype(pyobj);
                case "NDarray": return (T)(object)new Cupy.NDarray(pyobj);
                case "NDarray`1":
                    switch (typeof(T).GenericTypeArguments[0].Name)
                    {
                        case "Byte": return (T)(object)new Cupy.NDarray<byte>(pyobj);
                        case "Short": return (T)(object)new Cupy.NDarray<short>(pyobj);
                        case "Boolean": return (T)(object)new Cupy.NDarray<bool>(pyobj);
                        case "Int32": return (T)(object)new Cupy.NDarray<int>(pyobj);
                        case "Int64": return (T)(object)new Cupy.NDarray<long>(pyobj);
                        case "Single": return (T)(object)new Cupy.NDarray<float>(pyobj);
                        case "Double": return (T)(object)new Cupy.NDarray<double>(pyobj);
                        default:
                            throw new NotImplementedException(
                                $"Type NDarray<{typeof(T).GenericTypeArguments[0].Name}> missing. Add it to 'ToCsharpConversions'");
                    }

                    break;
                case "NDarray[]":
                    var po = pyobj as PyObject;
                    var len = po.Length();
                    var rv = new Cupy.NDarray[len];
                    for (var i = 0; i < len; i++)
                        rv[i] = ToCsharpNp<Cupy.NDarray>(po[i]);
                    return (T)(object)rv;
                case "Matrix": return (T)(object)new Matrix(pyobj);
                default:
                    var pyClass = $"{pyobj.__class__}";
                    if (pyClass == "<class 'str'>") return (T)(object)pyobj.ToString();
                    if (pyClass.StartsWith("<class 'Pillow")) return (pyobj.item() as PyObject).As<T>();
                    if (pyClass.StartsWith("<class 'PIL")) return (pyobj.item() as PyObject).As<T>();
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

        public static Variable as_variable(object obj)
        {
            if (obj is Variable v)
            {
                return v;
            }
            else
            {
                return new Variable(new DeZero.NET.NDarray((PyObject)obj));
            }
        }
    }
}
