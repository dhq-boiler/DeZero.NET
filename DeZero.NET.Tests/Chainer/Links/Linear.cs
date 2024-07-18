using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Cupy;
using Python.Runtime;

namespace DeZero.NET.Tests.Chainer.Links
{
    internal class Linear : PythonObject
    {
        public Linear(int inSize, int outSize)
        {
            dynamic chainerLinks = Py.Import("chainer.links");
            this.self = chainerLinks.Linear(inSize, outSize);
        }

        public NDarray W
        {
            get
            {
                dynamic __self__ = self;
                dynamic py = __self__.W;
                return ToCsharp<NDarray>(py);
            }
        }

        public NDarray b
        {
            get
            {
                dynamic __self__ = self;
                dynamic py = __self__.b;
                return ToCsharp<NDarray>(py);
            }
        }

        public NDarray __call__(NDarray x)
        {
            if (Gpu.Available && Gpu.Use)
            {
                var __self__ = self;
                var pyargs = ToTuple(new object[]
                {
                    cpExtensions.asnumpy(x.CupyNDarray).PyObject,
                });
                dynamic py = __self__.InvokeMethod("__call__", pyargs);
                return ToCsharp<NDarray>(py);
            }
            else
            {
                var __self__ = self;
                var pyargs = ToTuple(new object[]
                {
                    x.NumpyNDarray.PyObject,
                });
                dynamic py = __self__.InvokeMethod("__call__", pyargs);
                return ToCsharp<NDarray>(py);
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

        internal static T ToCsharp<T>(dynamic pyobj)
        {
            switch (typeof(T).Name)
            {
                // types from 'ToCsharpConversions'
                case "Dtype": return (T)(object)new Dtype(pyobj);
                case "NDarray": return (T)(object)new NDarray(pyobj.array);
                case "NDarray`1":
                    switch (typeof(T).GenericTypeArguments[0].Name)
                    {
                        case "Byte": return (T)(object)new NDarray<byte>(pyobj.array);
                        case "Short": return (T)(object)new NDarray<short>(pyobj.array);
                        case "Boolean": return (T)(object)new NDarray<bool>(pyobj.array);
                        case "Int32": return (T)(object)new NDarray<int>(pyobj.array);
                        case "Int64": return (T)(object)new NDarray<long>(pyobj.array);
                        case "Single": return (T)(object)new NDarray<float>(pyobj.array);
                        case "Double": return (T)(object)new NDarray<double>(pyobj.array);
                        default:
                            throw new NotImplementedException(
                                $"Type NDarray<{typeof(T).GenericTypeArguments[0].Name}> missing. Add it to 'ToCsharpConversions'");
                    }

                    break;
                case "NDarray[]":
                    var po = pyobj as PyObject;
                    var len = po.Length();
                    var rv = new NDarray[len];
                    for (var i = 0; i < len; i++)
                        rv[i] = ToCsharp<NDarray>(po[i]);
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
                dict[new PyString(pair.Key)] = pair.Value.self;
            return dict;
        }
    }
}
