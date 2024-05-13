using Python.Runtime;

namespace DeZero.NET.Tests.Chainer
{
    internal static class CF
    {
        private static PyObject Instance { get; } = Py.Import("chainer.functions");

        public static NDarray fixed_batch_normalization(NDarray x, NDarray gamma, NDarray beta, NDarray mean,
            NDarray var)
        {
            if (Gpu.Available && Gpu.Use)
            {
                var __self__ = Instance;
                var pyargs = ToTuple(new object[]
                {
                    x.CupyNDarray.PyObject,
                    gamma.CupyNDarray.PyObject,
                    beta.CupyNDarray.PyObject,
                    mean.CupyNDarray.PyObject,
                    var.CupyNDarray.PyObject,
                });
                dynamic py = __self__.InvokeMethod("fixed_batch_normalization", pyargs);
                return new NDarray(ToCsharp<NDarray>(py).data);
            }
            else
            {
                var __self__ = Instance;
                var pyargs = ToTuple(new object[]
                {
                    x.NumpyNDarray.PyObject,
                    gamma.NumpyNDarray.PyObject,
                    beta.NumpyNDarray.PyObject,
                    mean.NumpyNDarray.PyObject,
                    var.NumpyNDarray.PyObject,
                });
                dynamic py = __self__.InvokeMethod("fixed_batch_normalization", pyargs);
                return new NDarray(ToCsharp<NDarray>(py).data);
            }
        }

        public static NDarray batch_normalization(NDarray x, NDarray gamma, NDarray beta, double eps = 2e-05, NDarray running_mean = null,
            NDarray running_var = null, double decay = 0.9, Axis axis = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                var __self__ = Instance;
                var pyargs = ToTuple(new object[]
                {
                    x.CupyNDarray.PyObject,
                    gamma.CupyNDarray.PyObject,
                    beta.CupyNDarray.PyObject,
                });
                var kwargs = new PyDict();
                if (eps != null) kwargs["eps"] = ToPython(eps);
                if (running_mean is not null) kwargs["running_mean"] = ToPython(running_mean.CupyNDarray.PyObject);
                if (running_var is not null) kwargs["running_var"] = ToPython(running_var.CupyNDarray.PyObject);
                kwargs["decay"] = ToPython(decay);
                if (axis != null) kwargs["axis"] = ToPython(axis.CupyAxis);
                dynamic py = __self__.InvokeMethod("batch_normalization", pyargs, kwargs);
                return new NDarray(ToCsharp<NDarray>(py).data);
            }
            else
            {
                var __self__ = Instance;
                var pyargs = ToTuple(new object[]
                {
                    x.NumpyNDarray.PyObject,
                    gamma.NumpyNDarray.PyObject,
                    beta.NumpyNDarray.PyObject,
                });
                var kwargs = new PyDict();
                if (eps != null) kwargs["eps"] = ToPython(eps);
                if (running_mean is not null) kwargs["running_mean"] = ToPython(running_mean.NumpyNDarray.PyObject);
                if (running_var is not null) kwargs["running_var"] = ToPython(running_var.NumpyNDarray.PyObject);
                kwargs["decay"] = ToPython(decay);
                if (axis != null) kwargs["axis"] = ToPython(axis.NumpyAxis);
                dynamic py = __self__.InvokeMethod("batch_normalization", pyargs, kwargs);
                return new NDarray(ToCsharp<NDarray>(py).data);
            }
        }

        public static NDarray convolution_2d(NDarray x, NDarray W, NDarray b = null, (int, int)? stride = null,
            (int, int)? pad = null, (int, int)? outsize = null)
        {
            if (!stride.HasValue)
            {
                stride = (1, 1);
            }

            if (!pad.HasValue)
            {
                pad = (0, 0);
            }

            if (Gpu.Available && Gpu.Use)
            {
                var __self__ = Instance;
                var pyargs = ToTuple(new object[]
                {
                    x.CupyNDarray.PyObject,
                    W.CupyNDarray.PyObject,
                    b?.CupyNDarray?.PyObject,
                }.Where(x => x is not null)
                    .ToArray());
                var kwargs = new PyDict();
                if (stride is not null) kwargs["stride"] = ToPython(stride);
                if (pad is not null) kwargs["pad"] = ToPython(pad);
                if (outsize is not null) kwargs["outsize"] = ToPython(outsize);
                dynamic py = __self__.InvokeMethod("convolution_2d", pyargs, kwargs);
                return new NDarray(ToCsharp<NDarray>(py).data);
            }
            else
            {
                var __self__ = Instance;
                var pyargs = ToTuple(new object[]
                    {
                        x.NumpyNDarray.PyObject,
                        W.NumpyNDarray.PyObject,
                        b?.NumpyNDarray?.PyObject,
                    }.Where(x => x is not null)
                    .ToArray());
                var kwargs = new PyDict();
                if (stride is not null) kwargs["stride"] = ToPython(stride);
                if (pad is not null) kwargs["pad"] = ToPython(pad);
                if (outsize is not null) kwargs["outsize"] = ToPython(outsize);
                dynamic py = __self__.InvokeMethod("convolution_2d", pyargs, kwargs);
                return new NDarray(ToCsharp<NDarray>(py).data);
            }
        }

        public static NDarray convolution_2d(NDarray x, NDarray W, NDarray b = null, int stride = 1,
            int pad = 0, (int, int)? outsize = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                var __self__ = Instance;
                var pyargs = ToTuple(new object[]
                    {
                        x.CupyNDarray.PyObject,
                        W.CupyNDarray.PyObject,
                        b?.CupyNDarray?.PyObject,
                    }.Where(x => x is not null)
                    .ToArray());
                var kwargs = new PyDict();
                kwargs["stride"] = ToPython(stride);
                kwargs["pad"] = ToPython(pad);
                if (outsize is not null) kwargs["outsize"] = ToPython(outsize);
                dynamic py = __self__.InvokeMethod("convolution_2d", pyargs, kwargs);
                return new NDarray(ToCsharp<NDarray>(py).data);
            }
            else
            {
                var __self__ = Instance;
                var pyargs = ToTuple(new object[]
                    {
                        x.NumpyNDarray.PyObject,
                        W.NumpyNDarray.PyObject,
                        b?.NumpyNDarray?.PyObject,
                    }.Where(x => x is not null)
                    .ToArray());
                var kwargs = new PyDict();
                kwargs["stride"] = ToPython(stride);
                kwargs["pad"] = ToPython(pad);
                if (outsize is not null) kwargs["outsize"] = ToPython(outsize);
                dynamic py = __self__.InvokeMethod("convolution_2d", pyargs, kwargs);
                return new NDarray(ToCsharp<NDarray>(py).data);
            }
        }
        
        public static NDarray deconvolution_2d(NDarray x, NDarray W, NDarray? b, (int s_y, int s_x)? stride = null, (int h_p, int w_p)? pad = null)
        {
            if (!stride.HasValue)
            {
                stride = (1, 1);
            }

            if (!pad.HasValue)
            {
                pad = (0, 0);
            }

            if (Gpu.Available && Gpu.Use)
            {
                var __self__ = Instance;
                var pyargs = ToTuple(new object[]
                    {
                        x.CupyNDarray.PyObject,
                        W.CupyNDarray.PyObject,
                        b?.CupyNDarray?.PyObject,
                    }.Where(x => x is not null)
                    .ToArray());
                var kwargs = new PyDict();
                kwargs["stride"] = ToPython(stride);
                kwargs["pad"] = ToPython(pad);
                dynamic py = __self__.InvokeMethod("deconvolution_2d", pyargs, kwargs);
                return new NDarray(ToCsharp<NDarray>(py).data);
            }
            else
            {
                var __self__ = Instance;
                var pyargs = ToTuple(new object[]
                    {
                        x.NumpyNDarray.PyObject,
                        W.NumpyNDarray.PyObject,
                        b?.NumpyNDarray?.PyObject,
                    }.Where(x => x is not null)
                    .ToArray());
                var kwargs = new PyDict();
                kwargs["stride"] = ToPython(stride);
                kwargs["pad"] = ToPython(pad);
                dynamic py = __self__.InvokeMethod("deconvolution_2d", pyargs, kwargs);
                return new NDarray(ToCsharp<NDarray>(py).data);
            }
        }

        public static NDarray deconvolution_2d(NDarray x, NDarray W, NDarray? b, int stride, int pad)
        {
            if (Gpu.Available && Gpu.Use)
            {
                var __self__ = Instance;
                var pyargs = ToTuple(new object[]
                    {
                        x.CupyNDarray.PyObject,
                        W.CupyNDarray.PyObject,
                        b?.CupyNDarray?.PyObject,
                    }.Where(x => x is not null)
                    .ToArray());
                var kwargs = new PyDict();
                kwargs["stride"] = ToPython(stride);
                kwargs["pad"] = ToPython(pad);
                dynamic py = __self__.InvokeMethod("deconvolution_2d", pyargs, kwargs);
                return new NDarray(ToCsharp<NDarray>(py).data);
            }
            else
            {
                var __self__ = Instance;
                var pyargs = ToTuple(new object[]
                    {
                        x.NumpyNDarray.PyObject,
                        W.NumpyNDarray.PyObject,
                        b?.NumpyNDarray?.PyObject,
                    }.Where(x => x is not null)
                    .ToArray());
                var kwargs = new PyDict();
                kwargs["stride"] = ToPython(stride);
                kwargs["pad"] = ToPython(pad);
                dynamic py = __self__.InvokeMethod("deconvolution_2d", pyargs, kwargs);
                return new NDarray(ToCsharp<NDarray>(py).data);
            }
        }

        public static NDarray linear(NDarray x, NDarray W, NDarray b = null, int n_batch_axes = 1)
        {
            if (Gpu.Available && Gpu.Use)
            {
                var __self__ = Instance;
                var pyargs = ToTuple(new object[]
                    {
                        x.CupyNDarray.PyObject,
                        W.CupyNDarray.PyObject,
                        b?.CupyNDarray?.PyObject,
                    }.Where(x => x is not null)
                    .ToArray());
                var kwargs = new PyDict();
                kwargs["n_batch_axes"] = ToPython(n_batch_axes);
                dynamic py = __self__.InvokeMethod("linear", pyargs, kwargs);
                return new NDarray(ToCsharp<NDarray>(py).data);
            }
            else
            {
                var __self__ = Instance;
                var pyargs = ToTuple(new object[]
                    {
                        x.NumpyNDarray.PyObject,
                        W.NumpyNDarray.PyObject,
                        b?.NumpyNDarray?.PyObject,
                    }.Where(x => x is not null)
                    .ToArray());
                var kwargs = new PyDict();
                kwargs["n_batch_axes"] = ToPython(n_batch_axes);
                dynamic py = __self__.InvokeMethod("linear", pyargs, kwargs);
                return new NDarray(ToCsharp<NDarray>(py).data);
            }
        }

        public static NDarray max_pooling_2d(NDarray x, int ksize, int? stride = null, int pad = 0, bool cover_all = true, bool return_indices = false)
        {
            if (Gpu.Available && Gpu.Use)
            {
                var __self__ = Instance;
                var pyargs = ToTuple(new object[]
                    {
                        x.CupyNDarray.PyObject,
                        ksize.ToPython(),
                    }.Where(x => x is not null)
                    .ToArray());
                var kwargs = new PyDict();
                if (stride is not null) kwargs["stride"] = stride?.ToPython();
                kwargs["pad"] = ToPython(pad);
                kwargs["cover_all"] = ToPython(cover_all);
                kwargs["return_indices"] = ToPython(return_indices);
                dynamic py = __self__.InvokeMethod("max_pooling_2d", pyargs, kwargs);
                return new NDarray(ToCsharp<NDarray>(py).data);
            }
            else
            {
                var __self__ = Instance;
                var pyargs = ToTuple(new object[]
                    {
                        x.NumpyNDarray.PyObject,
                        ksize.ToPython(),
                    }.Where(x => x is not null)
                    .ToArray());
                var kwargs = new PyDict();
                if (stride is not null) kwargs["stride"] = stride?.ToPython();
                kwargs["pad"] = ToPython(pad);
                kwargs["cover_all"] = ToPython(cover_all);
                kwargs["return_indices"] = ToPython(return_indices);
                dynamic py = __self__.InvokeMethod("max_pooling_2d", pyargs, kwargs);
                return new NDarray(ToCsharp<NDarray>(py).data);
            }
        }

        public static NDarray average_pooling_2d(NDarray x, int ksize, int? stride = null, int pad = 0)
        {
            if (Gpu.Available && Gpu.Use)
            {
                var __self__ = Instance;
                var pyargs = ToTuple(new object[]
                    {
                        x.CupyNDarray.PyObject,
                        ksize.ToPython(),
                    }.Where(x => x is not null)
                    .ToArray());
                var kwargs = new PyDict();
                if (stride is not null) kwargs["stride"] = stride?.ToPython();
                kwargs["pad"] = ToPython(pad);
                dynamic py = __self__.InvokeMethod("average_pooling_2d", pyargs, kwargs);
                return new NDarray(ToCsharp<NDarray>(py).data);
            }
            else
            {
                var __self__ = Instance;
                var pyargs = ToTuple(new object[]
                    {
                        x.NumpyNDarray.PyObject,
                        ksize.ToPython(),
                    }.Where(x => x is not null)
                    .ToArray());
                var kwargs = new PyDict();
                if (stride is not null) kwargs["stride"] = stride?.ToPython();
                kwargs["pad"] = ToPython(pad);
                dynamic py = __self__.InvokeMethod("average_pooling_2d", pyargs, kwargs);
                return new NDarray(ToCsharp<NDarray>(py).data);
            }
        }

        public static NDarray leaky_relu(NDarray x, double slope)
        {
            if (Gpu.Available && Gpu.Use)
            {
                var __self__ = Instance;
                var pyargs = ToTuple(new object[]
                    {
                        x.CupyNDarray.PyObject,
                        slope.ToPython(),
                    }.Where(x => x is not null)
                    .ToArray());
                var kwargs = new PyDict();
                dynamic py = __self__.InvokeMethod("leaky_relu", pyargs, kwargs);
                return new NDarray(ToCsharp<NDarray>(py).data);
            }
            else
            {
                var __self__ = Instance;
                var pyargs = ToTuple(new object[]
                    {
                        x.NumpyNDarray.PyObject,
                        slope.ToPython(),
                    }.Where(x => x is not null)
                    .ToArray());
                var kwargs = new PyDict();
                dynamic py = __self__.InvokeMethod("leaky_relu", pyargs, kwargs);
                return new NDarray(ToCsharp<NDarray>(py).data);
            }
        }

        public static NDarray sigmoid(NDarray x)
        {
            if (Gpu.Available && Gpu.Use)
            {
                var __self__ = Instance;
                var pyargs = ToTuple(new object[]
                    {
                        x.CupyNDarray.PyObject,
                    }.Where(x => x is not null)
                    .ToArray());
                var kwargs = new PyDict();
                dynamic py = __self__.InvokeMethod("sigmoid", pyargs, kwargs);
                return new NDarray(ToCsharp<NDarray>(py).data);
            }
            else
            {
                var __self__ = Instance;
                var pyargs = ToTuple(new object[]
                    {
                        x.NumpyNDarray.PyObject,
                    }.Where(x => x is not null)
                    .ToArray());
                var kwargs = new PyDict();
                dynamic py = __self__.InvokeMethod("sigmoid", pyargs, kwargs);
                return new NDarray(ToCsharp<NDarray>(py).data);
            }
        }

        public static NDarray softmax_cross_entropy(NDarray x, NDarray t)
        {
            if (Gpu.Available && Gpu.Use)
            {
                var __self__ = Instance;
                var pyargs = ToTuple(new object[]
                    {
                        x.CupyNDarray.PyObject,
                        t.CupyNDarray.PyObject,
                    }.Where(x => x is not null)
                    .ToArray());
                var kwargs = new PyDict();
                dynamic py = __self__.InvokeMethod("softmax_cross_entropy", pyargs, kwargs);
                return new NDarray(ToCsharp<NDarray>(py).data);
            }
            else
            {
                var __self__ = Instance;
                var pyargs = ToTuple(new object[]
                    {
                        x.NumpyNDarray.PyObject,
                        t.NumpyNDarray.PyObject,
                    }.Where(x => x is not null)
                    .ToArray());
                var kwargs = new PyDict();
                dynamic py = __self__.InvokeMethod("softmax_cross_entropy", pyargs, kwargs);
                return new NDarray(ToCsharp<NDarray>(py).data);
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
                case ValueTuple<int, int> o: return ToTuple(new object[] { o.Item1, o.Item2 });
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
                case "Dtype": return (T)(object)new Dtype(pyobj);
                case "NDarray": return (T)(object)new NDarray(pyobj);
                case "NDarray`1":
                    switch (typeof(T).GenericTypeArguments[0].Name)
                    {
                        case "Byte": return (T)(object)new NDarray<byte>(pyobj);
                        case "Short": return (T)(object)new NDarray<short>(pyobj);
                        case "Boolean": return (T)(object)new NDarray<bool>(pyobj);
                        case "Int32": return (T)(object)new NDarray<int>(pyobj);
                        case "Int64": return (T)(object)new NDarray<long>(pyobj);
                        case "Single": return (T)(object)new NDarray<float>(pyobj);
                        case "Double": return (T)(object)new NDarray<double>(pyobj);
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
