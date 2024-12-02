using Cupy;
using Python.Included;
using Python.Runtime;
using System.Diagnostics;

namespace DeZero.NET
{
    namespace matplotlib
    {
        public static partial class pyplot
        {
            private static Lazy<PyObject> _lazy_self;

            static pyplot()
            {
                ReInitializeLazySelf();
            }

            public static PyObject self => _lazy_self.Value;

            public static dynamic dynamic_self => self;
            private static bool IsInitialized => self != null;

            private static void ReInitializeLazySelf()
            {
                _lazy_self = new Lazy<PyObject>(() =>
                    {
                        try
                        {
                            return InstallAndImport();
                        }
                        catch (Exception)
                        {
                            // retry to fix the installation by forcing a repair, if Python.Included is used.
                            return InstallAndImport(true);
                        }
                    }
                );
            }

            private static PyObject InstallAndImport(bool force = false)
            {
#if PYTHON_INCLUDED
            Installer.SetupPython(force).Wait();
#endif
#if !PYTHON_INCLUDED
                PythonEngine.Initialize();

                //if (!Installer.IsModuleInstalled("numpy"))
                //{
                //    Installer.PipInstallModule("numpy", "1.24.2", true).GetAwaiter().GetResult();
                //}

                //if (!Installer.IsModuleInstalled("cupy"))
                //{
                //    Installer.PipInstallModule("cupy_cuda12x", force: true).GetAwaiter().GetResult();
                //}

                if (!Installer.IsModuleInstalled("matplotlib"))
                {
                    Installer.PipInstallModule("matplotlib", force: true).GetAwaiter().GetResult();
                }

                Debug.Assert(Installer.IsModuleInstalled("matplotlib"));
#endif
                PythonEngine.AddShutdownHandler(() => ReInitializeLazySelf());
                PythonEngine.Initialize();
                var pyplot = Py.Import("matplotlib.pyplot");
                return pyplot;
            }


            public static void Dispose()
            {
                self?.Dispose();
            }


            //auto-generated
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
                    case PyObject o: return o;
                    // sequence types
                    case Line2D o: return o.PyObject;
                    case Line2D[] o: return ToTuple(o);
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

            //auto-generated
            internal static T ToCsharp<T>(dynamic pyobj)
            {
                switch (typeof(T).Name)
                {
                    // types from 'ToCsharpConversions'
                    case "Image": return (T)(object)new Image(pyobj);
                    case "Line2D": return (T)(object)new Line2D(pyobj);
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
                    {
                        var po = pyobj as PyObject;
                        var len = po.Length();
                        var rv = new NDarray[len];
                        for (var i = 0; i < len; i++)
                            rv[i] = ToCsharp<NDarray>(po[i]);
                        return (T)(object)rv;
                    }
                    case "Line2D[]":
                    {
                        var po = pyobj as PyObject;
                        var len = po.Length();
                        var rv = new Line2D[len];
                        for (var i = 0; i < len; i++)
                            rv[i] = ToCsharp<Line2D>(po[i]);
                        return (T)(object)rv;
                    }
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

            //auto-generated
            internal static T SharpToSharp<T>(object obj)
            {
                if (obj == null) return default;
                switch (obj)
                {
                    // from 'SharpToSharpConversions':
                    case Array a:
                        if (typeof(T) == typeof(NDarray)) return (T)(object)ConvertArrayToNDarray(a);
                        break;
                }

                throw new NotImplementedException(
                    $"Type is not yet supported: {obj.GetType().Name}. Add it to 'SharpToSharpConversions'");
            }

            //auto-generated: SpecialConversions
            private static NDarray ConvertArrayToNDarray(Array a)
            {
                switch (a)
                {
                    case bool[] arr: return xp.array(arr);
                    case int[] arr: return xp.array(arr);
                    case float[] arr: return xp.array(arr);
                    case double[] arr: return xp.array(arr);
                    case int[,] arr:
                        return xp.array(arr.Cast<int>().ToArray()).reshape(arr.GetLength(0), arr.GetLength(1));
                    case float[,] arr:
                        return xp.array(arr.Cast<float>().ToArray()).reshape(arr.GetLength(0), arr.GetLength(1));
                    case double[,] arr:
                        return xp.array(arr.Cast<double>().ToArray()).reshape(arr.GetLength(0), arr.GetLength(1));
                    case bool[,] arr:
                        return xp.array(arr.Cast<bool>().ToArray()).reshape(arr.GetLength(0), arr.GetLength(1));
                    default:
                        throw new NotImplementedException(
                            $"Type {a.GetType()} not supported yet in ConvertArrayToNDarray.");
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

            #region Functions

            public static Line2D[] plot(int[] x, double[] y, string fmt = null, string data = "None", string label = null, double? linewidth = null, string color = null)
            {
                var __self__ = self;
                using var args = ToTuple(new Object[] { x.Select(i => (double)i).ToArray(), y, fmt, data });
                using var kwargs = new PyDict();
                if (label != null) kwargs["label"] = ToPython(label);
                if (linewidth != null) kwargs["linewidth"] = ToPython(linewidth);
                if (color != null) kwargs["color"] = ToPython(color);
                var py = self.InvokeMethod("plot", args, kwargs);
                args.Dispose();
                return ToCsharp<Line2D[]>(py);
            }

            public static Line2D[] plot(double[] x, double[] y, string fmt = null, string data = "None", string label = null, double? linewidth = null, string color = null)
            {
                var __self__ = self;
                using var args = ToTuple(new Object[] { x, y, fmt, data });
                using var kwargs = new PyDict();
                if (label != null) kwargs["label"] = ToPython(label);
                if (linewidth != null) kwargs["linewidth"] = ToPython(linewidth);
                if (color != null) kwargs["color"] = ToPython(color);
                var py = self.InvokeMethod("plot", args, kwargs);
                args.Dispose();
                return ToCsharp<Line2D[]>(py);
            }

            public static Line2D[] plot(NDarray array, string fmt = null, string data = "None", string label = null, double? linewidth = null, string color = null)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    var __self__ = self;
                    using var args = ToTuple(new Object[] { array.CupyNDarray.asnumpy(), fmt, data });
                    using var kwargs = new PyDict();
                    if (label != null) kwargs["label"] = ToPython(label);
                    if (linewidth != null) kwargs["linewidth"] = ToPython(linewidth);
                    if (color != null) kwargs["color"] = ToPython(color);
                    var py = self.InvokeMethod("plot", args, kwargs);
                    args.Dispose();
                    return ToCsharp<Line2D[]>(py);
                }
                else
                {
                    var __self__ = self;
                    using var args = ToTuple(new Object[] { array.NumpyNDarray, fmt, data });
                    using var kwargs = new PyDict();
                    if (label != null) kwargs["label"] = ToPython(label);
                    if (linewidth != null) kwargs["linewidth"] = ToPython(linewidth);
                    if (color != null) kwargs["color"] = ToPython(color);
                    var py = self.InvokeMethod("plot", args, kwargs);
                    args.Dispose();
                    return ToCsharp<Line2D[]>(py);
                }
            }

            public static Line2D[] plot(NDarray array1, NDarray array2, string fmt = null, string data = "None", string label = null, double? linewidth = null, string color = null)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    var __self__ = self;
                    using var args = ToTuple(new Object[] { array1.CupyNDarray.asnumpy(), array2.CupyNDarray.asnumpy(), fmt, data});
                    using var kwargs = new PyDict();
                    if (label != null) kwargs["label"] = ToPython(label);
                    if (linewidth != null) kwargs["linewidth"] = ToPython(linewidth);
                    if (color != null) kwargs["color"] = ToPython(color);
                    var py = self.InvokeMethod("plot", args, kwargs);
                    args.Dispose();
                    return ToCsharp<Line2D[]>(py);
                }
                else
                {
                    var __self__ = self;
                    using var args = ToTuple(new Object[] { array1.NumpyNDarray, array2.NumpyNDarray, fmt, data });
                    using var kwargs = new PyDict();
                    if (label != null) kwargs["label"] = ToPython(label);
                    if (linewidth != null) kwargs["linewidth"] = ToPython(linewidth);
                    if (color != null) kwargs["color"] = ToPython(color);
                    var py = self.InvokeMethod("plot", args, kwargs);
                    args.Dispose();
                    return ToCsharp<Line2D[]>(py);
                }
            }

            public static void title(string title)
            {
                var __self__ = self;
                using var args = ToTuple(new Object[] { title });
                using var py = self.InvokeMethod("title", args);
                args.Dispose();
            }

            public static void xlabel(string xlabel)
            {
                var __self__ = self;
                using var args = ToTuple(new Object[] { xlabel });
                using var py = self.InvokeMethod("xlabel", args);
                args.Dispose();
            }

            public static void ylabel(string ylabel)
            {
                var __self__ = self;
                using var args = ToTuple(new Object[] { ylabel });
                using var py = self.InvokeMethod("ylabel", args);
                args.Dispose();
            }

            public static void xscale(string scale)
            {
                var __self__ = self;
                using var args = ToTuple(new Object[] { scale });
                using var py = self.InvokeMethod("xscale", args);
                args.Dispose();
            }

            public static void yscale(string scale)
            {
                var __self__ = self;
                using var args = ToTuple(new Object[] { scale });
                using var py = self.InvokeMethod("yscale", args);
                args.Dispose();
            }

            public static void xlim(double xmin, double xmax)
            {
                var __self__ = self;
                using var args = ToTuple(new Object[] { xmin, xmax });
                using var py = self.InvokeMethod("xlim", args);
                args.Dispose();
            }

            public static void ylim(double ymin, double ymax)
            {
                var __self__ = self;
                using var args = ToTuple(new Object[] { ymin, ymax });
                using var py = self.InvokeMethod("ylim", args);
                args.Dispose();
            }

            public static void legend(string loc = null, double[] bbox_to_anchor = null)
            {
                var __self__ = self;
                using var args = ToTuple(new Object[] { loc, bbox_to_anchor is null ? null : ToTuple(bbox_to_anchor) });
                if (loc is null && bbox_to_anchor is null)
                {
                    using var py = self.InvokeMethod("legend");
                }
                else
                {
                    using var py = self.InvokeMethod("legend", args);
                }

                args.Dispose();
            }

            public static void legend(Line2D[] handles, string[] labels)
            {
                var __self__ = self;
                using var args = ToTuple(new Object[] { });
                using var kwargs = new PyDict();
                if (handles != null) kwargs["handles"] = ToPython(handles);
                if (labels != null) kwargs["labels"] = ToPython(labels);
                using var py = self.InvokeMethod("legend", args, kwargs);
                args.Dispose();
            }

            public static void imshow(NDarray array, string cmap = null, string interpolation = null)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    var __self__ = self;
                    using var args = ToTuple(new Object[] { array.CupyNDarray.asnumpy() });
                    using var kwargs = new PyDict();
                    if (cmap is not null) kwargs["cmap"] = ToPython(cmap);
                    if (interpolation is not null) kwargs["interpolation"] = ToPython(interpolation);
                    using var py = self.InvokeMethod("imshow", args, kwargs);
                    args.Dispose();
                }
                else
                {
                    var __self__ = self;
                    using var args = ToTuple(new Object[] { array.NumpyNDarray });
                    using var kwargs = new PyDict();
                    if (cmap is not null) kwargs["cmap"] = ToPython(cmap);
                    if (interpolation is not null) kwargs["interpolation"] = ToPython(interpolation);
                    using var py = self.InvokeMethod("imshow", args, kwargs);
                    args.Dispose();
                }
            }

            public static void show()
            {
                var __self__ = self;
                using var args = ToTuple(new Object[] {  });
                using var py = self.InvokeMethod("show", args);
                args.Dispose();
            }

            public static void axis(string option)
            {
                var __self__ = self;
                using var args = ToTuple(new Object[] { option.ToPython() });
                using var kwargs = new PyDict();
                using var py = self.InvokeMethod("axis", args, kwargs);
                args.Dispose();
            }

            public static void axis((int xmin, int xmax, int ymin, int ymax)? arg)
            {
                var __self__ = self;
                using var args = ToTuple(new Object[] { arg.ToPython() });
                using var kwargs = new PyDict();
                using var py = self.InvokeMethod("axis", args, kwargs);
                args.Dispose();
            }

            public static void axis((int xmin, int xmax, int ymin, int ymax)? arg, string option)
            {
                var __self__ = self;
                using var args = ToTuple(new Object[] { arg.ToPython(), option.ToPython() });
                using var kwargs = new PyDict();
                using var py = self.InvokeMethod("axis", args, kwargs);
                args.Dispose();
            }

            public static void clf()
            {
                var __self__ = self;
                using var args = ToTuple(new Object[] { });
                using var py = self.InvokeMethod("clf", args);
                args.Dispose();
            }

            public static void pause(double interval)
            {
                var __self__ = self;
                using var args = ToTuple(new Object[] { interval });
                using var py = self.InvokeMethod("pause", args);
                args.Dispose();
            }

            public static void grid(bool visible = true)
            {
                var __self__ = self;
                using var args = ToTuple(new Object[] { visible });
                using var py = self.InvokeMethod("grid", args);
                args.Dispose();
            }

            public static void figure(int? num = null)
            {
                var __self__ = self;
                using var args = ToTuple(new Object[] { });
                using var kwargs = new PyDict();
                if (num != null) kwargs["num"] = ToPython(num);
                using var py = self.InvokeMethod("figure", args, kwargs);
                args.Dispose();
            }

            public static void subplot(int nrows, int ncols, int index)
            {
                var __self__ = self;
                // matplotlibのsubplotは3つの引数を1つの数字として受け取ることもできます
                // 例：subplot(211) は subplot(2,1,1) と同じ
                int pos = (nrows * 100) + (ncols * 10) + index;
                using var args = ToTuple(new Object[] { pos });
                using var py = self.InvokeMethod("subplot", args);
                args.Dispose();
            }

            // オーバーロード - より柔軟な使用のため
            public static void subplot(int pos)
            {
                var __self__ = self;
                using var args = ToTuple(new Object[] { pos });
                using var py = self.InvokeMethod("subplot", args);
                args.Dispose();
            }

            public static void text(double x,
                                    double y,
                                    string s,
                                    string color = null,
                                    double? fontsize = null,
                                    string fontweight = null,
                                    string ha = null,  // horizontalalignment
                                    string va = null,  // verticalalignment
                                    double? alpha = null,
                                    string backgroundcolor = null,
                                    double? rotation = null,
                                    string fontfamily = null,
                                    string bbox = null)
            {
                var __self__ = self;
                using var args = ToTuple(new Object[] { x, y, s });
                using var kwargs = new PyDict();

                // オプションパラメータの設定
                if (color != null) kwargs["color"] = ToPython(color);
                if (fontsize != null) kwargs["fontsize"] = ToPython(fontsize);
                if (fontweight != null) kwargs["fontweight"] = ToPython(fontweight);
                if (ha != null) kwargs["horizontalalignment"] = ToPython(ha);
                if (va != null) kwargs["verticalalignment"] = ToPython(va);
                if (alpha != null) kwargs["alpha"] = ToPython(alpha);
                if (backgroundcolor != null) kwargs["backgroundcolor"] = ToPython(backgroundcolor);
                if (rotation != null) kwargs["rotation"] = ToPython(rotation);
                if (fontfamily != null) kwargs["fontfamily"] = ToPython(fontfamily);
                //if (bbox != null) kwargs["bbox"] = ToPython(bbox);

                using var py = self.InvokeMethod("text", args, kwargs);
                args.Dispose();
            }

            // bboxパラメータ用のオーバーロード
            public static void text(double x,
                                    double y,
                                    string s,
                                    Dictionary<string, object> bbox_props,
                                    string color = null,
                                    double? fontsize = null,
                                    string fontweight = null,
                                    string ha = null,
                                    string va = null,
                                    double? alpha = null,
                                    double? rotation = null)
            {
                var __self__ = self;
                using var args = ToTuple(new Object[] { x, y, s });
                using var kwargs = new PyDict();

                // bbox_propsをPyDictに変換
                if (bbox_props != null)
                {
                    var bbox_dict = new PyDict();
                    foreach (var kvp in bbox_props)
                    {
                        bbox_dict[new PyString(kvp.Key)] = ToPython(kvp.Value);
                    }
                    kwargs["bbox"] = bbox_dict;
                }

                if (color != null) kwargs["color"] = ToPython(color);
                if (fontsize != null) kwargs["fontsize"] = ToPython(fontsize);
                if (fontweight != null) kwargs["fontweight"] = ToPython(fontweight);
                if (ha != null) kwargs["horizontalalignment"] = ToPython(ha);
                if (va != null) kwargs["verticalalignment"] = ToPython(va);
                if (alpha != null) kwargs["alpha"] = ToPython(alpha);
                if (rotation != null) kwargs["rotation"] = ToPython(rotation);

                using var py = self.InvokeMethod("text", args, kwargs);
                args.Dispose();
            }

            // トランスフォーム付きのオーバーロード
            public static void text(double x,
                                    double y,
                                    string s,
                                    string transform,
                                    string color = null,
                                    double? fontsize = null,
                                    string fontweight = null,
                                    string ha = null,
                                    string va = null)
            {
                var __self__ = self;
                using var args = ToTuple(new Object[] { x, y, s });
                using var kwargs = new PyDict();

                kwargs["transform"] = ToPython(transform);
                if (color != null) kwargs["color"] = ToPython(color);
                if (fontsize != null) kwargs["fontsize"] = ToPython(fontsize);
                if (fontweight != null) kwargs["fontweight"] = ToPython(fontweight);
                if (ha != null) kwargs["horizontalalignment"] = ToPython(ha);
                if (va != null) kwargs["verticalalignment"] = ToPython(va);

                using var py = self.InvokeMethod("text", args, kwargs);
                args.Dispose();
            }

            public static void axhline(
                double y,
                double? xmin = null,
                double? xmax = null,
                string color = null,
                string linestyle = null,
                string label = null,
                double? linewidth = null,
                double? alpha = null)
            {
                var __self__ = self;
                using var args = ToTuple(new Object[] { y });
                using var kwargs = new PyDict();

                // オプションパラメータの設定
                if (xmin.HasValue) kwargs["xmin"] = ToPython(xmin.Value);
                if (xmax.HasValue) kwargs["xmax"] = ToPython(xmax.Value);
                if (color != null) kwargs["color"] = ToPython(color);
                if (linestyle != null) kwargs["linestyle"] = ToPython(linestyle);
                if (label != null) kwargs["label"] = ToPython(label);
                if (linewidth.HasValue) kwargs["linewidth"] = ToPython(linewidth.Value);
                if (alpha.HasValue) kwargs["alpha"] = ToPython(alpha.Value);

                using var py = self.InvokeMethod("axhline", args, kwargs);
                args.Dispose();
            }

            //// 簡略化したオーバーロード
            //public static void axhline(double y, string color, string linestyle = "--", double alpha = 0.5)
            //{
            //    axhline(y, color: color, linestyle: linestyle, alpha: alpha);
            //}

            #endregion
        }

        public partial class PythonObject : IDisposable
        {
            public PyObject self; // can not be made readonly because of NDarray(IntPtr ... )

            public PythonObject(PyObject pyobject)
            {
                self = pyobject;
            }

            public PythonObject(PythonObject t)
            {
                self = t.PyObject;
            }

            protected PythonObject()
            {
            } // required for some constructors

            public dynamic PyObject => self;

            public IntPtr Handle => self.Handle;


            /// <summary>
            ///     An object to simplify the interaction of the array with the ctypes module.
            /// </summary>
            //public PyObject ctypes => self.GetAttr("ctypes"); // TODO: wrap ctypes
            public PyObject ctypes => Cupy.ctypes.self; //.GetAttr("ctypes");

            public void Dispose()
            {
                self?.Dispose();
            }

            public override bool Equals(object obj)
            {
                if (obj == null)
                    return false;
                switch (obj)
                {
                    case PythonObject other:
                        return self.Equals(other.self);
                    case PyObject other:
                        return self.Equals(other);
                }

                return base.Equals(obj);
            }

            public override int GetHashCode()
            {
                return self.GetHashCode();
            }

            public override string ToString()
            {
                return self.ToString();
            }

            public static PythonObject Create<T>(string python_class)
            {
                throw new NotImplementedException();
            }
        }

        public partial class Image : PythonObject
        {
            protected Image()
            {
            }

            // these are manual overrides of functions or properties that can not be automatically generated

            public Image(PyObject pyobj) : base(pyobj)
            {
            }
        }

        public partial class Line2D : PythonObject
        {
            public Line2D(PyObject pyobj) : base(pyobj)
            {
            }
        }
    }
}
