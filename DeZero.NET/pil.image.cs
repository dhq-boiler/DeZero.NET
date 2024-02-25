using Cupy;
using DeZero.NET.PIL;
using Microsoft.VisualBasic;
using Numpy;
using Python.Included;
using Python.Runtime;
using System.Diagnostics;
using DeZero.NET;

namespace DeZero.NET
{
    namespace PIL
    {
        public static partial class PILImage
        {
            private static Lazy<PyObject> _lazy_self;

            static PILImage()
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

                if (!Installer.IsModuleInstalled("Pillow"))
                {
                    Installer.PipInstallModule("Pillow", force: true).GetAwaiter().GetResult();
                }

#endif
                PythonEngine.AddShutdownHandler(() => ReInitializeLazySelf());
                PythonEngine.Initialize();
                var Image = Py.Import("PIL.Image");
                return Image;
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

            public static Image open(string fp, string mode = "r", string formats = "None")
            {
                var __self__ = self;
                var args = ToTuple(new object[]
                {
                    fp, mode
                });
                var py = self.InvokeMethod("open", args);
                args.Dispose();
                return ToCsharp<Image>(py);
            }

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
    }

    public static partial class xp
    {
        public static NDarray array(Image image)
        {
            if (Gpu.Available && Gpu.Use)
            {
                var __self__ = Py.Import("cupy");
                var args = new PyTuple(new PyObject[]
                    { image.PyObject });
                dynamic py = __self__.InvokeMethod("array", args);
                return PILImage.ToCsharp<NDarray>(py);
            }
            else
            {
                var __self__ = Py.Import("numpy");
                var args = new PyTuple(new PyObject[]
                    { image.PyObject });
                dynamic py = __self__.InvokeMethod("array", args);
                return PILImage.ToCsharp<NDarray>(py);
            }
        }
    }
}
