using System.Diagnostics;
using Cupy;
using Numpy;
using Python.Runtime;
using System.Globalization;
using System.Numerics;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text.RegularExpressions;
using cp = Cupy;
using np = Numpy;

namespace DeZero.NET
{
    public static class Gpu
    {
        private static bool _Avaiable = false;

        public static bool Available
        {
            get
            {
                try
                {
                    if (!_Avaiable && Use)
                    {
                        if (string.IsNullOrEmpty(Runtime.PythonDLL))
                        {
                            Console.Error.WriteLine(@"Please set Runtime.PythonDLL (e.g. 'C:\Users\USERNAME\AppData\Local\Programs\Python\Python311\python311.dll')");
                            return false;
                        }
                        PythonEngine.Initialize();
                        _Avaiable = true;
                    }

                    return _Avaiable;
                }
                catch (Exception e)
                {
                    return false;
                }
            }
        }

        public static bool Use { get; set; } = false;

        static Gpu()
        {

        }
    }

    public enum ArrayMode
    {
        Unspecified,
        np,
        cp
    }

    public class NDarray
    {
        public Numpy.NDarray NumpyNDarray { get; internal set; }
        public Cupy.NDarray CupyNDarray { get; internal set; }

        public object Array => (object)CupyNDarray ?? (object)NumpyNDarray;

        protected NDarray()
        {
        }

        public NDarray(PyObject pyobj)
        {
            if (Gpu.Available && Gpu.Use)
            {
                CupyNDarray = new Cupy.NDarray(pyobj);
            }
            else
            {
                NumpyNDarray = new Numpy.NDarray(pyobj);
            }
        }

        public NDarray(byte obj)
        {
            if (Gpu.Available && Gpu.Use)
            {
                CupyNDarray = new Cupy.NDarray(obj);
            }
            else
            {
                throw new NotSupportedException();
            }
        }

        public NDarray(int obj)
        {
            if (Gpu.Available && Gpu.Use)
            {
                CupyNDarray = new Cupy.NDarray(obj);
            }
            else
            {
                throw new NotSupportedException();
            }
        }

        public NDarray(long obj)
        {
            if (Gpu.Available && Gpu.Use)
            {
                CupyNDarray = new Cupy.NDarray(obj);
            }
            else
            {
                throw new NotSupportedException();
            }
        }

        public NDarray(float obj)
        {
            if (Gpu.Available && Gpu.Use)
            {
                CupyNDarray = new Cupy.NDarray(obj);
            }
            else
            {
                throw new NotSupportedException();
            }
        }

        public NDarray(double obj)
        {
            if (Gpu.Available && Gpu.Use)
            {
                CupyNDarray = new Cupy.NDarray(obj);
            }
            else
            {
                throw new NotSupportedException();
            }
        }

        public NDarray(bool obj)
        {
            if (Gpu.Available && Gpu.Use)
            {
                CupyNDarray = new Cupy.NDarray(obj);
            }
            else
            {
                throw new NotSupportedException();
            }
        }

        private bool _autoDispose = false;
        private ArrayMode _saveTarget = ArrayMode.Unspecified;

        public NDarray(Numpy.NDarray t, bool autoDispose = false)
        {
            _autoDispose = autoDispose;
            _saveTarget = ArrayMode.np;
            if (Gpu.Available && Gpu.Use)
            {
                CupyNDarray = new Cupy.NDarray(t.PyObject);
            }
            else
            {
                NumpyNDarray = new Numpy.NDarray(t);
            }
        }

        public NDarray(Cupy.NDarray t, bool autoDispose = false)
        {
            _autoDispose = autoDispose;
            _saveTarget = ArrayMode.cp;
            if (Gpu.Available && Gpu.Use)
            {
                CupyNDarray = new Cupy.NDarray(t);
            }
            else
            {
                NumpyNDarray = cpExtensions.asnumpy(CupyNDarray);
            }
        }

        public static NDarray operator +(NDarray a, NDarray b)
        {
            return xp.add(b, a);
        }

        public static NDarray operator -(NDarray a, NDarray b)
        {
            return xp.subtract(b, a);
        }

        public static NDarray operator *(NDarray a, NDarray b)
        {
            return xp.multiply(b, a);
        }

        public static NDarray operator /(NDarray a, NDarray b)
        {
            return xp.divide(b, a);
        }

        public static NDarray operator +(NDarray a, int b)
        {
            if (Gpu.Available && Gpu.Use && a.CupyNDarray is not null)
            {
                dynamic arr = a.CupyNDarray.PyObject;
                arr += b;
                return new NDarray(arr);
            }
            else
            {
                dynamic arr = a.NumpyNDarray.PyObject;
                arr += b;
                return new NDarray(arr);
            }
        }

        public static NDarray operator +(NDarray a, long b)
        {
            if (Gpu.Available && Gpu.Use && a.CupyNDarray is not null)
            {
                dynamic arr = a.CupyNDarray.PyObject;
                arr += b;
                return new NDarray(arr);
            }
            else
            {
                dynamic arr = a.NumpyNDarray.PyObject;
                arr += b;
                return new NDarray(arr);
            }
        }

        public static NDarray operator +(NDarray a, float b)
        {
            if (Gpu.Available && Gpu.Use && a.CupyNDarray is not null)
            {
                dynamic arr = a.CupyNDarray.PyObject;
                arr += b;
                return new NDarray(arr);
            }
            else
            {
                dynamic arr = a.NumpyNDarray.PyObject;
                arr += b;
                return new NDarray(arr);
            }
        }

        public static NDarray operator +(NDarray a, double b)
        {
            if (Gpu.Available && Gpu.Use && a.CupyNDarray is not null)
            {
                dynamic arr = a.CupyNDarray.PyObject;
                arr += b;
                return new NDarray(arr);
            }
            else
            {
                dynamic arr = a.NumpyNDarray.PyObject;
                arr += b;
                return new NDarray(arr);
            }
        }

        public static NDarray operator -(NDarray a, int b)
        {
            if (Gpu.Available && Gpu.Use && a.CupyNDarray is not null)
            {
                dynamic arr = a.CupyNDarray.PyObject;
                arr -= b;
                return new NDarray(arr);
            }
            else
            {
                dynamic arr = a.NumpyNDarray.PyObject;
                arr -= b;
                return new NDarray(arr);
            }
        }

        public static NDarray operator -(NDarray a, long b)
        {
            if (Gpu.Available && Gpu.Use && a.CupyNDarray is not null)
            {
                dynamic arr = a.CupyNDarray.PyObject;
                arr -= b;
                return new NDarray(arr);
            }
            else
            {
                dynamic arr = a.NumpyNDarray.PyObject;
                arr -= b;
                return new NDarray(arr);
            }
        }

        public static NDarray operator -(NDarray a, float b)
        {
            if (Gpu.Available && Gpu.Use && a.CupyNDarray is not null)
            {
                dynamic arr = a.CupyNDarray.PyObject;
                arr -= b;
                return new NDarray(arr);
            }
            else
            {
                dynamic arr = a.NumpyNDarray.PyObject;
                arr -= b;
                return new NDarray(arr);
            }
        }

        public static NDarray operator -(NDarray a, double b)
        {
            if (Gpu.Available && Gpu.Use && a.CupyNDarray is not null)
            {
                dynamic arr = a.CupyNDarray.PyObject;
                arr -= b;
                return new NDarray(arr);
            }
            else
            {
                dynamic arr = a.NumpyNDarray.PyObject;
                arr -= b;
                return new NDarray(arr);
            }
        }

        public static NDarray operator *(NDarray a, int b)
        {
            if (Gpu.Available && Gpu.Use && a.CupyNDarray is not null)
            {
                dynamic arr = a.CupyNDarray.PyObject;
                arr *= b;
                return new NDarray(arr);
            }
            else
            {
                dynamic arr = a.NumpyNDarray.PyObject;
                arr *= b;
                return new NDarray(arr);
            }
        }

        public static NDarray operator *(NDarray a, long b)
        {
            if (Gpu.Available && Gpu.Use && a.CupyNDarray is not null)
            {
                dynamic arr = a.CupyNDarray.PyObject;
                arr *= b;
                return new NDarray(arr);
            }
            else
            {
                dynamic arr = a.NumpyNDarray.PyObject;
                arr *= b;
                return new NDarray(arr);
            }
        }

        public static NDarray operator *(NDarray a, float b)
        {
            if (Gpu.Available && Gpu.Use && a.CupyNDarray is not null)
            {
                dynamic arr = a.CupyNDarray.PyObject;
                arr *= b;
                return new NDarray(arr);
            }
            else
            {
                dynamic arr = a.NumpyNDarray.PyObject;
                arr *= b;
                return new NDarray(arr);
            }
        }

        public static NDarray operator *(NDarray a, double b)
        {
            if (Gpu.Available && Gpu.Use && a.CupyNDarray is not null)
            {
                dynamic arr = a.CupyNDarray.PyObject;
                arr *= b;
                return new NDarray(arr);
            }
            else
            {
                dynamic arr = a.NumpyNDarray.PyObject;
                arr *= b;
                return new NDarray(arr);
            }
        }

        public static NDarray operator *(double a, NDarray b)
        {
            if (Gpu.Available && Gpu.Use && b.CupyNDarray is not null)
            {
                dynamic arr = b.CupyNDarray.PyObject;
                arr *= b;
                return new NDarray(arr);
            }
            else
            {
                dynamic arr = b.NumpyNDarray.PyObject;
                arr *= b;
                return new NDarray(arr);
            }
        }

        public static NDarray operator /(NDarray a, int b)
        {
            if (Gpu.Available && Gpu.Use && a.CupyNDarray is not null)
            {
                dynamic arr = a.CupyNDarray.PyObject;
                arr /= b;
                return new NDarray(arr);
            }
            else
            {
                dynamic arr = a.NumpyNDarray.PyObject;
                arr /= b;
                return new NDarray(arr);
            }
        }

        public static NDarray operator /(NDarray a, long b)
        {
            if (Gpu.Available && Gpu.Use && a.CupyNDarray is not null)
            {
                dynamic arr = a.CupyNDarray.PyObject;
                arr /= b;
                return new NDarray(arr);
            }
            else
            {
                dynamic arr = a.NumpyNDarray.PyObject;
                arr /= b;
                return new NDarray(arr);
            }
        }

        public static NDarray operator /(NDarray a, float b)
        {
            if (Gpu.Available && Gpu.Use && a.CupyNDarray is not null)
            {
                dynamic arr = a.CupyNDarray.PyObject;
                arr /= b;
                return new NDarray(arr);
            }
            else
            {
                dynamic arr = a.NumpyNDarray.PyObject;
                arr /= b;
                return new NDarray(arr);
            }
        }

        public static NDarray operator /(NDarray a, double b)
        {
            if (Gpu.Available && Gpu.Use && a.CupyNDarray is not null)
            {
                dynamic arr = a.CupyNDarray.PyObject;
                arr /= b;
                return new NDarray(arr);
            }
            else
            {
                dynamic arr = a.NumpyNDarray.PyObject;
                arr /= b;
                return new NDarray(arr);
            }
        }

        public static NDarray operator -(NDarray a)
        {
            return a.negative();
        }
        
        public NDarray T => (Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp ? new NDarray(CupyNDarray.T) : new NDarray(NumpyNDarray.T);

        public PyObject ctypes => (Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp ? CupyNDarray.ctypes : NumpyNDarray.ctypes;

        public PyObject data => (Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp ? CupyNDarray.data : NumpyNDarray.data;

        public Dtype dtype => (Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp
            ? new Dtype(CupyNDarray.dtype)
            : new Dtype(NumpyNDarray.dtype);

        public Flags flags => (Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp ? new Flags(CupyNDarray.flags) : new Flags(NumpyNDarray.flags);

        public PyObject flat => (Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp ? CupyNDarray.flat : NumpyNDarray.flat;

        public NDarray imag => (Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp ? new NDarray(CupyNDarray.imag) : new NDarray(NumpyNDarray.imag);

        public int itemsize => (Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp ? CupyNDarray.itemsize : NumpyNDarray.itemsize;

        public int len => (Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp ? CupyNDarray.len : NumpyNDarray.len;

        public int nbytes => (Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp ? CupyNDarray.nbytes : NumpyNDarray.nbytes;

        public int ndim => (Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp ? CupyNDarray.ndim : NumpyNDarray.ndim;

        public NDarray real => (Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp ? new NDarray(CupyNDarray.real) : new NDarray(NumpyNDarray.real);

        public string repr => (Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp ? CupyNDarray.repr : NumpyNDarray.repr;

        public Shape shape => this.ToShape((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp);

        public int size => (Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp ? CupyNDarray.size : NumpyNDarray.size;

        public string str => (Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp ? CupyNDarray.str : NumpyNDarray.str;

        public int[] strides => (Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp ? CupyNDarray.strides : NumpyNDarray.strides;

        public IntPtr Handle => (Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp ? CupyNDarray.Handle : NumpyNDarray.Handle;

        public dynamic PyObject => (Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp ? CupyNDarray.PyObject : NumpyNDarray.PyObject;

        public PyObject self => (Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp ? CupyNDarray.self : NumpyNDarray.self;

        public NDarray this[int index] => (Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp ? new NDarray(CupyNDarray[index]) : new NDarray(NumpyNDarray[index]);
        public NDarray this[(int x, int y) index] => (Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp ? new NDarray(CupyNDarray[index.x, index.y]) : new NDarray(NumpyNDarray[index.x, index.y]);

        public NDarray this[PyObject index]
        {
            get
            {
                try
                {
                    var tuple2 = (Tuple<int, int>)ToCsharp<Tuple<int, int>>(index);
                    if (tuple2 is not null)
                    {
                        if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                        {
                            return new NDarray(CupyNDarray[tuple2.Item1, tuple2.Item2]);
                        }
                        else
                        {
                            return new NDarray(NumpyNDarray[tuple2.Item1, tuple2.Item2]);
                        }
                    }
                }
                catch (Exception e)
                {
                }

                try
                {
                    var tuple3 = (Tuple<int, int, int>)ToCsharp<Tuple<int, int, int>>(index);
                    if (tuple3 is not null)
                    {
                        if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                        {
                            return new NDarray(CupyNDarray[tuple3.Item1, tuple3.Item2, tuple3.Item3]);
                        }
                        else
                        {
                            return new NDarray(NumpyNDarray[tuple3.Item1, tuple3.Item2, tuple3.Item3]);
                        }
                    }
                }
                catch (Exception e)
                {
                }

                throw new NotSupportedException();
            }
            set
            {
                try
                {
                    var tuple2 = (Tuple<int, int>)ToCsharp<Tuple<int, int>>(index);
                    if (tuple2 is not null)
                    {
                        if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                        {
                            CupyNDarray[tuple2.Item1, tuple2.Item2] = value.CupyNDarray;
                        }
                        else
                        {
                            NumpyNDarray[tuple2.Item1, tuple2.Item2] = value.NumpyNDarray;
                        }
                    }

                    return;
                }
                catch (Exception e)
                {
                }

                try
                {
                    var tuple3 = (Tuple<int, int, int>)ToCsharp<Tuple<int, int, int>>(index);
                    if (tuple3 is not null)
                    {
                        if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                        {
                            CupyNDarray[tuple3.Item1, tuple3.Item2, tuple3.Item3] = value.CupyNDarray;
                        }
                        else
                        {
                            NumpyNDarray[tuple3.Item1, tuple3.Item2, tuple3.Item3] = value.NumpyNDarray;
                        }
                    }

                    return;
                }
                catch (Exception e)
                {
                }

                throw new NotSupportedException();
            }
        }

        public NDarray Switch(bool deleteOriginal = true)
        {
            if (this.CupyNDarray is null && this.NumpyNDarray is not null)
            {
                //CupyNDarray = new Cupy.NDarray(NumpyNDarray.PyObject).copy();
                CupyNDarray = NumpyNDarray.asarray();
                if (deleteOriginal)
                {
                    using (Py.GIL())
                    {
                        NumpyNDarray.Dispose();
                        NumpyNDarray = null;
                    }
                }

                return this;
            }
            else if (this.CupyNDarray is not null && this.NumpyNDarray is null)
            {
                NumpyNDarray = cpExtensions.asnumpy(CupyNDarray).copy();
                if (deleteOriginal)
                {
                    using (Py.GIL())
                    {
                        CupyNDarray.Dispose();
                        CupyNDarray = null;
                    }
                }

                return this;
            }

            throw new InvalidOperationException();
        }

        public static PyTuple ToTuple(Array input)
        {
            var array = new PyObject[input.Length];
            for (var i = 0; i < input.Length; i++) array[i] = ToPython(input.GetValue(i));
            return new PyTuple(array);
        }

        //auto-generated
        public static PyObject ToPython(object obj)
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

        public static object ToCsharp<T>(dynamic pyobj)
        {
            if (typeof(T).Name == "NDarray" || !pyobj.ToString().Contains(",") && !pyobj.ToString().Contains(". ") && !pyobj.ToString().Contains(" "))
            {
                return ToCsharpInternal<T>(pyobj);
            }

            return ToCsharpInternalArray<T>(pyobj);
        }

        private static object ToCsharpInternalArray<T>(dynamic pyobj)
        {
            switch (typeof(T).Name)
            {
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
                case "NDarray[]":
                    var po = pyobj as PyObject;
                    var len = po.Length();
                    var rv = new NDarray[len];
                    for (var i = 0; i < len; i++)
                        rv[i] = (NDarray)ToCsharp<NDarray>(po[i]);
                    return (object)rv;
                case "Boolean[]":
                    {
                        var _po = GetPo(pyobj);
                        int _len = GetLen(_po);
                        var _rv = new Boolean[_len];
                        for (var i = 0; i < _len; i++)
                            _rv[i] = (bool)ToCsharp<Boolean>(_po[i]);
                        return (object)_rv;
                    }
                case "Int16":
                case "Int16[]":
                    {
                        var _po = GetPo(pyobj);
                        int _len = GetLen(_po);
                        var _rv = new Int16[_len];
                        for (var i = 0; i < _len; i++)
                            _rv[i] = (Int16)ToCsharp<Int16>(_po[i]);
                        return (object)_rv;
                    }
                case "Int32":
                case "Int32[]":
                    {
                        var _po = GetPo(pyobj);
                        int _len = GetLen(_po);
                        var _rv = new int[_len];
                        for (var i = 0; i < _len; i++)
                            _rv[i] = (Int32)ToCsharp<Int32>(_po[i]);
                        return (object)_rv;
                    }
                case "Int64":
                case "Int64[]":
                    {
                        var _po = GetPo(pyobj);
                        int _len = GetLen(_po);
                        var _rv = new Int64[_len];
                        for (var i = 0; i < _len; i++)
                            _rv[i] = (Int64)ToCsharp<Int64>(_po[i]);
                        return (object)_rv;
                    }
                case "UInt16":
                case "UInt16[]":
                    {
                        var _po = GetPo(pyobj);
                        int _len = GetLen(_po);
                        var _rv = new UInt16[_len];
                        for (var i = 0; i < _len; i++)
                            _rv[i] = (UInt16)ToCsharp<UInt16>(_po[i]);
                        return (object)_rv;
                    }
                case "UInt32":
                case "UInt32[]":
                    {
                        var _po = GetPo(pyobj);
                        int _len = GetLen(_po);
                        var _rv = new UInt32[_len];
                        for (var i = 0; i < _len; i++)
                            _rv[i] = (UInt32)ToCsharp<UInt32>(_po[i]);
                        return (object)_rv;
                    }
                case "UInt64":
                case "UInt64[]":
                    {
                        var _po = GetPo(pyobj);
                        int _len = GetLen(_po);
                        var _rv = new UInt64[_len];
                        for (var i = 0; i < _len; i++)
                            _rv[i] = (UInt64)ToCsharp<UInt64>(_po[i]);
                        return (object)_rv;
                    }
                case "Single":
                case "Single[]":
                    {
                        var _po = GetPo(pyobj);
                        int _len = GetLen(_po);
                        var _rv = new float[_len];
                        for (var i = 0; i < _len; i++)
                            _rv[i] = (Single)ToCsharp<float>(_po[i]);
                        return (object)_rv;
                    }
                case "Double":
                case "Double[]":
                    {
                        var _po = GetPo(pyobj);
                        int _len = GetLen(_po);
                        var _rv = new double[_len];
                        for (var i = 0; i < _len; i++)
                            _rv[i] = (Double)ToCsharp<double>(_po[i]);
                        return (object)_rv;
                    }
                case "Complex":
                case "Complex[]":
                    {
                        var _po = GetPo(pyobj);
                        int _len = GetLen(_po);
                        var _rv = new Complex[_len];
                        for (var i = 0; i < _len; i++)
                            _rv[i] = (Complex)ToCsharp<Complex>(_po[i]);
                        return (object)_rv;
                    }
                case "Int32[][]":
                    {
                        var _po = GetPo(pyobj);
                        int _len = GetLen(_po);
                        var __len = ToCsharp<Int32>(pyobj[0].len);
                        var _rv = new int[_len][];
                        for (var i = 0; i < _len; i++)
                        {
                            _rv[i] = new Int32[__len];
                            for (var j = 0; j < __len; j++)
                                _rv[i][j] = (Int32)ToCsharp<Int32>(_po[i][j]);
                        }
                        return (object)_rv;
                    }
                case "Int32[,]":
                    {
                        var _po = GetPo(pyobj);
                        var _len = ToCsharp<int>(pyobj.len);
                        var _rv = CreateInitialMultidimensionalArray<int>(_len, pyobj[0].len);
                        for (var i = 0; i < _len; i++)
                        {
                            if (_len == 1)
                            {
                                _rv[i, 0] = (int)ToCsharp<int>(_po);
                            }
                            else
                            {
                                var elements = (int[])ToCsharp<int[]>(_po[i]);
                                for (int j = 0; j < elements.Length; j++)
                                {
                                    _rv[i, j] = elements[j];
                                }
                            }
                        }
                        return (object)_rv;
                    }
                case "Single[,]":
                    {
                        var _po = GetPo(pyobj);
                        var _len = ToCsharp<int>(pyobj.len);
                        var _rv = CreateInitialMultidimensionalArray<float>(_len, pyobj[0].len);
                        for (var i = 0; i < _len; i++)
                        {
                            if (_len == 1)
                            {
                                _rv[i, 0] = (float)ToCsharp<float>(_po);
                            }
                            else
                            {
                                var elements = (float[])ToCsharp<float[]>(_po[i]);
                                for (int j = 0; j < elements.Length; j++)
                                {
                                    _rv[i, j] = elements[j];
                                }
                            }
                        }
                        return (object)_rv;
                    }
                case "Tuple`2":
                {
                    var tuple = ToPython(pyobj);
                    return new Tuple<int, int>((int)ToCsharp<int>(tuple[0]), (int)ToCsharp<int>(tuple[1]));
                }
                case "Tuple`3":
                {
                    var tuple = ToPython(pyobj);
                    return new Tuple<int, int, int>((int)ToCsharp<int>(tuple[0]), (int)ToCsharp<int>(tuple[1]), (int)ToCsharp<int>(tuple[2]));
                }
                default:
                    var pyClass = $"{pyobj.__class__}";
                    if (pyClass == "<class 'str'>") return (T)(object)pyobj.ToString();
                    if (pyClass.StartsWith("<class 'Cupy")) return (pyobj.item() as PyObject).As<T>();
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

        private static T ToCsharpInternal<T>(dynamic pyobj)
        {
            switch (typeof(T).Name)
            {
                // types from 'ToCsharpConversions'
                case "Dtype": return (T)(object)new Dtype(pyobj);
                case "Matrix": return (T)(object)new Matrix(pyobj);
                case "Boolean": return Boolean.Parse(pyobj.ToString());
                case "Int16": return Int16.Parse(pyobj.ToString());
                case "Int32": return Int32.Parse(pyobj.ToString());
                case "Int64": return Int64.Parse(pyobj.ToString());
                case "UInt16": return UInt16.Parse(pyobj.ToString());
                case "UInt32": return UInt32.Parse(pyobj.ToString());
                case "UInt64": return UInt64.Parse(pyobj.ToString());
                case "Single": return float.Parse(pyobj.ToString());
                case "Double": return double.Parse(pyobj.ToString());
                case "Complex": return ParseComplex(pyobj.ToString());
                case "NDarray": return (T)(object)new NDarray(pyobj);
            }

            throw new NotSupportedException();
        }

        private static Complex ParseComplex(string input)
        {
            Regex regex = new Regex(@"\s*([-+]?\d+\.?\d*)\s*([-+]\s*\d+\.?\d*)[ij]\s*", RegexOptions.IgnoreCase);
            Match match = regex.Match(input);

            if (match.Success)
            {
                double realPart = double.Parse(match.Groups[1].Value, CultureInfo.InvariantCulture);
                double imaginaryPart = double.Parse(match.Groups[2].Value, CultureInfo.InvariantCulture);
                return new Complex(realPart, imaginaryPart);
            }
            else
            {
                throw new FormatException("Invalid complex number format.");
            }
        }

        private static T[,] CreateInitialMultidimensionalArray<T>(int iCount, int jCount)
        {
            T[,] ret = new T[iCount, jCount];
            for (int i = 0; i < iCount; i++)
            {
                for (int j = 0; j < jCount; j++)
                {
                    ret[i, j] = default(T);
                }
            }
            return ret;
        }

        private static int GetLen(dynamic _po)
        {
            int _len = 0;
            if ((_po as PyObject).HasAttr("__len__"))
            {
                var a = ToCsharp<NDarray>(_po);
                _len = a.len;
            }
            else
            {
                _len = _po.size;
            }

            return _len;
        }

        private static PyObject GetPo(dynamic pyobj)
        {
            return (pyobj is PythonObject) ? (pyobj as PythonObject).PyObject : (PyObject)pyobj;
        }
        private static PyDict ToDict(Dictionary<string, NDarray> d)
        {
            var dict = new PyDict();
            foreach (var pair in d)
                dict[new PyString(pair.Key)] = pair.Value.self;
            return dict;
        }

        public bool Equals(NDarray other)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return CupyNDarray.Equals(other.CupyNDarray);
            else
                return NumpyNDarray.Equals(other.NumpyNDarray);
        }

        public override bool Equals(object obj)
        {
            if (obj is Variable v)
            {
                return xp.equal(this, v.Data).all();
            }
            else if (obj is NDarray arr)
            {
                return xp.equal(this, arr).all();
            }
            return false;
        }

        public override string ToString() => repr;

        public void __setstate__(int version, Shape shape, Dtype dtype, bool isFortran, string rawdata)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                CupyNDarray.__setstate__(version, shape.CupyShape, dtype.CupyDtype, isFortran, rawdata);
            else
                NumpyNDarray.__setstate__(version, shape.NumpyShape, dtype.NumpyDtype, isFortran, rawdata);
        }

        public NDarray abs(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.abs(@out.CupyNDarray, where.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.abs(@out.NumpyNDarray, where.NumpyNDarray));
        }

        public T asscalar<T>()
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return CupyNDarray.asscalar<T>();
            else
                return NumpyNDarray.asscalar<T>();
        }

        public NDarray astype(Dtype dtype, string order = null, string casting = null, bool? subok = null, bool? copy = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
            {
                return new NDarray(CupyNDarray.astype(dtype.CupyDtype, order, casting, subok, copy));
            }
            else
            {
                try
                {
                    Push(ArrayMode.np);
                    return new NDarray(NumpyNDarray.astype(dtype.NumpyDtype, order, casting, subok, copy));
                }
                finally
                {
                    Pop();
                }
            }
        }

        public NDarray byteswap(bool? inplace = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.byteswap(inplace));
            else
                return new NDarray(NumpyNDarray.byteswap(inplace));
        }

        public NDarray copy(string order = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.copy(order));
            else
                return new NDarray(NumpyNDarray.copy(order));
        }

        public NDarray divmod(ValueType obj)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.divmod(obj));
            else
                return new NDarray(NumpyNDarray.divmod(obj));
        }

        public void dump(string file)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                CupyNDarray.dump(file);
            else
                NumpyNDarray.dump(file);
        }

        public void dumps()
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                CupyNDarray.dumps();
            else
                NumpyNDarray.dumps();
        }

        public void Dispose()
        {
            CupyNDarray?.Dispose();
            NumpyNDarray?.Dispose();
        }

        public NDarray<bool> equals(ValueType valueType)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray<bool>(CupyNDarray.equals(valueType));
            else
                return new NDarray<bool>(NumpyNDarray.equals(valueType));
        }

        public NDarray<bool> equals(NDarray obj)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray<bool>(CupyNDarray.equals(obj.CupyNDarray));
            else
                return new NDarray<bool>(NumpyNDarray.equals(obj.NumpyNDarray));
        }

        public void fill(ValueType value)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                CupyNDarray.fill(value);
            else
                NumpyNDarray.fill(value);
        }

        public NDarray flatten(string order = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.flatten(order));
            else
                return new NDarray(NumpyNDarray.flatten(order));
        }

        public NDarray floordiv(NDarray a, ValueType obj)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.floordiv(a.CupyNDarray, obj));
            else
                return new NDarray(NumpyNDarray.floordiv(a.NumpyNDarray, obj));
        }

        public void getfield(Dtype dtype, int offset)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                CupyNDarray.getfield(dtype.CupyDtype, offset);
            else
                NumpyNDarray.getfield(dtype.NumpyDtype, offset);
        }

        public T GetData<T>()
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return CupyNDarray.GetData<T>();
            else
            {
                var type = typeof(T);
                var elementType = type.GetElementType();

                MethodInfo method = NumpyNDarray.GetType().GetMethod(nameof(GetData));
                MethodInfo generic = method.MakeGenericMethod(elementType);
                return (T)generic.Invoke(NumpyNDarray, null);
            }
        }

        public int GetHashCode()
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return CupyNDarray.GetHashCode();
            else
                return NumpyNDarray.GetHashCode();
        }

        public NDarray iadd(NDarray obj)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.iadd(obj.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.iadd(obj.NumpyNDarray));
        }

        public NDarray iadd(ValueType obj)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.iadd(obj));
            else
                return new NDarray(NumpyNDarray.iadd(obj));
        }

        public NDarray iand(NDarray obj)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.iand(obj.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.iand(obj.NumpyNDarray));
        }

        public NDarray iand(ValueType obj)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.iand(obj));
            else
                return new NDarray(NumpyNDarray.iand(obj));
        }

        public NDarray idiv(NDarray obj)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.idiv(obj.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.idiv(obj.NumpyNDarray));
        }

        public NDarray idiv(ValueType obj)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.idiv(obj));
            else
                return new NDarray(NumpyNDarray.idiv(obj));
        }

        public NDarray ifloordiv(NDarray obj)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.ifloordiv(obj.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.ifloordiv(obj.NumpyNDarray));
        }

        public NDarray ifloordiv(ValueType obj)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.ifloordiv(obj));
            else
                return new NDarray(NumpyNDarray.ifloordiv(obj));
        }

        public NDarray ilshift(NDarray obj)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.ilshift(obj.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.ilshift(obj.NumpyNDarray));
        }

        public NDarray ilshift(int obj)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.ilshift(obj));
            else
                return new NDarray(NumpyNDarray.ilshift(obj));
        }

        public NDarray imod(NDarray obj)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.imod(obj.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.imod(obj.NumpyNDarray));
        }

        public NDarray imod(ValueType obj)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.imod(obj));
            else
                return new NDarray(NumpyNDarray.imod(obj));
        }

        public NDarray imul(NDarray obj)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.imul(obj.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.imul(obj.NumpyNDarray));
        }

        public NDarray imul(ValueType obj)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.imul(obj));
            else
                return new NDarray(NumpyNDarray.imul(obj));
        }

        public NDarray ior(NDarray obj)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.ior(obj.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.ior(obj.NumpyNDarray));
        }

        public NDarray ior(ValueType obj)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.ior(obj));
            else
                return new NDarray(NumpyNDarray.ior(obj));
        }

        public NDarray ipow(NDarray obj)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.ipow(obj.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.ipow(obj.NumpyNDarray));
        }

        public NDarray ipow(ValueType obj)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.ipow(obj));
            else
                return new NDarray(NumpyNDarray.ipow(obj));
        }

        public NDarray irshift(NDarray obj)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.irshift(obj.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.irshift(obj.NumpyNDarray));
        }

        public NDarray irshift(int obj)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.irshift(obj));
            else
                return new NDarray(NumpyNDarray.irshift(obj));
        }

        public NDarray isub(NDarray obj)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.isub(obj.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.isub(obj.NumpyNDarray));
        }

        public NDarray isub(ValueType obj)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.isub(obj));
            else
                return new NDarray(NumpyNDarray.isub(obj));
        }

        public T item<T>(params int[] args)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return CupyNDarray.item<T>(args);
            else
                return NumpyNDarray.item<T>(args);
        }

        public void itemset(params object[] args)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                CupyNDarray.itemset(args);
            else
                NumpyNDarray.itemset(args);
        }

        public NDarray itruediv(NDarray obj)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.itruediv(obj.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.itruediv(obj.NumpyNDarray));
        }

        public NDarray itruediv(ValueType obj)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.itruediv(obj));
            else
                return new NDarray(NumpyNDarray.itruediv(obj));
        }

        public NDarray ixor(NDarray obj)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.ixor(obj.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.ixor(obj.NumpyNDarray));
        }

        public NDarray ixor(ValueType obj)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.ixor(obj));
            else
                return new NDarray(NumpyNDarray.ixor(obj));
        }

        public NDarray max(int[] axis = null, NDarray @out = null, bool? keepdims = null, ValueType initial = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.max(axis, @out.CupyNDarray, keepdims, initial));
            else
                return new NDarray(NumpyNDarray.max(axis, @out.NumpyNDarray, keepdims, initial));
        }

        public NDarray min(int[] axis = null, NDarray @out = null, bool? keepdims = null, ValueType initial = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.min(axis, @out.CupyNDarray, keepdims));
            else
                return new NDarray(NumpyNDarray.min(axis, @out.NumpyNDarray, keepdims, initial));
        }

        public NDarray<bool> not_equals(ValueType valueType)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray<bool>(CupyNDarray.not_equals(valueType));
            else
                return new NDarray<bool>(NumpyNDarray.not_equals(valueType));
        }

        public NDarray<bool> not_equals(NDarray obj)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray<bool>(CupyNDarray.not_equals(obj.CupyNDarray));
            else
                return new NDarray<bool>(NumpyNDarray.not_equals(obj.NumpyNDarray));
        }

        public NDarray pow(ValueType obj)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.pow(obj));
            else
                return new NDarray(NumpyNDarray.pow(obj));
        }

        public NDarray reshape(params int[] newshape)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.reshape(newshape));
            else
                return new NDarray(NumpyNDarray.reshape(newshape));
        }

        public void resize(Shape new_shape, bool? refcheck = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                CupyNDarray.resize(new_shape.CupyShape, refcheck);
            else
                NumpyNDarray.resize(new_shape.NumpyShape, refcheck);
        }

        public void setflags(bool? write = null, bool? align = null, bool? uic = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                CupyNDarray.setflags(write, align, uic);
            else
                NumpyNDarray.setflags(write, align, uic);
        }

        public byte[] tobytes(string order = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return CupyNDarray.tobytes(order);
            else
                return NumpyNDarray.tobytes(order);
        }

        public void tobytes(string fid, string sep, string format)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                CupyNDarray.tofile(fid, sep, format);
            else
                NumpyNDarray.tofile(fid, sep, format);
        }

        public byte[] tostring(string order = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return CupyNDarray.tostring(order);
            else
                return NumpyNDarray.tostring(order);
        }

        public NDarray transpose(params int[] axes)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.transpose(axes));
            else
                return new NDarray(NumpyNDarray.transpose(axes));
        }

        public void view(Dtype dtype = null, Type type = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                CupyNDarray.view(dtype.CupyDtype, type);
            else
                NumpyNDarray.view(dtype.NumpyDtype, type);
        }

        #region Extension Methods

        public NDarray prod(Axis axis = null, Dtype dtype = null, NDarray @out = null, bool? keepdims = null, ValueType initial = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.prod(CupyNDarray, axis?.CupyAxis, dtype?.CupyDtype, @out?.CupyNDarray, keepdims));
            else
                return new NDarray(np.np.prod(NumpyNDarray, axis?.NumpyAxis, dtype?.NumpyDtype, @out?.NumpyNDarray, keepdims, initial));
        }

        public Dtype GetDtype(object obj)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new Dtype(Cupy.DtypeExtensions.GetDtype(CupyNDarray));
            else
                return new Dtype(Numpy.DtypeExtensions.GetDtype(NumpyNDarray));
        }

        public NDarray absolute(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.absolute(@out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.absolute(@out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray add(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.add(@out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.add(@out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray all(Axis axis, NDarray @out = null, bool? keepdims = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.all(axis.CupyAxis, @out?.CupyNDarray, keepdims));
            else
                return new NDarray(NumpyNDarray.all(axis.NumpyAxis, @out?.NumpyNDarray, keepdims));
        }

        public NDarray allclose(NDarray a, float rtol = 1e-05f, float atol = 1e-08f, bool equal_nan = false)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.allclose(CupyNDarray, a.CupyNDarray, rtol, atol, equal_nan));
            else
                return new NDarray(np.np.allclose(NumpyNDarray, a.NumpyNDarray, rtol, atol, equal_nan));
        }

        public NDarray amax(Axis axis = null, NDarray @out = null, bool? keepdims = null, ValueType initial = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.amax(CupyNDarray, axis?.CupyAxis, @out?.CupyNDarray, keepdims, initial));
            else
                return new NDarray(np.np.amax(NumpyNDarray, axis?.NumpyAxis, @out?.NumpyNDarray, keepdims, initial));
        }

        public NDarray amin(Axis axis = null, NDarray @out = null, bool? keepdims = null, ValueType initial = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.amin(CupyNDarray, axis?.CupyAxis, @out?.CupyNDarray, keepdims));
            else
                return new NDarray(np.np.amin(NumpyNDarray, axis?.NumpyAxis, @out?.NumpyNDarray, keepdims, initial));
        }

        public NDarray angle(bool deg = false)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.angle(CupyNDarray, deg));
            else
                return new NDarray(np.np.angle(NumpyNDarray, deg));
        }

        public bool any()
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return cp.cp.any(CupyNDarray);
            else
                return np.np.any(NumpyNDarray);
        }

        public NDarray any(Axis axis, NDarray @out = null, bool? keepdims = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.any(CupyNDarray, axis?.CupyAxis, @out?.CupyNDarray, keepdims));
            else
                return new NDarray(np.np.any(NumpyNDarray, axis?.NumpyAxis, @out?.NumpyNDarray, keepdims));
        }

        public NDarray append(NDarray values, int? axis = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.append(CupyNDarray, values?.CupyNDarray, axis));
            else
                return new NDarray(np.np.append(NumpyNDarray, values?.NumpyNDarray, axis));
        }

        public NDarray arccos(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.arccos(CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(np.np.arccos(NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray arccosh(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.arccosh(CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(np.np.arccosh(NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray arcsin(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.arcsin(CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(np.np.arcsin(NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray arcsinh(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.arcsinh(CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(np.np.arcsinh(NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray arctan(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.arctan(CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(np.np.arctan(NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray arctan2(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.arctan2(CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(np.np.arctan2(NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray arctanh(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.arctanh(CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(np.np.arctanh(NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray argmax(int? axis = null, NDarray @out = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.argmax(CupyNDarray, axis, @out?.CupyNDarray));
            else
                return new NDarray(np.np.argmax(NumpyNDarray, axis, @out?.NumpyNDarray));
        }

        public NDarray argmin(int? axis = null, NDarray @out = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.argmin(CupyNDarray, axis, @out?.CupyNDarray));
            else
                return new NDarray(np.np.argmin(NumpyNDarray, axis, @out?.NumpyNDarray));
        }

        public NDarray argpartition(int[] kth, int? axis = -1, string kind = "introselect", string order = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.argpartition(CupyNDarray, kth, axis, kind, order));
            else
                return new NDarray(np.np.argpartition(NumpyNDarray, kth, axis, kind, order));
        }

        public NDarray argsort(int? axis = -1, string kind = "quicksort", string order = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.argsort(CupyNDarray, axis, kind, order));
            else
                return new NDarray(np.np.argsort(NumpyNDarray, axis, kind, order));
        }

        public NDarray argwhere()
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.argwhere(CupyNDarray));
            else
                return new NDarray(np.np.argwhere(NumpyNDarray));
        }

        public NDarray around(int? decimals = 0, NDarray @out = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.around(CupyNDarray, decimals, @out?.CupyNDarray));
            else
                return new NDarray(np.np.around(NumpyNDarray, decimals, @out?.NumpyNDarray));
        }

        public bool array_equal(NDarray a2)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return cp.cp.array_equal(CupyNDarray, a2.CupyNDarray);
            else
                return np.np.array_equal(NumpyNDarray, a2.NumpyNDarray);
        }

        public bool array_equiv(NDarray a2)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return cp.cp.array_equiv(CupyNDarray, a2.CupyNDarray);
            else
                return np.np.array_equiv(NumpyNDarray, a2.NumpyNDarray);
        }

        public string array_repr(int? max_line_width = null, int? precision = null, bool? suppress_small = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return cp.cp.array_repr(CupyNDarray, max_line_width, precision, suppress_small);
            else
                return np.np.array_repr(NumpyNDarray, max_line_width, precision, suppress_small);
        }

        public void array_str(int? max_line_width = null, int? precision = null, bool? suppress_small = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                cp.cp.array_str(CupyNDarray, max_line_width, precision, suppress_small);
            else
                np.np.array_str(NumpyNDarray, max_line_width, precision, suppress_small);
        }

        public NDarray asarray_chkfinite(Dtype dtype = null, string order = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.asarray_chkfinite(CupyNDarray, dtype?.CupyDtype, order));
            else
                return new NDarray(np.np.asarray_chkfinite(NumpyNDarray, dtype?.NumpyDtype, order));
        }

        public NDarray asfarray(Dtype dtype = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.asfarray(CupyNDarray, dtype?.CupyDtype));
            else
                return new NDarray(np.np.asfarray(NumpyNDarray, dtype?.NumpyDtype));
        }

        public NDarray asfortranarray(Dtype dtype = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.asfortranarray(CupyNDarray, dtype?.CupyDtype));
            else
                return new NDarray(np.np.asfortranarray(NumpyNDarray, dtype?.NumpyDtype));
        }

        public NDarray<double> average(Axis axis, NDarray weights = null, bool? returned = false)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray<double>(cp.cp.average(CupyNDarray, axis.CupyAxis, weights?.CupyNDarray, returned));
            else
                return new NDarray<double>(np.np.average(NumpyNDarray, axis.NumpyAxis, weights?.NumpyNDarray, returned));
        }

        public NDarray bincount(NDarray weights = null, int? minlength = 0)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.bincount(CupyNDarray, weights?.CupyNDarray, minlength));
            else
                return new NDarray(np.np.bincount(NumpyNDarray, weights?.NumpyNDarray, minlength));
        }

        public NDarray bitwise_and(NDarray x1, NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.bitwise_and(CupyNDarray, x1.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(np.np.bitwise_and(NumpyNDarray, x1.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray bitwise_or(NDarray x1, NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.bitwise_or(CupyNDarray, x1.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(np.np.bitwise_or(NumpyNDarray, x1.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray bitwise_xor(NDarray x1, NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.bitwise_xor(CupyNDarray, x1.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(np.np.bitwise_xor(NumpyNDarray, x1.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray broadcast(NDarray in1)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.broadcast(CupyNDarray, in1.CupyNDarray));
            else
                return new NDarray(np.np.broadcast(NumpyNDarray, in1.NumpyNDarray));
        }

        public NDarray broadcast_to(Shape shape, bool? subok = false)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.broadcast_to(CupyNDarray, shape.CupyShape, subok));
            else
                return new NDarray(np.np.broadcast_to(NumpyNDarray, shape.NumpyShape, subok));
        }

        public NDarray cbrt(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.cbrt(CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(np.np.cbrt(NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray ceil(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.ceil(CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(np.np.ceil(NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray clip(NDarray a_min, NDarray a_max, NDarray @out = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.clip(CupyNDarray, a_min.CupyNDarray, a_max.CupyNDarray, @out?.CupyNDarray));
            else
                return new NDarray(np.np.clip(NumpyNDarray, a_min.NumpyNDarray, a_max.NumpyNDarray, @out?.NumpyNDarray));
        }

        public Dtype common_type(NDarray array1)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new Dtype(cp.cp.common_type(CupyNDarray, array1.CupyNDarray));
            else
                return new Dtype(np.np.common_type(NumpyNDarray, array1.NumpyNDarray));
        }

        public NDarray conj(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.conj(CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(np.np.conj(NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray convolve(NDarray v, string mode = "full")
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.convolve(CupyNDarray, v.CupyNDarray, mode));
            else
                return new NDarray(np.np.convolve(NumpyNDarray, v.NumpyNDarray, mode));
        }

        public NDarray copysign(NDarray x2, NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.copysign(CupyNDarray, x2.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(np.np.copysign(NumpyNDarray, x2.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray corrcoef(NDarray y = null, bool? rowvar = true)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.corrcoef(CupyNDarray, y?.CupyNDarray, rowvar));
            else
                return new NDarray(np.np.corrcoef(NumpyNDarray, y?.NumpyNDarray, rowvar));
        }

        public NDarray correlate(NDarray a, string mode = "valid")
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.correlate(CupyNDarray, a?.CupyNDarray, mode));
            else
                return new NDarray(np.np.correlate(NumpyNDarray, a?.NumpyNDarray, mode));
        }

        public NDarray cos(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.cos(CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(np.np.cos(NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray cosh(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.cosh(CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(np.np.cosh(NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray count_nonzero()
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.count_nonzero(CupyNDarray));
            else
                return new NDarray(np.np.count_nonzero(NumpyNDarray));
        }

        public NDarray count_nonzero(Axis axis)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.count_nonzero(CupyNDarray, axis.CupyAxis));
            else
                return new NDarray(np.np.count_nonzero(NumpyNDarray, axis.NumpyAxis));
        }

        public NDarray cov(NDarray y = null, bool? rowvar = true, bool? bias = false, int? ddof = null, NDarray fweights = null, NDarray aweights = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.cov(CupyNDarray, y?.CupyNDarray, rowvar, bias, ddof, fweights?.CupyNDarray, aweights?.CupyNDarray));
            else
                return new NDarray(np.np.cov(NumpyNDarray, y?.NumpyNDarray, rowvar, bias, ddof, fweights?.NumpyNDarray, aweights?.NumpyNDarray));
        }

        public NDarray cross(NDarray b, int? axisa = -1, int? axisb = -1, int? axisc = -1, int? axis = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.cross(CupyNDarray, b.CupyNDarray, axisa, axisb, axisc, axis));
            else
                return new NDarray(np.np.cross(NumpyNDarray, b.NumpyNDarray, axisa, axisb, axisc, axis));
        }

        public NDarray cumprod(int? axis = null, Dtype dtype = null, NDarray @out = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.cumprod(CupyNDarray, axis, dtype?.CupyDtype, @out?.CupyNDarray));
            else
                return new NDarray(np.np.cumprod(NumpyNDarray, axis, dtype?.NumpyDtype, @out?.NumpyNDarray));
        }

        public NDarray cumsum(int? axis = null, Dtype dtype = null, NDarray @out = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.cumsum(CupyNDarray, axis, dtype?.CupyDtype, @out?.CupyNDarray));
            else
                return new NDarray(np.np.cumsum(NumpyNDarray, axis, dtype?.NumpyDtype, @out?.NumpyNDarray));
        }

        public NDarray deg2rad(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.deg2rad(CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(np.np.deg2rad(NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray degrees(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.degrees(CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(np.np.degrees(NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray delete(Slice obj, int? axis = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.delete(CupyNDarray, obj.CupySlice, axis));
            else
                return new NDarray(np.np.delete(NumpyNDarray, obj.NumpySlice, axis));
        }

        public void diag_indices_from()
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                cp.cp.diag_indices_from(CupyNDarray);
            else
                np.np.diag_indices_from(NumpyNDarray);
        }

        public NDarray diagonal(int? offset = 0, int? axis1 = 0, int? axis2 = 1)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.diagonal(CupyNDarray, offset, axis1, axis2));
            else
                return new NDarray(np.np.diagonal(NumpyNDarray, offset, axis1, axis2));
        }

        public NDarray diff(int? n = 1, int? axis = -1, NDarray append = null, NDarray prepend = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.diff(CupyNDarray, n, axis, append?.CupyNDarray, prepend?.CupyNDarray));
            else
                return new NDarray(np.np.diff(NumpyNDarray, n, axis, append?.NumpyNDarray, prepend?.NumpyNDarray));
        }

        public NDarray digitize(NDarray bins, bool? right = false)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.digitize(CupyNDarray, bins.CupyNDarray, right));
            else
                return new NDarray(np.np.digitize(NumpyNDarray, bins.NumpyNDarray, right));
        }

        public NDarray divide(NDarray x2, NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.divide(CupyNDarray, x2.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(np.np.divide(NumpyNDarray, x2.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray dot(NDarray b, NDarray @out = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.dot(CupyNDarray, b.CupyNDarray, @out?.CupyNDarray));
            else
                return new NDarray(np.np.dot(NumpyNDarray, b.NumpyNDarray, @out?.NumpyNDarray));
        }

        public NDarray ediff1d(NDarray to_end = null, NDarray to_begin = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.ediff1d(CupyNDarray, to_end?.CupyNDarray, to_begin?.CupyNDarray));
            else
                return new NDarray(np.np.ediff1d(NumpyNDarray, to_end?.NumpyNDarray, to_begin?.NumpyNDarray));
        }

        public NDarray equal(NDarray x1, NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.equal(CupyNDarray, x1.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(np.np.equal(NumpyNDarray, x1.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray exp(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.exp(CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(np.np.exp(NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray exp2(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.exp2(CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(np.np.exp2(NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray expand_dims(int axis)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.expand_dims(CupyNDarray, axis));
            else
                return new NDarray(np.np.expand_dims(NumpyNDarray, axis));
        }

        public NDarray expm1(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.expm1(CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(np.np.expm1(NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray extract(NDarray arr)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.extract(CupyNDarray, arr.CupyNDarray));
            else
                return new NDarray(np.np.extract(NumpyNDarray, arr.NumpyNDarray));
        }

        public NDarray fabs(NDarray arr)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.fabs(CupyNDarray, arr.CupyNDarray));
            else
                return new NDarray(np.np.fabs(NumpyNDarray, arr.NumpyNDarray));
        }

        public void fill_diagonal(ValueType val, bool wrap = false)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                cp.cp.fill_diagonal(CupyNDarray, val, wrap);
            else
                np.np.fill_diagonal(NumpyNDarray, val, wrap);
        }

        public NDarray fix(NDarray y = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.fix(CupyNDarray, y?.CupyNDarray));
            else
                return new NDarray(np.np.fix(NumpyNDarray, y?.NumpyNDarray));
        }

        public NDarray flatnonzero()
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.flatnonzero(CupyNDarray));
            else
                return new NDarray(np.np.flatnonzero(NumpyNDarray));
        }

        public NDarray flip(Axis axis = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.flip(CupyNDarray, axis?.CupyAxis));
            else
                return new NDarray(np.np.flip(NumpyNDarray, axis?.NumpyAxis));
        }

        public NDarray fliplr()
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.fliplr(CupyNDarray));
            else
                return new NDarray(np.np.fliplr(NumpyNDarray));
        }

        public NDarray flipud()
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.flipud(CupyNDarray));
            else
                return new NDarray(np.np.flipud(NumpyNDarray));
        }

        public NDarray float_power(NDarray x2, NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.float_power(CupyNDarray, x2.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(np.np.float_power(NumpyNDarray, x2.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray floor(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.floor(CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(np.np.floor(NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray floor_divide(NDarray x2, NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.floor_divide(CupyNDarray, x2.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(np.np.floor_divide(NumpyNDarray, x2.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray fmax(NDarray x1, NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.fmax(CupyNDarray, x1.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(np.np.fmax(NumpyNDarray, x1.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray fmin(NDarray x1, NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.fmin(CupyNDarray, x1.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(np.np.fmin(NumpyNDarray, x1.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray fmod(NDarray x2, NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.fmod(CupyNDarray, x2.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(np.np.fmod(NumpyNDarray, x2.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public (NDarray, NDarray) frexp(NDarray out1 = null, NDarray out2 = null, NDarray @out = null,
            NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
            {
                var ret = cp.cp.frexp(CupyNDarray, out1?.CupyNDarray, out2?.CupyNDarray, @out?.CupyNDarray,
                    where?.CupyNDarray);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2));
            }
            else
            {
                var ret = np.np.frexp(NumpyNDarray, out1?.NumpyNDarray, out2?.NumpyNDarray, @out?.NumpyNDarray,
                    where?.NumpyNDarray);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2));
            }
        }

        public NDarray fv(NDarray nper, NDarray pmt, NDarray pv, string when = "end")
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.fv(CupyNDarray, nper.CupyNDarray, pmt.CupyNDarray, pv.CupyNDarray, when));
            else
                return new NDarray(np.np.fv(NumpyNDarray, nper.NumpyNDarray, pmt.NumpyNDarray, pv.NumpyNDarray, when));
        }

        public NDarray gcd(NDarray x1)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.gcd(CupyNDarray, x1.CupyNDarray));
            else
                return new NDarray(np.np.gcd(NumpyNDarray, x1.NumpyNDarray));
        }

        public NDarray greater(NDarray x1, NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.greater(CupyNDarray, x1.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(np.np.greater(NumpyNDarray, x1.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray greater_equal(NDarray x1, NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.greater_equal(CupyNDarray, x1.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(np.np.greater_equal(NumpyNDarray, x1.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray heaviside(NDarray x2, NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.heaviside(CupyNDarray, x2.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(np.np.heaviside(NumpyNDarray, x2.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public (NDarray, NDarray) histogram(int? bins = null, (float, float)? range = null, bool? normed = null, NDarray weights = null, bool? density = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
            {
                var ret = cp.cp.histogram(CupyNDarray, bins, range, normed, weights?.CupyNDarray, density);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2));
            }
            else
            {
                var ret = np.np.histogram(NumpyNDarray, bins, range, normed, weights?.NumpyNDarray, density);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2));
            }
        }

        public (NDarray, NDarray) histogram(NDarray bins = null, (float, float)? range = null, bool? normed = null, NDarray weights = null, bool? density = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
            {
                var ret = cp.cp.histogram(CupyNDarray, bins?.CupyNDarray, range, normed, weights?.CupyNDarray, density);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2));
            }
            else
            {
                var ret = np.np.histogram(NumpyNDarray, bins?.NumpyNDarray, range, normed, weights?.NumpyNDarray, density);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2));
            }
        }

        public (NDarray, NDarray) histogram(List<string> bins = null, (float, float)? range = null, bool? normed = null, NDarray weights = null, bool? density = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
            {
                var ret = cp.cp.histogram(CupyNDarray, bins, range, normed, weights?.CupyNDarray, density);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2));
            }
            else
            {
                var ret = np.np.histogram(NumpyNDarray, bins, range, normed, weights?.NumpyNDarray, density);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2));
            }
        }

        public (NDarray, NDarray) histogram2d(NDarray y, int? bins = null, (float, float)? range = null, bool? density = null, bool? normed = null, NDarray weights = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
            {
                var ret = cp.cp.histogram2d(CupyNDarray, y.CupyNDarray, bins, range, density, normed, weights?.CupyNDarray);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2));
            }
            else
            {
                var ret = np.np.histogram2d(NumpyNDarray, y.NumpyNDarray, bins, range, density, normed, weights?.NumpyNDarray);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2));
            }
        }

        public (NDarray, NDarray) histogram2d(NDarray y, NDarray bins = null, (float, float)? range = null, bool? density = null, bool? normed = null, NDarray weights = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
            {
                var ret = cp.cp.histogram2d(CupyNDarray, y.CupyNDarray, bins?.CupyNDarray, range, density, normed, weights?.CupyNDarray);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2));
            }
            else
            {
                var ret = np.np.histogram2d(NumpyNDarray, y.NumpyNDarray, bins?.NumpyNDarray, range, density, normed, weights?.NumpyNDarray);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2));
            }
        }

        public (NDarray, NDarray) histogram2d(NDarray y, List<string> bins = null, (float, float)? range = null, bool? density = null, bool? normed = null, NDarray weights = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
            {
                var ret = cp.cp.histogram2d(CupyNDarray, y.CupyNDarray, bins, range, density, normed, weights?.CupyNDarray);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2));
            }
            else
            {
                var ret = np.np.histogram2d(NumpyNDarray, y.NumpyNDarray, bins, range, density, normed, weights?.NumpyNDarray);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2));
            }
        }

        public NDarray<float> histogram_bin_edges(int? bins = null, (float, float)? range = null, NDarray weights = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray<float>(cp.cp.histogram_bin_edges(CupyNDarray, bins, range, weights?.CupyNDarray));
            else
                return new NDarray<float>(np.np.histogram_bin_edges(NumpyNDarray, bins, range, weights?.NumpyNDarray));
        }

        public NDarray<float> histogram_bin_edges(NDarray bins = null, (float, float)? range = null, NDarray weights = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray<float>(cp.cp.histogram_bin_edges(CupyNDarray, bins?.CupyNDarray, range, weights?.CupyNDarray));
            else
                return new NDarray<float>(np.np.histogram_bin_edges(NumpyNDarray, bins?.NumpyNDarray, range, weights?.NumpyNDarray));
        }

        public NDarray<float> histogram_bin_edges(List<string> bins = null, (float, float)? range = null, NDarray weights = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray<float>(cp.cp.histogram_bin_edges(CupyNDarray, bins, range, weights?.CupyNDarray));
            else
                return new NDarray<float>(np.np.histogram_bin_edges(NumpyNDarray, bins, range, weights?.NumpyNDarray));
        }

        public (NDarray, NDarray) histogramdd(int? bins = null, (float, float)? range = null, bool? density = null, bool? normed = null, NDarray weights = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
            {
                var ret = cp.cp.histogramdd(CupyNDarray, bins, range, density, normed, weights?.CupyNDarray);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2));
            }
            else
            {
                var ret = np.np.histogramdd(NumpyNDarray, bins, range, density, normed, weights?.NumpyNDarray);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2));
            }
        }

        public (NDarray, NDarray) histogramdd(NDarray bins = null, (float, float)? range = null, bool? density = null, bool? normed = null, NDarray weights = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
            {
                var ret = cp.cp.histogramdd(CupyNDarray, bins.CupyNDarray, range, density, normed, weights?.CupyNDarray);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2));
            }
            else
            {
                var ret = np.np.histogramdd(NumpyNDarray, bins.NumpyNDarray, range, density, normed, weights?.NumpyNDarray);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2));
            }
        }

        public (NDarray, NDarray) histogramdd(List<string> bins = null, (float, float)? range = null, bool? density = null, bool? normed = null, NDarray weights = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
            {
                var ret = cp.cp.histogramdd(CupyNDarray, bins, range, density, normed, weights?.CupyNDarray);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2));
            }
            else
            {
                var ret = np.np.histogramdd(NumpyNDarray, bins, range, density, normed, weights?.NumpyNDarray);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2));
            }
        }

        public NDarray hypot(NDarray x1, NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(cp.cp.hypot(CupyNDarray, x1.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(np.np.hypot(NumpyNDarray, x1.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray i0()
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.i0());
            else
                return new NDarray(NumpyNDarray.i0());
        }

        public NDarray in1d(NDarray ar2, bool? assume_unique = false, bool? invert = false)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.in1d(ar2.CupyNDarray, assume_unique, invert));
            else
                return new NDarray(NumpyNDarray.in1d(ar2.NumpyNDarray, assume_unique, invert));
        }

        public NDarray inner(NDarray a)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.inner(a.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.inner(a.NumpyNDarray));
        }

        public (NDarray, NDarray, NDarray) intersect1d(NDarray a)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
            {
                var ret = CupyNDarray.intersect1d(a.CupyNDarray);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2), new NDarray(ret.Item3));
            }
            else
            {
                var ret = NumpyNDarray.intersect1d(a.NumpyNDarray);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2), new NDarray(ret.Item3));
            }
        }

        public NDarray invert(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.invert(@out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.invert(@out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray ipmt(NDarray per, NDarray nper, NDarray pv, NDarray fv = null, string when = "end")
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.ipmt(per.CupyNDarray, nper.CupyNDarray, pv.CupyNDarray, fv?.CupyNDarray, when));
            else
                return new NDarray(NumpyNDarray.ipmt(per.NumpyNDarray, nper.NumpyNDarray, pv.NumpyNDarray, fv?.NumpyNDarray, when));
        }

        public NDarray irr()
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.irr());
            else
                return new NDarray(NumpyNDarray.irr());
        }

        public NDarray isclose(NDarray a, float rtol = 1e-05f, float atol = 1e-08f, bool equal_nan = false)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.isclose(a.CupyNDarray, rtol, atol, equal_nan));
            else
                return new NDarray(NumpyNDarray.isclose(a.NumpyNDarray, rtol, atol, equal_nan));
        }

        public NDarray iscomplex()
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.iscomplex());
            else
                return new NDarray(NumpyNDarray.iscomplex());
        }

        public NDarray isfinite()
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.isfinite());
            else
                return new NDarray(NumpyNDarray.isfinite());
        }

        public NDarray isfortran()
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.isfortran());
            else
                return new NDarray(NumpyNDarray.isfortran());
        }

        public NDarray isin(NDarray test_elements, bool? assume_unique = false, bool? invert = false)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.isin(test_elements.CupyNDarray, assume_unique, invert));
            else
                return new NDarray(NumpyNDarray.isin(test_elements.NumpyNDarray, assume_unique, invert));
        }

        public NDarray<bool> isinf(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray<bool>(CupyNDarray.isinf(@out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray<bool>(NumpyNDarray.isinf(@out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray isnan(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.isnan(@out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.isnan(@out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray isnat(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.isnat(@out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.isnat(@out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray isneginf(NDarray @out = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.isneginf(@out?.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.isneginf(@out?.NumpyNDarray));
        }

        public NDarray isposinf(NDarray y = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.isposinf(y?.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.isposinf(y?.NumpyNDarray));
        }

        public NDarray isreal()
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.isreal());
            else
                return new NDarray(NumpyNDarray.isreal());
        }

        public NDarray kron(NDarray a)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.kron(a.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.kron(a.NumpyNDarray));
        }

        public NDarray lcm(NDarray x1)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.lcm(x1.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.lcm(x1.NumpyNDarray));
        }

        public NDarray ldexp(NDarray x2, NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.ldexp(x2.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.ldexp(x2.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray less(NDarray x1, NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.less(x1.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.less(x1.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray less_equal(NDarray x1, NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.less_equal(x1.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.less_equal(x1.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray lexsort(int? axis = -1)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.lexsort(axis));
            else
                return new NDarray(NumpyNDarray.lexsort(axis));
        }

        public NDarray log(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.log(@out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.log(@out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray log10(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.log10(@out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.log10(@out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray log1p(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.log1p(@out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.log1p(@out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray log2(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.log2(@out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.log2(@out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray logaddexp(NDarray x1, NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.logaddexp(x1.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.logaddexp(x1.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray logaddexp2(NDarray x1, NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.logaddexp2(x1.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.logaddexp2(x1.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray logical_and(NDarray x1, NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.logical_and(x1.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.logical_and(x1.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray logical_not(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.logical_not(@out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.logical_not(@out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray logical_or(NDarray x1, NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.logical_or(x1.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.logical_or(x1.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray logical_xor(NDarray x1, NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.logical_xor(x1.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.logical_xor(x1.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray matmul(NDarray x1, NDarray @out = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.matmul(x1.CupyNDarray, @out?.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.matmul(x1.NumpyNDarray, @out?.NumpyNDarray));
        }

        public NDarray maximum(NDarray x1, NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.maximum(x1.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.maximum(x1.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray<double> mean(Axis axis, Dtype dtype = null, NDarray @out = null, bool? keepdims = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray<double>(CupyNDarray.mean(axis.CupyAxis, dtype?.CupyDtype, @out?.CupyNDarray, keepdims));
            else
                return new NDarray<double>(NumpyNDarray.mean(axis.NumpyAxis, dtype?.NumpyDtype, @out?.NumpyNDarray, keepdims));
        }

        public NDarray<double> median(Axis axis, NDarray @out = null, bool? overwrite_input = false, bool? keepdims = false)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray<double>(CupyNDarray.median(axis.CupyAxis, @out?.CupyNDarray, overwrite_input, keepdims));
            else
                return new NDarray<double>(NumpyNDarray.median(axis.NumpyAxis, @out?.NumpyNDarray, overwrite_input, keepdims));
        }

        public Dtype min_scalar_type()
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new Dtype(CupyNDarray.min_scalar_type());
            else
                return new Dtype(NumpyNDarray.min_scalar_type());
        }

        public NDarray minimum(NDarray x1, NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.minimum(x1.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.minimum(x1.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray mirr(ValueType finance_rate, ValueType reinvest_rate)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.mirr(finance_rate, reinvest_rate));
            else
                return new NDarray(NumpyNDarray.mirr(finance_rate, reinvest_rate));
        }

        public NDarray mod(NDarray x2, NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.mod(x2.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.mod(x2.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public (NDarray, NDarray) modf(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
            {
                var ret = CupyNDarray.modf(@out?.CupyNDarray, where?.CupyNDarray);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2));
            }
            else
            {
                var ret = NumpyNDarray.modf(@out?.NumpyNDarray, where?.NumpyNDarray);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2));
            }
        }

        public NDarray moveaxis(int[] source, int[] destination)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.moveaxis(source, destination));
            else
                return new NDarray(NumpyNDarray.moveaxis(source, destination));
        }

        public NDarray msort()
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.msort());
            else
                return new NDarray(NumpyNDarray.msort());
        }

        public NDarray multiply(NDarray x1, NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.multiply(x1.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.multiply(x1.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public void nper(NDarray pmt, NDarray pv, NDarray fv = null, string when = "end")
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                CupyNDarray.nper(pmt.CupyNDarray, pv.CupyNDarray, fv?.CupyNDarray, when);
            else
                NumpyNDarray.nper(pmt.NumpyNDarray, pv.NumpyNDarray, fv?.NumpyNDarray, when);
        }

        public NDarray nan_to_num(bool? copy = true)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.nan_to_num(copy));
            else
                return new NDarray(NumpyNDarray.nan_to_num(copy));
        }

        public NDarray nanargmax(int? axis = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.nanargmax(axis));
            else
                return new NDarray(NumpyNDarray.nanargmax(axis));
        }

        public NDarray nanargmin(int? axis = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.nanargmin(axis));
            else
                return new NDarray(NumpyNDarray.nanargmin(axis));
        }

        public NDarray nancumprod(int? axis = null, Dtype dtype = null, NDarray @out = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.nancumprod(axis, dtype?.CupyDtype, @out?.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.nancumprod(axis, dtype?.NumpyDtype, @out?.NumpyNDarray));
        }

        public NDarray nancumsum(int? axis = null, Dtype dtype = null, NDarray @out = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.nancumsum(axis, dtype?.CupyDtype, @out?.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.nancumsum(axis, dtype?.NumpyDtype, @out?.NumpyNDarray));
        }

        public NDarray nanmax(Axis axis, NDarray @out = null, bool? keepdims = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.nanmax(axis.CupyAxis, @out?.CupyNDarray, keepdims));
            else
                return new NDarray(NumpyNDarray.nanmax(axis.NumpyAxis, @out?.NumpyNDarray, keepdims));
        }

        public NDarray<double> nanmean(Axis axis, Dtype dtype = null, NDarray @out = null, bool? keepdims = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray<double>(CupyNDarray.nanmean(axis.CupyAxis, dtype?.CupyDtype, @out?.CupyNDarray, keepdims));
            else
                return new NDarray<double>(NumpyNDarray.nanmean(axis.NumpyAxis, dtype?.NumpyDtype, @out?.NumpyNDarray, keepdims));
        }

        public NDarray<double> nanmedian(Axis axis, NDarray @out = null, bool? overwrite_input = false, bool? keepdims = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray<double>(CupyNDarray.nanmedian(axis.CupyAxis, @out?.CupyNDarray, overwrite_input, keepdims));
            else
                return new NDarray<double>(NumpyNDarray.nanmedian(axis.NumpyAxis, @out?.NumpyNDarray, overwrite_input, keepdims));
        }

        public NDarray nanmin(Axis axis, NDarray @out = null, bool? keepdims = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.nanmin(axis.CupyAxis, @out?.CupyNDarray, keepdims));
            else
                return new NDarray(NumpyNDarray.nanmin(axis.NumpyAxis, @out?.NumpyNDarray, keepdims));
        }

        public NDarray<double> nanpercentile(NDarray<float> q, Axis axis, NDarray @out = null, bool? overwrite_input = false, string interpolation = "linear", bool? keepdims = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray<double>(cp.cp.nanpercentile(CupyNDarray, q.CupyNDarray, axis.CupyAxis, @out?.CupyNDarray, overwrite_input, interpolation, keepdims));
            else
                return new NDarray<double>(np.np.nanpercentile(NumpyNDarray, q.NumpyNDarray, axis.NumpyAxis, @out?.NumpyNDarray, overwrite_input, interpolation, keepdims));
        }

        public NDarray nanpercentile(NDarray<float> q, NDarray @out = null, bool? overwrite_input = false, string interpolation = "linear")
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.nanpercentile(q.CupyNDarray, @out?.CupyNDarray, overwrite_input, interpolation));
            else
                return new NDarray(NumpyNDarray.nanpercentile(q.NumpyNDarray, @out?.NumpyNDarray, overwrite_input, interpolation));
        }

        public NDarray nanstd(Axis axis, Dtype dtype = null, NDarray @out = null, int? ddof = 0, bool? keepdims = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.nanstd(axis.CupyAxis, dtype?.CupyDtype, @out?.CupyNDarray, ddof, keepdims));
            else
                return new NDarray(NumpyNDarray.nanstd(axis.NumpyAxis, dtype?.NumpyDtype, @out?.NumpyNDarray, ddof, keepdims));
        }

        public NDarray nansum(Axis axis = null, Dtype dtype = null, NDarray @out = null, bool? keepdims = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.nansum(axis.CupyAxis, dtype?.CupyDtype, @out?.CupyNDarray, keepdims));
            else
                return new NDarray(NumpyNDarray.nansum(axis.NumpyAxis, dtype?.NumpyDtype, @out?.NumpyNDarray, keepdims));
        }

        public NDarray nanvar(Axis axis, Dtype dtype = null, NDarray @out = null, int? ddof = 0, bool? keepdims = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.nanvar(axis.CupyAxis, dtype?.CupyDtype, @out?.CupyNDarray, ddof, keepdims));
            else
                return new NDarray(NumpyNDarray.nanvar(axis.NumpyAxis, dtype?.NumpyDtype, @out?.NumpyNDarray, ddof, keepdims));
        }

        public void ndenumerate()
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                CupyNDarray.ndenumerate();
            else
                NumpyNDarray.ndenumerate();
        }

        public NDarray negative(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.negative(@out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.negative(@out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray nextafter(NDarray x2, NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.nextafter(x2.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.nextafter(x2.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray[] nonzero(NDarray x2)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return cp.cp.nonzero(this.CupyNDarray).Select(x => new NDarray(x)).ToArray();
            else
                return np.np.nonzero(this.NumpyNDarray).Select(x => new NDarray(x)).ToArray();
        }

        public NDarray not_equal(NDarray x1, NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.not_equal(x1.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.not_equal(x1.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray outer(NDarray b, NDarray @out = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.outer(b.CupyNDarray, @out?.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.outer(b.NumpyNDarray, @out?.NumpyNDarray));
        }

        public NDarray pv(NDarray nper, NDarray pmt, NDarray fv = null, string when = "end")
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.pv(nper.CupyNDarray, pmt.CupyNDarray, fv?.CupyNDarray, when));
            else
                return new NDarray(NumpyNDarray.pv(nper.NumpyNDarray, pmt.NumpyNDarray, fv?.NumpyNDarray, when));
        }

        public NDarray packbits(int? axis = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.packbits(axis));
            else
                return new NDarray(NumpyNDarray.packbits(axis));
        }

        public NDarray pad(NDarray pad_width, string mode, int[] stat_length = null, int[] constant_values = null, int[] end_values = null, string reflect_type = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.pad(pad_width.CupyNDarray, mode, stat_length, constant_values, end_values, reflect_type));
            else
                return new NDarray(NumpyNDarray.pad(pad_width.NumpyNDarray, mode, stat_length, constant_values, end_values, reflect_type));
        }

        public NDarray partition(int[] kth, int? axis = -1, string kind = "introselect", string order = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.partition(kth, axis, kind, order));
            else
                return new NDarray(NumpyNDarray.partition(kth, axis, kind, order));
        }

        public NDarray percentile(NDarray<float> q, Axis axis, NDarray @out = null, bool? overwrite_input = false, string interpolation = "linear", bool? keepdims = false)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.percentile(q.CupyNDarray, axis.CupyAxis, @out?.CupyNDarray, overwrite_input, interpolation, keepdims));
            else
                return new NDarray(NumpyNDarray.percentile(q.NumpyNDarray, axis.NumpyAxis, @out?.NumpyNDarray, overwrite_input, interpolation, keepdims));
        }

        public NDarray percentile(NDarray<float> q, NDarray @out = null, bool? overwrite_input = false, string interpolation = "linear")
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.percentile(q.CupyNDarray, @out?.CupyNDarray, overwrite_input, interpolation));
            else
                return new NDarray(NumpyNDarray.percentile(q.NumpyNDarray, @out?.NumpyNDarray, overwrite_input, interpolation));
        }

        public void place(NDarray mask, NDarray vals)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                CupyNDarray.place(mask.CupyNDarray, vals.CupyNDarray);
            else
                NumpyNDarray.place(mask.NumpyNDarray, vals.NumpyNDarray);
        }

        public NDarray pmt(NDarray nper, NDarray pv, NDarray fv = null, string when = "end")
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.pmt(nper.CupyNDarray, pv.CupyNDarray, fv?.CupyNDarray, when));
            else
                return new NDarray(NumpyNDarray.pmt(nper.NumpyNDarray, pv.NumpyNDarray, fv?.NumpyNDarray, when));
        }

        public NDarray positive()
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.positive());
            else
                return new NDarray(NumpyNDarray.positive());
        }

        public NDarray power(NDarray x2, NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.power(x2.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.power(x2.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public void ppmt(NDarray per, NDarray nper, NDarray pv, NDarray fv = null, string when = "end")
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                CupyNDarray.ppmt(per.CupyNDarray, nper.CupyNDarray, pv.CupyNDarray, fv?.CupyNDarray, when);
            else
                NumpyNDarray.ppmt(per.NumpyNDarray, nper.NumpyNDarray, pv.NumpyNDarray, fv?.NumpyNDarray, when);
        }

        public NDarray prod(Axis axis = null, Dtype dtype = null, NDarray @out = null, bool? keepdims = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.prod(axis?.CupyAxis, dtype?.CupyDtype, @out?.CupyNDarray, keepdims));
            else
                return new NDarray(NumpyNDarray.prod(axis?.NumpyAxis, dtype?.NumpyDtype, @out?.NumpyNDarray, keepdims));
        }

        public NDarray ptp(Axis axis = null, NDarray @out = null, bool? keepdims = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.ptp(axis?.CupyAxis, @out?.CupyNDarray, keepdims));
            else
                return new NDarray(NumpyNDarray.ptp(axis?.NumpyAxis, @out?.NumpyNDarray, keepdims));
        }

        public void put(NDarray ind, NDarray v, string mode = "raise")
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                CupyNDarray.put(ind.CupyNDarray, v.CupyNDarray, mode);
            else
                NumpyNDarray.put(ind.NumpyNDarray, v.NumpyNDarray, mode);
        }

        public void put_along_axis(NDarray indices, NDarray[] values, int axis)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                CupyNDarray.put_along_axis(indices.CupyNDarray, values.Select(v => v.CupyNDarray).ToArray(), axis);
            else
                NumpyNDarray.put_along_axis(indices.NumpyNDarray, values.Select(v => v.NumpyNDarray).ToArray(), axis);
        }

        public void putmask(NDarray mask, NDarray values)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                CupyNDarray.putmask(mask.CupyNDarray, values.CupyNDarray);
            else
                NumpyNDarray.putmask(mask.NumpyNDarray, values.NumpyNDarray);
        }

        public NDarray quantile(NDarray<float> q, Axis axis, NDarray @out = null, bool? overwrite_input = false, string interpolation = "linear", bool? keepdims = false)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.quantile(q.CupyNDarray, axis.CupyAxis, @out?.CupyNDarray, overwrite_input, interpolation, keepdims));
            else
                return new NDarray(NumpyNDarray.quantile(q.NumpyNDarray, axis.NumpyAxis, @out?.NumpyNDarray, overwrite_input, interpolation, keepdims));
        }

        public NDarray rad2deg(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.rad2deg(@out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.rad2deg(@out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray radians(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.radians(@out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.radians(@out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public void rate(NDarray pmt, NDarray pv, NDarray fv, string when = "end", double? guess = null, double? tol = null, int? maxiter = 100)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                CupyNDarray.rate(pmt.CupyNDarray, pv.CupyNDarray, fv.CupyNDarray, when, guess, tol, maxiter);
            else
                NumpyNDarray.rate(pmt.NumpyNDarray, pv.NumpyNDarray, fv.NumpyNDarray, when, guess, tol, maxiter);
        }

        public NDarray ravel(string order = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.ravel(order));
            else
                return new NDarray(NumpyNDarray.ravel(order));
        }

        public NDarray real_if_close(float tol = 100)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.real_if_close(tol));
            else
                return new NDarray(NumpyNDarray.real_if_close(tol));
        }

        public NDarray reciprocal(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.reciprocal(@out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.reciprocal(@out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray remainder(NDarray x2, NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.remainder(x2.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.remainder(x2.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray repeat(int[] repeats, int? axis = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.repeat(repeats, axis));
            else
                return new NDarray(NumpyNDarray.repeat(repeats, axis));
        }

        public NDarray require(Dtype dtype, string[] requirements = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.require(dtype.CupyDtype, requirements));
            else
                return new NDarray(NumpyNDarray.require(dtype.NumpyDtype, requirements));
        }

        public NDarray right_shift(NDarray x2, NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.right_shift(x2.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.right_shift(x2.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray rint(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.rint(@out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.rint(@out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray roll(int[] shift, Axis axis = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.roll(shift, axis?.CupyAxis));
            else
                return new NDarray(NumpyNDarray.roll(shift, axis?.NumpyAxis));
        }

        public NDarray rollaxis(int axis, int? start = 0)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.rollaxis(axis, start));
            else
                return new NDarray(NumpyNDarray.rollaxis(axis, start));
        }

        public NDarray roots()
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.roots());
            else
                return new NDarray(NumpyNDarray.roots());
        }

        public NDarray rot90(int k = 1, int[] axes = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.rot90(k, axes));
            else
                return new NDarray(NumpyNDarray.rot90(k, axes));
        }

        public NDarray<int> searchsorted(NDarray v, string side = "left", NDarray sorter = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray<int>(CupyNDarray.searchsorted(v.CupyNDarray, side, sorter?.CupyNDarray));
            else
                return new NDarray<int>(NumpyNDarray.searchsorted(v.NumpyNDarray, side, sorter?.NumpyNDarray));
        }

        public NDarray setdiff1d(NDarray ar2, bool assume_unique = false)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.setdiff1d(ar2.CupyNDarray, assume_unique));
            else
                return new NDarray(NumpyNDarray.setdiff1d(ar2.NumpyNDarray, assume_unique));
        }

        public NDarray setxor1d(NDarray ar1, bool assume_unique = false)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.setxor1d(ar1.CupyNDarray, assume_unique));
            else
                return new NDarray(NumpyNDarray.setxor1d(ar1.NumpyNDarray, assume_unique));
        }

        public NDarray sign(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.sign(@out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.sign(@out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray signbit(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.signbit(@out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.signbit(@out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray sin(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.sin(@out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.sin(@out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray sinc()
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.sinc());
            else
                return new NDarray(NumpyNDarray.sinc());
        }

        public NDarray sinh(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.sinh(@out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.sinh(@out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray sort(int? axis = -1, string kind = "quicksort", string order = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.sort(axis, kind, order));
            else
                return new NDarray(NumpyNDarray.sort(axis, kind, order));
        }

        public NDarray sort_complex()
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.sort_complex());
            else
                return new NDarray(NumpyNDarray.sort_complex());
        }

        public NDarray spacing(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.spacing(@out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.spacing(@out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray[] split(int[] indices_or_sections, int? axis = 0)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return CupyNDarray.split(indices_or_sections, axis).Select(x => new NDarray(x)).ToArray();
            else
                return NumpyNDarray.split(indices_or_sections, axis).Select(x => new NDarray(x)).ToArray();
        }

        public NDarray[] split(int indices_or_sections, int? axis = 0)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return CupyNDarray.split(indices_or_sections, axis).Select(x => new NDarray(x)).ToArray();
            else
                return NumpyNDarray.split(indices_or_sections, axis).Select(x => new NDarray(x)).ToArray();
        }

        public NDarray sqrt(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.sqrt(@out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.sqrt(@out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray square(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.square(@out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.square(@out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray squeeze(Axis axis = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.squeeze(axis?.CupyAxis));
            else
                return new NDarray(NumpyNDarray.squeeze(axis?.NumpyAxis));
        }

        public NDarray std(Axis axis, Dtype dtype = null, NDarray @out = null, int? ddof = 0, bool? keepdims = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.std(axis.CupyAxis, dtype?.CupyDtype, @out?.CupyNDarray, ddof, keepdims));
            else
                return new NDarray(NumpyNDarray.std(axis.NumpyAxis, dtype?.NumpyDtype, @out?.NumpyNDarray, ddof, keepdims));
        }

        public NDarray subtract(NDarray x1, NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.subtract(x1.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.subtract(x1.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray sum(Axis axis = null, Dtype dtype = null, NDarray @out = null, bool? keepdims = null, ValueType initial = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.sum(axis?.CupyAxis, dtype?.CupyDtype, @out?.CupyNDarray, keepdims, initial));
            else
                return new NDarray(NumpyNDarray.sum(axis?.NumpyAxis, dtype?.NumpyDtype, @out?.NumpyNDarray, keepdims, initial));
        }

        public NDarray swapaxes(int axis1, int axis2)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.swapaxes(axis1, axis2));
            else
                return new NDarray(NumpyNDarray.swapaxes(axis1, axis2));
        }

        public NDarray take_along_axis(NDarray indices, int? axis = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.take_along_axis(indices.CupyNDarray, axis));
            else
                return new NDarray(NumpyNDarray.take_along_axis(indices.NumpyNDarray, axis));
        }

        public NDarray tan(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.tan(@out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.tan(@out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray tanh(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.tanh(@out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.tanh(@out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray tensordot(NDarray a, int[] axes = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.tensordot(a.CupyNDarray, axes));
            else
                return new NDarray(NumpyNDarray.tensordot(a.NumpyNDarray, axes));
        }

        public NDarray tile(NDarray reps)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.tile(reps.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.tile(reps.NumpyNDarray));
        }

        public NDarray trace(int? offset = 0, int? axis2 = null, int? axis1 = null, Dtype dtype = null, NDarray @out = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.trace(offset, axis2, axis1, dtype?.CupyDtype, @out?.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.trace(offset, axis2, axis1, dtype?.NumpyDtype, @out?.NumpyNDarray));
        }

        public float trapz(NDarray x = null, float? dx = 1.0f, int? axis = -1)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return CupyNDarray.trapz(x?.CupyNDarray, dx, axis);
            else
                return NumpyNDarray.trapz(x?.NumpyNDarray, dx, axis);
        }

        public void tril_indices_from(int? k = 0)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                CupyNDarray.tril_indices_from(k);
            else
                NumpyNDarray.tril_indices_from(k);
        }

        public NDarray trim_zeros(string trim = "fb")
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.trim_zeros(trim));
            else
                return new NDarray(NumpyNDarray.trim_zeros(trim));
        }

        public NDarray[] triu_indices_from(int? k = 0)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return CupyNDarray.triu_indices_from(k).Select(x => new NDarray(x)).ToArray();
            else
                return NumpyNDarray.triu_indices_from(k).Select(x => new NDarray(x)).ToArray();
        }

        public NDarray true_divide(NDarray x2, NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.true_divide(x2.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.true_divide(x2.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray trunc(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.trunc(@out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.trunc(@out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray union1d(NDarray ar1)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.union1d(ar1.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.union1d(ar1.NumpyNDarray));
        }

        public NDarray[] unique(bool return_index, bool return_inverse, bool return_counts, int? axis = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return CupyNDarray.unique(return_index, return_inverse, return_counts, axis).Select(x => new NDarray(x)).ToArray();
            else
                return NumpyNDarray.unique(return_index, return_inverse, return_counts, axis).Select(x => new NDarray(x)).ToArray();
        }

        public NDarray unpackbits(int? axis = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.unpackbits(axis));
            else
                return new NDarray(NumpyNDarray.unpackbits(axis));
        }

        public NDarray[] unravel_index(Shape shape, string order = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return CupyNDarray.unravel_index(shape.CupyShape, order).Select(x => new NDarray(x)).ToArray();
            else
                return NumpyNDarray.unravel_index(shape.NumpyShape, order).Select(x => new NDarray(x)).ToArray();
        }

        public NDarray unwrap(float? discont = 3.141592653589793f, int? axis = -1)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.unwrap(discont, axis));
            else
                return new NDarray(NumpyNDarray.unwrap(discont, axis));
        }

        public NDarray<double> var(Axis axis, Dtype dtype = null, NDarray @out = null, int? ddof = 0, bool? keepdims = null)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray<double>(CupyNDarray.var(axis.CupyAxis, dtype?.CupyDtype, @out?.CupyNDarray, ddof, keepdims));
            else
                return new NDarray<double>(NumpyNDarray.var(axis.NumpyAxis, dtype?.NumpyDtype, @out?.NumpyNDarray, ddof, keepdims));
        }

        public double var(Dtype dtype = null, NDarray @out = null, int? ddof = 0)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return CupyNDarray.var(dtype?.CupyDtype, @out?.CupyNDarray, ddof);
            else
                return NumpyNDarray.var(dtype?.NumpyDtype, @out?.NumpyNDarray, ddof);
        }

        public NDarray vdot(NDarray b)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.vdot(b.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.vdot(b.NumpyNDarray));
        }

        public NDarray where(NDarray y, NDarray x)
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return new NDarray(CupyNDarray.where(y.CupyNDarray, x.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.where(y.NumpyNDarray, x.NumpyNDarray));
        }

        public NDarray[] where()
        {
            if ((Gpu.Available && Gpu.Use) || TryPeek() == ArrayMode.cp)
                return CupyNDarray.where().Select(x => new NDarray(x)).ToArray();
            else
                return NumpyNDarray.where().Select(x => new NDarray(x)).ToArray();
        }



        #endregion //Extension Methods

        private Stack<ArrayMode> _arrayMode = new();

        public void Push(ArrayMode arrayMode)
        {
            _arrayMode.Push(arrayMode);
            switch (arrayMode)
            {
                case ArrayMode.cp when NumpyNDarray is not null && CupyNDarray is null:
                case ArrayMode.np when CupyNDarray is not null && NumpyNDarray is null:
                    Switch(false);
                    break;
            }
        }

        public void Pop()
        {
            var arrayMode = _arrayMode.Pop();
            switch (arrayMode)
            {
                case ArrayMode.cp when CupyNDarray is not null:
                    NumpyNDarray = cpExtensions.asnumpy(CupyNDarray);
                    if (_autoDispose && _saveTarget != ArrayMode.cp)
                    {
                        CupyNDarray.Dispose();
                        CupyNDarray = null;
                    }
                    break;
                case ArrayMode.np when NumpyNDarray is not null:
                    CupyNDarray = cpExtensions.asarray(NumpyNDarray);
                    if (_autoDispose && _saveTarget != ArrayMode.np)
                    {
                        DisposeNpNDarray();
                    }
                    break;
            }
        }

        [DebuggerStepThrough]
        private void DisposeNpNDarray()
        {
            NumpyNDarray.Dispose();
            NumpyNDarray = null;
        }

        public ArrayMode TryPeek()
        {
            if (_arrayMode.TryPeek(out var ret))
            {
                return ret;
            }
            else
            {
                return ArrayMode.Unspecified;
            }
        }
    }

    public class NDarray<T> : NDarray
    {

        internal new Numpy.NDarray<T> NumpyNDarray
        {
            get => (Numpy.NDarray<T>)base.NumpyNDarray;
            set => base.NumpyNDarray = value;
        }

        internal new Cupy.NDarray<T> CupyNDarray
        {
            get => (Cupy.NDarray<T>)base.CupyNDarray;
            set => base.CupyNDarray = value;
        }

        public NDarray(Numpy.NDarray<T> t)
        {
            if (!Gpu.Available || !Gpu.Use)
            {
                NumpyNDarray = new Numpy.NDarray<T>(t);
            }
            else
            {
                throw new NotSupportedException();
            }
        }

        public NDarray(Cupy.NDarray<T> t)
        {
            if (Gpu.Available && Gpu.Use)
            {
                CupyNDarray = new Cupy.NDarray<T>(t);
            }
            else
            {
                throw new NotSupportedException();
            }
        }
    }

    public class Dtype : IDisposable
    {
        public Numpy.Dtype NumpyDtype { get; private set; }
        public Cupy.Dtype CupyDtype { get; private set; }

        public Dtype(Numpy.Dtype dtype)
        {
            NumpyDtype = dtype;
        }

        public Dtype(Cupy.Dtype dtype)
        {
            CupyDtype = dtype;
        }

        public IntPtr Handle => Gpu.Available && Gpu.Use ? CupyDtype.Handle : NumpyDtype.Handle;

        public dynamic PyObject => Gpu.Available && Gpu.Use ? CupyDtype.PyObject : NumpyDtype.PyObject;

        public PyObject self => Gpu.Available && Gpu.Use ? CupyDtype.self : NumpyDtype.self;

        public void Dispose()
        {
            if (Gpu.Available && Gpu.Use)
                CupyDtype.Dispose();
            else
                NumpyDtype.Dispose();
        }

        public bool Equals(object obj)
        {
            if (Gpu.Available && Gpu.Use)
                return CupyDtype.Equals(obj);
            else
                return NumpyDtype.Equals(obj);
        }

        public int GetHashCode()
        {
            if (Gpu.Available && Gpu.Use)
                return CupyDtype.GetHashCode();
            else
                return NumpyDtype.GetHashCode();
        }

        public T SharpToSharp<T>(object obj)
        {
            if (Gpu.Available && Gpu.Use)
                return CupyDtype.SharpToSharp<T>(obj);
            else
                return NumpyDtype.SharpToSharp<T>(obj);
        }

        public T ToCsharp<T>(object obj)
        {
            if (Gpu.Available && Gpu.Use)
                return (T)CupyDtype.ToCsharp<T>(obj);
            else
                return (T)NumpyDtype.ToCsharp<T>(obj);
        }

        public string ToString()
        {
            if (Gpu.Available && Gpu.Use)
                return CupyDtype.ToString();
            else
                return NumpyDtype.ToString();
        }

        public PyTuple ToTuple(Array input)
        {
            if (Gpu.Available && Gpu.Use)
                return CupyDtype.ToTuple(input);
            else
                return NumpyDtype.ToTuple(input);
        }
    }


    public class Flags : IDisposable
    {
        private Numpy.Models.Flags NumpyFlags;
        private Cupy.Models.Flags CupyFlags;

        public Flags(Numpy.Models.Flags flags)
        {
            NumpyFlags = flags;
        }

        public Flags(Cupy.Models.Flags flags)
        {
            CupyFlags = flags;
        }


        public IntPtr Handle => Gpu.Available && Gpu.Use ? CupyFlags.Handle : NumpyFlags.Handle;

        public dynamic PyObject => Gpu.Available && Gpu.Use ? CupyFlags.PyObject : NumpyFlags.PyObject;

        public PyObject self => Gpu.Available && Gpu.Use ? CupyFlags.self : NumpyFlags.self;

        public void Dispose()
        {
            if (Gpu.Available && Gpu.Use)
                CupyFlags.Dispose();
            else
                NumpyFlags.Dispose();
        }

        public bool Equals(object obj)
        {
            if (Gpu.Available && Gpu.Use)
                return CupyFlags.Equals(obj);
            else
                return NumpyFlags.Equals(obj);
        }

        public int GetHashCode()
        {
            if (Gpu.Available && Gpu.Use)
                return CupyFlags.GetHashCode();
            else
                return NumpyFlags.GetHashCode();
        }

        public T SharpToSharp<T>(object obj)
        {
            if (Gpu.Available && Gpu.Use)
                return CupyFlags.SharpToSharp<T>(obj);
            else
                return NumpyFlags.SharpToSharp<T>(obj);
        }

        public T ToCsharp<T>(dynamic obj)
        {
            if (Gpu.Available && Gpu.Use)
                return CupyFlags.ToCsharp<T>(obj);
            else
                return NumpyFlags.ToCsharp<T>(obj);
        }

        public string ToString()
        {
            if (Gpu.Available && Gpu.Use)
                return CupyFlags.ToString();
            else
                return NumpyFlags.ToString();
        }

        public PyTuple ToTuple(Array input)
        {
            if (Gpu.Available && Gpu.Use)
                return CupyFlags.ToTuple(input);
            else
                return NumpyFlags.ToTuple(input);
        }
    }

    public class Shape
    {
        public Numpy.Models.Shape NumpyShape { get; private set; }
        public Cupy.Models.Shape CupyShape { get; private set; }

        public Shape(Numpy.Models.Shape shape)
        {
            NumpyShape = shape;
        }

        public Shape(Cupy.Models.Shape shape)
        {
            CupyShape = shape;
        }

        public int[] Dimensions => Gpu.Available && Gpu.Use ? CupyShape.Dimensions : NumpyShape.Dimensions;

        public object shape => Gpu.Available && Gpu.Use ? CupyShape : NumpyShape;

        public int this[int n] => Gpu.Available && Gpu.Use ? CupyShape[n] : NumpyShape[n];


        public override bool Equals(object obj)
        {
            if (obj is Shape s)
            {
                if (Gpu.Available && Gpu.Use)
                    return CupyShape.Dimensions.SequenceEqual(s.Dimensions);
                else
                    return NumpyShape.Dimensions.SequenceEqual(s.Dimensions);
            }
            else
            {
                return false;
            }
        }

        public override int GetHashCode()
        {
            if (Gpu.Available && Gpu.Use)
                return CupyShape.GetHashCode();
            else
                return NumpyShape.GetHashCode();
        }

        public static bool operator ==(Shape a, Shape b)
        {
            return a.Equals(b);
        }

        public static bool operator !=(Shape a, Shape b)
        {
            return !a.Equals(b);
        }

        //public T SharpToSharp<T>(object obj)
        //{
        //    if (Gpu.Available && Gpu.Use)
        //    {
        //        return CupyShape.SharpToSharp<T>(obj);
        //    }
        //    else
        //        return NumpyShape.SharpToSharp<T>(obj);
        //}

        //public T ToCsharp<T>(object obj)
        //{
        //    if (Gpu.Available && Gpu.Use)
        //        return CupyShape.ToCsharp<T>(obj);
        //    else
        //        return NumpyShape.ToCsharp<T>(obj);
        //}

        public string ToString()
        {
            if (Gpu.Available && Gpu.Use)
                return CupyShape.ToString();
            else
                return NumpyShape.ToString();
        }

        //public PyTuple ToTuple(Array input)
        //{
        //    if (Gpu.Available && Gpu.Use)
        //        return CupyShape.ToTuple(input);
        //    else
        //        return NumpyShape.ToTuple(input);
        //}
    }

    public class Axis
    {
        public Numpy.Models.Axis NumpyAxis { get; private set; }
        public Cupy.Models.Axis CupyAxis { get; private set; }

        public Axis(Numpy.Models.Axis axis)
        {
            NumpyAxis = axis;
        }

        public Axis(Cupy.Models.Axis axis)
        {
            CupyAxis = axis;
        }

        public Axis(int[] axes)
        {
            if (Gpu.Available && Gpu.Use)
                CupyAxis = new Cupy.Models.Axis(axes);
            else
                NumpyAxis = new Numpy.Models.Axis(axes);
        }

        public int[] Axes => Gpu.Available && Gpu.Use ? CupyAxis.Axes : NumpyAxis.Axes;

        public bool Equals(object obj)
        {
            if (Gpu.Available && Gpu.Use)
                return CupyAxis.Equals(obj);
            else
                return NumpyAxis.Equals(obj);
        }

        public int GetHashCode()
        {
            if (Gpu.Available && Gpu.Use)
                return CupyAxis.GetHashCode();
            else
                return NumpyAxis.GetHashCode();
        }

        public string ToString()
        {
            if (Gpu.Available && Gpu.Use)
                return CupyAxis.ToString();
            else
                return NumpyAxis.ToString();
        }
    }

    public class Slice
    {
        public Cupy.Models.Slice CupySlice { get; private set; }
        public Numpy.Models.Slice NumpySlice { get; private set; }

        public Slice(Cupy.Models.Slice slice)
        {
            CupySlice = slice;
        }

        public Slice(Numpy.Models.Slice slice)
        {
            NumpySlice = slice;
        }

        public bool IsIndex => Gpu.Available && Gpu.Use ? CupySlice.IsIndex : NumpySlice.IsIndex;

        public int? Length => Gpu.Available && Gpu.Use ? CupySlice.Length : NumpySlice.Length;

        public int? Start => Gpu.Available && Gpu.Use ? CupySlice.Start : NumpySlice.Start;

        public int Step => Gpu.Available && Gpu.Use ? CupySlice.Step : NumpySlice.Step;
        
        public int? Stop => Gpu.Available && Gpu.Use ? CupySlice.Stop : NumpySlice.Stop;

        public int GetAbsStart(int dim)
        {
            if (Gpu.Available && Gpu.Use)
                return CupySlice.GetAbsStart(dim);
            else
                return NumpySlice.GetAbsStart(dim);
        }

        public int GetAbsStep()
        {
            if (Gpu.Available && Gpu.Use)
                return CupySlice.GetAbsStep();
            else
                return NumpySlice.GetAbsStep();
        }

        public int GetAbsStop(int dim)
        {
            if (Gpu.Available && Gpu.Use)
                return CupySlice.GetAbsStop(dim);
            else
                return NumpySlice.GetAbsStop(dim);
        }

        public int GetSize(int dim)
        {
            if (Gpu.Available && Gpu.Use)
                return CupySlice.GetSize(dim);
            else
                return NumpySlice.GetSize(dim);
        }

        public override bool Equals(object? obj)
        {
            if (Gpu.Available && Gpu.Use)
                return CupySlice.Equals(obj);
            else
                return NumpySlice.Equals(obj);
        }

        public override int GetHashCode()
        {
            if (Gpu.Available && Gpu.Use)
                return CupySlice.GetHashCode();
            else
                return NumpySlice.GetHashCode();
        }

        public override string ToString()
        {
            if (Gpu.Available && Gpu.Use)
                return CupySlice.ToString();
            else
                return NumpySlice.ToString();
        }

        public PyObject ToPython()
        {
            if (Gpu.Available && Gpu.Use)
                return CupySlice.ToPython();
            else
                return NumpySlice.ToPython();
        }
    }

    public class Constants
    {
        public Cupy.Models.Constants CupyConstants { get; internal set; }
        public Numpy.Models.Constants NumpyConstants { get; internal set; }

        public Constants(Cupy.Models.Constants constants)
        {
            CupyConstants = constants;
        }

        public Constants(Numpy.Models.Constants constants)
        {
            NumpyConstants = constants;
        }

        public Constants(string value)
        {
            switch (value)
            {
                case "inf":
                    CupyConstants = Cupy.Models.Constants.inf;
                    NumpyConstants = Numpy.Models.Constants.inf;
                    break;
                case "neg_inf":
                    CupyConstants = Cupy.Models.Constants.neg_inf;
                    NumpyConstants = Numpy.Models.Constants.neg_inf;
                    break;
            }
        }

        public static readonly Constants inf = new Constants("inf");
        public static readonly Constants neg_inf = new Constants("neg_inf");
    }

    public class Matrix
    {
        public Cupy.Models.Matrix CupyMatrix { get; internal set; }
        public Numpy.Models.Matrix NumpyMatrix { get; internal set; }

        public Matrix(Cupy.Models.Matrix matrix)
        {
            CupyMatrix = matrix;
        }

        public Matrix(Numpy.Models.Matrix matrix)
        {
            NumpyMatrix = matrix;
        }
    }

    public class MemMapMode
    {
        public Cupy.Models.MemMapMode CupyMemMapMode { get; internal set; }
        public Numpy.Models.MemMapMode NumpyMemMapMode { get; internal set; }

        public MemMapMode(Cupy.Models.MemMapMode cupyMemMapMode)
        {
            CupyMemMapMode = cupyMemMapMode;
        }

        public MemMapMode(Numpy.Models.MemMapMode numpyMemMapMode)
        {
            NumpyMemMapMode = numpyMemMapMode;
        }
    }
}
