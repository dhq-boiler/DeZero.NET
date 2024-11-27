using Cupy;
using DeZero.NET.Core;
using DeZero.NET.Extensions;
using Numpy;
using Python.Runtime;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Globalization;
using System.Numerics;
using System.Reflection;
using System.Text.RegularExpressions;
using cp = Cupy;
using NotSupportedException = System.NotSupportedException;
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

        /// <summary>
        /// Class for using IDisposable objects that temporarily disable the GPU in a using statement.
        /// When the Dispose method is called, it restores the GPU to its original state of use.
        /// </summary>
        public class TemporaryDisable : IDisposable
        {
            public bool LocalUse { get; set; }

            public TemporaryDisable()
            {
                LocalUse = Use;
                Use = false;
            }

            public void Dispose()
            {
                Use = LocalUse;
            }
        }
    }

    public static class Preferences
    {
        public static IDisposable TestMode() => new UsingConfig("Train", false);
    }

    public enum ArrayMode
    {
        Unspecified,
        np,
        cp
    }

    public class NDarray : IDisposable, IDeZeroObject
    {
        private bool _disposed;
        private readonly object _disposeLock = new object();
        private static readonly object _gilLock = new object();
        private static int _destructorCallCount = 0;
        private const int MAX_RETRY_COUNT = 3;
        private const int GIL_TIMEOUT_MS = 100;

        public Numpy.NDarray NumpyNDarray { get; internal set; }
        public Cupy.NDarray CupyNDarray { get; internal set; }

        public object Array => (object)CupyNDarray ?? (object)NumpyNDarray;

        public Numpy.NDarray ToNumpyNDarray
        {
            get
            {
                if (NumpyNDarray is not null)
                {
                    return NumpyNDarray;
                }
                else
                {
                    NumpyNDarray?.Dispose();
                    return (NumpyNDarray = CupyNDarray.asnumpy());
                }
            }
        }

        public Cupy.NDarray ToCupyNDarray
        {
            get
            {
                if (cp.cp.isscalar(CupyNDarray))
                {
                    return CupyNDarray;
                }
                else
                {
                    if (CupyNDarray is null && NumpyNDarray is not null)
                    {
                        CupyNDarray?.Dispose();
                        return CupyNDarray = cpExtensions.asarray(NumpyNDarray);
                    }
                    else
                    {
                        using dynamic __str__ = CupyNDarray.PyObject.__str__;
                        string str = (string)ToCsharp<string>(__str__);
                        if (str.StartsWith("variable("))
                        {
                            using dynamic array = CupyNDarray.PyObject.array;
                            return CupyNDarray = new Cupy.NDarray(array);
                        }

                        using dynamic flat = CupyNDarray?.flat;
                        if (flat?.ToString()?.StartsWith("<numpy.flatiter") == true)
                        {
                            using dynamic __str__2 = ToNumpyNDarray.PyObject.__str__;
                            str = (string)ToCsharp<string>(__str__2);
                            if (str.StartsWith("variable("))
                            {
                                using dynamic array = ToNumpyNDarray.PyObject.array;
                                var temp = cpExtensions.asarray(new Numpy.NDarray(array));
                                CupyNDarray?.Dispose();
                                return CupyNDarray = temp;
                            }

                            var _temp = ToNumpyNDarray.asarray();
                            CupyNDarray?.Dispose();
                            return CupyNDarray = _temp;
                        }

                        return CupyNDarray;
                    }
                }
            }
        }

        public string NpzIndex { get; set; }


        protected NDarray()
        {
        }

        public NDarray(PyObject pyobj, string npzIndex = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                CupyNDarray = new Cupy.NDarray(pyobj);
            }
            else
            {
                NumpyNDarray = new Numpy.NDarray(pyobj);
            }
            NpzIndex = npzIndex;
            VRAMLeakDetector.TrackAllocation(this);
        }

        public NDarray(byte obj)
        {
            if (Gpu.Available && Gpu.Use)
            {
                CupyNDarray = new Cupy.NDarray(obj);
            }
            else
            {
                NumpyNDarray = np.np.array(obj);
            }
            VRAMLeakDetector.TrackAllocation(this);
        }

        public NDarray(int obj)
        {
            if (Gpu.Available && Gpu.Use)
            {
                CupyNDarray = new Cupy.NDarray(obj);
            }
            else
            {
                NumpyNDarray = np.np.array(obj);
            }
            VRAMLeakDetector.TrackAllocation(this);
        }

        public NDarray(long obj)
        {
            if (Gpu.Available && Gpu.Use)
            {
                CupyNDarray = new Cupy.NDarray(obj);
            }
            else
            {
                NumpyNDarray = np.np.array(obj);
            }
            VRAMLeakDetector.TrackAllocation(this);
        }

        public NDarray(float obj)
        {
            if (Gpu.Available && Gpu.Use)
            {
                CupyNDarray = new Cupy.NDarray(obj);
            }
            else
            {
                NumpyNDarray = np.np.array(obj);
            }
            VRAMLeakDetector.TrackAllocation(this);
        }

        public NDarray(double obj)
        {
            if (Gpu.Available && Gpu.Use)
            {
                CupyNDarray = new Cupy.NDarray(obj);
            }
            else
            {
                NumpyNDarray = np.np.array(obj);
            }
            VRAMLeakDetector.TrackAllocation(this);
        }

        public NDarray(bool obj)
        {
            if (Gpu.Available && Gpu.Use)
            {
                CupyNDarray = new Cupy.NDarray(obj);
            }
            else
            {
                NumpyNDarray = np.np.array(obj);
            }
            VRAMLeakDetector.TrackAllocation(this);
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
                NumpyNDarray = t;
            }
            VRAMLeakDetector.TrackAllocation(this);
        }

        public NDarray(Cupy.NDarray t, bool autoDispose = false)
        {
            _autoDispose = autoDispose;
            _saveTarget = ArrayMode.cp;
            if (Gpu.Available && Gpu.Use)
            {
                CupyNDarray = t;
            }
            else
            {
                NumpyNDarray = cpExtensions.asnumpy(CupyNDarray);
            }
            VRAMLeakDetector.TrackAllocation(this);
        }
        
        public static NDarray operator +(NDarray a, NDarray b)
        {
            return xp.add(a, b);
        }

        public static NDarray operator -(NDarray a, NDarray b)
        {
            return xp.subtract(a, b);
        }

        public static NDarray operator *(NDarray a, NDarray b)
        {
            return xp.multiply(a, b);
        }

        public static NDarray operator /(NDarray a, NDarray b)
        {
            return xp.divide(a, b);
        }

        public static NDarray operator +(NDarray a, int b)
        {
            if (Gpu.Available && Gpu.Use && a.CupyNDarray is not null)
            {
                dynamic arr = a.ToCupyNDarray.PyObject;
                arr += b;
                return new NDarray(arr);
            }
            else
            {
                dynamic arr = a.ToNumpyNDarray.PyObject;
                arr += b;
                return new NDarray(arr);
            }
        }

        public static NDarray operator +(NDarray a, long b)
        {
            if (Gpu.Available && Gpu.Use && a.CupyNDarray is not null)
            {
                dynamic arr = a.ToCupyNDarray.PyObject;
                arr += b;
                return new NDarray(arr);
            }
            else
            {
                dynamic arr = a.ToNumpyNDarray.PyObject;
                arr += b;
                return new NDarray(arr);
            }
        }

        public static NDarray operator +(NDarray a, float b)
        {
            if (Gpu.Available && Gpu.Use && a.CupyNDarray is not null)
            {
                dynamic arr = a.ToCupyNDarray.PyObject;
                arr += b;
                return new NDarray(arr);
            }
            else
            {
                dynamic arr = a.ToNumpyNDarray.PyObject;
                arr += b;
                return new NDarray(arr);
            }
        }

        public static NDarray operator +(NDarray a, double b)
        {
            if (Gpu.Available && Gpu.Use && a.CupyNDarray is not null)
            {
                dynamic arr = a.ToCupyNDarray.PyObject;
                arr += b;
                return new NDarray(arr);
            }
            else
            {
                dynamic arr = a.ToNumpyNDarray.PyObject;
                arr += b;
                return new NDarray(arr);
            }
        }

        public static NDarray operator -(NDarray a, int b)
        {
            if (Gpu.Available && Gpu.Use && a.CupyNDarray is not null)
            {
                dynamic arr = a.ToCupyNDarray.PyObject;
                arr -= b;
                return new NDarray(arr);
            }
            else
            {
                dynamic arr = a.ToNumpyNDarray.PyObject;
                arr -= b;
                return new NDarray(arr);
            }
        }

        public static NDarray operator -(NDarray a, long b)
        {
            if (Gpu.Available && Gpu.Use && a.CupyNDarray is not null)
            {
                dynamic arr = a.ToCupyNDarray.PyObject;
                arr -= b;
                return new NDarray(arr);
            }
            else
            {
                dynamic arr = a.ToNumpyNDarray.PyObject;
                arr -= b;
                return new NDarray(arr);
            }
        }

        public static NDarray operator -(NDarray a, float b)
        {
            if (Gpu.Available && Gpu.Use && a.CupyNDarray is not null)
            {
                dynamic arr = a.ToCupyNDarray.PyObject;
                arr -= b;
                return new NDarray(arr);
            }
            else
            {
                dynamic arr = a.ToNumpyNDarray.PyObject;
                arr -= b;
                return new NDarray(arr);
            }
        }

        public static NDarray operator -(NDarray a, double b)
        {
            if (Gpu.Available && Gpu.Use && a.CupyNDarray is not null)
            {
                dynamic arr = a.ToCupyNDarray.PyObject;
                arr -= b;
                return new NDarray(arr);
            }
            else
            {
                dynamic arr = a.ToNumpyNDarray.PyObject;
                arr -= b;
                return new NDarray(arr);
            }
        }

        public static NDarray operator *(NDarray a, int b)
        {
            if (Gpu.Available && Gpu.Use && a.CupyNDarray is not null)
            {
                dynamic arr = a.ToCupyNDarray.PyObject;
                arr *= b;
                return new NDarray(arr);
            }
            else
            {
                dynamic arr = a.ToNumpyNDarray.PyObject;
                arr *= b;
                return new NDarray(arr);
            }
        }

        public static NDarray operator *(NDarray a, long b)
        {
            if (Gpu.Available && Gpu.Use && a.CupyNDarray is not null)
            {
                dynamic arr = a.ToCupyNDarray.PyObject;
                arr *= b;
                return new NDarray(arr);
            }
            else
            {
                dynamic arr = a.ToNumpyNDarray.PyObject;
                arr *= b;
                return new NDarray(arr);
            }
        }

        public static NDarray operator *(NDarray a, float b)
        {
            if (Gpu.Available && Gpu.Use && a.CupyNDarray is not null)
            {
                dynamic arr = a.ToCupyNDarray.PyObject;
                arr *= b;
                return new NDarray(arr);
            }
            else
            {
                dynamic arr = a.ToNumpyNDarray.PyObject;
                arr *= b;
                return new NDarray(arr);
            }
        }

        public static NDarray operator *(NDarray a, double b)
        {
            if (Gpu.Available && Gpu.Use && a.CupyNDarray is not null)
            {
                dynamic arr = a.ToCupyNDarray.PyObject;
                arr *= b;
                return new NDarray(arr);
            }
            else
            {
                dynamic arr = a.ToNumpyNDarray.PyObject;
                arr *= b;
                return new NDarray(arr);
            }
        }

        public static NDarray operator *(double a, NDarray b)
        {
            if (Gpu.Available && Gpu.Use && b.CupyNDarray is not null)
            {
                dynamic arr = b.ToCupyNDarray;
                var ret = a * arr;
                return new NDarray(ret);
            }
            else
            {
                dynamic arr = b.ToNumpyNDarray;
                var ret = a * arr;
                return new NDarray(ret);
            }
        }

        public static NDarray operator /(NDarray a, int b)
        {
            if (Gpu.Available && Gpu.Use && a.CupyNDarray is not null)
            {
                dynamic arr = a.ToCupyNDarray.PyObject;
                arr /= b;
                return new NDarray(arr);
            }
            else
            {
                dynamic arr = a.ToNumpyNDarray.PyObject;
                arr /= b;
                return new NDarray(arr);
            }
        }

        public static NDarray operator /(NDarray a, long b)
        {
            if (Gpu.Available && Gpu.Use && a.CupyNDarray is not null)
            {
                dynamic arr = a.ToCupyNDarray.PyObject;
                arr /= b;
                return new NDarray(arr);
            }
            else
            {
                dynamic arr = a.ToNumpyNDarray.PyObject;
                arr /= b;
                return new NDarray(arr);
            }
        }

        public static NDarray operator /(NDarray a, float b)
        {
            if (Gpu.Available && Gpu.Use && a.CupyNDarray is not null)
            {
                dynamic arr = a.ToCupyNDarray.PyObject;
                arr /= b;
                return new NDarray(arr);
            }
            else
            {
                dynamic arr = a.ToNumpyNDarray.PyObject;
                arr /= b;
                return new NDarray(arr);
            }
        }

        public static NDarray operator /(NDarray a, double b)
        {
            if (Gpu.Available && Gpu.Use && a.CupyNDarray is not null)
            {
                dynamic arr = a.ToCupyNDarray.PyObject;
                arr /= b;
                return new NDarray(arr);
            }
            else
            {
                dynamic arr = a.ToNumpyNDarray.PyObject;
                arr /= b;
                return new NDarray(arr);
            }
        }

        public static NDarray operator /(int a, NDarray b)
        {
            if (Gpu.Available && Gpu.Use && b.CupyNDarray is not null)
            {
                var _a = new NDarray(a);
                _a /= b;
                return _a;
            }
            else
            {
                var _a = new NDarray(a);
                _a /= b;
                return _a;
            }
        }

        public static NDarray operator /(long a, NDarray b)
        {
            if (Gpu.Available && Gpu.Use && b.CupyNDarray is not null)
            {
                var _a = new NDarray(a);
                _a /= b;
                return _a;
            }
            else
            {
                var _a = new NDarray(a);
                _a /= b;
                return _a;
            }
        }

        public static NDarray operator /(float a, NDarray b)
        {
            if (Gpu.Available && Gpu.Use && b.CupyNDarray is not null)
            {
                var _a = new NDarray(a);
                _a /= b;
                return _a;
            }
            else
            {
                var _a = new NDarray(a);
                _a /= b;
                return _a;
            }
        }

        public static NDarray operator /(double a, NDarray b)
        {
            if (Gpu.Available && Gpu.Use && b.CupyNDarray is not null)
            {
                var _a = new NDarray(a);
                _a /= b;
                return _a;
            }
            else
            {
                var _a = new NDarray(a);
                _a /= b;
                return _a;
            }
        }

        public static NDarray operator >(NDarray a, NDarray b)
        {
            return xp.greater(a, b);
        }

        public static NDarray operator >(NDarray a, int b)
        {
            return xp.greater(a, xp.array(b));
        }

        public static NDarray operator >(NDarray a, long b)
        {
            return xp.greater(a, xp.array(b));
        }

        public static NDarray operator >(NDarray a, float b)
        {
            return xp.greater(a, xp.array(b));
        }

        public static NDarray operator >(NDarray a, double b)
        {
            return xp.greater(a, xp.array(b));
        }

        public static NDarray operator >=(NDarray a, NDarray b)
        {
            return xp.greater_equal(a, b);
        }

        public static NDarray operator >=(NDarray a, int b)
        {
            return xp.greater_equal(a, xp.array(b));
        }

        public static NDarray operator >=(NDarray a, long b)
        {
            return xp.greater_equal(a, xp.array(b));
        }

        public static NDarray operator >=(NDarray a, float b)
        {
            return xp.greater_equal(a, xp.array(b));
        }

        public static NDarray operator >=(NDarray a, double b)
        {
            return xp.greater_equal(a, xp.array(b));
        }

        public static NDarray operator <(NDarray a, NDarray b)
        {
            return xp.less(a, b);
        }

        public static NDarray operator <(NDarray a, int b)
        {
            return xp.less(a, xp.array(b));
        }

        public static NDarray operator <(NDarray a, long b)
        {
            return xp.less(a, xp.array(b));
        }

        public static NDarray operator <(NDarray a, float b)
        {
            return xp.less(a, xp.array(b));
        }

        public static NDarray operator <(NDarray a, double b)
        {
            return xp.less(a, xp.array(b));
        }

        public static NDarray operator <=(NDarray a, NDarray b)
        {
            return xp.less_equal(a, b);
        }

        public static NDarray operator <=(NDarray a, int b)
        {
            return xp.less_equal(a, xp.array(b));
        }

        public static NDarray operator <=(NDarray a, long b)
        {
            return xp.less_equal(a, xp.array(b));
        }

        public static NDarray operator <=(NDarray a, float b)
        {
            return xp.less_equal(a, xp.array(b));
        }

        public static NDarray operator <=(NDarray a, double b)
        {
            return xp.less_equal(a, xp.array(b));
        }

        public static NDarray operator ==(NDarray a, NDarray b)
        {
            return xp.equal(a, b);
        }

        public static NDarray operator !=(NDarray a, NDarray b)
        {
            return xp.not_equal(a, b);
        }

        public static NDarray operator -(NDarray a)
        {
            return a.negative();
        }

        private NDarray Sugar(Func<Cupy.NDarray> cpFunc, Func<Numpy.NDarray> npFunc)
        {
            if (Gpu.Available && Gpu.Use && CupyNDarray is not null)
            {
                return new NDarray(cpFunc());
            }
            else
            {
                return new NDarray(npFunc());
            }
        }

        [DebuggerStepThrough]
        private Dtype Sugar(Func<Cupy.Dtype> cpFunc, Func<Numpy.Dtype> npFunc)
        {
            if (Gpu.Available && Gpu.Use && CupyNDarray is not null)
            {
                return new Dtype(cpFunc());
            }
            else
            {
                return new Dtype(npFunc());
            }
        }

        private Flags Sugar(Func<Cupy.Models.Flags> cpFunc, Func<Numpy.Models.Flags> npFunc)
        {
            if (Gpu.Available && Gpu.Use && CupyNDarray is not null)
            {
                return new Flags(cpFunc());
            }
            else
            {
                return new Flags(npFunc());
            }
        }

        private Shape Sugar(Func<Cupy.Models.Shape> cpFunc, Func<Numpy.Models.Shape> npFunc)
        {
            if (Gpu.Available && Gpu.Use && CupyNDarray is not null)
            {
                return new Shape(cpFunc());
            }
            else
            {
                return new Shape(npFunc());
            }
        }

        private PyObject Sugar(Func<PyObject> cpFunc, Func<PyObject> npFunc)
        {
            if (Gpu.Available && Gpu.Use && CupyNDarray is not null)
            {
                return cpFunc();
            }
            else
            {
                return npFunc();
            }
        }

        private T Sugar<T>(Func<T> cpFunc, Func<T> npFunc)
        {
            if (Gpu.Available && Gpu.Use && CupyNDarray is not null)
            {
                return cpFunc();
            }
            else
            {
                return npFunc();
            }
        }

        //public Cupy.NDarray SafeCupyNDarray
        //{
        //    get
        //    {
        //        if (CupyNDarray is not null)
        //        {
        //            return CupyNDarray;
        //        }
        //        else if (NumpyNDarray is not null)
        //        {
        //            return ToCupyNDarray;
        //        }
        //        else
        //        {
        //            throw new InvalidOperationException();
        //        }
        //    }
        //}

        //public Numpy.NDarray SafeNumpyNDarray
        //{
        //    get
        //    {
        //        if (NumpyNDarray is not null)
        //        {
        //            return NumpyNDarray;
        //        }
        //        else if (CupyNDarray is not null)
        //        {
        //            return ToNumpyNDarray;
        //        }
        //        else
        //        {
        //            throw new InvalidOperationException();
        //        }
        //    }
        //}

        //private NDarray Sugar<CupyT, NumpyT>(Func<CupyT> cpFunc, Func<NumpyT> npFunc)
        //{
        //    if (Gpu.Available && Gpu.Use && CupyNDarray is not null)
        //    {
        //        return new NDarray((object)cpFunc());
        //    }
        //    else
        //    {
        //        return (T)(object)npFunc();
        //    }
        //}

        //public NDarray T => CupyNDarray is not null ? new NDarray(CupyNDarray.T) : new NDarray(NumpyNDarray.T);
        public NDarray T => Sugar(() => ToCupyNDarray.T, () => ToNumpyNDarray.T);

        //public PyObject ctypes => CupyNDarray is not null ? CupyNDarray.ctypes : NumpyNDarray.ctypes;
        public PyObject ctypes => Sugar(() => ToCupyNDarray.ctypes, () => ToNumpyNDarray.ctypes);

        //public PyObject data => CupyNDarray is not null ? CupyNDarray.data : NumpyNDarray.data;
        public PyObject data => Sugar(() => ToCupyNDarray.data, () => ToNumpyNDarray.data);

        //public Dtype dtype => CupyNDarray is not null
        //    ? new Dtype(CupyNDarray.dtype)
        //    : new Dtype(NumpyNDarray.dtype);
        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        public Dtype dtype => Sugar(() => xp.isscalar(ToCupyNDarray) ? null : ToCupyNDarray.dtype, () => xp.isscalar(ToNumpyNDarray) ? null : ToNumpyNDarray.dtype);

        //public Flags flags => CupyNDarray is not null ? new Flags(CupyNDarray.flags) : new Flags(NumpyNDarray.flags);
        public Flags flags => Sugar(() => ToCupyNDarray.flags, () => ToNumpyNDarray.flags);

        //public PyObject flat => CupyNDarray is not null ? CupyNDarray.flat : NumpyNDarray.flat;
        public PyObject flat => Sugar(() => ToCupyNDarray.flat, () => ToNumpyNDarray.flat);

        //public NDarray imag => CupyNDarray is not null ? new NDarray(CupyNDarray.imag) : new NDarray(NumpyNDarray.imag);
        public NDarray imag => Sugar(() => ToCupyNDarray.imag, () => ToNumpyNDarray.imag);

        //public int itemsize => CupyNDarray is not null ? CupyNDarray.itemsize : NumpyNDarray.itemsize;
        public int itemsize => Sugar(() => ToCupyNDarray.itemsize, () => ToNumpyNDarray.itemsize);

        //public int len => CupyNDarray is not null ? CupyNDarray.len : NumpyNDarray.len;
        public int len => Sugar(() => ToCupyNDarray.len, () => ToNumpyNDarray.len);

        //public int nbytes => CupyNDarray is not null ? CupyNDarray.nbytes : NumpyNDarray.nbytes;
        public int nbytes => Sugar(() => ToCupyNDarray.nbytes, () => ToNumpyNDarray.nbytes);

        //public int ndim => CupyNDarray?.ndim ?? NumpyNDarray.ndim;
        public int ndim => Sugar(() => ToCupyNDarray.ndim, () => ToNumpyNDarray.ndim);

        //public NDarray real => CupyNDarray is not null ? new NDarray(CupyNDarray.real) : new NDarray(NumpyNDarray.real);
        public NDarray real => Sugar(() => ToCupyNDarray.real, () => ToNumpyNDarray.real);

        //public string repr => CupyNDarray is not null ? CupyNDarray.repr : NumpyNDarray.repr;
        public string repr => Sugar(() => ToCupyNDarray.repr, () => ToNumpyNDarray.repr);

        //public Shape shape => this.ToShape((Gpu.Available && Gpu.Use));
        //public Shape shape => Sugar(() => new NDarray(SafeCupyNDarray).shape.CupyShape, () => new NDarray(SafeNumpyNDarray).shape.NumpyShape);
        public Shape shape => Sugar(() => ToCupyNDarray.isarray() ? ToCupyNDarray.shape : new Cupy.Models.Shape(), 
            () => ToNumpyNDarray.isarray() ? ToNumpyNDarray.shape : new Numpy.Models.Shape());

        //public int size => CupyNDarray?.size ?? NumpyNDarray?.size ?? default;
        public int size => Sugar(() => ToCupyNDarray.size, () => ToNumpyNDarray.size);

        //public string str => CupyNDarray is not null ? CupyNDarray.str : NumpyNDarray.str;
        public string str => Sugar(() => ToCupyNDarray.str, () => ToNumpyNDarray.str);

        //public int[] strides => CupyNDarray is not null ? CupyNDarray.strides : NumpyNDarray.strides;
        public int[] strides => Sugar(() => ToCupyNDarray.strides, () => ToNumpyNDarray.strides);

        //public IntPtr Handle => CupyNDarray is not null ? CupyNDarray.Handle : NumpyNDarray.Handle;
        public IntPtr Handle => Sugar(() => ToCupyNDarray.Handle, () => ToNumpyNDarray.Handle);

        //public dynamic PyObject => CupyNDarray is not null ? CupyNDarray.PyObject : NumpyNDarray.PyObject;
        public dynamic PyObject => Sugar(() => ToCupyNDarray.PyObject, () => ToNumpyNDarray.PyObject);

        //public PyObject self => CupyNDarray is not null ? CupyNDarray.self : NumpyNDarray.self;
        public PyObject self => Sugar(() => ToCupyNDarray.self, () => ToNumpyNDarray.self);

        public Slice ToSlice =>
            len == 3
                ? Sugar(
                    () => new Slice(ToCupyNDarray[0].asscalar<int>(), ToCupyNDarray[1].asscalar<int>(),
                        ToCupyNDarray[2].asscalar<int>()),
                    () => new Slice(ToNumpyNDarray[0].asscalar<int>(), ToNumpyNDarray[1].asscalar<int>(),
                        ToNumpyNDarray[2].asscalar<int>()))
                : Sugar(
                    () => new Slice(ToCupyNDarray[0].asscalar<int>(), ToCupyNDarray[1].asscalar<int>()),
                    () => new Slice(ToNumpyNDarray[0].asscalar<int>(), ToNumpyNDarray[1].asscalar<int>()));

        //public NDarray this[int index] => CupyNDarray is not null ? new NDarray(CupyNDarray[index]) : new NDarray(NumpyNDarray[index]);
        public NDarray this[int index]
        {
            get => Sugar(() => ToCupyNDarray[index], () => ToNumpyNDarray[index]);
            set => Sugar(() => ToCupyNDarray[index] = value.ToCupyNDarray, () => ToNumpyNDarray[index] = value.ToNumpyNDarray);
        }

        //public NDarray this[(int x, int y) index] => CupyNDarray is not null ? new NDarray(CupyNDarray[index.x, index.y]) : new NDarray(NumpyNDarray[index.x, index.y]);
        public NDarray this[(int x, int y) index] => Sugar(() => ToCupyNDarray[index.x, index.y], () => ToNumpyNDarray[index.x, index.y]);

        public NDarray this[string arrayName]
        {
            get
            {
                if (Gpu.Available && Gpu.Use && CupyNDarray is not null)
                {
                    throw new NotSupportedException();
                }
                else
                {
                    return new NDarray(ToNumpyNDarray.PyObject[arrayName]);
                }
            }
            set
            {
                if (Gpu.Available && Gpu.Use && CupyNDarray is not null)
                {
                    CupyNDarray[arrayName] = value.ToCupyNDarray;
                }
                else
                {
                    NumpyNDarray[arrayName] = value.ToNumpyNDarray;
                }
            }
        }

        public NDarray this[params NDarray[] index]
        {
            get
            {
                if (Gpu.Available && Gpu.Use && CupyNDarray is not null)
                {
                    return new NDarray(ToCupyNDarray[index.Select(x => x.ToCupyNDarray).ToArray()]);
                }
                else
                {
                    return new NDarray(ToNumpyNDarray[index.Select(x => x.ToNumpyNDarray).ToArray()]);
                }
            }
            set
            {
                if (Gpu.Available && Gpu.Use && CupyNDarray is not null)
                {
                    ToCupyNDarray[index.Select(x => x.ToCupyNDarray).ToArray()] = value.ToCupyNDarray;
                }
                else
                {
                    ToNumpyNDarray[index.Select(x => x.ToNumpyNDarray).ToArray()] = value.ToNumpyNDarray;
                }
            }
        }

        public NDarray this[PyObject index]
        {
            get
            {
                try
                {
                    var tuple2 = (Tuple<int, int>)ToCsharp<Tuple<int, int>>(index);
                    if (tuple2 is not null)
                    {
                        if (CupyNDarray is not null)
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
                        if ((Gpu.Available && Gpu.Use))
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
                        if (CupyNDarray is not null)
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
                        if ((Gpu.Available && Gpu.Use))
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

        public NDarray this[params Slice[] slice]
        {
            get
            {
                if (Gpu.Available && Gpu.Use && CupyNDarray is not null)
                {
                    return new NDarray(this.slice(CupyNDarray, slice));
                    //return new NDarray(CupyNDarray[slice.Select(x => x.CupySlice).ToArray()]);
                }
                else
                {
                    return new NDarray(this.slice(NumpyNDarray, slice));
                    //return new NDarray(NumpyNDarray[slice.Select(x => x.NumpySlice).ToArray()]);
                }
            }
            set
            {
                //TODO
                if (Gpu.Available && Gpu.Use && CupyNDarray is not null)
                {
                    CupyNDarray[slice.Select(x => x.CupySlice).ToArray()] = value.CupyNDarray;
                }
                else
                {
                    NumpyNDarray[slice.Select(x => x.NumpySlice).ToArray()] = value.NumpyNDarray;
                }
            }
        }

        //public int this[int index]
        //{
        //    get
        //    {
        //        if (Gpu.Available && Gpu.Use && CupyNDarray is not null)
        //        {
        //            return CupyNDarray[index].asscalar<int>();
        //        }
        //        else
        //        {
        //            return NumpyNDarray[index].asscalar<int>();
        //        }
        //    }
        //    set
        //    {
        //        if (Gpu.Available && Gpu.Use && CupyNDarray is not null)
        //        {
        //            CupyNDarray[index] = new cp.NDarray(value);
        //        }
        //        else
        //        {
        //            NumpyNDarray[index] = new np.NDarray(value.ToPython());
        //        }
        //    }
        //}

        private Cupy.NDarray slice(Cupy.NDarray arr, params Slice[] slices)
        {
            Cupy.NDarray ret = arr;
            List<Cupy.NDarray> list = new List<Cupy.NDarray>();

            var i = 0;
            var currentSlice = slices[0];
            var start = currentSlice.Start ?? 0;
            var stop = currentSlice.Stop ?? ret.len;
            var step = currentSlice.Step;

            if (start < stop)
            {
                for (int j = start; j < stop; j += step) // Changed the loop condition and increment
                {
                    list.Add(ret[j]); // Added the element to the list directly
                }
            }
            else
            {
                for (int j = start; j > stop; j -= step) // Changed the loop condition and decrement
                {
                    list.Add(ret[j]); // Added the element to the list directly
                }

                list.Reverse(); // Reversed the list
            }

            return cp.cp.array(list.ToArray()).astype(ret.dtype);
        }

        private Numpy.NDarray slice(Numpy.NDarray arr, params Slice[] slices)
        {
            Numpy.NDarray ret = arr;
            List<double> list = new List<double>();

            var i = 0;
            var currentSlice = slices[0];
            var start = currentSlice.Start ?? 0;
            var stop = currentSlice.Stop ?? ret.len - 1;
            var step = currentSlice.Step;

            if (start < stop)
            {
                for (int j = start; j < stop; j += step) // Changed the loop condition and increment
                {
                    list.Add(ret[j].asscalar<double>()); // Added the element to the list directly
                }
            }
            else
            {
                for (int j = start; j >= stop; j -= step) // Changed the loop condition and decrement
                {
                    list.Add(ret[j].asscalar<double>()); // Added the element to the list directly
                }

                list.Reverse(); // Reversed the list
            }

            return np.np.array(list.ToArray()).astype(ret.dtype);
        }

        private int Dig(List<double> list, np.NDarray ret, int i, int[] branches, Slice currentSlice, Slice[] slices)
        {
            var start = currentSlice.Start ?? 0;
            var stop = currentSlice.Stop ?? ret.len - 1;
            var step = currentSlice.Step;
            int r = 0;
            if (start < stop)
            {
                for (int j = 0; j < ret.len; j++)
                {
                    if (i + 1 < slices.Length - 1)
                    {
                        r = Dig(list, ret[j], i + 1, [j], slices[i + 1], slices.Skip(i + 1).ToArray());
                    }
                    else
                    {
                        list.Add(ret[j].asscalar<double>());
                    }
                }
            }
            else
            {
                var offset = 0;
                for (int j = start; j < ret.len;)
                {
                    if (i + 1 < slices.Length - 1)
                    {
                        r = Dig(list, ret[j], i + 1, [j], slices[i + 1], slices.Skip(i + 1).ToArray());
                    }
                    else
                    {
                        list.Add(ret[j].asscalar<double>());
                    }
                    j += Math.Max(step, 1);
                }

                for (int j = offset; j <= stop;)
                {
                    if (i + 1 < slices.Length - 1)
                    {
                        r = Dig(list, A(ret.@base, [branches[0], j]), i + 1, [j], slices[i + 1], slices.Skip(i + 1).ToArray());
                    }
                    else
                    {
                        list.Add(A(ret.@base, [branches[0], j]).asscalar<double>());
                    }
                    if (j == offset)
                    {
                        r = 1;
                    }
                    j += Math.Max(step, 1);
                }
            }

            return r;
        }

        private int Dig(List<double> list, cp.NDarray ret, int i, int[] branches, Slice currentSlice, Slice[] slices)
        {
            var start = currentSlice.Start ?? 0;
            var stop = currentSlice.Stop ?? ret.len - 1;
            var step = currentSlice.Step;
            int r = 0;
            if (start < stop)
            {
                for (int j = 0; j < ret.len; j++)
                {
                    if (i + 1 < slices.Length - 1)
                    {
                        r = Dig(list, ret[j], i + 1, [j], slices[i + 1], slices.Skip(i + 1).ToArray());
                    }
                    else
                    {
                        list.Add(ret[j].asscalar<double>());
                    }
                }
            }
            else
            {
                var offset = 0;
                for (int j = start; j < ret.len;)
                {
                    if (i + 1 < slices.Length - 1)
                    {
                        r = Dig(list, ret[j], i + 1, [j], slices[i + 1], slices.Skip(i + 1).ToArray());
                    }
                    else
                    {
                        list.Add(ret[j].asscalar<double>());
                    }
                    j += Math.Max(step, 1);
                }

                for (int j = offset; j <= stop;)
                {
                    if (i + 1 < slices.Length - 1)
                    {
                        r = Dig(list, A(ret.@base, [branches[0], j]), i + 1, [j], slices[i + 1], slices.Skip(i + 1).ToArray());
                    }
                    else
                    {
                        list.Add(A(ret.@base, [branches[0], j]).asscalar<double>());
                    }
                    if (j == offset)
                    {
                        r = 1;
                    }
                    j += Math.Max(step, 1);
                }
            }

            return r;
        }

        private np.NDarray A(np.NDarray retBase, int[] index)
        {
            for (int i = 0; i < index.Length; i++)
            {
                if (i == 0)
                {
                    retBase = retBase[index[i] + 1];
                }
                else
                {
                    retBase = retBase[index[i]];
                }
            }

            return retBase;
        }

        private cp.NDarray A(cp.NDarray retBase, int[] index)
        {
            for (int i = 0; i < index.Length; i++)
            {
                if (i == 0)
                {
                    retBase = retBase[index[i] + 1];
                }
                else
                {
                    retBase = retBase[index[i]];
                }
            }

            return retBase;
        }

        public NDarray this[params int[] index]
        {
            get
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(ToCupyNDarray[index.ToArray()]);
                }
                else
                {
                    return new NDarray(ToNumpyNDarray[index.ToArray()]);
                }
            }
            set
            {
                if (Gpu.Available && Gpu.Use && CupyNDarray is not null)
                {
                    CupyNDarray[index.ToArray()] = value.ToCupyNDarray;
                }
                else
                {
                    NumpyNDarray[index.ToArray()] = value.ToNumpyNDarray;
                }
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
            else if (this.CupyNDarray is not null && this.NumpyNDarray is not null)
            {
                this.CupyNDarray = cp.cp.asarray(this.CupyNDarray).copy();
                this.NumpyNDarray = np.np.asarray(this.NumpyNDarray).copy();
                return this;
            }
            else
            {
                throw new InvalidOperationException();
            }
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
                case (string, string)[] o: return ToDict(o);
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
                case Dtype o: return o.PyObject;
                case NDarray o: return o.PyObject;
                default:
                    throw new NotImplementedException(
                        $"Type is not yet supported: {obj.GetType().Name}. Add it to 'ToPythonConversions'");
            }
        }

        public static object ToCsharp<T>(dynamic pyobj)
        {
            if (typeof(T).Name == "NDarray" || typeof(T).Name == "String" || !pyobj.ToString().Contains(",") && !pyobj.ToString().Contains(". ") && !pyobj.ToString().Contains(" "))
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
                case "Tuple`1":
                {
                    var tuple = ToPython(pyobj);
                    if (Regex.IsMatch(tuple.ToString(), @"\(\d+?,\)"))
                    {
                        return new Tuple<int>((int)ToCsharp<int>(tuple[0]));
                    }
                    return new Tuple<int, int>((int)ToCsharp<int>(tuple[0]), (int)ToCsharp<int>(tuple[1]));
                }
                case "Tuple`2":
                {
                    var tuple = ToPython(pyobj);
                    if (Regex.IsMatch(tuple.ToString(), @"\(\d+?,\)"))
                    {
                        return new Tuple<int>((int)ToCsharp<int>(tuple[0]));
                    }
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
                //case "Dtype": return (T)(object)new T(pyobj);
                case "String": return (T)(object)pyobj.ToString();
                case "Dtype": return (T)Activator.CreateInstance(typeof(T), BindingFlags.Instance | BindingFlags.Public, null, [pyobj], null);
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
                using var a = ToCsharp<NDarray>(_po);
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

        private static PyDict ToDict((string, string)[] d)
        {
            var dict = new PyDict();
            foreach (var pair in d)
                dict[new PyString(pair.Item1)] = new PyString(pair.Item2);
            return dict;
        }

        public bool Equals(NDarray other)
        {
            if ((Gpu.Available && Gpu.Use))
                return ToCupyNDarray.Equals(other.ToCupyNDarray);
            else
                return ToNumpyNDarray.Equals(other.ToNumpyNDarray);
        }

        public override bool Equals(object obj)
        {
            if (obj is Variable v)
            {
                return xp.equal(this, v.Data.Value).all();
            }
            else if (obj is NDarray arr)
            {
                return xp.equal(this, arr).all();
            }
            return false;
        }

        //public override string ToString() => repr;

        public void __setstate__(int version, Shape shape, Dtype dtype, bool isFortran, string rawdata)
        {
            if ((Gpu.Available && Gpu.Use))
                CupyNDarray.__setstate__(version, shape.CupyShape, dtype.CupyDtype, isFortran, rawdata);
            else
                NumpyNDarray.__setstate__(version, shape.NumpyShape, dtype.NumpyDtype, isFortran, rawdata);
        }

        public NDarray abs(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.abs(@out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(ToNumpyNDarray.abs(@out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public T asscalar<T>()
        {
            if ((Gpu.Available && Gpu.Use))
                return ToCupyNDarray.asscalar<T>();
            else
                return ToNumpyNDarray.asscalar<T>();
        }

        public NDarray astype(Dtype dtype, string order = null, string casting = null, bool? subok = null, bool? copy = null)
        {
            if ((Gpu.Available && Gpu.Use))
            {
                return new NDarray(ToCupyNDarray.astype(dtype.CupyDtype, order, casting, subok, copy));
            }
            else
            {
                return new NDarray(ToNumpyNDarray.astype(dtype.NumpyDtype, order, casting, subok, copy));
            }
        }

        public NDarray byteswap(bool? inplace = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.byteswap(inplace));
            else
                return new NDarray(ToNumpyNDarray.byteswap(inplace));
        }

        public NDarray copy(string order = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.copy(order));
            else
                return new NDarray(ToNumpyNDarray.copy(order));
        }

        public NDarray divmod(ValueType obj)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.divmod(obj));
            else
                return new NDarray(ToNumpyNDarray.divmod(obj));
        }

        public void dump(string file)
        {
            if ((Gpu.Available && Gpu.Use))
                CupyNDarray.dump(file);
            else
                NumpyNDarray.dump(file);
        }

        public void dumps()
        {
            if ((Gpu.Available && Gpu.Use))
                CupyNDarray.dumps();
            else
                NumpyNDarray.dumps();
        }

        public NDarray<bool> equals(ValueType valueType)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray<bool>(ToCupyNDarray.equals(valueType));
            else
                return new NDarray<bool>(ToNumpyNDarray.equals(valueType));
        }

        public NDarray<bool> equals(NDarray obj)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray<bool>(ToCupyNDarray.equals(obj.ToCupyNDarray));
            else
                return new NDarray<bool>(ToNumpyNDarray.equals(obj.ToNumpyNDarray));
        }

        public void fill(ValueType value)
        {
            if ((Gpu.Available && Gpu.Use))
                CupyNDarray.fill(value);
            else
                NumpyNDarray.fill(value);
        }

        public NDarray flatten(string order = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.flatten(order));
            else
                return new NDarray(ToNumpyNDarray.flatten(order));
        }

        public NDarray floordiv(NDarray a, ValueType obj)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.floordiv(a.ToCupyNDarray, obj));
            else
                return new NDarray(ToNumpyNDarray.floordiv(a.ToNumpyNDarray, obj));
        }

        public void getfield(Dtype dtype, int offset)
        {
            if ((Gpu.Available && Gpu.Use))
                ToCupyNDarray.getfield(dtype.CupyDtype, offset);
            else
                ToNumpyNDarray.getfield(dtype.NumpyDtype, offset);
        }

        public T GetData<T>()
        {
            if ((Gpu.Available && Gpu.Use))
                return ToCupyNDarray.GetData<T>();
            else
            {
                var type = typeof(T);
                var elementType = type;

                // NumPy配列から値を取得
                dynamic npArray = ToNumpyNDarray.GetData<T>();

                // NumPy型からネイティブ型への変換
                if (npArray.GetType().ToString().Contains("numpy"))
                {
                    // item()メソッドを使用してPythonのネイティブ型に変換
                    var pythonValue = npArray.InvokeMethod("item");
                    // Convert.ChangeTypeを使用してT型に変換
                    return (T)Convert.ChangeType(pythonValue, typeof(T));
                }

                return (T)npArray;
            }
        }

        public int GetHashCode()
        {
            if ((Gpu.Available && Gpu.Use))
                return ToCupyNDarray.GetHashCode();
            else
                return ToNumpyNDarray.GetHashCode();
        }

        public NDarray iadd(NDarray obj)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.iadd(obj.ToCupyNDarray));
            else
                return new NDarray(ToNumpyNDarray.iadd(obj.ToNumpyNDarray));
        }

        public NDarray iadd(ValueType obj)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.iadd(obj));
            else
                return new NDarray(ToNumpyNDarray.iadd(obj));
        }

        public NDarray iand(NDarray obj)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.iand(obj.ToCupyNDarray));
            else
                return new NDarray(ToNumpyNDarray.iand(obj.ToNumpyNDarray));
        }

        public NDarray iand(ValueType obj)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.iand(obj));
            else
                return new NDarray(ToNumpyNDarray.iand(obj));
        }

        public NDarray idiv(NDarray obj)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.idiv(obj.ToCupyNDarray));
            else
                return new NDarray(ToNumpyNDarray.idiv(obj.ToNumpyNDarray));
        }

        public NDarray idiv(ValueType obj)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.idiv(obj));
            else
                return new NDarray(ToNumpyNDarray.idiv(obj));
        }

        public NDarray ifloordiv(NDarray obj)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.ifloordiv(obj.ToCupyNDarray));
            else
                return new NDarray(ToNumpyNDarray.ifloordiv(obj.ToNumpyNDarray));
        }

        public NDarray ifloordiv(ValueType obj)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.ifloordiv(obj));
            else
                return new NDarray(ToNumpyNDarray.ifloordiv(obj));
        }

        public NDarray ilshift(NDarray obj)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.ilshift(obj.ToCupyNDarray));
            else
                return new NDarray(ToNumpyNDarray.ilshift(obj.ToNumpyNDarray));
        }

        public NDarray ilshift(int obj)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.ilshift(obj));
            else
                return new NDarray(ToNumpyNDarray.ilshift(obj));
        }

        public NDarray imod(NDarray obj)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.imod(obj.ToCupyNDarray));
            else
                return new NDarray(ToNumpyNDarray.imod(obj.ToNumpyNDarray));
        }

        public NDarray imod(ValueType obj)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.imod(obj));
            else
                return new NDarray(ToNumpyNDarray.imod(obj));
        }

        public NDarray imul(NDarray obj)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.imul(obj.ToCupyNDarray));
            else
                return new NDarray(ToNumpyNDarray.imul(obj.ToNumpyNDarray));
        }

        public NDarray imul(ValueType obj)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.imul(obj));
            else
                return new NDarray(ToNumpyNDarray.imul(obj));
        }

        public NDarray ior(NDarray obj)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.ior(obj.ToCupyNDarray));
            else
                return new NDarray(ToNumpyNDarray.ior(obj.ToNumpyNDarray));
        }

        public NDarray ior(ValueType obj)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.ior(obj));
            else
                return new NDarray(ToNumpyNDarray.ior(obj));
        }

        public NDarray ipow(NDarray obj)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.ipow(obj.ToCupyNDarray));
            else
                return new NDarray(ToNumpyNDarray.ipow(obj.ToNumpyNDarray));
        }

        public NDarray ipow(ValueType obj)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.ipow(obj));
            else
                return new NDarray(ToNumpyNDarray.ipow(obj));
        }

        public NDarray irshift(NDarray obj)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.irshift(obj.ToCupyNDarray));
            else
                return new NDarray(ToNumpyNDarray.irshift(obj.ToNumpyNDarray));
        }

        public NDarray irshift(int obj)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.irshift(obj));
            else
                return new NDarray(ToNumpyNDarray.irshift(obj));
        }

        public NDarray isub(NDarray obj)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.isub(obj.ToCupyNDarray));
            else
                return new NDarray(ToNumpyNDarray.isub(obj.ToNumpyNDarray));
        }

        public NDarray isub(ValueType obj)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.isub(obj));
            else
                return new NDarray(ToNumpyNDarray.isub(obj));
        }

        public T item<T>(params int[] args)
        {
            if ((Gpu.Available && Gpu.Use))
                return ToCupyNDarray.item<T>(args);
            else
                return ToNumpyNDarray.item<T>(args);
        }

        public void itemset(params object[] args)
        {
            if ((Gpu.Available && Gpu.Use))
                ToCupyNDarray.itemset(args);
            else
                ToNumpyNDarray.itemset(args);
        }

        public NDarray itruediv(NDarray obj)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.itruediv(obj.ToCupyNDarray));
            else
                return new NDarray(ToNumpyNDarray.itruediv(obj.ToNumpyNDarray));
        }

        public NDarray itruediv(ValueType obj)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.itruediv(obj));
            else
                return new NDarray(ToNumpyNDarray.itruediv(obj));
        }

        public NDarray ixor(NDarray obj)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.ixor(obj.ToCupyNDarray));
            else
                return new NDarray(ToNumpyNDarray.ixor(obj.ToNumpyNDarray));
        }

        public NDarray ixor(ValueType obj)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.ixor(obj));
            else
                return new NDarray(ToNumpyNDarray.ixor(obj));
        }

        public NDarray max(int[] axis = null, NDarray @out = null, bool? keepdims = null, ValueType initial = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.max(axis, @out?.ToCupyNDarray, keepdims, initial));
            else
                return new NDarray(ToNumpyNDarray.max(axis, @out?.ToNumpyNDarray, keepdims, initial));
        }

        public NDarray min(int[] axis = null, NDarray @out = null, bool? keepdims = null, ValueType initial = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.min(axis, @out?.ToCupyNDarray, keepdims));
            else
                return new NDarray(ToNumpyNDarray.min(axis, @out?.ToNumpyNDarray, keepdims, initial));
        }

        public NDarray<bool> not_equals(ValueType valueType)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray<bool>(ToCupyNDarray.not_equals(valueType));
            else
                return new NDarray<bool>(ToNumpyNDarray.not_equals(valueType));
        }

        public NDarray<bool> not_equals(NDarray obj)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray<bool>(ToCupyNDarray.not_equals(obj.ToCupyNDarray));
            else
                return new NDarray<bool>(ToNumpyNDarray.not_equals(obj.ToNumpyNDarray));
        }

        public NDarray pow(ValueType obj)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.pow(obj));
            else
                return new NDarray(ToNumpyNDarray.pow(obj));
        }

        public NDarray reshape(params int[] newshape)
        {
            if ((Gpu.Available && Gpu.Use))
            {
                return new NDarray(ToCupyNDarray.reshape(newshape));
            }
            else
            {
                return new NDarray(ToNumpyNDarray.reshape(newshape));
            }
        }

        public void resize(Shape new_shape, bool? refcheck = null)
        {
            if ((Gpu.Available && Gpu.Use))
            {
                var gpuIsEnabled = Gpu.Available && Gpu.Use;
                Gpu.Use = false;
                var np_ndarray = ToNumpyNDarray;
                np_ndarray.resize(new Shape(new_shape.CupyShape.Dimensions).NumpyShape, false);
                Gpu.Use = gpuIsEnabled;
                CupyNDarray = cpExtensions.asarray(np_ndarray);
            }
            else
                NumpyNDarray.resize(new_shape.NumpyShape, refcheck);
        }

        public void setflags(bool? write = null, bool? align = null, bool? uic = null)
        {
            if ((Gpu.Available && Gpu.Use))
                CupyNDarray.setflags(write, align, uic);
            else
                NumpyNDarray.setflags(write, align, uic);
        }

        public byte[] tobytes(string order = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return ToCupyNDarray.tobytes(order);
            else
                return ToNumpyNDarray.tobytes(order);
        }

        public void tobytes(string fid, string sep, string format)
        {
            if ((Gpu.Available && Gpu.Use))
                CupyNDarray.tofile(fid, sep, format);
            else
                NumpyNDarray.tofile(fid, sep, format);
        }

        public byte[] tostring(string order = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return ToCupyNDarray.tostring(order);
            else
                return ToNumpyNDarray.tostring(order);
        }

        public NDarray transpose(params int[] axes)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.transpose(axes));
            else
                return new NDarray(ToNumpyNDarray.transpose(axes is not null ? (axes.Length == 0 ? null : axes) : null));
        }

        public void view(Dtype dtype = null, Type type = null)
        {
            if ((Gpu.Available && Gpu.Use))
                CupyNDarray.view(dtype.CupyDtype, type);
            else
                NumpyNDarray.view(dtype.NumpyDtype, type);
        }

        #region Extension Methods

        public NDarray prod(Axis axis = null, Dtype dtype = null, NDarray @out = null, bool? keepdims = null, ValueType initial = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.prod(ToCupyNDarray, axis?.CupyAxis, dtype?.CupyDtype, @out?.ToCupyNDarray, keepdims));
            else
                return new NDarray(np.np.prod(ToNumpyNDarray, axis?.NumpyAxis, dtype?.NumpyDtype, @out?.ToNumpyNDarray, keepdims, initial));
        }

        public Dtype GetDtype(object obj)
        {
            if ((Gpu.Available && Gpu.Use))
                return new Dtype(Cupy.DtypeExtensions.GetDtype(ToCupyNDarray));
            else
                return new Dtype(Numpy.DtypeExtensions.GetDtype(ToNumpyNDarray));
        }

        public NDarray absolute(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.absolute(@out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(ToNumpyNDarray.absolute(@out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public NDarray add(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.add(@out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(ToNumpyNDarray.add(@out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public NDarray all(Axis axis, NDarray @out = null, bool? keepdims = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.all(axis.CupyAxis, @out?.ToCupyNDarray, keepdims));
            else
                return new NDarray(ToNumpyNDarray.all(axis.NumpyAxis, @out?.ToNumpyNDarray, keepdims));
        }

        public NDarray allclose(NDarray a, float rtol = 1e-05f, float atol = 1e-08f, bool equal_nan = false)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.allclose(ToCupyNDarray, a.ToCupyNDarray, rtol, atol, equal_nan));
            else
                return new NDarray(np.np.allclose(ToNumpyNDarray, a.ToNumpyNDarray, rtol, atol, equal_nan));
        }

        public NDarray amax(Axis axis = null, NDarray @out = null, bool? keepdims = null, ValueType initial = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.amax(ToCupyNDarray, axis?.CupyAxis, @out?.ToCupyNDarray, keepdims, initial));
            else
                return new NDarray(np.np.amax(ToNumpyNDarray, axis?.NumpyAxis, @out?.ToNumpyNDarray, keepdims, initial));
        }

        public NDarray amin(Axis axis = null, NDarray @out = null, bool? keepdims = null, ValueType initial = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.amin(ToCupyNDarray, axis?.CupyAxis, @out?.ToCupyNDarray, keepdims));
            else
                return new NDarray(np.np.amin(ToNumpyNDarray, axis?.NumpyAxis, @out?.ToNumpyNDarray, keepdims, initial));
        }

        public NDarray angle(bool deg = false)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.angle(ToCupyNDarray, deg));
            else
                return new NDarray(np.np.angle(ToNumpyNDarray, deg));
        }

        public bool any()
        {
            if ((Gpu.Available && Gpu.Use))
                return cp.cp.any(ToCupyNDarray);
            else
                return np.np.any(ToNumpyNDarray);
        }

        public NDarray any(Axis axis, NDarray @out = null, bool? keepdims = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.any(ToCupyNDarray, axis?.CupyAxis, @out?.ToCupyNDarray, keepdims));
            else
                return new NDarray(np.np.any(ToNumpyNDarray, axis?.NumpyAxis, @out?.ToNumpyNDarray, keepdims));
        }

        public NDarray append(NDarray values, int? axis = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.append(ToCupyNDarray, values?.ToCupyNDarray, axis));
            else
                return new NDarray(np.np.append(ToNumpyNDarray, values?.ToNumpyNDarray, axis));
        }

        public NDarray arccos(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.arccos(ToCupyNDarray, @out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(np.np.arccos(ToNumpyNDarray, @out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public NDarray arccosh(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.arccosh(ToCupyNDarray, @out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(np.np.arccosh(ToNumpyNDarray, @out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public NDarray arcsin(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.arcsin(ToCupyNDarray, @out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(np.np.arcsin(ToNumpyNDarray, @out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public NDarray arcsinh(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.arcsinh(ToCupyNDarray, @out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(np.np.arcsinh(ToNumpyNDarray, @out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public NDarray arctan(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.arctan(ToCupyNDarray, @out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(np.np.arctan(ToNumpyNDarray, @out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public NDarray arctan2(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.arctan2(ToCupyNDarray, @out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(np.np.arctan2(ToNumpyNDarray, @out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public NDarray arctanh(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.arctanh(ToCupyNDarray, @out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(np.np.arctanh(ToNumpyNDarray, @out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public NDarray argmax(int? axis = null, NDarray @out = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.argmax(ToCupyNDarray, axis, @out?.ToCupyNDarray));
            else
                return new NDarray(np.np.argmax(ToNumpyNDarray, axis, @out?.ToNumpyNDarray));
        }

        public NDarray argmin(int? axis = null, NDarray @out = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.argmin(ToCupyNDarray, axis, @out?.ToCupyNDarray));
            else
                return new NDarray(np.np.argmin(ToNumpyNDarray, axis, @out?.ToNumpyNDarray));
        }

        public NDarray argpartition(int[] kth, int? axis = -1, string kind = "introselect", string order = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.argpartition(ToCupyNDarray, kth, axis, kind, order));
            else
                return new NDarray(np.np.argpartition(ToNumpyNDarray, kth, axis, kind, order));
        }

        public NDarray argsort(int? axis = -1, string kind = "quicksort", string order = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.argsort(ToCupyNDarray, axis, kind, order));
            else
                return new NDarray(np.np.argsort(ToNumpyNDarray, axis, kind, order));
        }

        public NDarray argwhere()
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.argwhere(ToCupyNDarray));
            else
                return new NDarray(np.np.argwhere(ToNumpyNDarray));
        }

        public NDarray around(int? decimals = 0, NDarray @out = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.around(ToCupyNDarray, decimals, @out?.ToCupyNDarray));
            else
                return new NDarray(np.np.around(ToNumpyNDarray, decimals, @out?.ToNumpyNDarray));
        }

        public bool array_equal(NDarray a2)
        {
            if ((Gpu.Available && Gpu.Use))
                return cp.cp.array_equal(ToCupyNDarray, a2.ToCupyNDarray);
            else
                return np.np.array_equal(ToNumpyNDarray, a2.ToNumpyNDarray);
        }

        public bool array_equiv(NDarray a2)
        {
            if ((Gpu.Available && Gpu.Use))
                return cp.cp.array_equiv(ToCupyNDarray, a2.ToCupyNDarray);
            else
                return np.np.array_equiv(ToNumpyNDarray, a2.ToNumpyNDarray);
        }

        public string array_repr(int? max_line_width = null, int? precision = null, bool? suppress_small = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return cp.cp.array_repr(ToCupyNDarray, max_line_width, precision, suppress_small);
            else
                return np.np.array_repr(ToNumpyNDarray, max_line_width, precision, suppress_small);
        }

        public void array_str(int? max_line_width = null, int? precision = null, bool? suppress_small = null)
        {
            if ((Gpu.Available && Gpu.Use))
                cp.cp.array_str(ToCupyNDarray, max_line_width, precision, suppress_small);
            else
                np.np.array_str(ToNumpyNDarray, max_line_width, precision, suppress_small);
        }

        public NDarray asarray_chkfinite(Dtype dtype = null, string order = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.asarray_chkfinite(ToCupyNDarray, dtype?.CupyDtype, order));
            else
                return new NDarray(np.np.asarray_chkfinite(ToNumpyNDarray, dtype?.NumpyDtype, order));
        }

        public NDarray asfarray(Dtype dtype = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.asfarray(ToCupyNDarray, dtype?.CupyDtype));
            else
                return new NDarray(np.np.asfarray(ToNumpyNDarray, dtype?.NumpyDtype));
        }

        public NDarray asfortranarray(Dtype dtype = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.asfortranarray(ToCupyNDarray, dtype?.CupyDtype));
            else
                return new NDarray(np.np.asfortranarray(ToNumpyNDarray, dtype?.NumpyDtype));
        }

        public NDarray<double> average(Axis axis, NDarray weights = null, bool? returned = false)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray<double>(cp.cp.average(ToCupyNDarray, axis.CupyAxis, weights?.ToCupyNDarray, returned));
            else
                return new NDarray<double>(np.np.average(ToNumpyNDarray, axis.NumpyAxis, weights?.ToNumpyNDarray, returned));
        }

        public NDarray bincount(NDarray weights = null, int? minlength = 0)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.bincount(ToCupyNDarray, weights?.ToCupyNDarray, minlength));
            else
                return new NDarray(np.np.bincount(ToNumpyNDarray, weights?.ToNumpyNDarray, minlength));
        }

        public NDarray bitwise_and(NDarray x1, NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.bitwise_and(ToCupyNDarray, x1.ToCupyNDarray, @out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(np.np.bitwise_and(ToNumpyNDarray, x1.ToNumpyNDarray, @out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public NDarray bitwise_or(NDarray x1, NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.bitwise_or(ToCupyNDarray, x1.ToCupyNDarray, @out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(np.np.bitwise_or(ToNumpyNDarray, x1.ToNumpyNDarray, @out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public NDarray bitwise_xor(NDarray x1, NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.bitwise_xor(ToCupyNDarray, x1.ToCupyNDarray, @out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(np.np.bitwise_xor(ToNumpyNDarray, x1.ToNumpyNDarray, @out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public NDarray broadcast(NDarray in1)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.broadcast(ToCupyNDarray, in1.ToCupyNDarray));
            else
                return new NDarray(np.np.broadcast(ToNumpyNDarray, in1.ToNumpyNDarray));
        }

        public NDarray broadcast_to(Shape shape, bool? subok = false)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.broadcast_to(ToCupyNDarray, shape.CupyShape, subok));
            else
                return new NDarray(np.np.broadcast_to(ToNumpyNDarray, shape.NumpyShape, subok));
        }

        public NDarray cbrt(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.cbrt(ToCupyNDarray, @out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(np.np.cbrt(ToNumpyNDarray, @out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public NDarray ceil(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.ceil(ToCupyNDarray, @out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(np.np.ceil(ToNumpyNDarray, @out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public NDarray clip(NDarray a_min, NDarray a_max, NDarray @out = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.clip(ToCupyNDarray, a_min.ToCupyNDarray, a_max.ToCupyNDarray, @out?.ToCupyNDarray));
            else
                return new NDarray(np.np.clip(ToNumpyNDarray, a_min.ToNumpyNDarray, a_max.ToNumpyNDarray, @out?.ToNumpyNDarray));
        }

        public Dtype common_type(NDarray array1)
        {
            if ((Gpu.Available && Gpu.Use))
                return new Dtype(cp.cp.common_type(ToCupyNDarray, array1.ToCupyNDarray));
            else
                return new Dtype(np.np.common_type(ToNumpyNDarray, array1.ToNumpyNDarray));
        }

        public NDarray conj(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.conj(ToCupyNDarray, @out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(np.np.conj(ToNumpyNDarray, @out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public NDarray convolve(NDarray v, string mode = "full")
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.convolve(ToCupyNDarray, v.ToCupyNDarray, mode));
            else
                return new NDarray(np.np.convolve(ToNumpyNDarray, v.ToNumpyNDarray, mode));
        }

        public NDarray copysign(NDarray x2, NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.copysign(ToCupyNDarray, x2.ToCupyNDarray, @out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(np.np.copysign(ToNumpyNDarray, x2.ToNumpyNDarray, @out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public NDarray corrcoef(NDarray y = null, bool? rowvar = true)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.corrcoef(ToCupyNDarray, y?.ToCupyNDarray, rowvar));
            else
                return new NDarray(np.np.corrcoef(ToNumpyNDarray, y?.ToNumpyNDarray, rowvar));
        }

        public NDarray correlate(NDarray a, string mode = "valid")
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.correlate(ToCupyNDarray, a?.ToCupyNDarray, mode));
            else
                return new NDarray(np.np.correlate(ToNumpyNDarray, a?.ToNumpyNDarray, mode));
        }

        public NDarray cos(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.cos(ToCupyNDarray, @out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(np.np.cos(ToNumpyNDarray, @out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public NDarray cosh(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.cosh(ToCupyNDarray, @out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(np.np.cosh(ToNumpyNDarray, @out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public NDarray count_nonzero()
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.count_nonzero(ToCupyNDarray));
            else
                return new NDarray(np.np.count_nonzero(ToNumpyNDarray));
        }

        public NDarray count_nonzero(Axis axis)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.count_nonzero(ToCupyNDarray, axis.CupyAxis));
            else
                return new NDarray(np.np.count_nonzero(ToNumpyNDarray, axis.NumpyAxis));
        }

        public NDarray cov(NDarray y = null, bool? rowvar = true, bool? bias = false, int? ddof = null, NDarray fweights = null, NDarray aweights = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.cov(ToCupyNDarray, y?.ToCupyNDarray, rowvar, bias, ddof, fweights?.ToCupyNDarray, aweights?.ToCupyNDarray));
            else
                return new NDarray(np.np.cov(ToNumpyNDarray, y?.ToNumpyNDarray, rowvar, bias, ddof, fweights?.ToNumpyNDarray, aweights?.ToNumpyNDarray));
        }

        public NDarray cross(NDarray b, int? axisa = -1, int? axisb = -1, int? axisc = -1, int? axis = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.cross(ToCupyNDarray, b.ToCupyNDarray, axisa, axisb, axisc, axis));
            else
                return new NDarray(np.np.cross(ToNumpyNDarray, b.ToNumpyNDarray, axisa, axisb, axisc, axis));
        }

        public NDarray cumprod(int? axis = null, Dtype dtype = null, NDarray @out = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.cumprod(ToCupyNDarray, axis, dtype?.CupyDtype, @out?.ToCupyNDarray));
            else
                return new NDarray(np.np.cumprod(ToNumpyNDarray, axis, dtype?.NumpyDtype, @out?.ToNumpyNDarray));
        }

        public NDarray cumsum(int? axis = null, Dtype dtype = null, NDarray @out = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.cumsum(ToCupyNDarray, axis, dtype?.CupyDtype, @out?.ToCupyNDarray));
            else
                return new NDarray(np.np.cumsum(ToNumpyNDarray, axis, dtype?.NumpyDtype, @out?.ToNumpyNDarray));
        }

        public NDarray deg2rad(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.deg2rad(ToCupyNDarray, @out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(np.np.deg2rad(ToNumpyNDarray, @out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public NDarray degrees(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.degrees(ToCupyNDarray, @out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(np.np.degrees(ToNumpyNDarray, @out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public NDarray delete(Slice obj, int? axis = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.delete(ToCupyNDarray, obj.CupySlice, axis));
            else
                return new NDarray(np.np.delete(ToNumpyNDarray, obj.NumpySlice, axis));
        }

        public void diag_indices_from()
        {
            if ((Gpu.Available && Gpu.Use))
                cp.cp.diag_indices_from(ToCupyNDarray);
            else
                np.np.diag_indices_from(ToNumpyNDarray);
        }

        public NDarray diagonal(int? offset = 0, int? axis1 = 0, int? axis2 = 1)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.diagonal(ToCupyNDarray, offset, axis1, axis2));
            else
                return new NDarray(np.np.diagonal(ToNumpyNDarray, offset, axis1, axis2));
        }

        public NDarray diff(int? n = 1, int? axis = -1, NDarray append = null, NDarray prepend = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.diff(ToCupyNDarray, n, axis, append?.ToCupyNDarray, prepend?.ToCupyNDarray));
            else
                return new NDarray(np.np.diff(ToNumpyNDarray, n, axis, append?.ToNumpyNDarray, prepend?.ToNumpyNDarray));
        }

        public NDarray digitize(NDarray bins, bool? right = false)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.digitize(ToCupyNDarray, bins.ToCupyNDarray, right));
            else
                return new NDarray(np.np.digitize(ToNumpyNDarray, bins.ToNumpyNDarray, right));
        }

        public NDarray divide(NDarray x2, NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.divide(ToCupyNDarray, x2.ToCupyNDarray, @out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(np.np.divide(ToNumpyNDarray, x2.ToNumpyNDarray, @out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public NDarray dot(NDarray b, NDarray @out = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.dot(b.ToCupyNDarray, @out?.ToCupyNDarray));
            //return new NDarray(cp.cp.dot(ToCupyNDarray, b.ToCupyNDarray, @out?.ToCupyNDarray));
            else
                return new NDarray(ToNumpyNDarray.dot(b.ToNumpyNDarray, @out?.ToNumpyNDarray));
                //return new NDarray(np.np.dot(ToNumpyNDarray, b.ToNumpyNDarray, @out?.ToNumpyNDarray));
        }

        public NDarray ediff1d(NDarray to_end = null, NDarray to_begin = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.ediff1d(ToCupyNDarray, to_end?.ToCupyNDarray, to_begin?.ToCupyNDarray));
            else
                return new NDarray(np.np.ediff1d(ToNumpyNDarray, to_end?.ToNumpyNDarray, to_begin?.ToNumpyNDarray));
        }

        public NDarray equal(NDarray x1, NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.equal(ToCupyNDarray, x1.ToCupyNDarray, @out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(np.np.equal(ToNumpyNDarray, x1.ToNumpyNDarray, @out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public NDarray exp(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.exp(ToCupyNDarray, @out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(np.np.exp(ToNumpyNDarray, @out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public NDarray exp2(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.exp2(ToCupyNDarray, @out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(np.np.exp2(ToNumpyNDarray, @out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public NDarray expand_dims(int axis)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.expand_dims(ToCupyNDarray, axis));
            else
                return new NDarray(np.np.expand_dims(ToNumpyNDarray, axis));
        }

        public NDarray expm1(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.expm1(ToCupyNDarray, @out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(np.np.expm1(ToNumpyNDarray, @out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public NDarray extract(NDarray arr)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.extract(ToCupyNDarray, arr.ToCupyNDarray));
            else
                return new NDarray(np.np.extract(ToNumpyNDarray, arr.ToNumpyNDarray));
        }

        public NDarray fabs(NDarray arr)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.fabs(ToCupyNDarray, arr.ToCupyNDarray));
            else
                return new NDarray(np.np.fabs(ToNumpyNDarray, arr.ToNumpyNDarray));
        }

        public void fill_diagonal(ValueType val, bool wrap = false)
        {
            if ((Gpu.Available && Gpu.Use))
                cp.cp.fill_diagonal(ToCupyNDarray, val, wrap);
            else
                np.np.fill_diagonal(ToNumpyNDarray, val, wrap);
        }

        public NDarray fix(NDarray y = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.fix(ToCupyNDarray, y?.ToCupyNDarray));
            else
                return new NDarray(np.np.fix(ToNumpyNDarray, y?.ToNumpyNDarray));
        }

        public NDarray flatnonzero()
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.flatnonzero(ToCupyNDarray));
            else
                return new NDarray(np.np.flatnonzero(ToNumpyNDarray));
        }

        public NDarray flip(Axis axis = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.flip(ToCupyNDarray, axis?.CupyAxis));
            else
                return new NDarray(np.np.flip(ToNumpyNDarray, axis?.NumpyAxis));
        }

        public NDarray fliplr()
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.fliplr(ToCupyNDarray));
            else
                return new NDarray(np.np.fliplr(ToNumpyNDarray));
        }

        public NDarray flipud()
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.flipud(ToCupyNDarray));
            else
                return new NDarray(np.np.flipud(ToNumpyNDarray));
        }

        public NDarray float_power(NDarray x2, NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.float_power(ToCupyNDarray, x2.ToCupyNDarray, @out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(np.np.float_power(ToNumpyNDarray, x2.ToNumpyNDarray, @out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public NDarray floor(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.floor(ToCupyNDarray, @out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(np.np.floor(ToNumpyNDarray, @out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public NDarray floor_divide(NDarray x2, NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.floor_divide(ToCupyNDarray, x2.ToCupyNDarray, @out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(np.np.floor_divide(ToNumpyNDarray, x2.ToNumpyNDarray, @out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public NDarray fmax(NDarray x1, NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.fmax(ToCupyNDarray, x1.ToCupyNDarray, @out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(np.np.fmax(ToNumpyNDarray, x1.ToNumpyNDarray, @out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public NDarray fmin(NDarray x1, NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.fmin(ToCupyNDarray, x1.ToCupyNDarray, @out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(np.np.fmin(ToNumpyNDarray, x1.ToNumpyNDarray, @out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public NDarray fmod(NDarray x2, NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.fmod(ToCupyNDarray, x2.ToCupyNDarray, @out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(np.np.fmod(ToNumpyNDarray, x2.ToNumpyNDarray, @out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public (NDarray, NDarray) frexp(NDarray out1 = null, NDarray out2 = null, NDarray @out = null,
            NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
            {
                var ret = cp.cp.frexp(ToCupyNDarray, out1?.ToCupyNDarray, out2?.ToCupyNDarray, @out?.ToCupyNDarray,
                    where?.ToCupyNDarray);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2));
            }
            else
            {
                var ret = np.np.frexp(ToNumpyNDarray, out1?.ToNumpyNDarray, out2?.ToNumpyNDarray, @out?.ToNumpyNDarray,
                    where?.ToNumpyNDarray);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2));
            }
        }

        public NDarray fv(NDarray nper, NDarray pmt, NDarray pv, string when = "end")
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.fv(ToCupyNDarray, nper.ToCupyNDarray, pmt.ToCupyNDarray, pv.ToCupyNDarray, when));
            else
                return new NDarray(np.np.fv(ToNumpyNDarray, nper.ToNumpyNDarray, pmt.ToNumpyNDarray, pv.ToNumpyNDarray, when));
        }

        public NDarray gcd(NDarray x1)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.gcd(ToCupyNDarray, x1.ToCupyNDarray));
            else
                return new NDarray(np.np.gcd(ToNumpyNDarray, x1.ToNumpyNDarray));
        }

        public NDarray greater(NDarray x1, NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.greater(ToCupyNDarray, x1.ToCupyNDarray, @out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(np.np.greater(ToNumpyNDarray, x1.ToNumpyNDarray, @out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public NDarray greater_equal(NDarray x1, NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.greater_equal(ToCupyNDarray, x1.ToCupyNDarray, @out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(np.np.greater_equal(ToNumpyNDarray, x1.ToNumpyNDarray, @out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public NDarray heaviside(NDarray x2, NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.heaviside(ToCupyNDarray, x2.ToCupyNDarray, @out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(np.np.heaviside(ToNumpyNDarray, x2.ToNumpyNDarray, @out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public (NDarray, NDarray) histogram(int? bins = null, (float, float)? range = null, bool? normed = null, NDarray weights = null, bool? density = null)
        {
            if ((Gpu.Available && Gpu.Use))
            {
                var ret = cp.cp.histogram(ToCupyNDarray, bins, range, normed, weights?.ToCupyNDarray, density);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2));
            }
            else
            {
                var ret = np.np.histogram(ToNumpyNDarray, bins, range, normed, weights?.ToNumpyNDarray, density);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2));
            }
        }

        public (NDarray, NDarray) histogram(NDarray bins = null, (float, float)? range = null, bool? normed = null, NDarray weights = null, bool? density = null)
        {
            if ((Gpu.Available && Gpu.Use))
            {
                var ret = cp.cp.histogram(ToCupyNDarray, bins?.ToCupyNDarray, range, normed, weights?.ToCupyNDarray, density);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2));
            }
            else
            {
                var ret = np.np.histogram(ToNumpyNDarray, bins?.ToNumpyNDarray, range, normed, weights?.ToNumpyNDarray, density);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2));
            }
        }

        public (NDarray, NDarray) histogram(List<string> bins = null, (float, float)? range = null, bool? normed = null, NDarray weights = null, bool? density = null)
        {
            if ((Gpu.Available && Gpu.Use))
            {
                var ret = cp.cp.histogram(ToCupyNDarray, bins, range, normed, weights?.ToCupyNDarray, density);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2));
            }
            else
            {
                var ret = np.np.histogram(ToNumpyNDarray, bins, range, normed, weights?.ToNumpyNDarray, density);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2));
            }
        }

        public (NDarray, NDarray) histogram2d(NDarray y, int? bins = null, (float, float)? range = null, bool? density = null, bool? normed = null, NDarray weights = null)
        {
            if ((Gpu.Available && Gpu.Use))
            {
                var ret = cp.cp.histogram2d(ToCupyNDarray, y.ToCupyNDarray, bins, range, density, normed, weights?.ToCupyNDarray);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2));
            }
            else
            {
                var ret = np.np.histogram2d(ToNumpyNDarray, y.ToNumpyNDarray, bins, range, density, normed, weights?.ToNumpyNDarray);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2));
            }
        }

        public (NDarray, NDarray) histogram2d(NDarray y, NDarray bins = null, (float, float)? range = null, bool? density = null, bool? normed = null, NDarray weights = null)
        {
            if ((Gpu.Available && Gpu.Use))
            {
                var ret = cp.cp.histogram2d(ToCupyNDarray, y.ToCupyNDarray, bins?.ToCupyNDarray, range, density, normed, weights?.ToCupyNDarray);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2));
            }
            else
            {
                var ret = np.np.histogram2d(ToNumpyNDarray, y.ToNumpyNDarray, bins?.ToNumpyNDarray, range, density, normed, weights?.ToNumpyNDarray);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2));
            }
        }

        public (NDarray, NDarray) histogram2d(NDarray y, List<string> bins = null, (float, float)? range = null, bool? density = null, bool? normed = null, NDarray weights = null)
        {
            if ((Gpu.Available && Gpu.Use))
            {
                var ret = cp.cp.histogram2d(ToCupyNDarray, y.ToCupyNDarray, bins, range, density, normed, weights?.ToCupyNDarray);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2));
            }
            else
            {
                var ret = np.np.histogram2d(ToNumpyNDarray, y.ToNumpyNDarray, bins, range, density, normed, weights?.ToNumpyNDarray);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2));
            }
        }

        public NDarray<float> histogram_bin_edges(int? bins = null, (float, float)? range = null, NDarray weights = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray<float>(cp.cp.histogram_bin_edges(ToCupyNDarray, bins, range, weights?.ToCupyNDarray));
            else
                return new NDarray<float>(np.np.histogram_bin_edges(ToNumpyNDarray, bins, range, weights?.ToNumpyNDarray));
        }

        public NDarray<float> histogram_bin_edges(NDarray bins = null, (float, float)? range = null, NDarray weights = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray<float>(cp.cp.histogram_bin_edges(ToCupyNDarray, bins?.ToCupyNDarray, range, weights?.ToCupyNDarray));
            else
                return new NDarray<float>(np.np.histogram_bin_edges(ToNumpyNDarray, bins?.ToNumpyNDarray, range, weights?.ToNumpyNDarray));
        }

        public NDarray<float> histogram_bin_edges(List<string> bins = null, (float, float)? range = null, NDarray weights = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray<float>(cp.cp.histogram_bin_edges(ToCupyNDarray, bins, range, weights?.ToCupyNDarray));
            else
                return new NDarray<float>(np.np.histogram_bin_edges(ToNumpyNDarray, bins, range, weights?.ToNumpyNDarray));
        }

        public (NDarray, NDarray) histogramdd(int? bins = null, (float, float)? range = null, bool? density = null, bool? normed = null, NDarray weights = null)
        {
            if ((Gpu.Available && Gpu.Use))
            {
                var ret = cp.cp.histogramdd(ToCupyNDarray, bins, range, density, normed, weights?.ToCupyNDarray);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2));
            }
            else
            {
                var ret = np.np.histogramdd(ToNumpyNDarray, bins, range, density, normed, weights?.ToNumpyNDarray);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2));
            }
        }

        public (NDarray, NDarray) histogramdd(NDarray bins = null, (float, float)? range = null, bool? density = null, bool? normed = null, NDarray weights = null)
        {
            if ((Gpu.Available && Gpu.Use))
            {
                var ret = cp.cp.histogramdd(ToCupyNDarray, bins.ToCupyNDarray, range, density, normed, weights?.ToCupyNDarray);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2));
            }
            else
            {
                var ret = np.np.histogramdd(ToNumpyNDarray, bins.ToNumpyNDarray, range, density, normed, weights?.ToNumpyNDarray);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2));
            }
        }

        public (NDarray, NDarray) histogramdd(List<string> bins = null, (float, float)? range = null, bool? density = null, bool? normed = null, NDarray weights = null)
        {
            if ((Gpu.Available && Gpu.Use))
            {
                var ret = cp.cp.histogramdd(ToCupyNDarray, bins, range, density, normed, weights?.ToCupyNDarray);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2));
            }
            else
            {
                var ret = np.np.histogramdd(ToNumpyNDarray, bins, range, density, normed, weights?.ToNumpyNDarray);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2));
            }
        }

        public NDarray hypot(NDarray x1, NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(cp.cp.hypot(ToCupyNDarray, x1.ToCupyNDarray, @out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(np.np.hypot(ToNumpyNDarray, x1.ToNumpyNDarray, @out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public NDarray i0()
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.i0());
            else
                return new NDarray(ToNumpyNDarray.i0());
        }

        public NDarray in1d(NDarray ar2, bool? assume_unique = false, bool? invert = false)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.in1d(ar2.ToCupyNDarray, assume_unique, invert));
            else
                return new NDarray(ToNumpyNDarray.in1d(ar2.ToNumpyNDarray, assume_unique, invert));
        }

        public NDarray inner(NDarray a)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.inner(a.ToCupyNDarray));
            else
                return new NDarray(ToNumpyNDarray.inner(a.ToNumpyNDarray));
        }

        public (NDarray, NDarray, NDarray) intersect1d(NDarray a)
        {
            if ((Gpu.Available && Gpu.Use))
            {
                var ret = ToCupyNDarray.intersect1d(a.ToCupyNDarray);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2), new NDarray(ret.Item3));
            }
            else
            {
                var ret = ToNumpyNDarray.intersect1d(a.ToNumpyNDarray);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2), new NDarray(ret.Item3));
            }
        }

        public NDarray invert(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.invert(@out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(ToNumpyNDarray.invert(@out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public NDarray ipmt(NDarray per, NDarray nper, NDarray pv, NDarray fv = null, string when = "end")
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.ipmt(per.ToCupyNDarray, nper.ToCupyNDarray, pv.ToCupyNDarray, fv?.ToCupyNDarray, when));
            else
                return new NDarray(ToNumpyNDarray.ipmt(per.ToNumpyNDarray, nper.ToNumpyNDarray, pv.ToNumpyNDarray, fv?.ToNumpyNDarray, when));
        }

        public NDarray irr()
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.irr());
            else
                return new NDarray(ToNumpyNDarray.irr());
        }

        public NDarray isclose(NDarray a, float rtol = 1e-05f, float atol = 1e-08f, bool equal_nan = false)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.isclose(a.ToCupyNDarray, rtol, atol, equal_nan));
            else
                return new NDarray(ToNumpyNDarray.isclose(a.ToNumpyNDarray, rtol, atol, equal_nan));
        }

        public NDarray iscomplex()
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.iscomplex());
            else
                return new NDarray(ToNumpyNDarray.iscomplex());
        }

        public NDarray isfinite()
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.isfinite());
            else
                return new NDarray(ToNumpyNDarray.isfinite());
        }

        public NDarray isfortran()
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.isfortran());
            else
                return new NDarray(ToNumpyNDarray.isfortran());
        }

        public NDarray isin(NDarray test_elements, bool? assume_unique = false, bool? invert = false)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.isin(test_elements.ToCupyNDarray, assume_unique, invert));
            else
                return new NDarray(ToNumpyNDarray.isin(test_elements.ToNumpyNDarray, assume_unique, invert));
        }

        public NDarray<bool> isinf(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray<bool>(ToCupyNDarray.isinf(@out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray<bool>(ToNumpyNDarray.isinf(@out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public NDarray isnan(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.isnan(@out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(ToNumpyNDarray.isnan(@out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public NDarray isnat(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.isnat(@out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(ToNumpyNDarray.isnat(@out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public NDarray isneginf(NDarray @out = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.isneginf(@out?.ToCupyNDarray));
            else
                return new NDarray(ToNumpyNDarray.isneginf(@out?.ToNumpyNDarray));
        }

        public NDarray isposinf(NDarray y = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.isposinf(y?.ToCupyNDarray));
            else
                return new NDarray(ToNumpyNDarray.isposinf(y?.ToNumpyNDarray));
        }

        public NDarray isreal()
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.isreal());
            else
                return new NDarray(ToNumpyNDarray.isreal());
        }

        public NDarray kron(NDarray a)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.kron(a.ToCupyNDarray));
            else
                return new NDarray(ToNumpyNDarray.kron(a.ToNumpyNDarray));
        }

        public NDarray lcm(NDarray x1)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.lcm(x1.ToCupyNDarray));
            else
                return new NDarray(ToNumpyNDarray.lcm(x1.ToNumpyNDarray));
        }

        public NDarray ldexp(NDarray x2, NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.ldexp(x2.ToCupyNDarray, @out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(ToNumpyNDarray.ldexp(x2.ToNumpyNDarray, @out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public NDarray less(NDarray x1, NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.less(x1.ToCupyNDarray, @out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(ToNumpyNDarray.less(x1.ToNumpyNDarray, @out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public NDarray less_equal(NDarray x1, NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.less_equal(x1.ToCupyNDarray, @out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(ToNumpyNDarray.less_equal(x1.ToNumpyNDarray, @out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public NDarray lexsort(int? axis = -1)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.lexsort(axis));
            else
                return new NDarray(ToNumpyNDarray.lexsort(axis));
        }

        public NDarray log(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.log(@out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(ToNumpyNDarray.log(@out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public NDarray log10(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.log10(@out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(ToNumpyNDarray.log10(@out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public NDarray log1p(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.log1p(@out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(ToNumpyNDarray.log1p(@out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public NDarray log2(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.log2(@out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(ToNumpyNDarray.log2(@out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public NDarray logaddexp(NDarray x1, NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.logaddexp(x1.ToCupyNDarray, @out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(ToNumpyNDarray.logaddexp(x1.ToNumpyNDarray, @out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public NDarray logaddexp2(NDarray x1, NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.logaddexp2(x1.ToCupyNDarray, @out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(ToNumpyNDarray.logaddexp2(x1.ToNumpyNDarray, @out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public NDarray logical_and(NDarray x1, NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.logical_and(x1.ToCupyNDarray, @out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(ToNumpyNDarray.logical_and(x1.ToNumpyNDarray, @out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public NDarray logical_not(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.logical_not(@out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(ToNumpyNDarray.logical_not(@out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public NDarray logical_or(NDarray x1, NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.logical_or(x1.ToCupyNDarray, @out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(ToNumpyNDarray.logical_or(x1.ToNumpyNDarray, @out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public NDarray logical_xor(NDarray x1, NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.logical_xor(x1.ToCupyNDarray, @out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(ToNumpyNDarray.logical_xor(x1.ToNumpyNDarray, @out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public NDarray matmul(NDarray x1, NDarray @out = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.matmul(x1.ToCupyNDarray, @out?.ToCupyNDarray));
            else
                return new NDarray(ToNumpyNDarray.matmul(x1.ToNumpyNDarray, @out?.ToNumpyNDarray));
        }

        public NDarray maximum(NDarray x1, NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.maximum(x1.ToCupyNDarray, @out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(ToNumpyNDarray.maximum(x1.ToNumpyNDarray, @out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public NDarray<double> mean(Axis axis = null, Dtype dtype = null, NDarray @out = null, bool? keepdims = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray<double>(ToCupyNDarray.mean(axis?.CupyAxis, dtype?.CupyDtype, @out?.ToCupyNDarray, keepdims));
            else if (CupyNDarray is not null)
                return new NDarray<double>(ToCupyNDarray.asnumpy().mean(axis?.NumpyAxis, dtype?.NumpyDtype, @out?.ToNumpyNDarray, keepdims));
            else
                return new NDarray<double>(ToNumpyNDarray.mean(axis?.NumpyAxis, dtype?.NumpyDtype, @out?.ToNumpyNDarray, keepdims));
        }

        public NDarray<double> median(Axis axis = null, NDarray @out = null, bool? overwrite_input = false, bool? keepdims = false)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray<double>(ToCupyNDarray.median(axis?.CupyAxis, @out?.ToCupyNDarray, overwrite_input, keepdims));
            else
                return new NDarray<double>(ToNumpyNDarray.median(axis?.NumpyAxis, @out?.ToNumpyNDarray, overwrite_input, keepdims));
        }

        public Dtype min_scalar_type()
        {
            if ((Gpu.Available && Gpu.Use))
                return new Dtype(ToCupyNDarray.min_scalar_type());
            else
                return new Dtype(ToNumpyNDarray.min_scalar_type());
        }

        public NDarray minimum(NDarray x1, NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.minimum(x1.ToCupyNDarray, @out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(ToNumpyNDarray.minimum(x1.ToNumpyNDarray, @out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public NDarray mirr(ValueType finance_rate, ValueType reinvest_rate)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.mirr(finance_rate, reinvest_rate));
            else
                return new NDarray(ToNumpyNDarray.mirr(finance_rate, reinvest_rate));
        }

        public NDarray mod(NDarray x2, NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.mod(x2.ToCupyNDarray, @out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(ToNumpyNDarray.mod(x2.ToNumpyNDarray, @out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public (NDarray, NDarray) modf(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
            {
                var ret = ToCupyNDarray.modf(@out?.ToCupyNDarray, where?.ToCupyNDarray);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2));
            }
            else
            {
                var ret = ToNumpyNDarray.modf(@out?.ToNumpyNDarray, where?.ToNumpyNDarray);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2));
            }
        }

        public NDarray moveaxis(int[] source, int[] destination)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.moveaxis(source, destination));
            else
                return new NDarray(ToNumpyNDarray.moveaxis(source, destination));
        }

        public NDarray msort()
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.msort());
            else
                return new NDarray(ToNumpyNDarray.msort());
        }

        public NDarray multiply(NDarray x1, NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.multiply(x1.ToCupyNDarray, @out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(ToNumpyNDarray.multiply(x1.ToNumpyNDarray, @out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public void nper(NDarray pmt, NDarray pv, NDarray fv = null, string when = "end")
        {
            if ((Gpu.Available && Gpu.Use))
                CupyNDarray.nper(pmt.ToCupyNDarray, pv.ToCupyNDarray, fv?.ToCupyNDarray, when);
            else
                NumpyNDarray.nper(pmt.ToNumpyNDarray, pv.ToNumpyNDarray, fv?.ToNumpyNDarray, when);
        }

        public NDarray nan_to_num(bool? copy = true)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.nan_to_num(copy));
            else
                return new NDarray(ToNumpyNDarray.nan_to_num(copy));
        }

        public NDarray nanargmax(int? axis = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.nanargmax(axis));
            else
                return new NDarray(ToNumpyNDarray.nanargmax(axis));
        }

        public NDarray nanargmin(int? axis = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.nanargmin(axis));
            else
                return new NDarray(ToNumpyNDarray.nanargmin(axis));
        }

        public NDarray nancumprod(int? axis = null, Dtype dtype = null, NDarray @out = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.nancumprod(axis, dtype?.CupyDtype, @out?.ToCupyNDarray));
            else
                return new NDarray(ToNumpyNDarray.nancumprod(axis, dtype?.NumpyDtype, @out?.ToNumpyNDarray));
        }

        public NDarray nancumsum(int? axis = null, Dtype dtype = null, NDarray @out = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.nancumsum(axis, dtype?.CupyDtype, @out?.ToCupyNDarray));
            else
                return new NDarray(ToNumpyNDarray.nancumsum(axis, dtype?.NumpyDtype, @out?.ToNumpyNDarray));
        }

        public NDarray nanmax(Axis axis, NDarray @out = null, bool? keepdims = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.nanmax(axis.CupyAxis, @out?.ToCupyNDarray, keepdims));
            else
                return new NDarray(ToNumpyNDarray.nanmax(axis.NumpyAxis, @out?.ToNumpyNDarray, keepdims));
        }

        public NDarray<double> nanmean(Axis axis, Dtype dtype = null, NDarray @out = null, bool? keepdims = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray<double>(ToCupyNDarray.nanmean(axis.CupyAxis, dtype?.CupyDtype, @out?.ToCupyNDarray, keepdims));
            else
                return new NDarray<double>(ToNumpyNDarray.nanmean(axis.NumpyAxis, dtype?.NumpyDtype, @out?.ToNumpyNDarray, keepdims));
        }

        public NDarray<double> nanmedian(Axis axis, NDarray @out = null, bool? overwrite_input = false, bool? keepdims = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray<double>(ToCupyNDarray.nanmedian(axis.CupyAxis, @out?.ToCupyNDarray, overwrite_input, keepdims));
            else
                return new NDarray<double>(ToNumpyNDarray.nanmedian(axis.NumpyAxis, @out?.ToNumpyNDarray, overwrite_input, keepdims));
        }

        public NDarray nanmin(Axis axis, NDarray @out = null, bool? keepdims = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.nanmin(axis.CupyAxis, @out?.ToCupyNDarray, keepdims));
            else
                return new NDarray(ToNumpyNDarray.nanmin(axis.NumpyAxis, @out?.ToNumpyNDarray, keepdims));
        }

        public NDarray<double> nanpercentile(NDarray<float> q, Axis axis, NDarray @out = null, bool? overwrite_input = false, string interpolation = "linear", bool? keepdims = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray<double>(cp.cp.nanpercentile(ToCupyNDarray, q.CupyNDarray, axis.CupyAxis, @out?.ToCupyNDarray, overwrite_input, interpolation, keepdims));
            else
                return new NDarray<double>(np.np.nanpercentile(ToNumpyNDarray, q.NumpyNDarray, axis.NumpyAxis, @out?.ToNumpyNDarray, overwrite_input, interpolation, keepdims));
        }

        public NDarray nanpercentile(NDarray<float> q, NDarray @out = null, bool? overwrite_input = false, string interpolation = "linear")
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.nanpercentile(q.CupyNDarray, @out?.ToCupyNDarray, overwrite_input, interpolation));
            else
                return new NDarray(ToNumpyNDarray.nanpercentile(q.NumpyNDarray, @out?.ToNumpyNDarray, overwrite_input, interpolation));
        }

        public NDarray nanstd(Axis axis, Dtype dtype = null, NDarray @out = null, int? ddof = 0, bool? keepdims = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.nanstd(axis.CupyAxis, dtype?.CupyDtype, @out?.ToCupyNDarray, ddof, keepdims));
            else
                return new NDarray(ToNumpyNDarray.nanstd(axis.NumpyAxis, dtype?.NumpyDtype, @out?.ToNumpyNDarray, ddof, keepdims));
        }

        public NDarray nansum(Axis axis = null, Dtype dtype = null, NDarray @out = null, bool? keepdims = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.nansum(axis.CupyAxis, dtype?.CupyDtype, @out?.ToCupyNDarray, keepdims));
            else
                return new NDarray(ToNumpyNDarray.nansum(axis.NumpyAxis, dtype?.NumpyDtype, @out?.ToNumpyNDarray, keepdims));
        }

        public NDarray nanvar(Axis axis, Dtype dtype = null, NDarray @out = null, int? ddof = 0, bool? keepdims = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.nanvar(axis.CupyAxis, dtype?.CupyDtype, @out?.ToCupyNDarray, ddof, keepdims));
            else
                return new NDarray(ToNumpyNDarray.nanvar(axis.NumpyAxis, dtype?.NumpyDtype, @out?.ToNumpyNDarray, ddof, keepdims));
        }

        public void ndenumerate()
        {
            if ((Gpu.Available && Gpu.Use))
                CupyNDarray.ndenumerate();
            else
                NumpyNDarray.ndenumerate();
        }

        public NDarray negative(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.negative(@out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(np.np.negative(ToNumpyNDarray, @out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public NDarray nextafter(NDarray x2, NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.nextafter(x2.ToCupyNDarray, @out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(ToNumpyNDarray.nextafter(x2.ToNumpyNDarray, @out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public NDarray[] nonzero(NDarray x2)
        {
            if ((Gpu.Available && Gpu.Use))
                return cp.cp.nonzero(this.ToCupyNDarray).Select(x => new NDarray(x)).ToArray();
            else
                return np.np.nonzero(this.ToNumpyNDarray).Select(x => new NDarray(x)).ToArray();
        }

        public NDarray not_equal(NDarray x1, NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.not_equal(x1.ToCupyNDarray, @out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(ToNumpyNDarray.not_equal(x1.ToNumpyNDarray, @out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public NDarray outer(NDarray b, NDarray @out = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.outer(b.ToCupyNDarray, @out?.ToCupyNDarray));
            else
                return new NDarray(ToNumpyNDarray.outer(b.ToNumpyNDarray, @out?.ToNumpyNDarray));
        }

        public NDarray pv(NDarray nper, NDarray pmt, NDarray fv = null, string when = "end")
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.pv(nper.ToCupyNDarray, pmt.ToCupyNDarray, fv?.ToCupyNDarray, when));
            else
                return new NDarray(ToNumpyNDarray.pv(nper.ToNumpyNDarray, pmt.ToNumpyNDarray, fv?.ToNumpyNDarray, when));
        }

        public NDarray packbits(int? axis = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.packbits(axis));
            else
                return new NDarray(ToNumpyNDarray.packbits(axis));
        }

        public NDarray pad(NDarray pad_width, string mode, int[] stat_length = null, int[] constant_values = null, int[] end_values = null, string reflect_type = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.pad(pad_width.ToCupyNDarray, mode, stat_length, constant_values, end_values, reflect_type));
            else
                return new NDarray(ToNumpyNDarray.pad(pad_width.ToNumpyNDarray, mode, stat_length, constant_values, end_values, reflect_type));
        }

        public NDarray partition(int[] kth, int? axis = -1, string kind = "introselect", string order = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.partition(kth, axis, kind, order));
            else
                return new NDarray(ToNumpyNDarray.partition(kth, axis, kind, order));
        }

        public NDarray percentile(NDarray<float> q, Axis axis, NDarray @out = null, bool? overwrite_input = false, string interpolation = "linear", bool? keepdims = false)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.percentile(q.CupyNDarray, axis.CupyAxis, @out?.ToCupyNDarray, overwrite_input, interpolation, keepdims));
            else
                return new NDarray(ToNumpyNDarray.percentile(q.NumpyNDarray, axis.NumpyAxis, @out?.ToNumpyNDarray, overwrite_input, interpolation, keepdims));
        }

        public NDarray percentile(NDarray<float> q, NDarray @out = null, bool? overwrite_input = false, string interpolation = "linear")
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.percentile(q.CupyNDarray, @out?.ToCupyNDarray, overwrite_input, interpolation));
            else
                return new NDarray(ToNumpyNDarray.percentile(q.NumpyNDarray, @out?.ToNumpyNDarray, overwrite_input, interpolation));
        }

        public void place(NDarray mask, NDarray vals)
        {
            if ((Gpu.Available && Gpu.Use))
                CupyNDarray.place(mask.ToCupyNDarray, vals.ToCupyNDarray);
            else
                NumpyNDarray.place(mask.ToNumpyNDarray, vals.ToNumpyNDarray);
        }

        public NDarray pmt(NDarray nper, NDarray pv, NDarray fv = null, string when = "end")
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.pmt(nper.ToCupyNDarray, pv.ToCupyNDarray, fv?.ToCupyNDarray, when));
            else
                return new NDarray(ToNumpyNDarray.pmt(nper.ToNumpyNDarray, pv.ToNumpyNDarray, fv?.ToNumpyNDarray, when));
        }

        public NDarray positive()
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.positive());
            else
                return new NDarray(ToNumpyNDarray.positive());
        }

        public NDarray power(NDarray x2, NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.power(x2.ToCupyNDarray, @out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(ToNumpyNDarray.power(x2.ToNumpyNDarray, @out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public void ppmt(NDarray per, NDarray nper, NDarray pv, NDarray fv = null, string when = "end")
        {
            if ((Gpu.Available && Gpu.Use))
                CupyNDarray.ppmt(per.ToCupyNDarray, nper.ToCupyNDarray, pv.ToCupyNDarray, fv?.ToCupyNDarray, when);
            else
                NumpyNDarray.ppmt(per.ToNumpyNDarray, nper.ToNumpyNDarray, pv.ToNumpyNDarray, fv?.ToNumpyNDarray, when);
        }

        public NDarray prod(Axis axis = null, Dtype dtype = null, NDarray @out = null, bool? keepdims = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.prod(axis?.CupyAxis, dtype?.CupyDtype, @out?.ToCupyNDarray, keepdims));
            else
                return new NDarray(ToNumpyNDarray.prod(axis?.NumpyAxis, dtype?.NumpyDtype, @out?.ToNumpyNDarray, keepdims));
        }

        public NDarray ptp(Axis axis = null, NDarray @out = null, bool? keepdims = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.ptp(axis?.CupyAxis, @out?.ToCupyNDarray, keepdims));
            else
                return new NDarray(ToNumpyNDarray.ptp(axis?.NumpyAxis, @out?.ToNumpyNDarray, keepdims));
        }

        public void put(NDarray ind, NDarray v, string mode = "raise")
        {
            if ((Gpu.Available && Gpu.Use))
                CupyNDarray.put(ind.ToCupyNDarray, v.ToCupyNDarray, mode);
            else
                NumpyNDarray.put(ind.ToNumpyNDarray, v.ToNumpyNDarray, mode);
        }

        public void put_along_axis(NDarray indices, NDarray[] values, int axis)
        {
            if ((Gpu.Available && Gpu.Use))
                CupyNDarray.put_along_axis(indices.ToCupyNDarray, values.Select(v => v.ToCupyNDarray).ToArray(), axis);
            else
                NumpyNDarray.put_along_axis(indices.ToNumpyNDarray, values.Select(v => v.ToNumpyNDarray).ToArray(), axis);
        }

        public void putmask(NDarray mask, NDarray values)
        {
            if ((Gpu.Available && Gpu.Use))
                CupyNDarray.putmask(mask.ToCupyNDarray, values.ToCupyNDarray);
            else
                NumpyNDarray.putmask(mask.ToNumpyNDarray, values.ToNumpyNDarray);
        }

        public NDarray quantile(NDarray<float> q, Axis axis, NDarray @out = null, bool? overwrite_input = false, string interpolation = "linear", bool? keepdims = false)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.quantile(q.CupyNDarray, axis.CupyAxis, @out?.ToCupyNDarray, overwrite_input, interpolation, keepdims));
            else
                return new NDarray(ToNumpyNDarray.quantile(q.NumpyNDarray, axis.NumpyAxis, @out?.ToNumpyNDarray, overwrite_input, interpolation, keepdims));
        }

        public NDarray rad2deg(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.rad2deg(@out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(ToNumpyNDarray.rad2deg(@out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public NDarray radians(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.radians(@out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(ToNumpyNDarray.radians(@out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public void rate(NDarray pmt, NDarray pv, NDarray fv, string when = "end", double? guess = null, double? tol = null, int? maxiter = 100)
        {
            if ((Gpu.Available && Gpu.Use))
                CupyNDarray.rate(pmt.ToCupyNDarray, pv.ToCupyNDarray, fv.ToCupyNDarray, when, guess, tol, maxiter);
            else
                NumpyNDarray.rate(pmt.ToNumpyNDarray, pv.ToNumpyNDarray, fv.ToNumpyNDarray, when, guess, tol, maxiter);
        }

        public NDarray ravel(string order = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.ravel(order));
            else
                return new NDarray(ToNumpyNDarray.ravel(order));
        }

        public NDarray reduced_view(Dtype dtype = null)
        {
            if (!Gpu.Available || !Gpu.Use) throw new NotSupportedException();
            return new NDarray(ToCupyNDarray.reduced_view(dtype?.CupyDtype));
        }

        public NDarray real_if_close(float tol = 100)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.real_if_close(tol));
            else
                return new NDarray(ToNumpyNDarray.real_if_close(tol));
        }

        public NDarray reciprocal(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.reciprocal(@out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(ToNumpyNDarray.reciprocal(@out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public NDarray remainder(NDarray x2, NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.remainder(x2.ToCupyNDarray, @out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(ToNumpyNDarray.remainder(x2.ToNumpyNDarray, @out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public NDarray repeat(int[] repeats, int? axis = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.repeat(repeats, axis));
            else
                return new NDarray(ToNumpyNDarray.repeat(repeats, axis));
        }

        public NDarray require(Dtype dtype, string[] requirements = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.require(dtype.CupyDtype, requirements));
            else
                return new NDarray(ToNumpyNDarray.require(dtype.NumpyDtype, requirements));
        }

        public NDarray right_shift(NDarray x2, NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.right_shift(x2.ToCupyNDarray, @out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(ToNumpyNDarray.right_shift(x2.ToNumpyNDarray, @out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public NDarray rint(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.rint(@out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(ToNumpyNDarray.rint(@out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public NDarray roll(int[] shift, Axis axis = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.roll(shift, axis?.CupyAxis));
            else
                return new NDarray(ToNumpyNDarray.roll(shift, axis?.NumpyAxis));
        }

        public NDarray rollaxis(int axis, int? start = 0)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.rollaxis(axis, start));
            else
                return new NDarray(ToNumpyNDarray.rollaxis(axis, start));
        }

        public NDarray roots()
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.roots());
            else
                return new NDarray(ToNumpyNDarray.roots());
        }

        public NDarray rot90(int k = 1, int[] axes = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.rot90(k, axes));
            else
                return new NDarray(ToNumpyNDarray.rot90(k, axes));
        }

        public NDarray round(int decimals = 0, NDarray @out = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(xp.round(ToCupyNDarray, decimals, @out?.ToCupyNDarray));
            else
                return new NDarray(xp.round(ToNumpyNDarray, decimals, @out?.ToNumpyNDarray));
        }

        public NDarray<int> searchsorted(NDarray v, string side = "left", NDarray sorter = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray<int>(ToCupyNDarray.searchsorted(v.ToCupyNDarray, side, sorter?.ToCupyNDarray));
            else
                return new NDarray<int>(ToNumpyNDarray.searchsorted(v.ToNumpyNDarray, side, sorter?.ToNumpyNDarray));
        }

        public NDarray setdiff1d(NDarray ar2, bool assume_unique = false)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.setdiff1d(ar2.ToCupyNDarray, assume_unique));
            else
                return new NDarray(ToNumpyNDarray.setdiff1d(ar2.ToNumpyNDarray, assume_unique));
        }

        public NDarray setxor1d(NDarray ar1, bool assume_unique = false)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.setxor1d(ar1.ToCupyNDarray, assume_unique));
            else
                return new NDarray(ToNumpyNDarray.setxor1d(ar1.ToNumpyNDarray, assume_unique));
        }

        public NDarray sign(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.sign(@out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(ToNumpyNDarray.sign(@out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public NDarray signbit(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.signbit(@out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(ToNumpyNDarray.signbit(@out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public NDarray sin(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.sin(@out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(ToNumpyNDarray.sin(@out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public NDarray sinc()
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.sinc());
            else
                return new NDarray(ToNumpyNDarray.sinc());
        }

        public NDarray sinh(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.sinh(@out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(ToNumpyNDarray.sinh(@out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public NDarray sort(int? axis = -1, string kind = "quicksort", string order = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.sort(axis, kind, order));
            else
                return new NDarray(ToNumpyNDarray.sort(axis, kind, order));
        }

        public NDarray sort_complex()
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.sort_complex());
            else
                return new NDarray(ToNumpyNDarray.sort_complex());
        }

        public NDarray spacing(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.spacing(@out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(ToNumpyNDarray.spacing(@out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public NDarray[] split(int[] indices_or_sections, int? axis = 0)
        {
            if ((Gpu.Available && Gpu.Use))
                return ToCupyNDarray.split(indices_or_sections, axis).Select(x => new NDarray(x)).ToArray();
            else
                return ToNumpyNDarray.split(indices_or_sections, axis).Select(x => new NDarray(x)).ToArray();
        }

        public NDarray[] split(int indices_or_sections, int? axis = 0)
        {
            if ((Gpu.Available && Gpu.Use))
                return ToCupyNDarray.split(indices_or_sections, axis).Select(x => new NDarray(x)).ToArray();
            else
                return ToNumpyNDarray.split([indices_or_sections], axis).Select(x => new NDarray(x)).ToArray();
        }

        public NDarray sqrt(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.sqrt(@out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(ToNumpyNDarray.sqrt(@out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public NDarray square(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.square(@out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(ToNumpyNDarray.square(@out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public NDarray squeeze(Axis axis = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.squeeze(axis?.CupyAxis));
            else
                return new NDarray(ToNumpyNDarray.squeeze(axis?.NumpyAxis));
        }

        public NDarray std(Axis axis, Dtype dtype = null, NDarray @out = null, int? ddof = 0, bool? keepdims = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.std(axis.CupyAxis, dtype?.CupyDtype, @out?.ToCupyNDarray, ddof, keepdims));
            else
                return new NDarray(ToNumpyNDarray.std(axis.NumpyAxis, dtype?.NumpyDtype, @out?.ToNumpyNDarray, ddof, keepdims));
        }

        public NDarray subtract(NDarray x1, NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.subtract(x1.ToCupyNDarray, @out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(ToNumpyNDarray.subtract(x1.ToNumpyNDarray, @out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public NDarray sum(Axis axis = null, Dtype dtype = null, NDarray @out = null, bool? keepdims = null, ValueType initial = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.sum(axis?.CupyAxis, dtype?.CupyDtype, @out?.ToCupyNDarray, keepdims, initial));
            else
                return new NDarray(ToNumpyNDarray.sum(axis?.NumpyAxis, dtype?.NumpyDtype, @out?.ToNumpyNDarray, keepdims, initial));
        }

        public NDarray swapaxes(int axis1, int axis2)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.swapaxes(axis1, axis2));
            else
                return new NDarray(ToNumpyNDarray.swapaxes(axis1, axis2));
        }

        public NDarray take(NDarray indices, int? axis = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                var self = Py.Import("cupy");
                var pyargs = ToTuple(new object[]
                {
                });
                var kwargs = new PyDict();
                kwargs["a"] = ToCupyNDarray.PyObject;
                kwargs["indices"] = indices.ToCupyNDarray.PyObject;
                if (axis is not null) kwargs["axis"] = ToPython(axis);
                dynamic py = self.InvokeMethod("take", pyargs, kwargs);
                return ToCsharp<NDarray>(py);
            }
            else
            {
                var self = Py.Import("numpy");
                var pyargs = ToTuple(new object[]
                {
                });
                var kwargs = new PyDict();
                kwargs["a"] = ToNumpyNDarray.PyObject;
                kwargs["indices"] = indices.ToNumpyNDarray.PyObject;
                if (axis is not null) kwargs["axis"] = ToPython(axis);
                dynamic py = self.InvokeMethod("take", pyargs, kwargs);
                return ToCsharp<NDarray>(py);
            }
        }

        public NDarray take_along_axis(NDarray indices, int? axis = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.take_along_axis(indices.ToCupyNDarray, axis));
            else
                return new NDarray(ToNumpyNDarray.take_along_axis(indices.ToNumpyNDarray, axis));
        }

        public NDarray tan(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.tan(@out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(ToNumpyNDarray.tan(@out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public NDarray tanh(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.tanh(@out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(ToNumpyNDarray.tanh(@out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public NDarray tensordot(NDarray a, int[] axes = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.tensordot(a.ToCupyNDarray, axes));
            else
                return new NDarray(ToNumpyNDarray.tensordot(a.ToNumpyNDarray, axes));
        }

        public NDarray tile(NDarray reps)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.tile(reps.ToCupyNDarray));
            else
                return new NDarray(ToNumpyNDarray.tile(reps.ToNumpyNDarray));
        }

        public NDarray trace(int? offset = 0, int? axis2 = null, int? axis1 = null, Dtype dtype = null, NDarray @out = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.trace(offset, axis2, axis1, dtype?.CupyDtype, @out?.ToCupyNDarray));
            else
                return new NDarray(ToNumpyNDarray.trace(offset, axis2, axis1, dtype?.NumpyDtype, @out?.ToNumpyNDarray));
        }

        public float trapz(NDarray x = null, float? dx = 1.0f, int? axis = -1)
        {
            if ((Gpu.Available && Gpu.Use))
                return ToCupyNDarray.trapz(x?.ToCupyNDarray, dx, axis);
            else
                return ToNumpyNDarray.trapz(x?.ToNumpyNDarray, dx, axis);
        }

        public void tril_indices_from(int? k = 0)
        {
            if ((Gpu.Available && Gpu.Use))
                CupyNDarray.tril_indices_from(k);
            else
                NumpyNDarray.tril_indices_from(k);
        }

        public NDarray trim_zeros(string trim = "fb")
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.trim_zeros(trim));
            else
                return new NDarray(ToNumpyNDarray.trim_zeros(trim));
        }

        public NDarray[] triu_indices_from(int? k = 0)
        {
            if ((Gpu.Available && Gpu.Use))
                return ToCupyNDarray.triu_indices_from(k).Select(x => new NDarray(x)).ToArray();
            else
                return ToNumpyNDarray.triu_indices_from(k).Select(x => new NDarray(x)).ToArray();
        }

        public NDarray true_divide(NDarray x2, NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.true_divide(x2.ToCupyNDarray, @out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(ToNumpyNDarray.true_divide(x2.ToNumpyNDarray, @out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public NDarray trunc(NDarray @out = null, NDarray where = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.trunc(@out?.ToCupyNDarray, where?.ToCupyNDarray));
            else
                return new NDarray(ToNumpyNDarray.trunc(@out?.ToNumpyNDarray, where?.ToNumpyNDarray));
        }

        public NDarray union1d(NDarray ar1)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.union1d(ar1.ToCupyNDarray));
            else
                return new NDarray(ToNumpyNDarray.union1d(ar1.ToNumpyNDarray));
        }

        public NDarray[] unique(bool return_index, bool return_inverse, bool return_counts, int? axis = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return ToCupyNDarray.unique(return_index, return_inverse, return_counts, axis).Select(x => new NDarray(x)).ToArray();
            else
                return ToNumpyNDarray.unique(return_index, return_inverse, return_counts, axis).Select(x => new NDarray(x)).ToArray();
        }

        public NDarray unpackbits(int? axis = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.unpackbits(axis));
            else
                return new NDarray(ToNumpyNDarray.unpackbits(axis));
        }

        public NDarray[] unravel_index(Shape shape, string order = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return ToCupyNDarray.unravel_index(shape.CupyShape, order).Select(x => new NDarray(x)).ToArray();
            else
                return ToNumpyNDarray.unravel_index(shape.NumpyShape, order).Select(x => new NDarray(x)).ToArray();
        }

        public NDarray unwrap(float? discont = 3.141592653589793f, int? axis = -1)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.unwrap(discont, axis));
            else
                return new NDarray(ToNumpyNDarray.unwrap(discont, axis));
        }

        public NDarray<double> var(Axis axis, Dtype dtype = null, NDarray @out = null, int? ddof = 0, bool? keepdims = null)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray<double>(ToCupyNDarray.var(axis.CupyAxis, dtype?.CupyDtype, @out?.ToCupyNDarray, ddof, keepdims));
            else if (CupyNDarray is not null)
                return new NDarray<double>(ToCupyNDarray.asnumpy().var(axis.NumpyAxis, dtype?.NumpyDtype, @out?.ToNumpyNDarray, ddof, keepdims));
            else
                return new NDarray<double>(ToNumpyNDarray.var(axis.NumpyAxis, dtype?.NumpyDtype, @out?.ToNumpyNDarray, ddof, keepdims));
        }

        public double var(Dtype dtype = null, NDarray @out = null, int? ddof = 0)
        {
            if ((Gpu.Available && Gpu.Use))
                return ToCupyNDarray.var(dtype?.CupyDtype, @out?.ToCupyNDarray, ddof);
            else
                return ToNumpyNDarray.var(dtype?.NumpyDtype, @out?.ToNumpyNDarray, ddof);
        }

        public NDarray vdot(NDarray b)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.vdot(b.ToCupyNDarray));
            else
                return new NDarray(ToNumpyNDarray.vdot(b.ToNumpyNDarray));
        }

        public NDarray where(NDarray y, NDarray x)
        {
            if ((Gpu.Available && Gpu.Use))
                return new NDarray(ToCupyNDarray.where(y.ToCupyNDarray, x.ToCupyNDarray));
            else
                return new NDarray(ToNumpyNDarray.where(y.ToNumpyNDarray, x.ToNumpyNDarray));
        }

        public NDarray[] where()
        {
            if ((Gpu.Available && Gpu.Use))
                return ToCupyNDarray.where().Select(x => new NDarray(x)).ToArray();
            else
                return ToNumpyNDarray.where().Select(x => new NDarray(x)).ToArray();
        }



        #endregion //Extension Methods

        //private Stack<ArrayMode> _arrayMode = new();

        //public void Push(ArrayMode arrayMode)
        //{
        //    _arrayMode.Push(arrayMode);
        //    switch (arrayMode)
        //    {
        //        case ArrayMode.cp when NumpyNDarray is not null && CupyNDarray is null:
        //        case ArrayMode.np when CupyNDarray is not null && NumpyNDarray is null:
        //            Switch(false);
        //            break;
        //    }
        //}

        //public void Pop()
        //{
        //    var arrayMode = _arrayMode.Pop();
        //    switch (arrayMode)
        //    {
        //        case ArrayMode.cp when CupyNDarray is not null:
        //            NumpyNDarray = cpExtensions.asnumpy(CupyNDarray);
        //            break;
        //        case ArrayMode.np when NumpyNDarray is not null:
        //            CupyNDarray = cpExtensions.asarray(NumpyNDarray);
        //            break;
        //    }
        //}

        [DebuggerStepThrough]
        private void DisposeNpNDarray()
        {
            NumpyNDarray?.Dispose();
            NumpyNDarray = null;
        }


        private void ReleaseUnmanagedResources()
        {
            NumpyNDarray?.Dispose();
            if (NumpyNDarray is not null) NumpyNDarray = null;
            if (CupyNDarray?.self is not null)
            {
                CupyNDarray?.Dispose();
            }
            if (CupyNDarray is not null) CupyNDarray = null;
        }

        private static readonly ConcurrentQueue<IDisposable> _finalizeQueue = new();
        private static readonly Thread _cleanupThread;

        static NDarray()
        {
            // クリーンアップスレッドの初期化
            _cleanupThread = new Thread(ProcessFinalizeQueue)
            {
                IsBackground = true,
                Name = "Python Resource Cleanup"
            };
            _cleanupThread.Start();
        }

        private static void ProcessFinalizeQueue()
        {
            while (true)
            {
                try
                {
                    // キューが空の場合は待機
                    if (_finalizeQueue.IsEmpty)
                    {
                        Thread.Sleep(100);
                        continue;
                    }

                    // GILを取得できる安全なコンテキストで処理
                    using (Py.GIL())
                    {
                        while (_finalizeQueue.TryDequeue(out var disposable))
                        {
                            if (disposable is not null)
                            {
                                try
                                {
                                    disposable.Dispose();
                                }
                                catch (Exception ex)
                                {
                                    Debug.WriteLine($"Error in cleanup thread: {ex.Message}");
                                }
                            }
                        }
                    }
                }
                catch (Exception ex)
                {
                    Debug.WriteLine($"Error in cleanup thread: {ex.Message}");
                    // 短い待機後に再試行
                    Thread.Sleep(1000);
                }
            }
        }

        ~NDarray()
        {
            // デストラクタではGILを取得せずに最小限の片付けのみを行う
            if (!_disposed)
            {
                if (NumpyNDarray is not null)
                {
                    _finalizeQueue.Enqueue(NumpyNDarray);
                    NumpyNDarray = null;
                }
                if (CupyNDarray is not null)
                {
                    _finalizeQueue.Enqueue(CupyNDarray);
                    CupyNDarray = null;
                }
            }
        }

        public void Dispose()
        {
            if (_disposed) return;

            lock (_disposeLock)
            {
                if (!_disposed)
                {
                    try
                    {
                        // GIL取得を試みる（タイムアウト付き）
                        if (TryAcquireGIL())
                        {
                            // NumpyNDarrayの解放
                            if (NumpyNDarray?.PyObject != null && NumpyNDarray.Handle != IntPtr.Zero)
                            {
                                NumpyNDarray.self?.Dispose();
                                NumpyNDarray.Dispose();
                                NumpyNDarray = null;
                            }

                            // CupyNDarrayの解放
                            if (CupyNDarray?.PyObject != null && CupyNDarray.Handle != IntPtr.Zero)
                            {
                                CupyNDarray.self?.Dispose();
                                CupyNDarray.Dispose();
                                CupyNDarray = null;
                            }
                        }
                        else
                        {
                            // GILが取得できない場合は、参照のクリアのみ行う
                            NumpyNDarray = null;
                            CupyNDarray = null;
                        }
                    }
                    catch (Exception ex)
                    {
                        Debug.WriteLine($"Error during NDarray disposal: {ex.Message}");
                    }
                    finally
                    {
                        _disposed = true;
                        GC.SuppressFinalize(this);
                    }
                }
            }
        }

        private bool TryAcquireGIL()
        {
            lock (_gilLock)
            {
                var sw = Stopwatch.StartNew();
                while (sw.ElapsedMilliseconds < GIL_TIMEOUT_MS)
                {
                    try
                    {
                        using (Py.GIL())
                        {
                            return true;
                        }
                    }
                    catch
                    {
                        Thread.Sleep(1);
                    }
                }
                return false;
            }
        }

#if DEBUG
        public string StackTrace { get; private set; }
#endif
    }

    public class NDarray<T> : NDarray where T : struct
    {

        internal new Numpy.NDarray<T> NumpyNDarray
        {
            get => new Numpy.NDarray<T>(base.NumpyNDarray);
            set => base.NumpyNDarray = value;
        }

        internal new Cupy.NDarray<T> CupyNDarray
        {
            get => new Cupy.NDarray<T>(base.CupyNDarray);
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
            VRAMLeakDetector.TrackAllocation(this);
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
            VRAMLeakDetector.TrackAllocation(this);
        }

        public NDarray(byte obj) : base(obj)
        {
        }

        public NDarray(int obj) : base(obj)
        {
        }

        public NDarray(long obj) : base(obj)
        {
        }

        public NDarray(float obj) : base(obj)
        {
        }

        public NDarray(double obj) : base(obj)
        {
        }

        public NDarray(bool obj) : base(obj)
        {
        }

        public NDarray(T[] array)
        {
            using var nd = xp.array<T>(array);
            if (nd.CupyNDarray is not null)
            {
                base.CupyNDarray = nd.CupyNDarray.copy();
            }
            else
            {
                base.NumpyNDarray = nd.NumpyNDarray.copy();
            }
            VRAMLeakDetector.TrackAllocation(this);
        }
    }

    public class Dtype : IDisposable
    {
        public Numpy.Dtype NumpyDtype { get; private set; }
        public Cupy.Dtype CupyDtype { get; private set; }

        public Dtype(Numpy.Dtype dtype)
        {
            NumpyDtype = dtype;
            VRAMLeakDetector.TrackAllocation(this);
        }

        public Dtype(Cupy.Dtype dtype)
        {
            CupyDtype = dtype;
            VRAMLeakDetector.TrackAllocation(this);
        }

        public Dtype(Numpy.Dtype npDtype, Cupy.Dtype cpDtype)
        {
            NumpyDtype = npDtype;
            CupyDtype = cpDtype;
            VRAMLeakDetector.TrackAllocation(this);
        }

        public Dtype(string dtype)
        {
            var to = DeZero.NET.Extensions.Extensions.dtype(dtype);
            NumpyDtype = to.NumpyDtype;
            CupyDtype = to.CupyDtype;
            VRAMLeakDetector.TrackAllocation(this);
        }

        public IntPtr Handle => Gpu.Available && Gpu.Use ? CupyDtype.Handle : NumpyDtype.Handle;

        public dynamic PyObject => Gpu.Available && Gpu.Use ? CupyDtype?.PyObject : NumpyDtype?.PyObject;

        public PyObject self => Gpu.Available && Gpu.Use ? CupyDtype.self : NumpyDtype.self;

        public static Dtype int8 => new Dtype(np.np.int8, cp.cp.int8);
        public static Dtype i1 => new Dtype(np.np.int8, cp.cp.int8);
        public static Dtype int16 => new Dtype(np.np.int16, cp.cp.int16);
        public static Dtype i2 => new Dtype(np.np.int16, cp.cp.int16);
        public static Dtype int32 => new Dtype(np.np.int32, cp.cp.int32);
        public static Dtype i4 => new Dtype(np.np.int32, cp.cp.int32);
        public static Dtype int64 => new Dtype(np.np.int64, cp.cp.int64);
        public static Dtype i8 => new Dtype(np.np.int64, cp.cp.int64);

        public static Dtype uint8 => new Dtype(np.np.uint8, cp.cp.uint8);
        public static Dtype u1 => new Dtype(np.np.uint8, cp.cp.uint8);
        public static Dtype uint16 => new Dtype(np.np.uint16, cp.cp.uint16);
        public static Dtype u2 => new Dtype(np.np.uint16, cp.cp.uint16);
        public static Dtype uint32 => new Dtype(np.np.uint32, cp.cp.uint32);
        public static Dtype u4 => new Dtype(np.np.uint32, cp.cp.uint32);
        public static Dtype uint64 => new Dtype(np.np.uint64, cp.cp.uint64);
        public static Dtype u8 => new Dtype(np.np.uint64, cp.cp.uint64);

        public static Dtype float16 => new Dtype(np.np.float16, cp.cp.float16);
        public static Dtype f2 => new Dtype(np.np.float16, cp.cp.float16);
        public static Dtype float32 => new Dtype(np.np.float32, cp.cp.float32);
        public static Dtype f4 => new Dtype(np.np.float32, cp.cp.float32);
        public static Dtype float64 => new Dtype(np.np.float64, cp.cp.float64);
        public static Dtype f8 => new Dtype(np.np.float64, cp.cp.float64);
        public static Dtype float128 => new Dtype(np.np.float128, cp.cp.float128);
        public static Dtype f16 => new Dtype(np.np.float128, cp.cp.float128);

        public static Dtype complex64 => new Dtype(np.np.complex64, cp.cp.complex64);
        public static Dtype c8 => new Dtype(np.np.complex64, cp.cp.complex64);
        public static Dtype complex128 => new Dtype(np.np.complex128, cp.cp.complex128);
        public static Dtype c16 => new Dtype(np.np.complex128, cp.cp.complex128);
        public static Dtype complex256 => new Dtype(np.np.complex256, cp.cp.complex256);
        public static Dtype c32 => new Dtype(np.np.complex256, cp.cp.complex256);

        public static Dtype bool_ => new Dtype(np.np.bool_, cp.cp.bool_);

        public static Dtype unicode_ => new Dtype(np.np.unicode_, cp.cp.unicode_);
        public static Dtype U => new Dtype(np.np.unicode_, cp.cp.unicode_);

        public static Dtype object_ => new Dtype(np.np.object_, cp.cp.object_);
        public static Dtype O => new Dtype(np.np.object_, cp.cp.object_);

        public static implicit operator Dtype(string dtype)
        {
            return DeZero.NET.Extensions.Extensions.dtype(dtype);
        }

        public static bool operator ==(Dtype a, Dtype b)
        {
            if (a is null && b is null) return true;
            if (a is null || b is null) return false;
            return a.Equals(b);
        }
        public static bool operator !=(Dtype a, Dtype b)
        {
            return !(a == b);
        }

        public override bool Equals(object obj)
        {
            if (obj is Dtype dtype)
            {
                if (Gpu.Available && Gpu.Use)
                    return CupyDtype.self == dtype.CupyDtype.self;
                else
                    return NumpyDtype.self == dtype.NumpyDtype.self;
            }
            else
            {
                return false;
            }
        }

        public override int GetHashCode()
        {
            if (Gpu.Available && Gpu.Use)
                return CupyDtype.GetHashCode();
            else
                return NumpyDtype.GetHashCode();
        }

        public override string ToString()
        {
            if (Gpu.Available && Gpu.Use)
                return CupyDtype?.ToString() ?? "scalar";
            else
                return NumpyDtype?.ToString() ?? "scalar";
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

        public PyTuple ToTuple(Array input)
        {
            if (Gpu.Available && Gpu.Use)
                return CupyDtype.ToTuple(input);
            else
                return NumpyDtype.ToTuple(input);
        }

        public static Dtype ToPython(object obj)
        {
            if (obj == null) return null;
            switch (obj)
            {
                case string o when o.Contains("class 'numpy.int8'"):
                case string p when p == "i1":
                    return xp.int8;
                case string o when o.Contains("class 'numpy.int16'"):
                case string p when p == "i2":
                    return xp.int16;
                case string o when o.Contains("class 'numpy.int32'"):
                case string p when p == "i4":
                    return xp.int32;
                case string o when o.Contains("class 'numpy.int64'"):
                case string p when p == "i8":
                    return xp.int64;
                case string o when o.Contains("class 'numpy.uint8'"):
                case string p when p == "u1":
                    return xp.uint8;
                case string o when o.Contains("class 'numpy.uint16'"):
                case string p when p == "u2":
                    return xp.uint16;
                case string o when o.Contains("class 'numpy.uint32'"):
                case string p when p == "u4":
                    return xp.uint32;
                case string o when o.Contains("class 'numpy.uint64'"):
                case string p when p == "u8":
                    return xp.uint64;
                case string o when o.Contains("class 'numpy.float16'"):
                case string q when q == "f2":
                    return xp.float16;
                case string o when o.Contains("class 'numpy.float32'"):
                case string p when p == "f":
                case string q when q == "f4":
                    return xp.float32;
                case string o when o.Contains("class 'numpy.float64'"):
                case string p when p == "f8":
                    return xp.float64;
                case string o when o.Contains("class 'numpy.complex64'"):
                case string p when p == "c8":
                    return xp.complex64;
                case string o when o.Contains("class 'numpy.complex128'"):
                case string p when p == "c16":
                    return xp.complex128;
                case string o when o.Contains("class 'numpy.complex256'"):
                case string p when p == "c32":
                    return xp.complex256;
                case string o when o.Contains("class 'numpy.bool'"):
                case string p when p == "?":
                    return xp.bool_;
                case string o when o.Contains("class 'numpy.unicode'"):
                case string p when p == "U":
                    return xp.unicode_;
                case string o when o.Contains("class 'numpy.object'"):
                case string p when p == "O":
                    return xp.object_;
                default:
                    throw new NotImplementedException(
                        $"Type is not yet supported: {obj.GetType().Name}. Add it to 'ToPythonConversions'");
            }
        }

        private void ReleaseUnmanagedResources()
        {
            NumpyDtype?.Dispose();
            NumpyDtype = null;
            CupyDtype?.Dispose();
            CupyDtype = null;
        }

        public void Dispose()
        {
            ReleaseUnmanagedResources();
            GC.SuppressFinalize(this);
        }
    }


    public class Flags : IDisposable
    {
        private Numpy.Models.Flags NumpyFlags;
        private Cupy.Models.Flags CupyFlags;

        public Flags(Numpy.Models.Flags flags)
        {
            NumpyFlags = flags;
            VRAMLeakDetector.TrackAllocation(this);
        }

        public Flags(Cupy.Models.Flags flags)
        {
            CupyFlags = flags;
            VRAMLeakDetector.TrackAllocation(this);
        }


        public IntPtr Handle => Gpu.Available && Gpu.Use ? CupyFlags.Handle : NumpyFlags.Handle;

        public dynamic PyObject => Gpu.Available && Gpu.Use ? CupyFlags.PyObject : NumpyFlags.PyObject;

        public PyObject self => Gpu.Available && Gpu.Use ? CupyFlags.self : NumpyFlags.self;


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

        private void ReleaseUnmanagedResources()
        {
            NumpyFlags?.Dispose();
            NumpyFlags = null;
            CupyFlags?.Dispose();
            CupyFlags = null;
        }

        public void Dispose()
        {
            ReleaseUnmanagedResources();
            GC.SuppressFinalize(this);
        }
    }

    public class Shape : IDisposable
    {
        public Numpy.Models.Shape NumpyShape { get; private set; }
        public Cupy.Models.Shape CupyShape { get; private set; }

        public Shape ToCupyShape => new Shape(CupyShape ?? new Cupy.Models.Shape(NumpyShape.Dimensions));

        public Shape ToNumpyShape => new Shape(NumpyShape ?? new Numpy.Models.Shape(CupyShape.Dimensions));

        public Shape(Numpy.Models.Shape shape)
        {
            NumpyShape = shape;
        }

        public Shape(Cupy.Models.Shape shape)
        {
            CupyShape = shape;
        }

        public Shape(int size)
        {
            if (Gpu.Available && Gpu.Use)
            {
                CupyShape = new Cupy.Models.Shape(size);
            }
            else
            {
                NumpyShape = new Numpy.Models.Shape(size);
            }
        }

        public Shape(params int[] size)
        {
            if (Gpu.Available && Gpu.Use)
            {
                CupyShape = new Cupy.Models.Shape(size);
            }
            else
            {
                NumpyShape = new Numpy.Models.Shape(size);
            }
        }

        public int[] Dimensions => Gpu.Available && Gpu.Use ? ToCupyShape.CupyShape.Dimensions : ToNumpyShape.NumpyShape.Dimensions;

        public object shape => Gpu.Available && Gpu.Use ? ToCupyShape : ToNumpyShape;

        public int this[int n]
        {
            get => Gpu.Available && Gpu.Use ? ToCupyShape.CupyShape[n] : ToNumpyShape.NumpyShape[n];
        }


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

        public static Shape operator +(Shape shape1, Shape shape2)
        {
            return new Shape(shape1.Dimensions.Zip(shape2.Dimensions).Select(x => x.First + x.Second).ToArray());
        }

        public static Shape operator -(Shape shape1, Shape shape2)
        {
            return new Shape(shape1.Dimensions.Zip(shape2.Dimensions).Select(x => x.First - x.Second).ToArray());
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

        public override string ToString()
        {
            if (Gpu.Available && Gpu.Use)
                return $"({string.Join(", ", ToCupyShape.Dimensions)})";
            else
                return $"({string.Join(", ", ToNumpyShape.Dimensions)})";
        }

        //public PyTuple ToTuple(Array input)
        //{
        //    if (Gpu.Available && Gpu.Use)
        //        return CupyShape.ToTuple(input);
        //    else
        //        return NumpyShape.ToTuple(input);
        //}

        public static implicit operator Shape(int size)
        {
            return new Shape(size);
        }

        public static implicit operator Shape(int[] size)
        {
            return new Shape(size);
        }

        public static Shape operator *(Shape shape, int n)
        { 
            return new Shape(shape.Dimensions.Select(d => d * n).ToArray());
        }

        private void ReleaseUnmanagedResources()
        {
            CupyShape = null;
            NumpyShape = null;
        }

        public void Dispose()
        {
            ReleaseUnmanagedResources();
            GC.SuppressFinalize(this);
        }
    }

    public class Axis : IDisposable
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

        public Axis(int axis)
        {
            if (Gpu.Available && Gpu.Use)
                CupyAxis = new Cupy.Models.Axis(axis);
            else
                NumpyAxis = new Numpy.Models.Axis(axis);
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
            var otherAxis = obj as Axis;
            if (Gpu.Available && Gpu.Use)
                return CupyAxis.Equals(otherAxis?.CupyAxis);
            else
                return NumpyAxis.Equals(otherAxis?.NumpyAxis);
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

        public static implicit operator Axis(int axis)
        {
            return new Axis(axis);
        }

        private void ReleaseUnmanagedResources()
        {
            CupyAxis = null;
            NumpyAxis = null;
        }

        public void Dispose()
        {
            ReleaseUnmanagedResources();
            GC.SuppressFinalize(this);
        }
    }

    public class Slice : IDisposable
    {
        public Cupy.Models.Slice CupySlice { get; private set; }
        public Numpy.Models.Slice NumpySlice { get; private set; }

        public Slice()
        {
            if (Gpu.Available && Gpu.Use)
                CupySlice = new Cupy.Models.Slice();
            else
                NumpySlice = new Numpy.Models.Slice();
        }

        public Slice(Cupy.Models.Slice slice)
        {
            CupySlice = slice;
        }

        public Slice(Numpy.Models.Slice slice)
        {
            NumpySlice = slice;
        }

        public Slice(int? start, int? stop)
        {
            if (Gpu.Available && Gpu.Use)
                CupySlice = new Cupy.Models.Slice(start, stop);
            else
                NumpySlice = new Numpy.Models.Slice(start, stop);
        }

        public Slice(int? start, int? stop, int step)
        {
            if (Gpu.Available && Gpu.Use)
                CupySlice = new Cupy.Models.Slice(start, stop, step);
            else
                NumpySlice = new Numpy.Models.Slice(start, stop, step);
        }

        public Slice(int index)
        {
            if (Gpu.Available && Gpu.Use)
                CupySlice = new Cupy.Models.Slice(index, index);
            else
                NumpySlice = new Numpy.Models.Slice(index, index);
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

        public static Slice[] operator *(Slice slice, int n)
        {
            List<Slice> ret = new();
            for (int i = 0; i < n; i++)
            {
                ret.Add(slice);
            }

            return ret.ToArray();
        }

        public static implicit operator Slice(int index)
        {
            return new Slice(index);
        }

        private void ReleaseUnmanagedResources()
        {
            CupySlice = null;
            NumpySlice = null;
        }

        public void Dispose()
        {
            ReleaseUnmanagedResources();
            GC.SuppressFinalize(this);
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

    public class Matrix : IDisposable
    {
        public Cupy.Models.Matrix CupyMatrix { get; internal set; }
        public Numpy.Models.Matrix NumpyMatrix { get; internal set; }
        public dynamic PyObject => Gpu.Available && Gpu.Use ? CupyMatrix.PyObject : NumpyMatrix.PyObject;

        public Matrix(Cupy.Models.Matrix matrix)
        {
            CupyMatrix = matrix;
            VRAMLeakDetector.TrackAllocation(this);
        }

        public Matrix(Numpy.Models.Matrix matrix)
        {
            NumpyMatrix = matrix;
            VRAMLeakDetector.TrackAllocation(this);
        }

        private void ReleaseUnmanagedResources()
        {
            CupyMatrix?.Dispose();
            CupyMatrix = null;
            NumpyMatrix?.Dispose();
            NumpyMatrix = null;
        }

        public void Dispose()
        {
            ReleaseUnmanagedResources();
            GC.SuppressFinalize(this);
        }
    }

    public class Ellipsis : Slice
    {
        public Ellipsis() : base(0, int.MaxValue, 1)
        {
        }
    }

    public class MemMapMode : IDisposable
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

        private void ReleaseUnmanagedResources()
        {
            CupyMemMapMode?.Dispose();
            CupyMemMapMode = null;
            NumpyMemMapMode?.Dispose();
            NumpyMemMapMode = null;
        }

        public void Dispose()
        {
            ReleaseUnmanagedResources();
            GC.SuppressFinalize(this);
        }
    }
}
