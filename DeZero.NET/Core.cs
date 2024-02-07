﻿using Python.Runtime;
using System;
using System.Linq.Expressions;
using System.Reflection;
using Cupy;
using Numpy;
using np = Numpy;
using cp = Cupy;

namespace DeZero.NET
{
    public static class Core
    {
        public static bool GpuAvailable
        {
            get
            {
                try
                {
                    Runtime.PythonDLL = @"C:\Users\boiler\AppData\Local\Programs\Python\Python311\python311.dll";
                    PythonEngine.Initialize();
                    return true;
                }
                catch (Exception e)
                {
                    return false;
                }
            }
        }

        public static bool UseGpu { get; set; } = false;

        static Core()
        {

        }
    }

    public class NDarray
    {
        protected Numpy.NDarray NumpyNDarray;
        protected Cupy.NDarray CupyNDarray;


        public NDarray(PyObject pyobj)
        {
            if (Core.GpuAvailable && Core.UseGpu)
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
            if (Core.GpuAvailable && Core.UseGpu)
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
            if (Core.GpuAvailable && Core.UseGpu)
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
            if (Core.GpuAvailable && Core.UseGpu)
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
            if (Core.GpuAvailable && Core.UseGpu)
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
            if (Core.GpuAvailable && Core.UseGpu)
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
            if (Core.GpuAvailable && Core.UseGpu)
            {
                CupyNDarray = new Cupy.NDarray(obj);
            }
            else
            {
                throw new NotSupportedException();
            }
        }

        public NDarray(Numpy.NDarray t)
        {
            if (!Core.GpuAvailable || !Core.UseGpu)
            {
                NumpyNDarray = new Numpy.NDarray(t);
            }
            else
            {
                throw new NotSupportedException();
            }
        }

        public NDarray(Cupy.NDarray t)
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                CupyNDarray = new Cupy.NDarray(t);
            }
            else
            {
                throw new NotSupportedException();
            }
        }
        
        public NDarray T => Core.GpuAvailable && Core.UseGpu ? new NDarray(CupyNDarray.T) : new NDarray(NumpyNDarray.T);

        public PyObject ctypes => Core.GpuAvailable && Core.UseGpu ? CupyNDarray.ctypes : NumpyNDarray.ctypes;

        public PyObject data => Core.GpuAvailable && Core.UseGpu ? CupyNDarray.data : NumpyNDarray.data;

        public Dtype dtype => Core.GpuAvailable && Core.UseGpu
            ? new Dtype(CupyNDarray.dtype)
            : new Dtype(NumpyNDarray.dtype);

        public Flags flags => Core.GpuAvailable && Core.UseGpu ? new Flags(CupyNDarray.flags) : new Flags(NumpyNDarray.flags);

        public PyObject flat => Core.GpuAvailable && Core.UseGpu ? CupyNDarray.flat : NumpyNDarray.flat;

        public NDarray imag => Core.GpuAvailable && Core.UseGpu ? new NDarray(CupyNDarray.imag) : new NDarray(NumpyNDarray.imag);

        public int itemsize => Core.GpuAvailable && Core.UseGpu ? CupyNDarray.itemsize : NumpyNDarray.itemsize;

        public int len => Core.GpuAvailable && Core.UseGpu ? CupyNDarray.len : NumpyNDarray.len;

        public int nbytes => Core.GpuAvailable && Core.UseGpu ? CupyNDarray.nbytes : NumpyNDarray.nbytes;

        public int ndim => Core.GpuAvailable && Core.UseGpu ? CupyNDarray.ndim : NumpyNDarray.ndim;

        public NDarray real => Core.GpuAvailable && Core.UseGpu ? new NDarray(CupyNDarray.real) : new NDarray(NumpyNDarray.real);

        public string repr => Core.GpuAvailable && Core.UseGpu ? CupyNDarray.repr : NumpyNDarray.repr;

        public Shape shape => Core.GpuAvailable && Core.UseGpu ? new Shape(CupyNDarray.shape) : new Shape(NumpyNDarray.shape);

        public int size => Core.GpuAvailable && Core.UseGpu ? CupyNDarray.size : NumpyNDarray.size;

        public string str => Core.GpuAvailable && Core.UseGpu ? CupyNDarray.str : NumpyNDarray.str;

        public int[] strides => Core.GpuAvailable && Core.UseGpu ? CupyNDarray.strides : NumpyNDarray.strides;

        public IntPtr Handle => Core.GpuAvailable && Core.UseGpu ? CupyNDarray.Handle : NumpyNDarray.Handle;

        public dynamic PyObject => Core.GpuAvailable && Core.UseGpu ? CupyNDarray.PyObject : NumpyNDarray.PyObject;

        public PyObject self => Core.GpuAvailable && Core.UseGpu ? CupyNDarray.self : NumpyNDarray.self;

        public NDarray this[int index] => Core.GpuAvailable && Core.UseGpu ? new NDarray(CupyNDarray[index]) : new NDarray(NumpyNDarray[index]);


        public bool Equals(NDarray other)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return CupyNDarray.Equals(other.CupyNDarray);
            else
                return NumpyNDarray.Equals(other.NumpyNDarray);
        }

        public void __setstate__(int version, Shape shape, Dtype dtype, bool isFortran, string rawdata)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                CupyNDarray.__setstate__(version, shape.CupyShape, dtype.CupyDtype, isFortran, rawdata);
            else
                NumpyNDarray.__setstate__(version, shape.NumpyShape, dtype.NumpyDtype, isFortran, rawdata);
        }

        public NDarray abs(NDarray @out = null, NDarray where = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(CupyNDarray.abs(@out.CupyNDarray, where.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.abs(@out.NumpyNDarray, where.NumpyNDarray));
        }

        public T asscalar<T>()
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return CupyNDarray.asscalar<T>();
            else
                return NumpyNDarray.asscalar<T>();
        }

        public NDarray astype(Dtype dtype, string order = null, string casting = null, bool? subok = null, bool? copy = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(CupyNDarray.astype(dtype.CupyDtype, order, casting, subok, copy));
            else
                return new NDarray(NumpyNDarray.astype(dtype.NumpyDtype, order, casting, subok, copy));
        }

        public NDarray byteswap(bool? inplace = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(CupyNDarray.byteswap(inplace));
            else
                return new NDarray(NumpyNDarray.byteswap(inplace));
        }

        public NDarray copy(string order = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(CupyNDarray.copy(order));
            else
                return new NDarray(NumpyNDarray.copy(order));
        }

        public NDarray divmod(ValueType obj)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(CupyNDarray.divmod(obj));
            else
                return new NDarray(NumpyNDarray.divmod(obj));
        }

        public void dump(string file)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                CupyNDarray.dump(file);
            else
                NumpyNDarray.dump(file);
        }

        public void dumps()
        {
            if (Core.GpuAvailable && Core.UseGpu)
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
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray<bool>(CupyNDarray.equals(valueType));
            else
                return new NDarray<bool>(NumpyNDarray.equals(valueType));
        }

        public NDarray<bool> equals(NDarray obj)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray<bool>(CupyNDarray.equals(obj.CupyNDarray));
            else
                return new NDarray<bool>(NumpyNDarray.equals(obj.NumpyNDarray));
        }

        public void fill(ValueType value)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                CupyNDarray.fill(value);
            else
                NumpyNDarray.fill(value);
        }

        public NDarray flatten(string order = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(CupyNDarray.flatten(order));
            else
                return new NDarray(NumpyNDarray.flatten(order));
        }

        public NDarray floordiv(NDarray a, ValueType obj)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(CupyNDarray.floordiv(a.CupyNDarray, obj));
            else
                return new NDarray(NumpyNDarray.floordiv(a.NumpyNDarray, obj));
        }

        public void getfield(Dtype dtype, int offset)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                CupyNDarray.getfield(dtype.CupyDtype, offset);
            else
                NumpyNDarray.getfield(dtype.NumpyDtype, offset);
        }

        public T GetData<T>()
        {
            if (Core.GpuAvailable && Core.UseGpu)
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
            if (Core.GpuAvailable && Core.UseGpu)
                return CupyNDarray.GetHashCode();
            else
                return NumpyNDarray.GetHashCode();
        }

        public NDarray iadd(NDarray obj)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(CupyNDarray.iadd(obj.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.iadd(obj.NumpyNDarray));
        }

        public NDarray iadd(ValueType obj)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(CupyNDarray.iadd(obj));
            else
                return new NDarray(NumpyNDarray.iadd(obj));
        }

        public NDarray iand(NDarray obj)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(CupyNDarray.iand(obj.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.iand(obj.NumpyNDarray));
        }

        public NDarray iand(ValueType obj)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(CupyNDarray.iand(obj));
            else
                return new NDarray(NumpyNDarray.iand(obj));
        }

        public NDarray idiv(NDarray obj)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(CupyNDarray.idiv(obj.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.idiv(obj.NumpyNDarray));
        }

        public NDarray idiv(ValueType obj)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(CupyNDarray.idiv(obj));
            else
                return new NDarray(NumpyNDarray.idiv(obj));
        }

        public NDarray ifloordiv(NDarray obj)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(CupyNDarray.ifloordiv(obj.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.ifloordiv(obj.NumpyNDarray));
        }

        public NDarray ifloordiv(ValueType obj)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(CupyNDarray.ifloordiv(obj));
            else
                return new NDarray(NumpyNDarray.ifloordiv(obj));
        }

        public NDarray ilshift(NDarray obj)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(CupyNDarray.ilshift(obj.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.ilshift(obj.NumpyNDarray));
        }

        public NDarray ilshift(int obj)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(CupyNDarray.ilshift(obj));
            else
                return new NDarray(NumpyNDarray.ilshift(obj));
        }

        public NDarray imod(NDarray obj)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(CupyNDarray.imod(obj.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.imod(obj.NumpyNDarray));
        }

        public NDarray imod(ValueType obj)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(CupyNDarray.imod(obj));
            else
                return new NDarray(NumpyNDarray.imod(obj));
        }

        public NDarray imul(NDarray obj)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(CupyNDarray.imul(obj.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.imul(obj.NumpyNDarray));
        }

        public NDarray imul(ValueType obj)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(CupyNDarray.imul(obj));
            else
                return new NDarray(NumpyNDarray.imul(obj));
        }

        public NDarray ior(NDarray obj)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(CupyNDarray.ior(obj.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.ior(obj.NumpyNDarray));
        }

        public NDarray ior(ValueType obj)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(CupyNDarray.ior(obj));
            else
                return new NDarray(NumpyNDarray.ior(obj));
        }

        public NDarray ipow(NDarray obj)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(CupyNDarray.ipow(obj.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.ipow(obj.NumpyNDarray));
        }

        public NDarray ipow(ValueType obj)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(CupyNDarray.ipow(obj));
            else
                return new NDarray(NumpyNDarray.ipow(obj));
        }

        public NDarray irshift(NDarray obj)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(CupyNDarray.irshift(obj.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.irshift(obj.NumpyNDarray));
        }

        public NDarray irshift(int obj)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(CupyNDarray.irshift(obj));
            else
                return new NDarray(NumpyNDarray.irshift(obj));
        }

        public NDarray isub(NDarray obj)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(CupyNDarray.isub(obj.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.isub(obj.NumpyNDarray));
        }

        public NDarray isub(ValueType obj)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(CupyNDarray.isub(obj));
            else
                return new NDarray(NumpyNDarray.isub(obj));
        }

        public T item<T>(params int[] args)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return CupyNDarray.item<T>(args);
            else
                return NumpyNDarray.item<T>(args);
        }

        public void itemset(params object[] args)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                CupyNDarray.itemset(args);
            else
                NumpyNDarray.itemset(args);
        }

        public NDarray itruediv(NDarray obj)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(CupyNDarray.itruediv(obj.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.itruediv(obj.NumpyNDarray));
        }

        public NDarray itruediv(ValueType obj)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(CupyNDarray.itruediv(obj));
            else
                return new NDarray(NumpyNDarray.itruediv(obj));
        }

        public NDarray ixor(NDarray obj)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(CupyNDarray.ixor(obj.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.ixor(obj.NumpyNDarray));
        }

        public NDarray ixor(ValueType obj)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(CupyNDarray.ixor(obj));
            else
                return new NDarray(NumpyNDarray.ixor(obj));
        }

        public NDarray max(int[] axis = null, NDarray @out = null, bool? keepdims = null, ValueType initial = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(CupyNDarray.max(axis, @out.CupyNDarray, keepdims, initial));
            else
                return new NDarray(NumpyNDarray.max(axis, @out.NumpyNDarray, keepdims, initial));
        }

        public NDarray min(int[] axis = null, NDarray @out = null, bool? keepdims = null, ValueType initial = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(CupyNDarray.min(axis, @out.CupyNDarray, keepdims));
            else
                return new NDarray(NumpyNDarray.min(axis, @out.NumpyNDarray, keepdims, initial));
        }

        public NDarray<bool> not_equals(ValueType valueType)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray<bool>(CupyNDarray.not_equals(valueType));
            else
                return new NDarray<bool>(NumpyNDarray.not_equals(valueType));
        }

        public NDarray<bool> not_equals(NDarray obj)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray<bool>(CupyNDarray.not_equals(obj.CupyNDarray));
            else
                return new NDarray<bool>(NumpyNDarray.not_equals(obj.NumpyNDarray));
        }

        public NDarray pow(ValueType obj)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(CupyNDarray.pow(obj));
            else
                return new NDarray(NumpyNDarray.pow(obj));
        }

        public NDarray reshape(params int[] newshape)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(CupyNDarray.reshape(newshape));
            else
                return new NDarray(NumpyNDarray.reshape(newshape));
        }

        public void resize(Shape new_shape, bool? refcheck = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                CupyNDarray.resize(new_shape.CupyShape, refcheck);
            else
                NumpyNDarray.resize(new_shape.NumpyShape, refcheck);
        }

        public void setflags(bool? write = null, bool? align = null, bool? uic = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                CupyNDarray.setflags(write, align, uic);
            else
                NumpyNDarray.setflags(write, align, uic);
        }

        public byte[] tobytes(string order = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return CupyNDarray.tobytes(order);
            else
                return NumpyNDarray.tobytes(order);
        }

        public void tobytes(string fid, string sep, string format)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                CupyNDarray.tofile(fid, sep, format);
            else
                NumpyNDarray.tofile(fid, sep, format);
        }

        public byte[] tostring(string order = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return CupyNDarray.tostring(order);
            else
                return NumpyNDarray.tostring(order);
        }

        public NDarray transpose(params int[] axes)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(CupyNDarray.transpose(axes));
            else
                return new NDarray(NumpyNDarray.transpose(axes));
        }

        public void view(Dtype dtype = null, Type type = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                CupyNDarray.view(dtype.CupyDtype, type);
            else
                NumpyNDarray.view(dtype.NumpyDtype, type);
        }
        protected NDarray()
        {
        }

        #region Extension Methods

        public double var(Dtype dtype = null, NDarray @out = null, int? ddof = 0)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return CupyNDarray.var(dtype.CupyDtype, @out.CupyNDarray, ddof);
            else
                return NumpyNDarray.var(dtype.NumpyDtype, @out.NumpyNDarray, ddof);
        }

        public NDarray i0()
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(CupyNDarray.i0());
            else
                return new NDarray(NumpyNDarray.i0());
        }

        public NDarray prod(Axis axis = null, Dtype dtype = null, NDarray @out = null, bool? keepdims = null, ValueType initial = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.prod(CupyNDarray, axis?.CupyAxis, dtype?.CupyDtype, @out?.CupyNDarray, keepdims));
            else
                return new NDarray(np.np.prod(NumpyNDarray, axis?.NumpyAxis, dtype?.NumpyDtype, @out?.NumpyNDarray, keepdims, initial));
        }

        public Dtype GetDtype(object obj)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new Dtype(Cupy.DtypeExtensions.GetDtype(CupyNDarray));
            else
                return new Dtype(Numpy.DtypeExtensions.GetDtype(NumpyNDarray));
        }

        public NDarray absolute(NDarray @out = null, NDarray where = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(CupyNDarray.absolute(@out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.absolute(@out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray add(NDarray @out = null, NDarray where = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(CupyNDarray.add(@out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(NumpyNDarray.add(@out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray all(Axis axis, NDarray @out = null, bool? keepdims = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(CupyNDarray.all(axis.CupyAxis, @out?.CupyNDarray, keepdims));
            else
                return new NDarray(NumpyNDarray.all(axis.NumpyAxis, @out?.NumpyNDarray, keepdims));
        }

        public NDarray allclose(NDarray a, float rtol = 1e-05f, float atol = 1e-08f, bool equal_nan = false)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.allclose(CupyNDarray, a.CupyNDarray, rtol, atol, equal_nan));
            else
                return new NDarray(np.np.allclose(NumpyNDarray, a.NumpyNDarray, rtol, atol, equal_nan));
        }

        public NDarray amax(Axis axis = null, NDarray @out = null, bool? keepdims = null, ValueType initial = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.amax(CupyNDarray, axis?.CupyAxis, @out?.CupyNDarray, keepdims, initial));
            else
                return new NDarray(np.np.amax(NumpyNDarray, axis?.NumpyAxis, @out?.NumpyNDarray, keepdims, initial));
        }

        public NDarray amin(Axis axis = null, NDarray @out = null, bool? keepdims = null, ValueType initial = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.amin(CupyNDarray, axis?.CupyAxis, @out?.CupyNDarray, keepdims));
            else
                return new NDarray(np.np.amin(NumpyNDarray, axis?.NumpyAxis, @out?.NumpyNDarray, keepdims, initial));
        }

        public NDarray angle(bool deg = false)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.angle(CupyNDarray, deg));
            else
                return new NDarray(np.np.angle(NumpyNDarray, deg));
        }

        public bool any()
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return cp.cp.any(CupyNDarray);
            else
                return np.np.any(NumpyNDarray);
        }

        public NDarray any(Axis axis, NDarray @out = null, bool? keepdims = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.any(CupyNDarray, axis?.CupyAxis, @out?.CupyNDarray, keepdims));
            else
                return new NDarray(np.np.any(NumpyNDarray, axis?.NumpyAxis, @out?.NumpyNDarray, keepdims));
        }

        public NDarray append(NDarray values, int? axis = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.append(CupyNDarray, values?.CupyNDarray, axis));
            else
                return new NDarray(np.np.append(NumpyNDarray, values?.NumpyNDarray, axis));
        }

        public NDarray arccos(NDarray @out = null, NDarray where = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.arccos(CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(np.np.arccos(NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray arccosh(NDarray @out = null, NDarray where = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.arccosh(CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(np.np.arccosh(NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray arcsin(NDarray @out = null, NDarray where = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.arcsin(CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(np.np.arcsin(NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray arcsinh(NDarray @out = null, NDarray where = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.arcsinh(CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(np.np.arcsinh(NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray arctan(NDarray @out = null, NDarray where = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.arctan(CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(np.np.arctan(NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray arctan2(NDarray @out = null, NDarray where = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.arctan2(CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(np.np.arctan2(NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray arctanh(NDarray @out = null, NDarray where = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.arctanh(CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(np.np.arctanh(NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray argmax(int? axis = null, NDarray @out = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.argmax(CupyNDarray, axis, @out?.CupyNDarray));
            else
                return new NDarray(np.np.argmax(NumpyNDarray, axis, @out?.NumpyNDarray));
        }

        public NDarray argmin(int? axis = null, NDarray @out = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.argmin(CupyNDarray, axis, @out?.CupyNDarray));
            else
                return new NDarray(np.np.argmin(NumpyNDarray, axis, @out?.NumpyNDarray));
        }

        public NDarray argpartition(int[] kth, int? axis = -1, string kind = "introselect", string order = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.argpartition(CupyNDarray, kth, axis, kind, order));
            else
                return new NDarray(np.np.argpartition(NumpyNDarray, kth, axis, kind, order));
        }

        public NDarray argsort(int? axis = -1, string kind = "quicksort", string order = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.argsort(CupyNDarray, axis, kind, order));
            else
                return new NDarray(np.np.argsort(NumpyNDarray, axis, kind, order));
        }

        public NDarray argwhere()
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.argwhere(CupyNDarray));
            else
                return new NDarray(np.np.argwhere(NumpyNDarray));
        }

        public NDarray around(int? decimals = 0, NDarray @out = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.around(CupyNDarray, decimals, @out?.CupyNDarray));
            else
                return new NDarray(np.np.around(NumpyNDarray, decimals, @out?.NumpyNDarray));
        }

        public bool array_equal(NDarray a2)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return cp.cp.array_equal(CupyNDarray, a2.CupyNDarray);
            else
                return np.np.array_equal(NumpyNDarray, a2.NumpyNDarray);
        }

        public bool array_equiv(NDarray a2)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return cp.cp.array_equiv(CupyNDarray, a2.CupyNDarray);
            else
                return np.np.array_equiv(NumpyNDarray, a2.NumpyNDarray);
        }

        public string array_repr(int? max_line_width = null, int? precision = null, bool? suppress_small = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return cp.cp.array_repr(CupyNDarray, max_line_width, precision, suppress_small);
            else
                return np.np.array_repr(NumpyNDarray, max_line_width, precision, suppress_small);
        }

        public void array_str(int? max_line_width = null, int? precision = null, bool? suppress_small = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                cp.cp.array_str(CupyNDarray, max_line_width, precision, suppress_small);
            else
                np.np.array_str(NumpyNDarray, max_line_width, precision, suppress_small);
        }

        public NDarray asarray_chkfinite(Dtype dtype = null, string order = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.asarray_chkfinite(CupyNDarray, dtype?.CupyDtype, order));
            else
                return new NDarray(np.np.asarray_chkfinite(NumpyNDarray, dtype?.NumpyDtype, order));
        }

        public NDarray asfarray(Dtype dtype = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.asfarray(CupyNDarray, dtype?.CupyDtype));
            else
                return new NDarray(np.np.asfarray(NumpyNDarray, dtype?.NumpyDtype));
        }

        public NDarray asfortranarray(Dtype dtype = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.asfortranarray(CupyNDarray, dtype?.CupyDtype));
            else
                return new NDarray(np.np.asfortranarray(NumpyNDarray, dtype?.NumpyDtype));
        }

        public NDarray<double> average(Axis axis, NDarray weights = null, bool? returned = false)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray<double>(cp.cp.average(CupyNDarray, axis.CupyAxis, weights?.CupyNDarray, returned));
            else
                return new NDarray<double>(np.np.average(NumpyNDarray, axis.NumpyAxis, weights?.NumpyNDarray, returned));
        }

        public NDarray bincount(NDarray weights = null, int? minlength = 0)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.bincount(CupyNDarray, weights?.CupyNDarray, minlength));
            else
                return new NDarray(np.np.bincount(NumpyNDarray, weights?.NumpyNDarray, minlength));
        }

        public NDarray bitwise_and(NDarray x1, NDarray @out = null, NDarray where = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.bitwise_and(CupyNDarray, x1.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(np.np.bitwise_and(NumpyNDarray, x1.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray bitwise_or(NDarray x1, NDarray @out = null, NDarray where = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.bitwise_or(CupyNDarray, x1.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(np.np.bitwise_or(NumpyNDarray, x1.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray bitwise_xor(NDarray x1, NDarray @out = null, NDarray where = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.bitwise_xor(CupyNDarray, x1.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(np.np.bitwise_xor(NumpyNDarray, x1.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray broadcast(NDarray in1)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.broadcast(CupyNDarray, in1.CupyNDarray));
            else
                return new NDarray(np.np.broadcast(NumpyNDarray, in1.NumpyNDarray));
        }

        public NDarray broadcast_to(Shape shape, bool? subok = false)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.broadcast_to(CupyNDarray, shape.CupyShape, subok));
            else
                return new NDarray(np.np.broadcast_to(NumpyNDarray, shape.NumpyShape, subok));
        }

        public NDarray cbrt(NDarray @out = null, NDarray where = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.cbrt(CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(np.np.cbrt(NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray ceil(NDarray @out = null, NDarray where = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.ceil(CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(np.np.ceil(NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray clip(NDarray a_min, NDarray a_max, NDarray @out = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.clip(CupyNDarray, a_min.CupyNDarray, a_max.CupyNDarray, @out?.CupyNDarray));
            else
                return new NDarray(np.np.clip(NumpyNDarray, a_min.NumpyNDarray, a_max.NumpyNDarray, @out?.NumpyNDarray));
        }

        public Dtype common_type(NDarray array1)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new Dtype(cp.cp.common_type(CupyNDarray, array1.CupyNDarray));
            else
                return new Dtype(np.np.common_type(NumpyNDarray, array1.NumpyNDarray));
        }

        public NDarray conj(NDarray @out = null, NDarray where = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.conj(CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(np.np.conj(NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray convolve(NDarray v, string mode = "full")
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.convolve(CupyNDarray, v.CupyNDarray, mode));
            else
                return new NDarray(np.np.convolve(NumpyNDarray, v.NumpyNDarray, mode));
        }

        public NDarray copysign(NDarray x2, NDarray @out = null, NDarray where = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.copysign(CupyNDarray, x2.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(np.np.copysign(NumpyNDarray, x2.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray corrcoef(NDarray y = null, bool? rowvar = true)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.corrcoef(CupyNDarray, y?.CupyNDarray, rowvar));
            else
                return new NDarray(np.np.corrcoef(NumpyNDarray, y?.NumpyNDarray, rowvar));
        }

        public NDarray correlate(NDarray a, string mode = "valid")
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.correlate(CupyNDarray, a?.CupyNDarray, mode));
            else
                return new NDarray(np.np.correlate(NumpyNDarray, a?.NumpyNDarray, mode));
        }

        public NDarray cos(NDarray @out = null, NDarray where = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.cos(CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(np.np.cos(NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray cosh(NDarray @out = null, NDarray where = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.cosh(CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(np.np.cosh(NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray count_nonzero()
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.count_nonzero(CupyNDarray));
            else
                return new NDarray(np.np.count_nonzero(NumpyNDarray));
        }

        public NDarray count_nonzero(Axis axis)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.count_nonzero(CupyNDarray, axis.CupyAxis));
            else
                return new NDarray(np.np.count_nonzero(NumpyNDarray, axis.NumpyAxis));
        }

        public NDarray cov(NDarray y = null, bool? rowvar = true, bool? bias = false, int? ddof = null, NDarray fweights = null, NDarray aweights = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.cov(CupyNDarray, y?.CupyNDarray, rowvar, bias, ddof, fweights?.CupyNDarray, aweights?.CupyNDarray));
            else
                return new NDarray(np.np.cov(NumpyNDarray, y?.NumpyNDarray, rowvar, bias, ddof, fweights?.NumpyNDarray, aweights?.NumpyNDarray));
        }

        public NDarray cross(NDarray b, int? axisa = -1, int? axisb = -1, int? axisc = -1, int? axis = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.cross(CupyNDarray, b.CupyNDarray, axisa, axisb, axisc, axis));
            else
                return new NDarray(np.np.cross(NumpyNDarray, b.NumpyNDarray, axisa, axisb, axisc, axis));
        }

        public NDarray cumprod(int? axis = null, Dtype dtype = null, NDarray @out = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.cumprod(CupyNDarray, axis, dtype?.CupyDtype, @out?.CupyNDarray));
            else
                return new NDarray(np.np.cumprod(NumpyNDarray, axis, dtype?.NumpyDtype, @out?.NumpyNDarray));
        }

        public NDarray cumsum(int? axis = null, Dtype dtype = null, NDarray @out = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.cumsum(CupyNDarray, axis, dtype?.CupyDtype, @out?.CupyNDarray));
            else
                return new NDarray(np.np.cumsum(NumpyNDarray, axis, dtype?.NumpyDtype, @out?.NumpyNDarray));
        }

        public NDarray deg2rad(NDarray @out = null, NDarray where = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.deg2rad(CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(np.np.deg2rad(NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray degrees(NDarray @out = null, NDarray where = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.degrees(CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(np.np.degrees(NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray delete(Slice obj, int? axis = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.delete(CupyNDarray, obj.CupySlice, axis));
            else
                return new NDarray(np.np.delete(NumpyNDarray, obj.NumpySlice, axis));
        }

        public void diag_indices_from()
        {
            if (Core.GpuAvailable && Core.UseGpu)
                cp.cp.diag_indices_from(CupyNDarray);
            else
                np.np.diag_indices_from(NumpyNDarray);
        }

        public NDarray diagonal(int? offset = 0, int? axis1 = 0, int? axis2 = 1)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.diagonal(CupyNDarray, offset, axis1, axis2));
            else
                return new NDarray(np.np.diagonal(NumpyNDarray, offset, axis1, axis2));
        }

        public NDarray diff(int? n = 1, int? axis = -1, NDarray append = null, NDarray prepend = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.diff(CupyNDarray, n, axis, append?.CupyNDarray, prepend?.CupyNDarray));
            else
                return new NDarray(np.np.diff(NumpyNDarray, n, axis, append?.NumpyNDarray, prepend?.NumpyNDarray));
        }

        public NDarray digitize(NDarray bins, bool? right = false)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.digitize(CupyNDarray, bins.CupyNDarray, right));
            else
                return new NDarray(np.np.digitize(NumpyNDarray, bins.NumpyNDarray, right));
        }

        public NDarray divide(NDarray x2, NDarray @out = null, NDarray where = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.divide(CupyNDarray, x2.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(np.np.divide(NumpyNDarray, x2.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray dot(NDarray b, NDarray @out = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.dot(CupyNDarray, b.CupyNDarray, @out?.CupyNDarray));
            else
                return new NDarray(np.np.dot(NumpyNDarray, b.NumpyNDarray, @out?.NumpyNDarray));
        }

        public NDarray ediff1d(NDarray to_end = null, NDarray to_begin = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.ediff1d(CupyNDarray, to_end?.CupyNDarray, to_begin?.CupyNDarray));
            else
                return new NDarray(np.np.ediff1d(NumpyNDarray, to_end?.NumpyNDarray, to_begin?.NumpyNDarray));
        }

        public NDarray equal(NDarray x1, NDarray @out = null, NDarray where = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.equal(CupyNDarray, x1.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(np.np.equal(NumpyNDarray, x1.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray exp(NDarray @out = null, NDarray where = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.exp(CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(np.np.exp(NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray exp2(NDarray @out = null, NDarray where = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.exp2(CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(np.np.exp2(NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray expand_dims(int axis)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.expand_dims(CupyNDarray, axis));
            else
                return new NDarray(np.np.expand_dims(NumpyNDarray, axis));
        }

        public NDarray expm1(NDarray @out = null, NDarray where = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.expm1(CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(np.np.expm1(NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray extract(NDarray arr)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.extract(CupyNDarray, arr.CupyNDarray));
            else
                return new NDarray(np.np.extract(NumpyNDarray, arr.NumpyNDarray));
        }

        public NDarray fabs(NDarray arr)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.fabs(CupyNDarray, arr.CupyNDarray));
            else
                return new NDarray(np.np.fabs(NumpyNDarray, arr.NumpyNDarray));
        }

        public void fill_diagonal(ValueType val, bool wrap = false)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                cp.cp.fill_diagonal(CupyNDarray, val, wrap);
            else
                np.np.fill_diagonal(NumpyNDarray, val, wrap);
        }

        public NDarray fix(NDarray y = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.fix(CupyNDarray, y?.CupyNDarray));
            else
                return new NDarray(np.np.fix(NumpyNDarray, y?.NumpyNDarray));
        }

        public NDarray flatnonzero()
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.flatnonzero(CupyNDarray));
            else
                return new NDarray(np.np.flatnonzero(NumpyNDarray));
        }

        public NDarray flip(Axis axis = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.flip(CupyNDarray, axis?.CupyAxis));
            else
                return new NDarray(np.np.flip(NumpyNDarray, axis?.NumpyAxis));
        }

        public NDarray fliplr()
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.fliplr(CupyNDarray));
            else
                return new NDarray(np.np.fliplr(NumpyNDarray));
        }

        public NDarray flipud()
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.flipud(CupyNDarray));
            else
                return new NDarray(np.np.flipud(NumpyNDarray));
        }

        public NDarray float_power(NDarray x2, NDarray @out = null, NDarray where = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.float_power(CupyNDarray, x2.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(np.np.float_power(NumpyNDarray, x2.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray floor(NDarray @out = null, NDarray where = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.floor(CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(np.np.floor(NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray floor_divide(NDarray x2, NDarray @out = null, NDarray where = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.floor_divide(CupyNDarray, x2.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(np.np.floor_divide(NumpyNDarray, x2.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray fmax(NDarray x1, NDarray @out = null, NDarray where = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.fmax(CupyNDarray, x1.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(np.np.fmax(NumpyNDarray, x1.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray fmin(NDarray x1, NDarray @out = null, NDarray where = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.fmin(CupyNDarray, x1.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(np.np.fmin(NumpyNDarray, x1.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray fmod(NDarray x2, NDarray @out = null, NDarray where = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.fmod(CupyNDarray, x2.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(np.np.fmod(NumpyNDarray, x2.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public (NDarray, NDarray) frexp(NDarray out1 = null, NDarray out2 = null, NDarray @out = null,
            NDarray where = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
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
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.fv(CupyNDarray, nper.CupyNDarray, pmt.CupyNDarray, pv.CupyNDarray, when));
            else
                return new NDarray(np.np.fv(NumpyNDarray, nper.NumpyNDarray, pmt.NumpyNDarray, pv.NumpyNDarray, when));
        }

        public NDarray gcd(NDarray x1)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.gcd(CupyNDarray, x1.CupyNDarray));
            else
                return new NDarray(np.np.gcd(NumpyNDarray, x1.NumpyNDarray));
        }

        public NDarray greater(NDarray x1, NDarray @out = null, NDarray where = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.greater(CupyNDarray, x1.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(np.np.greater(NumpyNDarray, x1.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray greater_equal(NDarray x1, NDarray @out = null, NDarray where = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.greater_equal(CupyNDarray, x1.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(np.np.greater_equal(NumpyNDarray, x1.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public NDarray heaviside(NDarray x2, NDarray @out = null, NDarray where = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray(cp.cp.heaviside(CupyNDarray, x2.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            else
                return new NDarray(np.np.heaviside(NumpyNDarray, x2.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
        }

        public (NDarray, NDarray) histogram(int? bins = null, (float, float)? range = null, bool? normed = null, NDarray weights = null, bool? density = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
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
            if (Core.GpuAvailable && Core.UseGpu)
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
            if (Core.GpuAvailable && Core.UseGpu)
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
            if (Core.GpuAvailable && Core.UseGpu)
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
            if (Core.GpuAvailable && Core.UseGpu)
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
            if (Core.GpuAvailable && Core.UseGpu)
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
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray<float>(cp.cp.histogram_bin_edges(CupyNDarray, bins, range, weights?.CupyNDarray));
            else
                return new NDarray<float>(np.np.histogram_bin_edges(NumpyNDarray, bins, range, weights?.NumpyNDarray));
        }

        public NDarray<float> histogram_bin_edges(NDarray bins = null, (float, float)? range = null, NDarray weights = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray<float>(cp.cp.histogram_bin_edges(CupyNDarray, bins?.CupyNDarray, range, weights?.CupyNDarray));
            else
                return new NDarray<float>(np.np.histogram_bin_edges(NumpyNDarray, bins?.NumpyNDarray, range, weights?.NumpyNDarray));
        }

        public NDarray<float> histogram_bin_edges(List<string> bins = null, (float, float)? range = null, NDarray weights = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return new NDarray<float>(cp.cp.histogram_bin_edges(CupyNDarray, bins, range, weights?.CupyNDarray));
            else
                return new NDarray<float>(np.np.histogram_bin_edges(NumpyNDarray, bins, range, weights?.NumpyNDarray));
        }

        public (NDarray, NDarray) histogramdd(int? bins = null, (float, float)? range = null, bool? density = null, bool? normed = null, NDarray weights = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
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
            if (Core.GpuAvailable && Core.UseGpu)
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
            if (Core.GpuAvailable && Core.UseGpu)
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

        #endregion //Extension Methods
    }

    public class NDarray<T> : NDarray
    {
        public NDarray(Numpy.NDarray<T> t)
        {
            if (!Core.GpuAvailable || !Core.UseGpu)
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
            if (Core.GpuAvailable && Core.UseGpu)
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

        public IntPtr Handle => Core.GpuAvailable && Core.UseGpu ? CupyDtype.Handle : NumpyDtype.Handle;

        public dynamic PyObject => Core.GpuAvailable && Core.UseGpu ? CupyDtype.PyObject : NumpyDtype.PyObject;

        public PyObject self => Core.GpuAvailable && Core.UseGpu ? CupyDtype.self : NumpyDtype.self;

        public void Dispose()
        {
            if (Core.GpuAvailable && Core.UseGpu)
                CupyDtype.Dispose();
            else
                NumpyDtype.Dispose();
        }

        public bool Equals(object obj)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return CupyDtype.Equals(obj);
            else
                return NumpyDtype.Equals(obj);
        }

        public int GetHashCode()
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return CupyDtype.GetHashCode();
            else
                return NumpyDtype.GetHashCode();
        }

        public T SharpToSharp<T>(object obj)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return CupyDtype.SharpToSharp<T>(obj);
            else
                return NumpyDtype.SharpToSharp<T>(obj);
        }

        public T ToCsharp<T>(object obj)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return CupyDtype.ToCsharp<T>(obj);
            else
                return NumpyDtype.ToCsharp<T>(obj);
        }

        public string ToString()
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return CupyDtype.ToString();
            else
                return NumpyDtype.ToString();
        }

        public PyTuple ToTuple(Array input)
        {
            if (Core.GpuAvailable && Core.UseGpu)
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


        public IntPtr Handle => Core.GpuAvailable && Core.UseGpu ? CupyFlags.Handle : NumpyFlags.Handle;

        public dynamic PyObject => Core.GpuAvailable && Core.UseGpu ? CupyFlags.PyObject : NumpyFlags.PyObject;

        public PyObject self => Core.GpuAvailable && Core.UseGpu ? CupyFlags.self : NumpyFlags.self;

        public void Dispose()
        {
            if (Core.GpuAvailable && Core.UseGpu)
                CupyFlags.Dispose();
            else
                NumpyFlags.Dispose();
        }

        public bool Equals(object obj)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return CupyFlags.Equals(obj);
            else
                return NumpyFlags.Equals(obj);
        }

        public int GetHashCode()
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return CupyFlags.GetHashCode();
            else
                return NumpyFlags.GetHashCode();
        }

        public T SharpToSharp<T>(object obj)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return CupyFlags.SharpToSharp<T>(obj);
            else
                return NumpyFlags.SharpToSharp<T>(obj);
        }

        public T ToCsharp<T>(object obj)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return CupyFlags.ToCsharp<T>(obj);
            else
                return NumpyFlags.ToCsharp<T>(obj);
        }

        public string ToString()
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return CupyFlags.ToString();
            else
                return NumpyFlags.ToString();
        }

        public PyTuple ToTuple(Array input)
        {
            if (Core.GpuAvailable && Core.UseGpu)
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

        public int[] Dimentions => Core.GpuAvailable && Core.UseGpu ? CupyShape.Dimensions : NumpyShape.Dimensions;

        public int this[int n] => Core.GpuAvailable && Core.UseGpu ? CupyShape[n] : NumpyShape[n];


        public bool Equals(object obj)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return CupyShape.Equals(obj);
            else
                return NumpyShape.Equals(obj);
        }

        public int GetHashCode()
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return CupyShape.GetHashCode();
            else
                return NumpyShape.GetHashCode();
        }

        //public T SharpToSharp<T>(object obj)
        //{
        //    if (Core.GpuAvailable && Core.UseGpu)
        //    {
        //        return CupyShape.SharpToSharp<T>(obj);
        //    }
        //    else
        //        return NumpyShape.SharpToSharp<T>(obj);
        //}

        //public T ToCsharp<T>(object obj)
        //{
        //    if (Core.GpuAvailable && Core.UseGpu)
        //        return CupyShape.ToCsharp<T>(obj);
        //    else
        //        return NumpyShape.ToCsharp<T>(obj);
        //}

        public string ToString()
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return CupyShape.ToString();
            else
                return NumpyShape.ToString();
        }

        //public PyTuple ToTuple(Array input)
        //{
        //    if (Core.GpuAvailable && Core.UseGpu)
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

        public int[] Axes => Core.GpuAvailable && Core.UseGpu ? CupyAxis.Axes : NumpyAxis.Axes;

        public bool Equals(object obj)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return CupyAxis.Equals(obj);
            else
                return NumpyAxis.Equals(obj);
        }

        public int GetHashCode()
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return CupyAxis.GetHashCode();
            else
                return NumpyAxis.GetHashCode();
        }

        public string ToString()
        {
            if (Core.GpuAvailable && Core.UseGpu)
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

        public bool IsIndex => Core.GpuAvailable && Core.UseGpu ? CupySlice.IsIndex : NumpySlice.IsIndex;

        public int? Length => Core.GpuAvailable && Core.UseGpu ? CupySlice.Length : NumpySlice.Length;

        public int? Start => Core.GpuAvailable && Core.UseGpu ? CupySlice.Start : NumpySlice.Start;

        public int Step => Core.GpuAvailable && Core.UseGpu ? CupySlice.Step : NumpySlice.Step;
        
        public int? Stop => Core.GpuAvailable && Core.UseGpu ? CupySlice.Stop : NumpySlice.Stop;

        public int GetAbsStart(int dim)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return CupySlice.GetAbsStart(dim);
            else
                return NumpySlice.GetAbsStart(dim);
        }

        public int GetAbsStep()
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return CupySlice.GetAbsStep();
            else
                return NumpySlice.GetAbsStep();
        }

        public int GetAbsStop(int dim)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return CupySlice.GetAbsStop(dim);
            else
                return NumpySlice.GetAbsStop(dim);
        }

        public int GetSize(int dim)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return CupySlice.GetSize(dim);
            else
                return NumpySlice.GetSize(dim);
        }

        public override bool Equals(object? obj)
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return CupySlice.Equals(obj);
            else
                return NumpySlice.Equals(obj);
        }

        public override int GetHashCode()
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return CupySlice.GetHashCode();
            else
                return NumpySlice.GetHashCode();
        }

        public override string ToString()
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return CupySlice.ToString();
            else
                return NumpySlice.ToString();
        }

        public PyObject ToPython()
        {
            if (Core.GpuAvailable && Core.UseGpu)
                return CupySlice.ToPython();
            else
                return NumpySlice.ToPython();
        }
    }
}
