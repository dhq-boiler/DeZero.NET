using Cupy;
using Numpy;
using Python.Runtime;

namespace DeZero.NET
{
    public static partial class xp
    {
        /// <summary>
        ///     compatible: Python bool
        /// </summary>
        public static Dtype bool_
        {
            get
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new Dtype(cp.bool_);
                }
                else
                {
                    return new Dtype(np.bool_);
                }
            }
            set
            {
                if (Gpu.Available && Gpu.Use)
                {
                    cp.bool_ = value.CupyDtype;
                }
                else
                {
                    np.bool_ = value.NumpyDtype;
                }
            }
        }

        /// <summary>
        ///     8 bits
        /// </summary>
        public static Dtype bool8
        {
            get
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new Dtype(cp.bool8);
                }
                else
                {
                    return new Dtype(np.bool8);
                }
            }
            set
            {
                if (Gpu.Available && Gpu.Use)
                {
                    cp.bool8 = value.CupyDtype;
                }
                else
                {
                    np.bool8 = value.NumpyDtype;
                }
            }
        }

        /// <summary>
        ///     compatible: C char
        /// </summary>
        public static Dtype @byte
        {
            get
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new Dtype(cp.@byte);
                }
                else
                {
                    return new Dtype(np.@byte);
                }
            }
            set
            {
                if (Gpu.Available && Gpu.Use)
                {
                    cp.@byte = value.CupyDtype;
                }
                else
                {
                    np.@byte = value.NumpyDtype;
                }
            }
        }

        /// <summary>
        ///     compatible: C short
        /// </summary>
        public static Dtype @short
        {
            get
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new Dtype(cp.@short);
                }
                else
                {
                    return new Dtype(np.@short);
                }
            }
            set
            {
                if (Gpu.Available && Gpu.Use)
                {
                    cp.@short = value.CupyDtype;
                }
                else
                {
                    np.@short = value.NumpyDtype;
                }
            }
        }

        /// <summary>
        ///     compatible: C int
        /// </summary>
        public static Dtype intc
        {
            get
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new Dtype(cp.intc);
                }
                else
                {
                    return new Dtype(np.intc);
                }
            }
            set
            {
                if (Gpu.Available && Gpu.Use)
                {
                    cp.intc = value.CupyDtype;
                }
                else
                {
                    np.intc = value.NumpyDtype;
                }
            }
        }

        /// <summary>
        ///     compatible: Python int
        /// </summary>
        public static Dtype int_
        {
            get
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new Dtype(cp.int_);
                }
                else
                {
                    return new Dtype(np.int_);
                }
            }
            set
            {
                if (Gpu.Available && Gpu.Use)
                {
                    cp.int_ = value.CupyDtype;
                }
                else
                {
                    np.int_ = value.NumpyDtype;
                }
            }
        }

        /// <summary>
        ///     compatible: C long long
        /// </summary>
        public static Dtype longlong
        {
            get
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new Dtype(cp.longlong);
                }
                else
                {
                    return new Dtype(np.longlong);
                }
            }
            set
            {
                if (Gpu.Available && Gpu.Use)
                {
                    cp.longlong = value.CupyDtype;
                }
                else
                {
                    np.longlong = value.NumpyDtype;
                }
            }
        }

        /// <summary>
        ///     large enough to fit a pointer
        /// </summary>
        public static Dtype intp
        {
            get
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new Dtype(cp.intp);
                }
                else
                {
                    return new Dtype(np.intp);
                }
            }
            set
            {
                if (Gpu.Available && Gpu.Use)
                {
                    cp.intp = value.CupyDtype;
                }
                else
                {
                    np.intp = value.NumpyDtype;
                }
            }
        }

        /// <summary>
        ///     8 bits
        /// </summary>
        public static Dtype int8
        {
            get
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new Dtype(cp.int8);
                }
                else
                {
                    return new Dtype(np.int8);
                }
            }
            set
            {
                if (Gpu.Available && Gpu.Use)
                {
                    cp.int8 = value.CupyDtype;
                }
                else
                {
                    np.int8 = value.NumpyDtype;
                }
            }
        }

        /// <summary>
        ///     16 bits
        /// </summary>
        public static Dtype int16
        {
            get
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new Dtype(cp.int16);
                }
                else
                {
                    return new Dtype(np.int16);
                }
            }
            set
            {
                if (Gpu.Available && Gpu.Use)
                {
                    cp.int16 = value.CupyDtype;
                }
                else
                {
                    np.int16 = value.NumpyDtype;
                }
            }
        }

        public static dynamic Int16(Int64 val)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return cp.Int16(val);
            }
            else
            {
                throw new NotSupportedException();
            }
        }

        /// <summary>
        ///     32 bits
        /// </summary>
        public static Dtype int32
        {
            get
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new Dtype(cp.int32);
                }
                else
                {
                    return new Dtype(np.int32);
                }
            }
            set
            {
                if (Gpu.Available && Gpu.Use)
                {
                    cp.int32 = value.CupyDtype;
                }
                else
                {
                    np.int32 = value.NumpyDtype;
                }
            }
        }

        public static dynamic Int32(Int64 val)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return cp.Int32(val);
            }
            else
            {
                throw new NotSupportedException();
            }
        }

        /// <summary>
        ///     64 bits
        /// </summary>
        public static Dtype int64
        {
            get
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new Dtype(cp.int64);
                }
                else
                {
                    return new Dtype(np.int64);
                }
            }
            set
            {
                if (Gpu.Available && Gpu.Use)
                {
                    cp.int64 = value.CupyDtype;
                }
                else
                {
                    np.int64 = value.NumpyDtype;
                }
            }
        }

        public static dynamic Int64(Int64 val)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return cp.Int64(val);
            }
            else
            {
                throw new NotSupportedException();
            }
        }

        /// <summary>
        ///     compatible: C unsigned char
        /// </summary>
        public static Dtype ubyte
        {
            get
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new Dtype(cp.ubyte);
                }
                else
                {
                    return new Dtype(np.ubyte);
                }
            }
            set
            {
                if (Gpu.Available && Gpu.Use)
                {
                    cp.ubyte = value.CupyDtype;
                }
                else
                {
                    np.ubyte = value.NumpyDtype;
                }
            }
        }

        /// <summary>
        ///     compatible: C unsigned short
        /// </summary>
        public static Dtype @ushort
        {
            get
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new Dtype(cp.@ushort);
                }
                else
                {
                    return new Dtype(np.@ushort);
                }
            }
            set
            {
                if (Gpu.Available && Gpu.Use)
                {
                    cp.@ushort = value.CupyDtype;
                }
                else
                {
                    np.@ushort = value.NumpyDtype;
                }
            }
        }

        /// <summary>
        ///     compatible: C unsigned int
        /// </summary>
        public static Dtype uintc
        {
            get
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new Dtype(cp.uintc);
                }
                else
                {
                    return new Dtype(np.uintc);
                }
            }
            set
            {
                if (Gpu.Available && Gpu.Use)
                {
                    cp.uintc = value.CupyDtype;
                }
                else
                {
                    np.uintc = value.NumpyDtype;
                }
            }
        }

        /// <summary>
        ///     compatible: Python int
        /// </summary>
        public static Dtype @uint
        {
            get
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new Dtype(cp.@uint);
                }
                else
                {
                    return new Dtype(np.@uint);
                }
            }
            set
            {
                if (Gpu.Available && Gpu.Use)
                {
                    cp.@uint = value.CupyDtype;
                }
                else
                {
                    np.@uint = value.NumpyDtype;
                }
            }
        }

        /// <summary>
        ///     compatible: C long long
        /// </summary>
        public static Dtype ulonglong
        {
            get
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new Dtype(cp.ulonglong);
                }
                else
                {
                    return new Dtype(np.ulonglong);
                }
            }
            set
            {
                if (Gpu.Available && Gpu.Use)
                {
                    cp.ulonglong = value.CupyDtype;
                }
                else
                {
                    np.ulonglong = value.NumpyDtype;
                }
            }
        }

        /// <summary>
        ///     large enough to fit a pointer
        /// </summary>
        public static Dtype uintp
        {
            get
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new Dtype(cp.uintp);
                }
                else
                {
                    return new Dtype(np.uintp);
                }
            }
            set
            {
                if (Gpu.Available && Gpu.Use)
                {
                    cp.uintp = value.CupyDtype;
                }
                else
                {
                    np.uintp = value.NumpyDtype;
                }
            }
        }

        /// <summary>
        ///     8 bits
        /// </summary>
        public static Dtype uint8
        {
            get
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new Dtype(cp.uint8);
                }
                else
                {
                    return new Dtype(np.uint8);
                }
            }
            set
            {
                if (Gpu.Available && Gpu.Use)
                {
                    cp.uint8 = value.CupyDtype;
                }
                else
                {
                    np.uint8 = value.NumpyDtype;
                }
            }
        }

        /// <summary>
        ///     16 bits
        /// </summary>
        public static Dtype uint16
        {
            get
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new Dtype(cp.uint16);
                }
                else
                {
                    return new Dtype(np.uint16);
                }
            }
            set
            {
                if (Gpu.Available && Gpu.Use)
                {
                    cp.uint16 = value.CupyDtype;
                }
                else
                {
                    np.uint16 = value.NumpyDtype;
                }
            }
        }

        /// <summary>
        ///     32 bits
        /// </summary>
        public static Dtype uint32
        {
            get
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new Dtype(cp.uint32);
                }
                else
                {
                    return new Dtype(np.uint32);
                }
            }
            set
            {
                if (Gpu.Available && Gpu.Use)
                {
                    cp.uint32 = value.CupyDtype;
                }
                else
                {
                    np.uint32 = value.NumpyDtype;
                }
            }
        }

        /// <summary>
        ///     64 bits
        /// </summary>
        public static Dtype uint64
        {
            get
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new Dtype(cp.uint64);
                }
                else
                {
                    return new Dtype(np.uint64);
                }
            }
            set
            {
                if (Gpu.Available && Gpu.Use)
                {
                    cp.uint64 = value.CupyDtype;
                }
                else
                {
                    np.uint64 = value.NumpyDtype;
                }
            }
        }

        /// <summary>
        ///     &#160;
        /// </summary>
        public static Dtype half
        {
            get
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new Dtype(cp.half);
                }
                else
                {
                    return new Dtype(np.half);
                }
            }
            set
            {
                if (Gpu.Available && Gpu.Use)
                {
                    cp.half = value.CupyDtype;
                }
                else
                {
                    np.half = value.NumpyDtype;
                }
            }
        }

        /// <summary>
        ///     compatible: C float
        /// </summary>
        public static Dtype single
        {
            get
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new Dtype(cp.single);
                }
                else
                {
                    return new Dtype(np.single);
                }
            }
            set
            {
                if (Gpu.Available && Gpu.Use)
                {
                    cp.single = value.CupyDtype;
                }
                else
                {
                    np.single = value.NumpyDtype;
                }
            }
        }

        /// <summary>
        ///     compatible: C double
        /// </summary>
        public static Dtype @double
        {
            get
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new Dtype(cp.@double);
                }
                else
                {
                    return new Dtype(np.@double);
                }
            }
            set
            {
                if (Gpu.Available && Gpu.Use)
                {
                    cp.@double = value.CupyDtype;
                }
                else
                {
                    np.@double = value.NumpyDtype;
                }
            }
        }

        /// <summary>
        ///     compatible: Python float
        /// </summary>
        public static Dtype float_
        {
            get
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new Dtype(cp.float_);
                }
                else
                {
                    return new Dtype(np.float_);
                }
            }
            set
            {
                if (Gpu.Available && Gpu.Use)
                {
                    cp.float_ = value.CupyDtype;
                }
                else
                {
                    np.float_ = value.NumpyDtype;
                }
            }
        }

        /// <summary>
        ///     compatible: C long float
        /// </summary>
        public static Dtype longfloat
        {
            get
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new Dtype(cp.longfloat);
                }
                else
                {
                    return new Dtype(np.longfloat);
                }
            }
            set
            {
                if (Gpu.Available && Gpu.Use)
                {
                    cp.longfloat = value.CupyDtype;
                }
                else
                {
                    np.longfloat = value.NumpyDtype;
                }
            }
        }

        /// <summary>
        ///     16 bits
        /// </summary>
        public static Dtype float16
        {
            get
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new Dtype(cp.float16);
                }
                else
                {
                    return new Dtype(np.float16);
                }
            }
            set
            {
                if (Gpu.Available && Gpu.Use)
                {
                    cp.float16 = value.CupyDtype;
                }
                else
                {
                    np.float16 = value.NumpyDtype;
                }
            }
        }

        /// <summary>
        ///     32 bits
        /// </summary>
        public static Dtype float32
        {
            get
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new Dtype(cp.float32);
                }
                else
                {
                    return new Dtype(np.float32);
                }
            }
            set
            {
                if (Gpu.Available && Gpu.Use)
                {
                    cp.float32 = value.CupyDtype;
                }
                else
                {
                    np.float32 = value.NumpyDtype;
                }
            }
        }

        /// <summary>
        ///     64 bits
        /// </summary>
        public static Dtype float64
        {
            get
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new Dtype(cp.float64);
                }
                else
                {
                    return new Dtype(np.float64);
                }
            }
            set
            {
                if (Gpu.Available && Gpu.Use)
                {
                    cp.float64 = value.CupyDtype;
                }
                else
                {
                    np.float64 = value.NumpyDtype;
                }
            }
        }

        /// <summary>
        ///     &#160;
        /// </summary>
        public static Dtype csingle
        {
            get
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new Dtype(cp.csingle);
                }
                else
                {
                    return new Dtype(np.csingle);
                }
            }
            set
            {
                if (Gpu.Available && Gpu.Use)
                {
                    cp.csingle = value.CupyDtype;
                }
                else
                {
                    np.csingle = value.NumpyDtype;
                }
            }
        }

        /// <summary>
        ///     compatible: Python complex
        /// </summary>
        public static Dtype complex_
        {
            get
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new Dtype(cp.complex_);
                }
                else
                {
                    return new Dtype(np.complex_);
                }
            }
            set
            {
                if (Gpu.Available && Gpu.Use)
                {
                    cp.complex_ = value.CupyDtype;
                }
                else
                {
                    np.complex_ = value.NumpyDtype;
                }
            }
        }

        /// <summary>
        ///     &#160;
        /// </summary>
        public static Dtype clongfloat
        {
            get
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new Dtype(cp.clongfloat);
                }
                else
                {
                    return new Dtype(np.clongfloat);
                }
            }
            set
            {
                if (Gpu.Available && Gpu.Use)
                {
                    cp.clongfloat = value.CupyDtype;
                }
                else
                {
                    np.clongfloat = value.NumpyDtype;
                }
            }
        }

        /// <summary>
        ///     two 32-bit floats
        /// </summary>
        public static Dtype complex64
        {
            get
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new Dtype(cp.complex64);
                }
                else
                {
                    return new Dtype(np.complex64);
                }
            }
            set
            {
                if (Gpu.Available && Gpu.Use)
                {
                    cp.complex64 = value.CupyDtype;
                }
                else
                {
                    np.complex64 = value.NumpyDtype;
                }
            }
        }

        /// <summary>
        ///     two 64-bit floats
        /// </summary>
        public static Dtype complex128
        {
            get
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new Dtype(cp.complex128);
                }
                else
                {
                    return new Dtype(np.complex128);
                }
            }
            set
            {
                if (Gpu.Available && Gpu.Use)
                {
                    cp.complex128 = value.CupyDtype;
                }
                else
                {
                    np.complex128 = value.NumpyDtype;
                }
            }
        }

        /// <summary>
        ///     two 96-bit floats,
        ///     platform?
        /// </summary>
        public static Dtype complex192
        {
            get
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new Dtype(cp.complex192);
                }
                else
                {
                    return new Dtype(np.complex192);
                }
            }
            set
            {
                if (Gpu.Available && Gpu.Use)
                {
                    cp.complex192 = value.CupyDtype;
                }
                else
                {
                    np.complex192 = value.NumpyDtype;
                }
            }
        }

        /// <summary>
        ///     two 128-bit floats,
        ///     platform?
        /// </summary>
        public static Dtype complex256
        {
            get
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new Dtype(cp.complex256);
                }
                else
                {
                    return new Dtype(np.complex256);
                }
            }
            set
            {
                if (Gpu.Available && Gpu.Use)
                {
                    cp.complex256 = value.CupyDtype;
                }
                else
                {
                    np.complex256 = value.NumpyDtype;
                }
            }
        }

        /// <summary>
        ///     any Python object
        /// </summary>
        public static Dtype object_
        {
            get
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new Dtype(cp.object_);
                }
                else
                {
                    return new Dtype(np.object_);
                }
            }
            set
            {
                if (Gpu.Available && Gpu.Use)
                {
                    cp.object_ = value.CupyDtype;
                }
                else
                {
                    np.object_ = value.NumpyDtype;
                }
            }
        }

        /// <summary>
        ///     compatible: Python bytes
        /// </summary>
        public static Dtype bytes_
        {
            get
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new Dtype(cp.bytes_);
                }
                else
                {
                    return new Dtype(np.bytes_);
                }
            }
            set
            {
                if (Gpu.Available && Gpu.Use)
                {
                    cp.bytes_ = value.CupyDtype;
                }
                else
                {
                    np.bytes_ = value.NumpyDtype;
                }
            }
        }

        /// <summary>
        ///     compatible: Python unicode/str
        /// </summary>
        public static Dtype unicode_
        {
            get
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new Dtype(cp.unicode_);
                }
                else
                {
                    return new Dtype(np.unicode_);
                }
            }
            set
            {
                if (Gpu.Available && Gpu.Use)
                {
                    cp.unicode_ = value.CupyDtype;
                }
                else
                {
                    np.unicode_ = value.NumpyDtype;
                }
            }
        }

        /// <summary>
        ///     &#160;
        /// </summary>
        public static Dtype @void
        {
            get
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new Dtype(cp.@void);
                }
                else
                {
                    return new Dtype(np.@void);
                }
            }
            set
            {
                if (Gpu.Available && Gpu.Use)
                {
                    cp.@void = value.CupyDtype;
                }
                else
                {
                    np.@void = value.NumpyDtype;
                }
            }
        }
    }
}
