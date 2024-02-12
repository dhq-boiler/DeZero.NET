using Cupy;
using DeZero.NET.PIL;
using Numpy;


namespace DeZero.NET
{
    public static partial class xp
    {
        public static NDarray<T> array<T>(params T[] data) where T : struct
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                return new NDarray<T>(cp.array<T>(data));
            }
            else
            {
                return new NDarray<T>(np.array<T>(data));
            }
        }

        public static NDarray array(NDarray @object, Dtype dtype = null, bool? copy = null, string order = null,
            bool? subok = null, int? ndmin = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                return new NDarray(cp.array(@object.CupyNDarray, dtype?.CupyDtype, copy, order, subok, ndmin));
            }
            else
            {
                return new NDarray(np.array(@object.NumpyNDarray, dtype?.NumpyDtype, copy, order, subok, ndmin));
            }
        }

        public static NDarray<T> array<T>(T[] @object, Dtype dtype = null, bool? copy = null, string order = null,
            bool? subok = null, int? ndmin = null) where T : struct
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                return new NDarray<T>(cp.array<T>(@object, dtype?.CupyDtype, copy, order, subok, ndmin));
            }
            else
            {
                return new NDarray<T>(np.array<T>(@object, dtype?.NumpyDtype, copy, order, subok, ndmin));
            }
        }

        public static NDarray<T> array<T>(T[,] @object, Dtype dtype = null, bool? copy = null, string order = null,
            bool? subok = null, int? ndmin = null) where T : struct
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                return new NDarray<T>(cp.array<T>(@object, dtype?.CupyDtype, copy, order, subok, ndmin));
            }
            else
            {
                return new NDarray<T>(np.array<T>(@object, dtype?.NumpyDtype, copy, order, subok, ndmin));
            }
        }

        public static NDarray<T> array<T>(T[,,] data, Dtype dtype = null, bool? copy = null, string order = null,
            bool? subok = null, int? ndmin = null) where T : struct
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                return new NDarray<T>(cp.array<T>(data, dtype?.CupyDtype, copy, order, subok, ndmin));
            }
            else
            {
                return new NDarray<T>(np.array<T>(data, dtype?.NumpyDtype, copy, order, subok, ndmin));
            }
        }

        public static NDarray<T> array<T>(T[,,,] data, Dtype dtype = null, bool? copy = null, string order = null,
            bool? subok = null, int? ndmin = null) where T : struct
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                return new NDarray<T>(cp.array<T>(data, dtype?.CupyDtype, copy, order, subok, ndmin));
            }
            else
            {
                return new NDarray<T>(np.array<T>(data, dtype?.NumpyDtype, copy, order, subok, ndmin));
            }
        }

        public static NDarray<T> array<T>(T[,,,,] data, Dtype dtype = null, bool? copy = null, string order = null,
            bool? subok = null, int? ndmin = null) where T : struct
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                return new NDarray<T>(cp.array<T>(data, dtype?.CupyDtype, copy, order, subok, ndmin));
            }
            else
            {
                return new NDarray<T>(np.array<T>(data, dtype?.NumpyDtype, copy, order, subok, ndmin));
            }
        }

        public static NDarray<T> array<T>(T[,,,,,] data, Dtype dtype = null, bool? copy = null, string order = null,
            bool? subok = null, int? ndmin = null) where T : struct
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                return new NDarray<T>(cp.array<T>(data, dtype?.CupyDtype, copy, order, subok, ndmin));
            }
            else
            {
                return new NDarray<T>(np.array<T>(data, dtype?.NumpyDtype, copy, order, subok, ndmin));
            }
        }

        public static NDarray array(string[] strings, int? itemsize = null, bool? copy = null, bool? unicode = null,
            string order = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                return new NDarray(cp.array(strings, itemsize, copy, unicode, order));
            }
            else
            {
                return new NDarray(np.array(strings, itemsize, copy, unicode, order));
            }
        }

        public static NDarray array<T>(NDarray<T>[] arrays, Dtype dtype = null, bool? copy = null, string order = null,
            bool? subok = null, int? ndmin = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                return new NDarray(cp.array<T>(arrays.Select(x => x.CupyNDarray).ToArray(), dtype?.CupyDtype, copy, order, subok, ndmin));
            }
            else
            {
                return new NDarray(np.array<T>(arrays.Select(x => x.NumpyNDarray).ToArray(), dtype?.NumpyDtype, copy, order, subok, ndmin));
            }
        }

        public static NDarray array<T>(IEnumerable<NDarray<T>> arrays, Dtype dtype = null, bool? copy = null,
            string order = null, bool? subok = null, int? ndmin = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                return new NDarray(cp.array<T>(arrays.Select(x => x.CupyNDarray).ToArray(), dtype?.CupyDtype, copy, order, subok, ndmin));
            }
            else
            {
                return new NDarray(np.array<T>(arrays.Select(x => x.NumpyNDarray).ToArray(), dtype?.NumpyDtype, copy, order, subok, ndmin));
            }
        }

        public static NDarray array(List<NDarray> arrays, Dtype dtype = null, bool? copy = null, string order = null,
            bool? subok = null, int? ndmin = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                return new NDarray(cp.array(arrays.Select(x => x.CupyNDarray).ToArray(), dtype?.CupyDtype, copy, order, subok, ndmin));
            }
            else
            {
                return new NDarray(np.array(arrays.Select(x => x.NumpyNDarray).ToArray(), dtype?.NumpyDtype, copy, order, subok, ndmin));
            }
        }

        public static NDarray array(NDarray[] arrays, Dtype dtype = null, bool? copy = null, string order = null,
            bool? subok = null, int? ndmin = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                return new NDarray(cp.array(arrays, dtype?.CupyDtype, copy, order, subok, ndmin));
            }
            else
            {
                return new NDarray(np.array(arrays, dtype?.NumpyDtype, copy, order, subok, ndmin));
            }
        }

        public static NDarray array(IEnumerable<NDarray> arrays, Dtype dtype = null, bool? copy = null,
            string order = null, bool? subok = null, int? ndmin = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                return new NDarray(cp.array(arrays.Select(x => x.CupyNDarray).ToArray(), dtype?.CupyDtype, copy, order, subok, ndmin));
            }
            else
            {
                return new NDarray(np.array(arrays.Select(x => x.NumpyNDarray).ToArray(), dtype?.NumpyDtype, copy, order, subok, ndmin));
            }
        }

        public static NDarray asarray(ValueType scalar, Dtype dtype = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                return new NDarray(cp.asarray(scalar, dtype?.CupyDtype));
            }
            else
            {
                return new NDarray(np.asarray(scalar, dtype?.NumpyDtype));
            }
        }

        /// <summary>
        ///     Convert an array of size 1 to its scalar equivalent.
        /// </summary>
        /// <returns>
        ///     Scalar representation of a. The output data type is the same type
        ///     returned by the input’s item method.
        /// </returns>
        public static T asscalar<T>(NDarray a)
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                return cp.asscalar<T>(a.CupyNDarray);
            }
            else
            {
                return np.asscalar<T>(a.NumpyNDarray);
            }
        }
    }
}
