using Cupy;
using Numpy;

namespace DeZero.NET
{
    public static partial class xp
    {
        /// <summary>
        ///     Join a sequence of arrays along an existing axis.<br></br>
        ///     Notes
        ///     When one or more of the arrays to be concatenated is a MaskedArray,
        ///     this function will return a MaskedArray object instead of an ndarray,
        ///     but the input masks are not preserved.<br></br>
        ///     In cases where a MaskedArray
        ///     is expected as input, use the ma.concatenate function from the masked
        ///     array module instead.
        /// </summary>
        /// <param name="arys">
        ///     The arrays must have the same shape, except in the dimension
        ///     corresponding to axis (the first, by default).
        /// </param>
        /// <param name="axis">
        ///     The axis along which the arrays will be joined.<br></br>
        ///     If axis is None,
        ///     arrays are flattened before use.<br></br>
        ///     Default is 0.
        /// </param>
        /// <param name="out">
        ///     If provided, the destination to place the result.<br></br>
        ///     The shape must be
        ///     correct, matching that of what concatenate would have returned if no
        ///     out argument were specified.
        /// </param>
        /// <returns>
        ///     The concatenated array.
        /// </returns>
        public static NDarray concatenate((NDarray, NDarray) arys, int? axis = 0, NDarray @out = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                return new NDarray(cp.concatenate((arys.Item1.CupyNDarray, arys.Item2.CupyNDarray), axis,
                    @out?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.concatenate((arys.Item1.NumpyNDarray, arys.Item2.NumpyNDarray), axis,
                    @out?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Join a sequence of arrays along an existing axis.<br></br>
        ///     Notes
        ///     When one or more of the arrays to be concatenated is a MaskedArray,
        ///     this function will return a MaskedArray object instead of an ndarray,
        ///     but the input masks are not preserved.<br></br>
        ///     In cases where a MaskedArray
        ///     is expected as input, use the ma.concatenate function from the masked
        ///     array module instead.
        /// </summary>
        /// <param name="arys">
        ///     The arrays must have the same shape, except in the dimension
        ///     corresponding to axis (the first, by default).
        /// </param>
        /// <param name="axis">
        ///     The axis along which the arrays will be joined.<br></br>
        ///     If axis is None,
        ///     arrays are flattened before use.<br></br>
        ///     Default is 0.
        /// </param>
        /// <param name="out">
        ///     If provided, the destination to place the result.<br></br>
        ///     The shape must be
        ///     correct, matching that of what concatenate would have returned if no
        ///     out argument were specified.
        /// </param>
        /// <returns>
        ///     The concatenated array.
        /// </returns>
        public static NDarray concatenate((NDarray, NDarray, NDarray) arys, int? axis = 0, NDarray @out = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                return new NDarray(cp.concatenate((arys.Item1.CupyNDarray, arys.Item2.CupyNDarray, arys.Item3.CupyNDarray), axis,
                    @out?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.concatenate((arys.Item1.NumpyNDarray, arys.Item2.NumpyNDarray, arys.Item3.NumpyNDarray), axis,
                    @out?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Join a sequence of arrays along an existing axis.<br></br>
        ///     Notes
        ///     When one or more of the arrays to be concatenated is a MaskedArray,
        ///     this function will return a MaskedArray object instead of an ndarray,
        ///     but the input masks are not preserved.<br></br>
        ///     In cases where a MaskedArray
        ///     is expected as input, use the ma.concatenate function from the masked
        ///     array module instead.
        /// </summary>
        /// <param name="arys">
        ///     The arrays must have the same shape, except in the dimension
        ///     corresponding to axis (the first, by default).
        /// </param>
        /// <param name="axis">
        ///     The axis along which the arrays will be joined.<br></br>
        ///     If axis is None,
        ///     arrays are flattened before use.<br></br>
        ///     Default is 0.
        /// </param>
        /// <param name="out">
        ///     If provided, the destination to place the result.<br></br>
        ///     The shape must be
        ///     correct, matching that of what concatenate would have returned if no
        ///     out argument were specified.
        /// </param>
        /// <returns>
        ///     The concatenated array.
        /// </returns>
        public static NDarray concatenate((NDarray, NDarray, NDarray, NDarray) arys, int? axis = 0, NDarray @out = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                return new NDarray(cp.concatenate((arys.Item1.CupyNDarray, arys.Item2.CupyNDarray, arys.Item3.CupyNDarray, arys.Item4.CupyNDarray), axis,
                    @out?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.concatenate((arys.Item1.NumpyNDarray, arys.Item2.NumpyNDarray, arys.Item3.NumpyNDarray, arys.Item4.NumpyNDarray), axis,
                    @out?.NumpyNDarray));
            }
        }
    }
}
