using Cupy;
using Numpy;

namespace DeZero.NET
{
    public static partial class xp
    {
        /// <summary>
        ///     Return a new array with sub-arrays along an axis deleted.<br></br>
        ///     For a one
        ///     dimensional array, this returns those entries not returned by
        ///     arr[obj].<br></br>
        ///     Notes
        ///     Often it is preferable to use a boolean mask.<br></br>
        ///     For example:
        ///     Is equivalent to cp.delete(arr, [0,2,4], axis=0), but allows further
        ///     use of mask.
        /// </summary>
        /// <param name="arr">
        ///     Input array.
        /// </param>
        /// <param name="obj">
        ///     Indicate which sub-arrays to remove.
        /// </param>
        /// <param name="axis">
        ///     The axis along which to delete the subarray defined by obj.<br></br>
        ///     If axis is None, obj is applied to the flattened array.
        /// </param>
        /// <returns>
        ///     A copy of arr with the elements specified by obj removed.<br></br>
        ///     Note
        ///     that delete does not occur in-place.<br></br>
        ///     If axis is None, out is
        ///     a flattened array.
        /// </returns>
        public static NDarray delete(NDarray arr, int obj, int? axis = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                throw new NotSupportedException();
            }
            else
            {
                return new NDarray(np.delete(arr.NumpyNDarray, obj, axis));
            }
        }

        /// <summary>
        ///     Return a new array with sub-arrays along an axis deleted.<br></br>
        ///     For a one
        ///     dimensional array, this returns those entries not returned by
        ///     arr[obj].<br></br>
        ///     Notes
        ///     Often it is preferable to use a boolean mask.<br></br>
        ///     For example:
        ///     Is equivalent to cp.delete(arr, [0,2,4], axis=0), but allows further
        ///     use of mask.
        /// </summary>
        /// <param name="arr">
        ///     Input array.
        /// </param>
        /// <param name="obj">
        ///     Indicate which sub-arrays to remove.
        /// </param>
        /// <param name="axis">
        ///     The axis along which to delete the subarray defined by obj.<br></br>
        ///     If axis is None, obj is applied to the flattened array.
        /// </param>
        /// <returns>
        ///     A copy of arr with the elements specified by obj removed.<br></br>
        ///     Note
        ///     that delete does not occur in-place.<br></br>
        ///     If axis is None, out is
        ///     a flattened array.
        /// </returns>
        public static NDarray delete(NDarray arr, int[] obj, int? axis = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                return new NDarray(cp.delete(arr.CupyNDarray, obj, axis));
            }
            else
            {
                return new NDarray(np.delete(arr.NumpyNDarray, obj, axis));
            }
        }
    }
}
