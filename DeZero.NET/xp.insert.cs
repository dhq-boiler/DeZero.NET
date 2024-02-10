using Numpy;

namespace DeZero.NET
{
    public static partial class xp
    {
        /// <summary>
        ///     Insert values along the given axis before the given indices.<br></br>
        ///     Notes
        ///     Note that for higher dimensional inserts obj=0 behaves very different
        ///     from obj=[0] just like arr[:,0,:] = values is different from
        ///     arr[:,[0],:] = values.
        /// </summary>
        /// <param name="arr">
        ///     Input array.
        /// </param>
        /// <param name="obj">
        ///     Object that defines the index or indices before which values is
        ///     inserted.<br></br>
        ///     Support for multiple insertions when obj is a single scalar or a
        ///     sequence with one element (similar to calling insert multiple
        ///     times).
        /// </param>
        /// <param name="values">
        ///     Values to insert into arr.<br></br>
        ///     If the type of values is different
        ///     from that of arr, values is converted to the type of arr.<br></br>
        ///     values should be shaped so that arr[...,obj,...] = values
        ///     is legal.
        /// </param>
        /// <param name="axis">
        ///     Axis along which to insert values.<br></br>
        ///     If axis is None then arr
        ///     is flattened first.
        /// </param>
        /// <returns>
        ///     A copy of arr with values inserted.<br></br>
        ///     Note that insert
        ///     does not occur in-place: a new array is returned.<br></br>
        ///     If
        ///     axis is None, out is a flattened array.
        /// </returns>
        public static NDarray insert(NDarray arr, int obj, NDarray values = null, int? axis = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                throw new NotSupportedException();
            }
            else
            {
                return new NDarray(np.insert(arr.NumpyNDarray, obj, values?.NumpyNDarray, axis));
            }
        }

        /// <summary>
        ///     Insert values along the given axis before the given indices.<br></br>
        ///     Notes
        ///     Note that for higher dimensional inserts obj=0 behaves very different
        ///     from obj=[0] just like arr[:,0,:] = values is different from
        ///     arr[:,[0],:] = values.
        /// </summary>
        /// <param name="arr">
        ///     Input array.
        /// </param>
        /// <param name="obj">
        ///     Object that defines the index or indices before which values is
        ///     inserted.<br></br>
        ///     Support for multiple insertions when obj is a single scalar or a
        ///     sequence with one element (similar to calling insert multiple
        ///     times).
        /// </param>
        /// <param name="values">
        ///     Values to insert into arr.<br></br>
        ///     If the type of values is different
        ///     from that of arr, values is converted to the type of arr.<br></br>
        ///     values should be shaped so that arr[...,obj,...] = values
        ///     is legal.
        /// </param>
        /// <param name="axis">
        ///     Axis along which to insert values.<br></br>
        ///     If axis is None then arr
        ///     is flattened first.
        /// </param>
        /// <returns>
        ///     A copy of arr with values inserted.<br></br>
        ///     Note that insert
        ///     does not occur in-place: a new array is returned.<br></br>
        ///     If
        ///     axis is None, out is a flattened array.
        /// </returns>
        public static NDarray insert(NDarray arr, NDarray obj, NDarray values = null, int? axis = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                throw new NotSupportedException();
            }
            else
            {
                return new NDarray(np.insert(arr.NumpyNDarray, obj.NumpyNDarray, values?.NumpyNDarray, axis));
            }
        }

        /// <summary>
        ///     Insert values along the given axis before the given indices.<br></br>
        ///     Notes
        ///     Note that for higher dimensional inserts obj=0 behaves very different
        ///     from obj=[0] just like arr[:,0,:] = values is different from
        ///     arr[:,[0],:] = values.
        /// </summary>
        /// <param name="arr">
        ///     Input array.
        /// </param>
        /// <param name="obj">
        ///     Object that defines the index or indices before which values is
        ///     inserted.<br></br>
        ///     Support for multiple insertions when obj is a single scalar or a
        ///     sequence with one element (similar to calling insert multiple
        ///     times).
        /// </param>
        /// <param name="values">
        ///     Values to insert into arr.<br></br>
        ///     If the type of values is different
        ///     from that of arr, values is converted to the type of arr.<br></br>
        ///     values should be shaped so that arr[...,obj,...] = values
        ///     is legal.
        /// </param>
        /// <param name="axis">
        ///     Axis along which to insert values.<br></br>
        ///     If axis is None then arr
        ///     is flattened first.
        /// </param>
        /// <returns>
        ///     A copy of arr with values inserted.<br></br>
        ///     Note that insert
        ///     does not occur in-place: a new array is returned.<br></br>
        ///     If
        ///     axis is None, out is a flattened array.
        /// </returns>
        public static NDarray insert(NDarray arr, Slice obj, NDarray values = null, int? axis = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                throw new NotSupportedException();
            }
            else
            {
                return new NDarray(np.insert(arr.NumpyNDarray, obj.NumpySlice, values?.NumpyNDarray, axis));
            }
        }

        /// <summary>
        ///     Insert values along the given axis before the given indices.<br></br>
        ///     Notes
        ///     Note that for higher dimensional inserts obj=0 behaves very different
        ///     from obj=[0] just like arr[:,0,:] = values is different from
        ///     arr[:,[0],:] = values.
        /// </summary>
        /// <param name="arr">
        ///     Input array.
        /// </param>
        /// <param name="obj">
        ///     Object that defines the index or indices before which values is
        ///     inserted.<br></br>
        ///     Support for multiple insertions when obj is a single scalar or a
        ///     sequence with one element (similar to calling insert multiple
        ///     times).
        /// </param>
        /// <param name="values">
        ///     Values to insert into arr.<br></br>
        ///     If the type of values is different
        ///     from that of arr, values is converted to the type of arr.<br></br>
        ///     values should be shaped so that arr[...,obj,...] = values
        ///     is legal.
        /// </param>
        /// <param name="axis">
        ///     Axis along which to insert values.<br></br>
        ///     If axis is None then arr
        ///     is flattened first.
        /// </param>
        /// <returns>
        ///     A copy of arr with values inserted.<br></br>
        ///     Note that insert
        ///     does not occur in-place: a new array is returned.<br></br>
        ///     If
        ///     axis is None, out is a flattened array.
        /// </returns>
        public static NDarray insert<T>(NDarray arr, int obj, T values, int? axis = null) where T : struct
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                throw new NotSupportedException();
            }
            else
            {
                return new NDarray(np.insert(arr.NumpyNDarray, obj, values, axis));
            }
        }
    }
}
