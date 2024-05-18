﻿using Cupy;
using Numpy;

namespace DeZero.NET
{
    public static partial class xp
    {
        /// <summary>
        ///     Return the minimum of an array or minimum along an axis.<br></br>
        ///     Notes
        ///     NaN values are propagated, that is if at least one item is NaN, the
        ///     corresponding min value will be NaN as well.<br></br>
        ///     To ignore NaN values
        ///     (MATLAB behavior), please use nanmin.<br></br>
        ///     Don’t use amin for element-wise comparison of 2 arrays; when
        ///     a.shape[0] is 2, minimum(a[0], a[1]) is faster than
        ///     amin(a, axis=0).
        /// </summary>
        /// <param name="a">
        ///     Input data.
        /// </param>
        /// <param name="axis">
        ///     Axis or axis along which to operate.<br></br>
        ///     By default, flattened input is
        ///     used.<br></br>
        ///     If this is a tuple of ints, the minimum is selected over multiple axis,
        ///     instead of a single axis or all the axis as before.
        /// </param>
        /// <param name="out">
        ///     Alternative output array in which to place the result.<br></br>
        ///     Must
        ///     be of the same shape and buffer length as the expected output.<br></br>
        ///     See doc.ufuncs (Section “Output arguments”) for more details.
        /// </param>
        /// <param name="keepdims">
        ///     If this is set to True, the axis which are reduced are left
        ///     in the result as dimensions with size one.<br></br>
        ///     With this option,
        ///     the result will broadcast correctly against the input array.<br></br>
        ///     If the default value is passed, then keepdims will not be
        ///     passed through to the amin method of sub-classes of
        ///     ndarray, however any non-default value will be.<br></br>
        ///     If the
        ///     sub-class’ method does not implement keepdims any
        ///     exceptions will be raised.
        /// </param>
        /// <param name="initial">
        ///     The maximum value of an output element.<br></br>
        ///     Must be present to allow
        ///     computation on empty slice.<br></br>
        ///     See reduce for details.
        /// </param>
        /// <returns>
        ///     Minimum of a.<br></br>
        ///     If axis is None, the result is a scalar value.<br></br>
        ///     If axis is given, the result is an array of dimension
        ///     a.ndim - 1.
        /// </returns>
        public static NDarray amin(this NDarray a, Axis axis = null, NDarray @out = null, bool? keepdims = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.amin(a.CupyNDarray, axis?.CupyAxis, @out?.CupyNDarray, keepdims));
            }
            else
            {
                return new NDarray(np.amin(a.NumpyNDarray, axis?.NumpyAxis, @out?.NumpyNDarray, keepdims));
            }
        }

        /// <summary>
        ///     Return the maximum of an array or maximum along an axis.<br></br>
        ///     Notes
        ///     NaN values are propagated, that is if at least one item is NaN, the
        ///     corresponding max value will be NaN as well.<br></br>
        ///     To ignore NaN values
        ///     (MATLAB behavior), please use nanmax.<br></br>
        ///     Don’t use amax for element-wise comparison of 2 arrays; when
        ///     a.shape[0] is 2, maximum(a[0], a[1]) is faster than
        ///     amax(a, axis=0).
        /// </summary>
        /// <param name="a">
        ///     Input data.
        /// </param>
        /// <param name="axis">
        ///     Axis or axis along which to operate.<br></br>
        ///     By default, flattened input is
        ///     used.<br></br>
        ///     If this is a tuple of ints, the maximum is selected over multiple axis,
        ///     instead of a single axis or all the axis as before.
        /// </param>
        /// <param name="out">
        ///     Alternative output array in which to place the result.<br></br>
        ///     Must
        ///     be of the same shape and buffer length as the expected output.<br></br>
        ///     See doc.ufuncs (Section “Output arguments”) for more details.
        /// </param>
        /// <param name="keepdims">
        ///     If this is set to True, the axis which are reduced are left
        ///     in the result as dimensions with size one.<br></br>
        ///     With this option,
        ///     the result will broadcast correctly against the input array.<br></br>
        ///     If the default value is passed, then keepdims will not be
        ///     passed through to the amax method of sub-classes of
        ///     ndarray, however any non-default value will be.<br></br>
        ///     If the
        ///     sub-class’ method does not implement keepdims any
        ///     exceptions will be raised.
        /// </param>
        /// <param name="initial">
        ///     The minimum value of an output element.<br></br>
        ///     Must be present to allow
        ///     computation on empty slice.<br></br>
        ///     See reduce for details.
        /// </param>
        /// <returns>
        ///     Maximum of a.<br></br>
        ///     If axis is None, the result is a scalar value.<br></br>
        ///     If axis is given, the result is an array of dimension
        ///     a.ndim - 1.
        /// </returns>
        public static NDarray amax(this NDarray a, Axis axis = null, NDarray @out = null, bool? keepdims = null,
            ValueType initial = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.amax(a.CupyNDarray, axis?.CupyAxis, @out?.CupyNDarray, keepdims, initial));
            }
            else
            {
                return new NDarray(np.amax(a.NumpyNDarray, axis?.NumpyAxis, @out?.NumpyNDarray, keepdims, initial));
            }
        }

        /// <summary>
        ///     Return minimum of an array or minimum along an axis, ignoring any NaNs.<br></br>
        ///     When all-NaN slices are encountered a RuntimeWarning is raised and
        ///     Nan is returned for that slice.<br></br>
        ///     Notes
        ///     Cupy uses the IEEE Standard for Binary Floating-Point for Arithmetic
        ///     (IEEE 754).<br></br>
        ///     This means that Not a Number is not equivalent to infinity.<br></br>
        ///     Positive infinity is treated as a very large number and negative
        ///     infinity is treated as a very small (i.e.<br></br>
        ///     negative) number.<br></br>
        ///     If the input has a integer type the function is equivalent to cp.min.
        /// </summary>
        /// <param name="a">
        ///     Array containing numbers whose minimum is desired.<br></br>
        ///     If a is not an
        ///     array, a conversion is attempted.
        /// </param>
        /// <param name="axis">
        ///     Axis or axis along which the minimum is computed.<br></br>
        ///     The default is to compute
        ///     the minimum of the flattened array.
        /// </param>
        /// <param name="out">
        ///     Alternate output array in which to place the result.<br></br>
        ///     The default
        ///     is None; if provided, it must have the same shape as the
        ///     expected output, but the type will be cast if necessary.<br></br>
        ///     See
        ///     doc.ufuncs for details.
        /// </param>
        /// <param name="keepdims">
        ///     If this is set to True, the axis which are reduced are left
        ///     in the result as dimensions with size one.<br></br>
        ///     With this option,
        ///     the result will broadcast correctly against the original a.<br></br>
        ///     If the value is anything but the default, then
        ///     keepdims will be passed through to the min method
        ///     of sub-classes of ndarray.<br></br>
        ///     If the sub-classes methods
        ///     does not implement keepdims any exceptions will be raised.
        /// </param>
        /// <returns>
        ///     An array with the same shape as a, with the specified axis
        ///     removed.<br></br>
        ///     If a is a 0-d array, or if axis is None, an ndarray
        ///     scalar is returned.<br></br>
        ///     The same dtype as a is returned.
        /// </returns>
        public static NDarray nanmin(this NDarray a, Axis axis = null, NDarray @out = null, bool? keepdims = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.nanmin(a.CupyNDarray, axis?.CupyAxis, @out?.CupyNDarray, keepdims));
            }
            else
            {
                return new NDarray(np.nanmin(a.NumpyNDarray, axis?.NumpyAxis, @out?.NumpyNDarray, keepdims));
            }
        }

        /// <summary>
        ///     Return the maximum of an array or maximum along an axis, ignoring any
        ///     NaNs.<br></br>
        ///     When all-NaN slices are encountered a RuntimeWarning is
        ///     raised and NaN is returned for that slice.<br></br>
        ///     Notes
        ///     Cupy uses the IEEE Standard for Binary Floating-Point for Arithmetic
        ///     (IEEE 754).<br></br>
        ///     This means that Not a Number is not equivalent to infinity.<br></br>
        ///     Positive infinity is treated as a very large number and negative
        ///     infinity is treated as a very small (i.e.<br></br>
        ///     negative) number.<br></br>
        ///     If the input has a integer type the function is equivalent to cp.max.
        /// </summary>
        /// <param name="a">
        ///     Array containing numbers whose maximum is desired.<br></br>
        ///     If a is not an
        ///     array, a conversion is attempted.
        /// </param>
        /// <param name="axis">
        ///     Axis or axis along which the maximum is computed.<br></br>
        ///     The default is to compute
        ///     the maximum of the flattened array.
        /// </param>
        /// <param name="out">
        ///     Alternate output array in which to place the result.<br></br>
        ///     The default
        ///     is None; if provided, it must have the same shape as the
        ///     expected output, but the type will be cast if necessary.<br></br>
        ///     See
        ///     doc.ufuncs for details.
        /// </param>
        /// <param name="keepdims">
        ///     If this is set to True, the axis which are reduced are left
        ///     in the result as dimensions with size one.<br></br>
        ///     With this option,
        ///     the result will broadcast correctly against the original a.<br></br>
        ///     If the value is anything but the default, then
        ///     keepdims will be passed through to the max method
        ///     of sub-classes of ndarray.<br></br>
        ///     If the sub-classes methods
        ///     does not implement keepdims any exceptions will be raised.
        /// </param>
        /// <returns>
        ///     An array with the same shape as a, with the specified axis removed.<br></br>
        ///     If a is a 0-d array, or if axis is None, an ndarray scalar is
        ///     returned.<br></br>
        ///     The same dtype as a is returned.
        /// </returns>
        public static NDarray nanmax(this NDarray a, Axis axis = null, NDarray @out = null, bool? keepdims = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.nanmax(a.CupyNDarray, axis?.CupyAxis, @out?.CupyNDarray, keepdims));
            }
            else
            {
                return new NDarray(np.nanmax(a.NumpyNDarray, axis?.NumpyAxis, @out?.NumpyNDarray, keepdims));
            }
        }

        /// <summary>
        ///     Range of values (maximum - minimum) along an axis.<br></br>
        ///     The name of the function comes from the acronym for ‘peak to peak’.
        /// </summary>
        /// <param name="a">
        ///     Input values.
        /// </param>
        /// <param name="axis">
        ///     Axis along which to find the peaks.<br></br>
        ///     By default, flatten the
        ///     array.<br></br>
        ///     axis may be negative, in
        ///     which case it counts from the last to the first axis.<br></br>
        ///     If this is a tuple of ints, a reduction is performed on multiple
        ///     axis, instead of a single axis or all the axis as before.
        /// </param>
        /// <param name="out">
        ///     Alternative output array in which to place the result.<br></br>
        ///     It must
        ///     have the same shape and buffer length as the expected output,
        ///     but the type of the output values will be cast if necessary.
        /// </param>
        /// <param name="keepdims">
        ///     If this is set to True, the axis which are reduced are left
        ///     in the result as dimensions with size one.<br></br>
        ///     With this option,
        ///     the result will broadcast correctly against the input array.<br></br>
        ///     If the default value is passed, then keepdims will not be
        ///     passed through to the ptp method of sub-classes of
        ///     ndarray, however any non-default value will be.<br></br>
        ///     If the
        ///     sub-class’ method does not implement keepdims any
        ///     exceptions will be raised.
        /// </param>
        /// <returns>
        ///     A new array holding the result, unless out was
        ///     specified, in which case a reference to out is returned.
        /// </returns>
        public static NDarray ptp(this NDarray a, Axis axis = null, NDarray @out = null, bool? keepdims = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.ptp(a.CupyNDarray, axis?.CupyAxis, @out?.CupyNDarray, keepdims));
            }
            else
            {
                return new NDarray(np.ptp(a.NumpyNDarray, axis?.NumpyAxis, @out?.NumpyNDarray, keepdims));
            }
        }

        /// <summary>
        ///     Compute the q-th percentile of the data along the specified axis.<br></br>
        ///     Returns the q-th percentile(s) of the array elements.<br></br>
        ///     Notes
        ///     Given a vector V of length N, the q-th percentile of
        ///     V is the value q/100 of the way from the minimum to the
        ///     maximum in a sorted copy of V.<br></br>
        ///     The values and distances of
        ///     the two nearest neighbors as well as the interpolation parameter
        ///     will determine the percentile if the normalized ranking does not
        ///     match the location of q exactly.<br></br>
        ///     This function is the same as
        ///     the median if q=50, the same as the minimum if q=0 and the
        ///     same as the maximum if q=100.
        /// </summary>
        /// <param name="a">
        ///     Input array or object that can be converted to an array.
        /// </param>
        /// <param name="q">
        ///     Percentile or sequence of percentiles to compute, which must be between
        ///     0 and 100 inclusive.
        /// </param>
        /// <param name="axis">
        ///     Axis or axis along which the percentiles are computed.<br></br>
        ///     The
        ///     default is to compute the percentile(s) along a flattened
        ///     version of the array.
        /// </param>
        /// <param name="out">
        ///     Alternative output array in which to place the result.<br></br>
        ///     It must
        ///     have the same shape and buffer length as the expected output,
        ///     but the type (of the output) will be cast if necessary.
        /// </param>
        /// <param name="overwrite_input">
        ///     If True, then allow the input array a to be modified by intermediate
        ///     calculations, to save memory.<br></br>
        ///     In this case, the contents of the input
        ///     a after this function completes is undefined.
        /// </param>
        /// <param name="interpolation">
        ///     This optional parameter specifies the interpolation method to
        ///     use when the desired percentile lies between two data points
        ///     i &lt; j:
        /// </param>
        /// <param name="keepdims">
        ///     If this is set to True, the axis which are reduced are left in
        ///     the result as dimensions with size one.<br></br>
        ///     With this option, the
        ///     result will broadcast correctly against the original array a.
        /// </param>
        /// <returns>
        ///     If q is a single percentile and axis=None, then the result
        ///     is a scalar.<br></br>
        ///     If multiple percentiles are given, first axis of
        ///     the result corresponds to the percentiles.<br></br>
        ///     The other axis are
        ///     the axis that remain after the reduction of a.<br></br>
        ///     If the input
        ///     contains integers or floats smaller than float64, the output
        ///     data-type is float64. Otherwise, the output data-type is the
        ///     same as that of the input.<br></br>
        ///     If out is specified, that array is
        ///     returned instead.
        /// </returns>
        public static NDarray<double> percentile(this NDarray a, NDarray<float> q, Axis axis, NDarray @out = null,
            bool? overwrite_input = false, string interpolation = "linear", bool? keepdims = false)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray<double>(cp.percentile(a.CupyNDarray, q.CupyNDarray, axis?.CupyAxis, @out?.CupyNDarray, overwrite_input, interpolation, keepdims));
            }
            else
            {
                return new NDarray<double>(np.percentile(a.NumpyNDarray, q.NumpyNDarray, axis?.NumpyAxis, @out?.NumpyNDarray, overwrite_input, interpolation, keepdims));
            }
        }

        /// <summary>
        ///     Compute the q-th percentile of the data along the specified axis.<br></br>
        ///     Returns the q-th percentile(s) of the array elements.<br></br>
        ///     Notes
        ///     Given a vector V of length N, the q-th percentile of
        ///     V is the value q/100 of the way from the minimum to the
        ///     maximum in a sorted copy of V.<br></br>
        ///     The values and distances of
        ///     the two nearest neighbors as well as the interpolation parameter
        ///     will determine the percentile if the normalized ranking does not
        ///     match the location of q exactly.<br></br>
        ///     This function is the same as
        ///     the median if q=50, the same as the minimum if q=0 and the
        ///     same as the maximum if q=100.
        /// </summary>
        /// <param name="a">
        ///     Input array or object that can be converted to an array.
        /// </param>
        /// <param name="q">
        ///     Percentile or sequence of percentiles to compute, which must be between
        ///     0 and 100 inclusive.
        /// </param>
        /// <param name="out">
        ///     Alternative output array in which to place the result.<br></br>
        ///     It must
        ///     have the same shape and buffer length as the expected output,
        ///     but the type (of the output) will be cast if necessary.
        /// </param>
        /// <param name="overwrite_input">
        ///     If True, then allow the input array a to be modified by intermediate
        ///     calculations, to save memory.<br></br>
        ///     In this case, the contents of the input
        ///     a after this function completes is undefined.
        /// </param>
        /// <param name="interpolation">
        ///     This optional parameter specifies the interpolation method to
        ///     use when the desired percentile lies between two data points
        ///     i &lt; j:
        /// </param>
        /// <returns>
        ///     If q is a single percentile and axis=None, then the result
        ///     is a scalar.<br></br>
        ///     If multiple percentiles are given, first axis of
        ///     the result corresponds to the percentiles.<br></br>
        ///     The other axis are
        ///     the axis that remain after the reduction of a.<br></br>
        ///     If the input
        ///     contains integers or floats smaller than float64, the output
        ///     data-type is float64. Otherwise, the output data-type is the
        ///     same as that of the input.<br></br>
        ///     If out is specified, that array is
        ///     returned instead.
        /// </returns>
        public static double percentile(this NDarray a, NDarray<float> q, NDarray @out = null,
            bool? overwrite_input = false, string interpolation = "linear")
        {
            if (Gpu.Available && Gpu.Use)
            {
                return cp.percentile(a.CupyNDarray, q.CupyNDarray, @out?.CupyNDarray, overwrite_input, interpolation);
            }
            else
            {
                return np.percentile(a.NumpyNDarray, q.NumpyNDarray, @out?.NumpyNDarray, overwrite_input, interpolation);
            }
        }

        /// <summary>
        ///     Compute the qth percentile of the data along the specified axis,
        ///     while ignoring nan values.<br></br>
        ///     Returns the qth percentile(s) of the array elements.<br></br>
        ///     Notes
        ///     Given a vector V of length N, the q-th percentile of
        ///     V is the value q/100 of the way from the minimum to the
        ///     maximum in a sorted copy of V.<br></br>
        ///     The values and distances of
        ///     the two nearest neighbors as well as the interpolation parameter
        ///     will determine the percentile if the normalized ranking does not
        ///     match the location of q exactly.<br></br>
        ///     This function is the same as
        ///     the median if q=50, the same as the minimum if q=0 and the
        ///     same as the maximum if q=100.
        /// </summary>
        /// <param name="a">
        ///     Input array or object that can be converted to an array, containing
        ///     nan values to be ignored.
        /// </param>
        /// <param name="q">
        ///     Percentile or sequence of percentiles to compute, which must be between
        ///     0 and 100 inclusive.
        /// </param>
        /// <param name="axis">
        ///     Axis or axis along which the percentiles are computed.<br></br>
        ///     The
        ///     default is to compute the percentile(s) along a flattened
        ///     version of the array.
        /// </param>
        /// <param name="out">
        ///     Alternative output array in which to place the result.<br></br>
        ///     It must
        ///     have the same shape and buffer length as the expected output,
        ///     but the type (of the output) will be cast if necessary.
        /// </param>
        /// <param name="overwrite_input">
        ///     If True, then allow the input array a to be modified by intermediate
        ///     calculations, to save memory.<br></br>
        ///     In this case, the contents of the input
        ///     a after this function completes is undefined.
        /// </param>
        /// <param name="interpolation">
        ///     This optional parameter specifies the interpolation method to
        ///     use when the desired percentile lies between two data points
        ///     i &lt; j:
        /// </param>
        /// <param name="keepdims">
        ///     If this is set to True, the axis which are reduced are left in
        ///     the result as dimensions with size one.<br></br>
        ///     With this option, the
        ///     result will broadcast correctly against the original array a.<br></br>
        ///     If this is anything but the default value it will be passed
        ///     through (in the special case of an empty array) to the
        ///     mean function of the underlying array.<br></br>
        ///     If the array is
        ///     a sub-class and mean does not have the kwarg keepdims this
        ///     will raise a RuntimeError.
        /// </param>
        /// <returns>
        ///     If q is a single percentile and axis=None, then the result
        ///     is a scalar.<br></br>
        ///     If multiple percentiles are given, first axis of
        ///     the result corresponds to the percentiles.<br></br>
        ///     The other axis are
        ///     the axis that remain after the reduction of a.<br></br>
        ///     If the input
        ///     contains integers or floats smaller than float64, the output
        ///     data-type is float64. Otherwise, the output data-type is the
        ///     same as that of the input.<br></br>
        ///     If out is specified, that array is
        ///     returned instead.
        /// </returns>
        public static NDarray<double> nanpercentile(this NDarray a, NDarray<float> q, Axis axis, NDarray @out = null,
            bool? overwrite_input = false, string interpolation = "linear", bool? keepdims = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray<double>(cp.nanpercentile(a.CupyNDarray, q.CupyNDarray, axis.CupyAxis, @out?.CupyNDarray, overwrite_input, interpolation, keepdims));
            }
            else
            {
                return new NDarray<double>(np.nanpercentile(a.NumpyNDarray, q.NumpyNDarray, axis.NumpyAxis, @out?.NumpyNDarray, overwrite_input, interpolation, keepdims));
            }
        }

        /// <summary>
        ///     Compute the qth percentile of the data along the specified axis,
        ///     while ignoring nan values.<br></br>
        ///     Returns the qth percentile(s) of the array elements.<br></br>
        ///     Notes
        ///     Given a vector V of length N, the q-th percentile of
        ///     V is the value q/100 of the way from the minimum to the
        ///     maximum in a sorted copy of V.<br></br>
        ///     The values and distances of
        ///     the two nearest neighbors as well as the interpolation parameter
        ///     will determine the percentile if the normalized ranking does not
        ///     match the location of q exactly.<br></br>
        ///     This function is the same as
        ///     the median if q=50, the same as the minimum if q=0 and the
        ///     same as the maximum if q=100.
        /// </summary>
        /// <param name="a">
        ///     Input array or object that can be converted to an array, containing
        ///     nan values to be ignored.
        /// </param>
        /// <param name="q">
        ///     Percentile or sequence of percentiles to compute, which must be between
        ///     0 and 100 inclusive.
        /// </param>
        /// <param name="out">
        ///     Alternative output array in which to place the result.<br></br>
        ///     It must
        ///     have the same shape and buffer length as the expected output,
        ///     but the type (of the output) will be cast if necessary.
        /// </param>
        /// <param name="overwrite_input">
        ///     If True, then allow the input array a to be modified by intermediate
        ///     calculations, to save memory.<br></br>
        ///     In this case, the contents of the input
        ///     a after this function completes is undefined.
        /// </param>
        /// <param name="interpolation">
        ///     This optional parameter specifies the interpolation method to
        ///     use when the desired percentile lies between two data points
        ///     i &lt; j:
        /// </param>
        /// <returns>
        ///     If q is a single percentile and axis=None, then the result
        ///     is a scalar.<br></br>
        ///     If multiple percentiles are given, first axis of
        ///     the result corresponds to the percentiles.<br></br>
        ///     The other axis are
        ///     the axis that remain after the reduction of a.<br></br>
        ///     If the input
        ///     contains integers or floats smaller than float64, the output
        ///     data-type is float64. Otherwise, the output data-type is the
        ///     same as that of the input.<br></br>
        ///     If out is specified, that array is
        ///     returned instead.
        /// </returns>
        public static double nanpercentile(this NDarray a, NDarray<float> q, NDarray @out = null,
            bool? overwrite_input = false, string interpolation = "linear")
        {
            if (Gpu.Available && Gpu.Use)
            {
                return cp.nanpercentile(a.CupyNDarray, q.CupyNDarray, @out?.CupyNDarray, overwrite_input, interpolation);
            }
            else
            {
                return np.nanpercentile(a.NumpyNDarray, q.NumpyNDarray, @out?.NumpyNDarray, overwrite_input, interpolation);
            }
        }

        /// <summary>
        ///     Compute the q-th quantile of the data along the specified axis.<br></br>
        ///     ..versionadded:: 1.15.0
        ///     Notes
        ///     Given a vector V of length N, the q-th quantile of
        ///     V is the value q of the way from the minimum to the
        ///     maximum in a sorted copy of V.<br></br>
        ///     The values and distances of
        ///     the two nearest neighbors as well as the interpolation parameter
        ///     will determine the quantile if the normalized ranking does not
        ///     match the location of q exactly.<br></br>
        ///     This function is the same as
        ///     the median if q=0.5, the same as the minimum if q=0.0 and the
        ///     same as the maximum if q=1.0.
        /// </summary>
        /// <param name="a">
        ///     Input array or object that can be converted to an array.
        /// </param>
        /// <param name="q">
        ///     Quantile or sequence of quantiles to compute, which must be between
        ///     0 and 1 inclusive.
        /// </param>
        /// <param name="axis">
        ///     Axis or axis along which the quantiles are computed.<br></br>
        ///     The
        ///     default is to compute the quantile(s) along a flattened
        ///     version of the array.
        /// </param>
        /// <param name="out">
        ///     Alternative output array in which to place the result.<br></br>
        ///     It must
        ///     have the same shape and buffer length as the expected output,
        ///     but the type (of the output) will be cast if necessary.
        /// </param>
        /// <param name="overwrite_input">
        ///     If True, then allow the input array a to be modified by intermediate
        ///     calculations, to save memory.<br></br>
        ///     In this case, the contents of the input
        ///     a after this function completes is undefined.
        /// </param>
        /// <param name="interpolation">
        ///     This optional parameter specifies the interpolation method to
        ///     use when the desired quantile lies between two data points
        ///     i &lt; j:
        /// </param>
        /// <param name="keepdims">
        ///     If this is set to True, the axis which are reduced are left in
        ///     the result as dimensions with size one.<br></br>
        ///     With this option, the
        ///     result will broadcast correctly against the original array a.
        /// </param>
        /// <returns>
        ///     If q is a single quantile and axis=None, then the result
        ///     is a scalar.<br></br>
        ///     If multiple quantiles are given, first axis of
        ///     the result corresponds to the quantiles.<br></br>
        ///     The other axis are
        ///     the axis that remain after the reduction of a.<br></br>
        ///     If the input
        ///     contains integers or floats smaller than float64, the output
        ///     data-type is float64. Otherwise, the output data-type is the
        ///     same as that of the input.<br></br>
        ///     If out is specified, that array is
        ///     returned instead.
        /// </returns>
        public static NDarray<double> quantile(this NDarray a, NDarray<float> q, Axis axis, NDarray @out = null,
            bool? overwrite_input = false, string interpolation = "linear", bool? keepdims = false)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray<double>(cp.quantile(a.CupyNDarray, q.CupyNDarray, axis.CupyAxis, @out?.CupyNDarray, overwrite_input, interpolation, keepdims));
            }
            else
            {
                return new NDarray<double>(np.quantile(a.NumpyNDarray, q.NumpyNDarray, axis.NumpyAxis, @out?.NumpyNDarray, overwrite_input, interpolation, keepdims));
            }
        }

        /// <summary>
        ///     Compute the q-th quantile of the data along the specified axis.<br></br>
        ///     ..versionadded:: 1.15.0
        ///     Notes
        ///     Given a vector V of length N, the q-th quantile of
        ///     V is the value q of the way from the minimum to the
        ///     maximum in a sorted copy of V.<br></br>
        ///     The values and distances of
        ///     the two nearest neighbors as well as the interpolation parameter
        ///     will determine the quantile if the normalized ranking does not
        ///     match the location of q exactly.<br></br>
        ///     This function is the same as
        ///     the median if q=0.5, the same as the minimum if q=0.0 and the
        ///     same as the maximum if q=1.0.
        /// </summary>
        /// <param name="a">
        ///     Input array or object that can be converted to an array.
        /// </param>
        /// <param name="q">
        ///     Quantile or sequence of quantiles to compute, which must be between
        ///     0 and 1 inclusive.
        /// </param>
        /// <param name="out">
        ///     Alternative output array in which to place the result.<br></br>
        ///     It must
        ///     have the same shape and buffer length as the expected output,
        ///     but the type (of the output) will be cast if necessary.
        /// </param>
        /// <param name="overwrite_input">
        ///     If True, then allow the input array a to be modified by intermediate
        ///     calculations, to save memory.<br></br>
        ///     In this case, the contents of the input
        ///     a after this function completes is undefined.
        /// </param>
        /// <param name="interpolation">
        ///     This optional parameter specifies the interpolation method to
        ///     use when the desired quantile lies between two data points
        ///     i &lt; j:
        /// </param>
        /// <returns>
        ///     If q is a single quantile and axis=None, then the result
        ///     is a scalar.<br></br>
        ///     If multiple quantiles are given, first axis of
        ///     the result corresponds to the quantiles.<br></br>
        ///     The other axis are
        ///     the axis that remain after the reduction of a.<br></br>
        ///     If the input
        ///     contains integers or floats smaller than float64, the output
        ///     data-type is float64. Otherwise, the output data-type is the
        ///     same as that of the input.<br></br>
        ///     If out is specified, that array is
        ///     returned instead.
        /// </returns>
        public static double quantile(this NDarray a, NDarray<float> q, NDarray @out = null,
            bool? overwrite_input = false, string interpolation = "linear")
        {
            if (Gpu.Available && Gpu.Use)
            {
                return cp.quantile(a.CupyNDarray, q.CupyNDarray, @out?.CupyNDarray, overwrite_input, interpolation);
            }
            else
            {
                return np.quantile(a.NumpyNDarray, q.NumpyNDarray, @out?.NumpyNDarray, overwrite_input, interpolation);
            }
        }

        /// <summary>
        ///     Compute the qth quantile of the data along the specified axis,
        ///     while ignoring nan values.<br></br>
        ///     Returns the qth quantile(s) of the array elements.<br></br>
        ///     .. versionadded:: 1.15.0
        /// </summary>
        /// <param name="a">
        ///     Input array or object that can be converted to an array, containing
        ///     nan values to be ignored
        /// </param>
        /// <param name="q">
        ///     Quantile or sequence of quantiles to compute, which must be between
        ///     0 and 1 inclusive.
        /// </param>
        /// <param name="axis">
        ///     Axis or axis along which the quantiles are computed.<br></br>
        ///     The
        ///     default is to compute the quantile(s) along a flattened
        ///     version of the array.
        /// </param>
        /// <param name="out">
        ///     Alternative output array in which to place the result.<br></br>
        ///     It must
        ///     have the same shape and buffer length as the expected output,
        ///     but the type (of the output) will be cast if necessary.
        /// </param>
        /// <param name="overwrite_input">
        ///     If True, then allow the input array a to be modified by intermediate
        ///     calculations, to save memory.<br></br>
        ///     In this case, the contents of the input
        ///     a after this function completes is undefined.
        /// </param>
        /// <param name="interpolation">
        ///     This optional parameter specifies the interpolation method to
        ///     use when the desired quantile lies between two data points
        ///     i &lt; j:
        /// </param>
        /// <param name="keepdims">
        ///     If this is set to True, the axis which are reduced are left in
        ///     the result as dimensions with size one.<br></br>
        ///     With this option, the
        ///     result will broadcast correctly against the original array a.<br></br>
        ///     If this is anything but the default value it will be passed
        ///     through (in the special case of an empty array) to the
        ///     mean function of the underlying array.<br></br>
        ///     If the array is
        ///     a sub-class and mean does not have the kwarg keepdims this
        ///     will raise a RuntimeError.
        /// </param>
        /// <returns>
        ///     If q is a single percentile and axis=None, then the result
        ///     is a scalar.<br></br>
        ///     If multiple quantiles are given, first axis of
        ///     the result corresponds to the quantiles.<br></br>
        ///     The other axis are
        ///     the axis that remain after the reduction of a.<br></br>
        ///     If the input
        ///     contains integers or floats smaller than float64, the output
        ///     data-type is float64. Otherwise, the output data-type is the
        ///     same as that of the input.<br></br>
        ///     If out is specified, that array is
        ///     returned instead.
        /// </returns>
        public static NDarray<double> nanquantile(this NDarray a, NDarray<float> q, Axis axis, NDarray @out = null,
            bool? overwrite_input = false, string interpolation = "linear", bool? keepdims = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray<double>(cp.nanquantile(a.CupyNDarray, q.CupyNDarray, axis.CupyAxis, @out?.CupyNDarray, overwrite_input, interpolation, keepdims));
            }
            else
            {
                return new NDarray<double>(np.nanquantile(a.NumpyNDarray, q.NumpyNDarray, axis.NumpyAxis, @out?.NumpyNDarray, overwrite_input, interpolation, keepdims));
            }
        }

        /// <summary>
        ///     Compute the qth quantile of the data along the specified axis,
        ///     while ignoring nan values.<br></br>
        ///     Returns the qth quantile(s) of the array elements.<br></br>
        ///     .. versionadded:: 1.15.0
        /// </summary>
        /// <param name="a">
        ///     Input array or object that can be converted to an array, containing
        ///     nan values to be ignored
        /// </param>
        /// <param name="q">
        ///     Quantile or sequence of quantiles to compute, which must be between
        ///     0 and 1 inclusive.
        /// </param>
        /// <param name="out">
        ///     Alternative output array in which to place the result.<br></br>
        ///     It must
        ///     have the same shape and buffer length as the expected output,
        ///     but the type (of the output) will be cast if necessary.
        /// </param>
        /// <param name="overwrite_input">
        ///     If True, then allow the input array a to be modified by intermediate
        ///     calculations, to save memory.<br></br>
        ///     In this case, the contents of the input
        ///     a after this function completes is undefined.
        /// </param>
        /// <param name="interpolation">
        ///     This optional parameter specifies the interpolation method to
        ///     use when the desired quantile lies between two data points
        ///     i &lt; j:
        /// </param>
        /// <returns>
        ///     If q is a single percentile and axis=None, then the result
        ///     is a scalar.<br></br>
        ///     If multiple quantiles are given, first axis of
        ///     the result corresponds to the quantiles.<br></br>
        ///     The other axis are
        ///     the axis that remain after the reduction of a.<br></br>
        ///     If the input
        ///     contains integers or floats smaller than float64, the output
        ///     data-type is float64. Otherwise, the output data-type is the
        ///     same as that of the input.<br></br>
        ///     If out is specified, that array is
        ///     returned instead.
        /// </returns>
        public static double nanquantile(this NDarray a, NDarray<float> q, NDarray @out = null,
            bool? overwrite_input = false, string interpolation = "linear")
        {
            if (Gpu.Available && Gpu.Use)
            {
                return cp.nanquantile(a.CupyNDarray, q.CupyNDarray, @out?.CupyNDarray, overwrite_input, interpolation);
            }
            else
            {
                return np.nanquantile(a.NumpyNDarray, q.NumpyNDarray, @out?.NumpyNDarray, overwrite_input, interpolation);
            }
        }

        /// <summary>
        ///     Compute the median along the specified axis.<br></br>
        ///     Returns the median of the array elements.<br></br>
        ///     Notes
        ///     Given a vector V of length N, the median of V is the
        ///     middle value of a sorted copy of V, V_sorted - i
        ///     e., V_sorted[(N-1)/2], when N is odd, and the average of the
        ///     two middle values of V_sorted when N is even.
        /// </summary>
        /// <param name="a">
        ///     Input array or object that can be converted to an array.
        /// </param>
        /// <param name="axis">
        ///     Axis or axis along which the medians are computed.<br></br>
        ///     The default
        ///     is to compute the median along a flattened version of the array.<br></br>
        ///     A sequence of axis is supported since version 1.9.0.
        /// </param>
        /// <param name="out">
        ///     Alternative output array in which to place the result.<br></br>
        ///     It must
        ///     have the same shape and buffer length as the expected output,
        ///     but the type (of the output) will be cast if necessary.
        /// </param>
        /// <param name="overwrite_input">
        ///     If True, then allow use of memory of input array a for
        ///     calculations.<br></br>
        ///     The input array will be modified by the call to
        ///     median.<br></br>
        ///     This will save memory when you do not need to preserve
        ///     the contents of the input array.<br></br>
        ///     Treat the input as undefined,
        ///     but it will probably be fully or partially sorted.<br></br>
        ///     Default is
        ///     False.<br></br>
        ///     If overwrite_input is True and a is not already an
        ///     ndarray, an error will be raised.
        /// </param>
        /// <param name="keepdims">
        ///     If this is set to True, the axis which are reduced are left
        ///     in the result as dimensions with size one.<br></br>
        ///     With this option,
        ///     the result will broadcast correctly against the original arr.
        /// </param>
        /// <returns>
        ///     A new array holding the result.<br></br>
        ///     If the input contains integers
        ///     or floats smaller than float64, then the output data-type is
        ///     cp.float64.  Otherwise, the data-type of the output is the
        ///     same as that of the input.<br></br>
        ///     If out is specified, that array is
        ///     returned instead.
        /// </returns>
        public static NDarray<double> median(this NDarray a, Axis axis, NDarray @out = null,
            bool? overwrite_input = false, bool? keepdims = false)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray<double>(cp.median(a.CupyNDarray, axis.CupyAxis, @out?.CupyNDarray, overwrite_input, keepdims));
            }
            else
            {
                return new NDarray<double>(np.median(a.NumpyNDarray, axis.NumpyAxis, @out?.NumpyNDarray, overwrite_input, keepdims));
            }
        }

        /// <summary>
        ///     Compute the median along the specified axis.<br></br>
        ///     Returns the median of the array elements.<br></br>
        ///     Notes
        ///     Given a vector V of length N, the median of V is the
        ///     middle value of a sorted copy of V, V_sorted - i
        ///     e., V_sorted[(N-1)/2], when N is odd, and the average of the
        ///     two middle values of V_sorted when N is even.
        /// </summary>
        /// <param name="a">
        ///     Input array or object that can be converted to an array.
        /// </param>
        /// <param name="out">
        ///     Alternative output array in which to place the result.<br></br>
        ///     It must
        ///     have the same shape and buffer length as the expected output,
        ///     but the type (of the output) will be cast if necessary.
        /// </param>
        /// <param name="overwrite_input">
        ///     If True, then allow use of memory of input array a for
        ///     calculations.<br></br>
        ///     The input array will be modified by the call to
        ///     median.<br></br>
        ///     This will save memory when you do not need to preserve
        ///     the contents of the input array.<br></br>
        ///     Treat the input as undefined,
        ///     but it will probably be fully or partially sorted.<br></br>
        ///     Default is
        ///     False.<br></br>
        ///     If overwrite_input is True and a is not already an
        ///     ndarray, an error will be raised.
        /// </param>
        /// <returns>
        ///     A new array holding the result.<br></br>
        ///     If the input contains integers
        ///     or floats smaller than float64, then the output data-type is
        ///     cp.float64.  Otherwise, the data-type of the output is the
        ///     same as that of the input.<br></br>
        ///     If out is specified, that array is
        ///     returned instead.
        /// </returns>
        public static double median(this NDarray a, NDarray @out = null, bool? overwrite_input = false)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return cp.median(a.CupyNDarray, @out?.CupyNDarray, overwrite_input);
            }
            else
            {
                return np.median(a.NumpyNDarray, @out?.NumpyNDarray, overwrite_input);
            }
        }

        /// <summary>
        ///     Compute the weighted average along the specified axis.
        /// </summary>
        /// <param name="a">
        ///     Array containing data to be averaged.<br></br>
        ///     If a is not an array, a
        ///     conversion is attempted.
        /// </param>
        /// <param name="axis">
        ///     Axis or axis along which to average a.<br></br>
        ///     The default,
        ///     axis=None, will average over all of the elements of the input array.<br></br>
        ///     If axis is negative it counts from the last to the first axis.<br></br>
        ///     If axis is a tuple of ints, averaging is performed on all of the axis
        ///     specified in the tuple instead of a single axis or all the axis as
        ///     before.
        /// </param>
        /// <param name="weights">
        ///     An array of weights associated with the values in a.<br></br>
        ///     Each value in
        ///     a contributes to the average according to its associated weight.<br></br>
        ///     The weights array can either be 1-D (in which case its length must be
        ///     the size of a along the given axis) or of the same shape as a.<br></br>
        ///     If weights=None, then all data in a are assumed to have a
        ///     weight equal to one.
        /// </param>
        /// <param name="returned">
        ///     Default is False.<br></br>
        ///     If True, the tuple (average, sum_of_weights)
        ///     is returned, otherwise only the average is returned.<br></br>
        ///     If weights=None, sum_of_weights is equivalent to the number of
        ///     elements over which the average is taken.
        /// </param>
        /// <returns>
        ///     Return the average along the specified axis.<br></br>
        ///     When returned is True,
        ///     return a tuple with the average as the first element and the sum
        ///     of the weights as the second element.<br></br>
        ///     sum_of_weights is of the
        ///     same type as retval.<br></br>
        ///     The result dtype follows a genereal pattern.<br></br>
        ///     If weights is None, the result dtype will be that of a , or float64
        ///     if a is integral.<br></br>
        ///     Otherwise, if weights is not None and a is non-
        ///     integral, the result type will be the type of lowest precision capable of
        ///     representing values of both a and weights.<br></br>
        ///     If a happens to be
        ///     integral, the previous rules still applies but the result dtype will
        ///     at least be float64.
        /// </returns>
        public static NDarray<double> average(this NDarray a, Axis axis, NDarray weights = null, bool? returned = false)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray<double>(cp.average(a.CupyNDarray, axis.CupyAxis, weights?.CupyNDarray, returned));
            }
            else
            {
                return new NDarray<double>(np.average(a.NumpyNDarray, axis.NumpyAxis, weights?.NumpyNDarray, returned));
            }
        }

        /// <summary>
        ///     Compute the weighted average along the specified axis.
        /// </summary>
        /// <param name="a">
        ///     Array containing data to be averaged.<br></br>
        ///     If a is not an array, a
        ///     conversion is attempted.
        /// </param>
        /// <param name="weights">
        ///     An array of weights associated with the values in a.<br></br>
        ///     Each value in
        ///     a contributes to the average according to its associated weight.<br></br>
        ///     The weights array can either be 1-D (in which case its length must be
        ///     the size of a along the given axis) or of the same shape as a.<br></br>
        ///     If weights=None, then all data in a are assumed to have a
        ///     weight equal to one.
        /// </param>
        /// <param name="returned">
        ///     Default is False.<br></br>
        ///     If True, the tuple (average, sum_of_weights)
        ///     is returned, otherwise only the average is returned.<br></br>
        ///     If weights=None, sum_of_weights is equivalent to the number of
        ///     elements over which the average is taken.
        /// </param>
        /// <returns>
        ///     Return the average along the specified axis.<br></br>
        ///     When returned is True,
        ///     return a tuple with the average as the first element and the sum
        ///     of the weights as the second element.<br></br>
        ///     sum_of_weights is of the
        ///     same type as retval.<br></br>
        ///     The result dtype follows a genereal pattern.<br></br>
        ///     If weights is None, the result dtype will be that of a , or float64
        ///     if a is integral.<br></br>
        ///     Otherwise, if weights is not None and a is non-
        ///     integral, the result type will be the type of lowest precision capable of
        ///     representing values of both a and weights.<br></br>
        ///     If a happens to be
        ///     integral, the previous rules still applies but the result dtype will
        ///     at least be float64.
        /// </returns>
        public static double average(this NDarray a, NDarray weights = null, bool? returned = false)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return cp.average(a.CupyNDarray, weights?.CupyNDarray, returned);
            }
            else
            {
                return np.average(a.NumpyNDarray, weights?.NumpyNDarray, returned);
            }
        }

        /// <summary>
        ///     Compute the arithmetic mean along the specified axis.<br></br>
        ///     Returns the average of the array elements.<br></br>
        ///     The average is taken over
        ///     the flattened array by default, otherwise over the specified axis.<br></br>
        ///     float64 intermediate and return values are used for integer inputs.<br></br>
        ///     Notes
        ///     The arithmetic mean is the sum of the elements along the axis divided
        ///     by the number of elements.<br></br>
        ///     Note that for floating-point input, the mean is computed using the
        ///     same precision the input has.<br></br>
        ///     Depending on the input data, this can
        ///     cause the results to be inaccurate, especially for float32 (see
        ///     example below).<br></br>
        ///     Specifying a higher-precision accumulator using the
        ///     dtype keyword can alleviate this issue.<br></br>
        ///     By default, float16 results are computed using float32 intermediates
        ///     for extra precision.
        /// </summary>
        /// <param name="a">
        ///     Array containing numbers whose mean is desired.<br></br>
        ///     If a is not an
        ///     array, a conversion is attempted.
        /// </param>
        /// <param name="axis">
        ///     Axis or axis along which the means are computed.<br></br>
        ///     The default is to
        ///     compute the mean of the flattened array.<br></br>
        ///     If this is a tuple of ints, a mean is performed over multiple axis,
        ///     instead of a single axis or all the axis as before.
        /// </param>
        /// <param name="dtype">
        ///     Type to use in computing the mean.<br></br>
        ///     For integer inputs, the default
        ///     is float64; for floating point inputs, it is the same as the
        ///     input dtype.
        /// </param>
        /// <param name="out">
        ///     Alternate output array in which to place the result.<br></br>
        ///     The default
        ///     is None; if provided, it must have the same shape as the
        ///     expected output, but the type will be cast if necessary.<br></br>
        ///     See doc.ufuncs for details.
        /// </param>
        /// <param name="keepdims">
        ///     If this is set to True, the axis which are reduced are left
        ///     in the result as dimensions with size one.<br></br>
        ///     With this option,
        ///     the result will broadcast correctly against the input array.<br></br>
        ///     If the default value is passed, then keepdims will not be
        ///     passed through to the mean method of sub-classes of
        ///     ndarray, however any non-default value will be.<br></br>
        ///     If the
        ///     sub-class’ method does not implement keepdims any
        ///     exceptions will be raised.
        /// </param>
        /// <returns>
        ///     If out=None, returns a new array containing the mean values,
        ///     otherwise a reference to the output array is returned.
        /// </returns>
        public static NDarray<double> mean(this NDarray a, Axis axis, Dtype dtype = null, NDarray @out = null,
            bool? keepdims = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray<double>(cp.mean(a.CupyNDarray, axis.CupyAxis, dtype?.CupyDtype, @out?.CupyNDarray,
                    keepdims));
            }
            else
            {
                return new NDarray<double>(np.mean(a.NumpyNDarray, axis.NumpyAxis, dtype?.NumpyDtype, @out?.NumpyNDarray, keepdims));
            }
        }

        /// <summary>
        ///     Compute the arithmetic mean along the specified axis.<br></br>
        ///     Returns the average of the array elements.<br></br>
        ///     The average is taken over
        ///     the flattened array by default, otherwise over the specified axis.<br></br>
        ///     float64 intermediate and return values are used for integer inputs.<br></br>
        ///     Notes
        ///     The arithmetic mean is the sum of the elements along the axis divided
        ///     by the number of elements.<br></br>
        ///     Note that for floating-point input, the mean is computed using the
        ///     same precision the input has.<br></br>
        ///     Depending on the input data, this can
        ///     cause the results to be inaccurate, especially for float32 (see
        ///     example below).<br></br>
        ///     Specifying a higher-precision accumulator using the
        ///     dtype keyword can alleviate this issue.<br></br>
        ///     By default, float16 results are computed using float32 intermediates
        ///     for extra precision.
        /// </summary>
        /// <param name="a">
        ///     Array containing numbers whose mean is desired.<br></br>
        ///     If a is not an
        ///     array, a conversion is attempted.
        /// </param>
        /// <param name="dtype">
        ///     Type to use in computing the mean.<br></br>
        ///     For integer inputs, the default
        ///     is float64; for floating point inputs, it is the same as the
        ///     input dtype.
        /// </param>
        /// <param name="out">
        ///     Alternate output array in which to place the result.<br></br>
        ///     The default
        ///     is None; if provided, it must have the same shape as the
        ///     expected output, but the type will be cast if necessary.<br></br>
        ///     See doc.ufuncs for details.
        /// </param>
        /// <returns>
        ///     If out=None, returns a new array containing the mean values,
        ///     otherwise a reference to the output array is returned.
        /// </returns>
        public static double mean(this NDarray a, Dtype dtype = null, NDarray @out = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return cp.mean(a.CupyNDarray, dtype?.CupyDtype, @out?.CupyNDarray);
            }
            else
            {
                return np.mean(a.NumpyNDarray, dtype?.NumpyDtype, @out?.NumpyNDarray);
            }
        }

        /// <summary>
        ///     Compute the standard deviation along the specified axis.<br></br>
        ///     Returns the standard deviation, a measure of the spread of a distribution,
        ///     of the array elements.<br></br>
        ///     The standard deviation is computed for the
        ///     flattened array by default, otherwise over the specified axis.<br></br>
        ///     Notes
        ///     The standard deviation is the square root of the average of the squared
        ///     deviations from the mean, i.e., std = sqrt(mean(abs(x - x.mean())**2)).<br></br>
        ///     The average squared deviation is normally calculated as
        ///     x.sum() / N, where N = len(x).<br></br>
        ///     If, however, ddof is specified,
        ///     the divisor N - ddof is used instead.<br></br>
        ///     In standard statistical
        ///     practice, ddof=1 provides an unbiased estimator of the variance
        ///     of the infinite population.<br></br>
        ///     ddof=0 provides a maximum likelihood
        ///     estimate of the variance for normally distributed variables.<br></br>
        ///     The
        ///     standard deviation computed in this function is the square root of
        ///     the estimated variance, so even with ddof=1, it will not be an
        ///     unbiased estimate of the standard deviation per se.<br></br>
        ///     Note that, for complex numbers, std takes the absolute
        ///     value before squaring, so that the result is always real and nonnegative.<br></br>
        ///     For floating-point input, the std is computed using the same
        ///     precision the input has.<br></br>
        ///     Depending on the input data, this can cause
        ///     the results to be inaccurate, especially for float32 (see example below).<br></br>
        ///     Specifying a higher-accuracy accumulator using the dtype keyword can
        ///     alleviate this issue.
        /// </summary>
        /// <param name="a">
        ///     Calculate the standard deviation of these values.
        /// </param>
        /// <param name="axis">
        ///     Axis or axis along which the standard deviation is computed.<br></br>
        ///     The
        ///     default is to compute the standard deviation of the flattened array.<br></br>
        ///     If this is a tuple of ints, a standard deviation is performed over
        ///     multiple axis, instead of a single axis or all the axis as before.
        /// </param>
        /// <param name="dtype">
        ///     Type to use in computing the standard deviation.<br></br>
        ///     For arrays of
        ///     integer type the default is float64, for arrays of float types it is
        ///     the same as the array type.
        /// </param>
        /// <param name="out">
        ///     Alternative output array in which to place the result.<br></br>
        ///     It must have
        ///     the same shape as the expected output but the type (of the calculated
        ///     values) will be cast if necessary.
        /// </param>
        /// <param name="ddof">
        ///     Means Delta Degrees of Freedom.<br></br>
        ///     The divisor used in calculations
        ///     is N - ddof, where N represents the number of elements.<br></br>
        ///     By default ddof is zero.
        /// </param>
        /// <param name="keepdims">
        ///     If this is set to True, the axis which are reduced are left
        ///     in the result as dimensions with size one.<br></br>
        ///     With this option,
        ///     the result will broadcast correctly against the input array.<br></br>
        ///     If the default value is passed, then keepdims will not be
        ///     passed through to the std method of sub-classes of
        ///     ndarray, however any non-default value will be.<br></br>
        ///     If the
        ///     sub-class’ method does not implement keepdims any
        ///     exceptions will be raised.
        /// </param>
        /// <returns>
        ///     If out is None, return a new array containing the standard deviation,
        ///     otherwise return a reference to the output array.
        /// </returns>
        public static NDarray<double> std(this NDarray a, Axis axis, Dtype dtype = null, NDarray @out = null,
            int? ddof = 0, bool? keepdims = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray<double>(cp.std(a.CupyNDarray, axis.CupyAxis, dtype?.CupyDtype, @out?.CupyNDarray,
                    ddof, keepdims));
            }
            else
            {
                return new NDarray<double>(np.std(a.NumpyNDarray, axis.NumpyAxis, dtype?.NumpyDtype, @out?.NumpyNDarray, ddof, keepdims));
            }
        }

        /// <summary>
        ///     Compute the standard deviation along the specified axis.<br></br>
        ///     Returns the standard deviation, a measure of the spread of a distribution,
        ///     of the array elements.<br></br>
        ///     The standard deviation is computed for the
        ///     flattened array by default, otherwise over the specified axis.<br></br>
        ///     Notes
        ///     The standard deviation is the square root of the average of the squared
        ///     deviations from the mean, i.e., std = sqrt(mean(abs(x - x.mean())**2)).<br></br>
        ///     The average squared deviation is normally calculated as
        ///     x.sum() / N, where N = len(x).<br></br>
        ///     If, however, ddof is specified,
        ///     the divisor N - ddof is used instead.<br></br>
        ///     In standard statistical
        ///     practice, ddof=1 provides an unbiased estimator of the variance
        ///     of the infinite population.<br></br>
        ///     ddof=0 provides a maximum likelihood
        ///     estimate of the variance for normally distributed variables.<br></br>
        ///     The
        ///     standard deviation computed in this function is the square root of
        ///     the estimated variance, so even with ddof=1, it will not be an
        ///     unbiased estimate of the standard deviation per se.<br></br>
        ///     Note that, for complex numbers, std takes the absolute
        ///     value before squaring, so that the result is always real and nonnegative.<br></br>
        ///     For floating-point input, the std is computed using the same
        ///     precision the input has.<br></br>
        ///     Depending on the input data, this can cause
        ///     the results to be inaccurate, especially for float32 (see example below).<br></br>
        ///     Specifying a higher-accuracy accumulator using the dtype keyword can
        ///     alleviate this issue.
        /// </summary>
        /// <param name="a">
        ///     Calculate the standard deviation of these values.
        /// </param>
        /// <param name="dtype">
        ///     Type to use in computing the standard deviation.<br></br>
        ///     For arrays of
        ///     integer type the default is float64, for arrays of float types it is
        ///     the same as the array type.
        /// </param>
        /// <param name="out">
        ///     Alternative output array in which to place the result.<br></br>
        ///     It must have
        ///     the same shape as the expected output but the type (of the calculated
        ///     values) will be cast if necessary.
        /// </param>
        /// <param name="ddof">
        ///     Means Delta Degrees of Freedom.<br></br>
        ///     The divisor used in calculations
        ///     is N - ddof, where N represents the number of elements.<br></br>
        ///     By default ddof is zero.
        /// </param>
        /// <returns>
        ///     If out is None, return a new array containing the standard deviation,
        ///     otherwise return a reference to the output array.
        /// </returns>
        public static double std(this NDarray a, Dtype dtype = null, NDarray @out = null, int? ddof = 0)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return cp.std(a.CupyNDarray, dtype?.CupyDtype, @out?.CupyNDarray, ddof);
            }
            else
            {
                return np.std(a.NumpyNDarray, dtype?.NumpyDtype, @out?.NumpyNDarray, ddof);
            }
        }

        /// <summary>
        ///     Compute the variance along the specified axis.<br></br>
        ///     Returns the variance of the array elements, a measure of the spread of a
        ///     distribution.<br></br>
        ///     The variance is computed for the flattened array by
        ///     default, otherwise over the specified axis.<br></br>
        ///     Notes
        ///     The variance is the average of the squared deviations from the mean,
        ///     i.e.,  var = mean(abs(x - x.mean())**2).<br></br>
        ///     The mean is normally calculated as x.sum() / N, where N = len(x).<br></br>
        ///     If, however, ddof is specified, the divisor N - ddof is used
        ///     instead.<br></br>
        ///     In standard statistical practice, ddof=1 provides an
        ///     unbiased estimator of the variance of a hypothetical infinite population.<br></br>
        ///     ddof=0 provides a maximum likelihood estimate of the variance for
        ///     normally distributed variables.<br></br>
        ///     Note that for complex numbers, the absolute value is taken before
        ///     squaring, so that the result is always real and nonnegative.<br></br>
        ///     For floating-point input, the variance is computed using the same
        ///     precision the input has.<br></br>
        ///     Depending on the input data, this can cause
        ///     the results to be inaccurate, especially for float32 (see example
        ///     below).<br></br>
        ///     Specifying a higher-accuracy accumulator using the dtype
        ///     keyword can alleviate this issue.
        /// </summary>
        /// <param name="a">
        ///     Array containing numbers whose variance is desired.<br></br>
        ///     If a is not an
        ///     array, a conversion is attempted.
        /// </param>
        /// <param name="axis">
        ///     Axis or axis along which the variance is computed.<br></br>
        ///     The default is to
        ///     compute the variance of the flattened array.<br></br>
        ///     If this is a tuple of ints, a variance is performed over multiple axis,
        ///     instead of a single axis or all the axis as before.
        /// </param>
        /// <param name="dtype">
        ///     Type to use in computing the variance.<br></br>
        ///     For arrays of integer type
        ///     the default is float32; for arrays of float types it is the same as
        ///     the array type.
        /// </param>
        /// <param name="out">
        ///     Alternate output array in which to place the result.<br></br>
        ///     It must have
        ///     the same shape as the expected output, but the type is cast if
        ///     necessary.
        /// </param>
        /// <param name="ddof">
        ///     “Delta Degrees of Freedom”: the divisor used in the calculation is
        ///     N - ddof, where N represents the number of elements.<br></br>
        ///     By
        ///     default ddof is zero.
        /// </param>
        /// <param name="keepdims">
        ///     If this is set to True, the axis which are reduced are left
        ///     in the result as dimensions with size one.<br></br>
        ///     With this option,
        ///     the result will broadcast correctly against the input array.<br></br>
        ///     If the default value is passed, then keepdims will not be
        ///     passed through to the var method of sub-classes of
        ///     ndarray, however any non-default value will be.<br></br>
        ///     If the
        ///     sub-class’ method does not implement keepdims any
        ///     exceptions will be raised.
        /// </param>
        /// <returns>
        ///     If out=None, returns a new array containing the variance;
        ///     otherwise, a reference to the output array is returned.
        /// </returns>
        public static NDarray<double> var(this NDarray a, Axis axis, Dtype dtype = null, NDarray @out = null,
            int? ddof = 0, bool? keepdims = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray<double>(cp.var(a.CupyNDarray, axis.CupyAxis, dtype?.CupyDtype, @out?.CupyNDarray,
                    ddof, keepdims));
            }
            else
            {
                return new NDarray<double>(np.var(a.NumpyNDarray, axis.NumpyAxis, dtype?.NumpyDtype, @out?.NumpyNDarray, ddof, keepdims));
            }
        }

        /// <summary>
        ///     Compute the variance along the specified axis.<br></br>
        ///     Returns the variance of the array elements, a measure of the spread of a
        ///     distribution.<br></br>
        ///     The variance is computed for the flattened array by
        ///     default, otherwise over the specified axis.<br></br>
        ///     Notes
        ///     The variance is the average of the squared deviations from the mean,
        ///     i.e.,  var = mean(abs(x - x.mean())**2).<br></br>
        ///     The mean is normally calculated as x.sum() / N, where N = len(x).<br></br>
        ///     If, however, ddof is specified, the divisor N - ddof is used
        ///     instead.<br></br>
        ///     In standard statistical practice, ddof=1 provides an
        ///     unbiased estimator of the variance of a hypothetical infinite population.<br></br>
        ///     ddof=0 provides a maximum likelihood estimate of the variance for
        ///     normally distributed variables.<br></br>
        ///     Note that for complex numbers, the absolute value is taken before
        ///     squaring, so that the result is always real and nonnegative.<br></br>
        ///     For floating-point input, the variance is computed using the same
        ///     precision the input has.<br></br>
        ///     Depending on the input data, this can cause
        ///     the results to be inaccurate, especially for float32 (see example
        ///     below).<br></br>
        ///     Specifying a higher-accuracy accumulator using the dtype
        ///     keyword can alleviate this issue.
        /// </summary>
        /// <param name="a">
        ///     Array containing numbers whose variance is desired.<br></br>
        ///     If a is not an
        ///     array, a conversion is attempted.
        /// </param>
        /// <param name="dtype">
        ///     Type to use in computing the variance.<br></br>
        ///     For arrays of integer type
        ///     the default is float32; for arrays of float types it is the same as
        ///     the array type.
        /// </param>
        /// <param name="out">
        ///     Alternate output array in which to place the result.<br></br>
        ///     It must have
        ///     the same shape as the expected output, but the type is cast if
        ///     necessary.
        /// </param>
        /// <param name="ddof">
        ///     “Delta Degrees of Freedom”: the divisor used in the calculation is
        ///     N - ddof, where N represents the number of elements.<br></br>
        ///     By
        ///     default ddof is zero.
        /// </param>
        /// <returns>
        ///     If out=None, returns a new array containing the variance;
        ///     otherwise, a reference to the output array is returned.
        /// </returns>
        public static double var(this NDarray a, Dtype dtype = null, NDarray @out = null, int? ddof = 0)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return cp.var(a.CupyNDarray, dtype?.CupyDtype, @out?.CupyNDarray, ddof);
            }
            else
            {
                return np.var(a.NumpyNDarray, dtype?.NumpyDtype, @out?.NumpyNDarray, ddof);
            }
        }

        /// <summary>
        ///     Compute the median along the specified axis, while ignoring NaNs.<br></br>
        ///     Returns the median of the array elements.<br></br>
        ///     Notes
        ///     Given a vector V of length N, the median of V is the
        ///     middle value of a sorted copy of V, V_sorted - i.e.,
        ///     V_sorted[(N-1)/2], when N is odd and the average of the two
        ///     middle values of V_sorted when N is even.
        /// </summary>
        /// <param name="a">
        ///     Input array or object that can be converted to an array.
        /// </param>
        /// <param name="axis">
        ///     Axis or axis along which the medians are computed.<br></br>
        ///     The default
        ///     is to compute the median along a flattened version of the array.<br></br>
        ///     A sequence of axis is supported since version 1.9.0.
        /// </param>
        /// <param name="out">
        ///     Alternative output array in which to place the result.<br></br>
        ///     It must
        ///     have the same shape and buffer length as the expected output,
        ///     but the type (of the output) will be cast if necessary.
        /// </param>
        /// <param name="overwrite_input">
        ///     If True, then allow use of memory of input array a for
        ///     calculations.<br></br>
        ///     The input array will be modified by the call to
        ///     median.<br></br>
        ///     This will save memory when you do not need to preserve
        ///     the contents of the input array.<br></br>
        ///     Treat the input as undefined,
        ///     but it will probably be fully or partially sorted.<br></br>
        ///     Default is
        ///     False.<br></br>
        ///     If overwrite_input is True and a is not already an
        ///     ndarray, an error will be raised.
        /// </param>
        /// <param name="keepdims">
        ///     If this is set to True, the axis which are reduced are left
        ///     in the result as dimensions with size one.<br></br>
        ///     With this option,
        ///     the result will broadcast correctly against the original a.<br></br>
        ///     If this is anything but the default value it will be passed
        ///     through (in the special case of an empty array) to the
        ///     mean function of the underlying array.<br></br>
        ///     If the array is
        ///     a sub-class and mean does not have the kwarg keepdims this
        ///     will raise a RuntimeError.
        /// </param>
        /// <returns>
        ///     A new array holding the result.<br></br>
        ///     If the input contains integers
        ///     or floats smaller than float64, then the output data-type is
        ///     cp.float64.  Otherwise, the data-type of the output is the
        ///     same as that of the input.<br></br>
        ///     If out is specified, that array is
        ///     returned instead.
        /// </returns>
        public static NDarray<double> nanmedian(this NDarray a, Axis axis, NDarray @out = null,
            bool? overwrite_input = false, bool? keepdims = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray<double>(cp.nanmedian(a.CupyNDarray, axis.CupyAxis, @out?.CupyNDarray, overwrite_input, keepdims));
            }
            else
            {
                return new NDarray<double>(np.nanmedian(a.NumpyNDarray, axis.NumpyAxis, @out?.NumpyNDarray, overwrite_input, keepdims));
            }
        }

        /// <summary>
        ///     Compute the median along the specified axis, while ignoring NaNs.<br></br>
        ///     Returns the median of the array elements.<br></br>
        ///     Notes
        ///     Given a vector V of length N, the median of V is the
        ///     middle value of a sorted copy of V, V_sorted - i.e.,
        ///     V_sorted[(N-1)/2], when N is odd and the average of the two
        ///     middle values of V_sorted when N is even.
        /// </summary>
        /// <param name="a">
        ///     Input array or object that can be converted to an array.
        /// </param>
        /// <param name="out">
        ///     Alternative output array in which to place the result.<br></br>
        ///     It must
        ///     have the same shape and buffer length as the expected output,
        ///     but the type (of the output) will be cast if necessary.
        /// </param>
        /// <param name="overwrite_input">
        ///     If True, then allow use of memory of input array a for
        ///     calculations.<br></br>
        ///     The input array will be modified by the call to
        ///     median.<br></br>
        ///     This will save memory when you do not need to preserve
        ///     the contents of the input array.<br></br>
        ///     Treat the input as undefined,
        ///     but it will probably be fully or partially sorted.<br></br>
        ///     Default is
        ///     False.<br></br>
        ///     If overwrite_input is True and a is not already an
        ///     ndarray, an error will be raised.
        /// </param>
        /// <returns>
        ///     A new array holding the result.<br></br>
        ///     If the input contains integers
        ///     or floats smaller than float64, then the output data-type is
        ///     cp.float64.  Otherwise, the data-type of the output is the
        ///     same as that of the input.<br></br>
        ///     If out is specified, that array is
        ///     returned instead.
        /// </returns>
        public static double nanmedian(this NDarray a, NDarray @out = null, bool? overwrite_input = false)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return cp.nanmedian(a.CupyNDarray, @out?.CupyNDarray, overwrite_input);
            }
            else
            {
                return np.nanmedian(a.NumpyNDarray, @out?.NumpyNDarray, overwrite_input);
            }
        }

        /// <summary>
        ///     Compute the arithmetic mean along the specified axis, ignoring NaNs.<br></br>
        ///     Returns the average of the array elements.<br></br>
        ///     The average is taken over
        ///     the flattened array by default, otherwise over the specified axis.<br></br>
        ///     float64 intermediate and return values are used for integer inputs.<br></br>
        ///     For all-NaN slices, NaN is returned and a RuntimeWarning is raised.<br></br>
        ///     Notes
        ///     The arithmetic mean is the sum of the non-NaN elements along the axis
        ///     divided by the number of non-NaN elements.<br></br>
        ///     Note that for floating-point input, the mean is computed using the same
        ///     precision the input has.<br></br>
        ///     Depending on the input data, this can cause
        ///     the results to be inaccurate, especially for float32.  Specifying a
        ///     higher-precision accumulator using the dtype keyword can alleviate
        ///     this issue.
        /// </summary>
        /// <param name="a">
        ///     Array containing numbers whose mean is desired.<br></br>
        ///     If a is not an
        ///     array, a conversion is attempted.
        /// </param>
        /// <param name="axis">
        ///     Axis or axis along which the means are computed.<br></br>
        ///     The default is to compute
        ///     the mean of the flattened array.
        /// </param>
        /// <param name="dtype">
        ///     Type to use in computing the mean.<br></br>
        ///     For integer inputs, the default
        ///     is float64; for inexact inputs, it is the same as the input
        ///     dtype.
        /// </param>
        /// <param name="out">
        ///     Alternate output array in which to place the result.<br></br>
        ///     The default
        ///     is None; if provided, it must have the same shape as the
        ///     expected output, but the type will be cast if necessary.<br></br>
        ///     See
        ///     doc.ufuncs for details.
        /// </param>
        /// <param name="keepdims">
        ///     If this is set to True, the axis which are reduced are left
        ///     in the result as dimensions with size one.<br></br>
        ///     With this option,
        ///     the result will broadcast correctly against the original a.<br></br>
        ///     If the value is anything but the default, then
        ///     keepdims will be passed through to the mean or sum methods
        ///     of sub-classes of ndarray.<br></br>
        ///     If the sub-classes methods
        ///     does not implement keepdims any exceptions will be raised.
        /// </param>
        /// <returns>
        ///     If out=None, returns a new array containing the mean values,
        ///     otherwise a reference to the output array is returned.<br></br>
        ///     Nan is
        ///     returned for slices that contain only NaNs.
        /// </returns>
        public static NDarray<double> nanmean(this NDarray a, Axis axis, Dtype dtype = null, NDarray @out = null,
            bool? keepdims = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray<double>(cp.nanmean(a.CupyNDarray, axis.CupyAxis, dtype?.CupyDtype, @out?.CupyNDarray, keepdims));
            }
            else
            {
                return new NDarray<double>(np.nanmean(a.NumpyNDarray, axis.NumpyAxis, dtype?.NumpyDtype, @out?.NumpyNDarray, keepdims));
            }
        }

        /// <summary>
        ///     Compute the arithmetic mean along the specified axis, ignoring NaNs.<br></br>
        ///     Returns the average of the array elements.<br></br>
        ///     The average is taken over
        ///     the flattened array by default, otherwise over the specified axis.<br></br>
        ///     float64 intermediate and return values are used for integer inputs.<br></br>
        ///     For all-NaN slices, NaN is returned and a RuntimeWarning is raised.<br></br>
        ///     Notes
        ///     The arithmetic mean is the sum of the non-NaN elements along the axis
        ///     divided by the number of non-NaN elements.<br></br>
        ///     Note that for floating-point input, the mean is computed using the same
        ///     precision the input has.<br></br>
        ///     Depending on the input data, this can cause
        ///     the results to be inaccurate, especially for float32.  Specifying a
        ///     higher-precision accumulator using the dtype keyword can alleviate
        ///     this issue.
        /// </summary>
        /// <param name="a">
        ///     Array containing numbers whose mean is desired.<br></br>
        ///     If a is not an
        ///     array, a conversion is attempted.
        /// </param>
        /// <param name="dtype">
        ///     Type to use in computing the mean.<br></br>
        ///     For integer inputs, the default
        ///     is float64; for inexact inputs, it is the same as the input
        ///     dtype.
        /// </param>
        /// <param name="out">
        ///     Alternate output array in which to place the result.<br></br>
        ///     The default
        ///     is None; if provided, it must have the same shape as the
        ///     expected output, but the type will be cast if necessary.<br></br>
        ///     See
        ///     doc.ufuncs for details.
        /// </param>
        /// <returns>
        ///     If out=None, returns a new array containing the mean values,
        ///     otherwise a reference to the output array is returned.<br></br>
        ///     Nan is
        ///     returned for slices that contain only NaNs.
        /// </returns>
        public static double nanmean(this NDarray a, Dtype dtype = null, NDarray @out = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return cp.nanmean(a.CupyNDarray, dtype?.CupyDtype, @out?.CupyNDarray);
            }
            else
            {
                return np.nanmean(a.NumpyNDarray, dtype?.NumpyDtype, @out?.NumpyNDarray);
            }
        }

        /// <summary>
        ///     Compute the standard deviation along the specified axis, while
        ///     ignoring NaNs.<br></br>
        ///     Returns the standard deviation, a measure of the spread of a
        ///     distribution, of the non-NaN array elements.<br></br>
        ///     The standard deviation is
        ///     computed for the flattened array by default, otherwise over the
        ///     specified axis.<br></br>
        ///     For all-NaN slices or slices with zero degrees of freedom, NaN is
        ///     returned and a RuntimeWarning is raised.<br></br>
        ///     Notes
        ///     The standard deviation is the square root of the average of the squared
        ///     deviations from the mean: std = sqrt(mean(abs(x - x.mean())**2)).<br></br>
        ///     The average squared deviation is normally calculated as
        ///     x.sum() / N, where N = len(x).<br></br>
        ///     If, however, ddof is
        ///     specified, the divisor N - ddof is used instead.<br></br>
        ///     In standard
        ///     statistical practice, ddof=1 provides an unbiased estimator of the
        ///     variance of the infinite population.<br></br>
        ///     ddof=0 provides a maximum
        ///     likelihood estimate of the variance for normally distributed variables.<br></br>
        ///     The standard deviation computed in this function is the square root of
        ///     the estimated variance, so even with ddof=1, it will not be an
        ///     unbiased estimate of the standard deviation per se.<br></br>
        ///     Note that, for complex numbers, std takes the absolute value before
        ///     squaring, so that the result is always real and nonnegative.<br></br>
        ///     For floating-point input, the std is computed using the same
        ///     precision the input has.<br></br>
        ///     Depending on the input data, this can cause
        ///     the results to be inaccurate, especially for float32 (see example
        ///     below).<br></br>
        ///     Specifying a higher-accuracy accumulator using the dtype
        ///     keyword can alleviate this issue.
        /// </summary>
        /// <param name="a">
        ///     Calculate the standard deviation of the non-NaN values.
        /// </param>
        /// <param name="axis">
        ///     Axis or axis along which the standard deviation is computed.<br></br>
        ///     The default is
        ///     to compute the standard deviation of the flattened array.
        /// </param>
        /// <param name="dtype">
        ///     Type to use in computing the standard deviation.<br></br>
        ///     For arrays of
        ///     integer type the default is float64, for arrays of float types it
        ///     is the same as the array type.
        /// </param>
        /// <param name="out">
        ///     Alternative output array in which to place the result.<br></br>
        ///     It must have
        ///     the same shape as the expected output but the type (of the
        ///     calculated values) will be cast if necessary.
        /// </param>
        /// <param name="ddof">
        ///     Means Delta Degrees of Freedom.<br></br>
        ///     The divisor used in calculations
        ///     is N - ddof, where N represents the number of non-NaN
        ///     elements.<br></br>
        ///     By default ddof is zero.
        /// </param>
        /// <param name="keepdims">
        ///     If this is set to True, the axis which are reduced are left
        ///     in the result as dimensions with size one.<br></br>
        ///     With this option,
        ///     the result will broadcast correctly against the original a.<br></br>
        ///     If this value is anything but the default it is passed through
        ///     as-is to the relevant functions of the sub-classes.<br></br>
        ///     If these
        ///     functions do not have a keepdims kwarg, a RuntimeError will
        ///     be raised.
        /// </param>
        /// <returns>
        ///     If out is None, return a new array containing the standard
        ///     deviation, otherwise return a reference to the output array.<br></br>
        ///     If
        ///     ddof is &gt;= the number of non-NaN elements in a slice or the slice
        ///     contains only NaNs, then the result for that slice is NaN.
        /// </returns>
        public static NDarray<double> nanstd(this NDarray a, Axis axis, Dtype dtype = null, NDarray @out = null,
            int? ddof = 0, bool? keepdims = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray<double>(cp.nanstd(a.CupyNDarray, axis.CupyAxis, dtype?.CupyDtype, @out?.CupyNDarray,
                    ddof, keepdims));
            }
            else
            {
                return new NDarray<double>(np.nanstd(a.NumpyNDarray, axis.NumpyAxis, dtype?.NumpyDtype, @out?.NumpyNDarray, ddof, keepdims));
            }
        }

        /// <summary>
        ///     Compute the standard deviation along the specified axis, while
        ///     ignoring NaNs.<br></br>
        ///     Returns the standard deviation, a measure of the spread of a
        ///     distribution, of the non-NaN array elements.<br></br>
        ///     The standard deviation is
        ///     computed for the flattened array by default, otherwise over the
        ///     specified axis.<br></br>
        ///     For all-NaN slices or slices with zero degrees of freedom, NaN is
        ///     returned and a RuntimeWarning is raised.<br></br>
        ///     Notes
        ///     The standard deviation is the square root of the average of the squared
        ///     deviations from the mean: std = sqrt(mean(abs(x - x.mean())**2)).<br></br>
        ///     The average squared deviation is normally calculated as
        ///     x.sum() / N, where N = len(x).<br></br>
        ///     If, however, ddof is
        ///     specified, the divisor N - ddof is used instead.<br></br>
        ///     In standard
        ///     statistical practice, ddof=1 provides an unbiased estimator of the
        ///     variance of the infinite population.<br></br>
        ///     ddof=0 provides a maximum
        ///     likelihood estimate of the variance for normally distributed variables.<br></br>
        ///     The standard deviation computed in this function is the square root of
        ///     the estimated variance, so even with ddof=1, it will not be an
        ///     unbiased estimate of the standard deviation per se.<br></br>
        ///     Note that, for complex numbers, std takes the absolute value before
        ///     squaring, so that the result is always real and nonnegative.<br></br>
        ///     For floating-point input, the std is computed using the same
        ///     precision the input has.<br></br>
        ///     Depending on the input data, this can cause
        ///     the results to be inaccurate, especially for float32 (see example
        ///     below).<br></br>
        ///     Specifying a higher-accuracy accumulator using the dtype
        ///     keyword can alleviate this issue.
        /// </summary>
        /// <param name="a">
        ///     Calculate the standard deviation of the non-NaN values.
        /// </param>
        /// <param name="dtype">
        ///     Type to use in computing the standard deviation.<br></br>
        ///     For arrays of
        ///     integer type the default is float64, for arrays of float types it
        ///     is the same as the array type.
        /// </param>
        /// <param name="out">
        ///     Alternative output array in which to place the result.<br></br>
        ///     It must have
        ///     the same shape as the expected output but the type (of the
        ///     calculated values) will be cast if necessary.
        /// </param>
        /// <param name="ddof">
        ///     Means Delta Degrees of Freedom.<br></br>
        ///     The divisor used in calculations
        ///     is N - ddof, where N represents the number of non-NaN
        ///     elements.<br></br>
        ///     By default ddof is zero.
        /// </param>
        /// <returns>
        ///     If out is None, return a new array containing the standard
        ///     deviation, otherwise return a reference to the output array.<br></br>
        ///     If
        ///     ddof is &gt;= the number of non-NaN elements in a slice or the slice
        ///     contains only NaNs, then the result for that slice is NaN.
        /// </returns>
        public static double nanstd(this NDarray a, Dtype dtype = null, NDarray @out = null, int? ddof = 0)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return cp.nanstd(a.CupyNDarray, dtype?.CupyDtype, @out?.CupyNDarray, ddof);
            }
            else
            {
                return np.nanstd(a.NumpyNDarray, dtype?.NumpyDtype, @out?.NumpyNDarray, ddof);
            }
        }

        /// <summary>
        ///     Compute the variance along the specified axis, while ignoring NaNs.<br></br>
        ///     Returns the variance of the array elements, a measure of the spread of
        ///     a distribution.<br></br>
        ///     The variance is computed for the flattened array by
        ///     default, otherwise over the specified axis.<br></br>
        ///     For all-NaN slices or slices with zero degrees of freedom, NaN is
        ///     returned and a RuntimeWarning is raised.<br></br>
        ///     Notes
        ///     The variance is the average of the squared deviations from the mean,
        ///     i.e.,  var = mean(abs(x - x.mean())**2).<br></br>
        ///     The mean is normally calculated as x.sum() / N, where N = len(x).<br></br>
        ///     If, however, ddof is specified, the divisor N - ddof is used
        ///     instead.<br></br>
        ///     In standard statistical practice, ddof=1 provides an
        ///     unbiased estimator of the variance of a hypothetical infinite
        ///     population.<br></br>
        ///     ddof=0 provides a maximum likelihood estimate of the
        ///     variance for normally distributed variables.<br></br>
        ///     Note that for complex numbers, the absolute value is taken before
        ///     squaring, so that the result is always real and nonnegative.<br></br>
        ///     For floating-point input, the variance is computed using the same
        ///     precision the input has.<br></br>
        ///     Depending on the input data, this can cause
        ///     the results to be inaccurate, especially for float32 (see example
        ///     below).<br></br>
        ///     Specifying a higher-accuracy accumulator using the dtype
        ///     keyword can alleviate this issue.<br></br>
        ///     For this function to work on sub-classes of ndarray, they must define
        ///     sum with the kwarg keepdims
        /// </summary>
        /// <param name="a">
        ///     Array containing numbers whose variance is desired.<br></br>
        ///     If a is not an
        ///     array, a conversion is attempted.
        /// </param>
        /// <param name="axis">
        ///     Axis or axis along which the variance is computed.<br></br>
        ///     The default is to compute
        ///     the variance of the flattened array.
        /// </param>
        /// <param name="dtype">
        ///     Type to use in computing the variance.<br></br>
        ///     For arrays of integer type
        ///     the default is float32; for arrays of float types it is the same as
        ///     the array type.
        /// </param>
        /// <param name="out">
        ///     Alternate output array in which to place the result.<br></br>
        ///     It must have
        ///     the same shape as the expected output, but the type is cast if
        ///     necessary.
        /// </param>
        /// <param name="ddof">
        ///     “Delta Degrees of Freedom”: the divisor used in the calculation is
        ///     N - ddof, where N represents the number of non-NaN
        ///     elements.<br></br>
        ///     By default ddof is zero.
        /// </param>
        /// <param name="keepdims">
        ///     If this is set to True, the axis which are reduced are left
        ///     in the result as dimensions with size one.<br></br>
        ///     With this option,
        ///     the result will broadcast correctly against the original a.
        /// </param>
        /// <returns>
        ///     If out is None, return a new array containing the variance,
        ///     otherwise return a reference to the output array.<br></br>
        ///     If ddof is &gt;= the
        ///     number of non-NaN elements in a slice or the slice contains only
        ///     NaNs, then the result for that slice is NaN.
        /// </returns>
        public static NDarray<double> nanvar(this NDarray a, Axis axis, Dtype dtype = null, NDarray @out = null,
            int? ddof = 0, bool? keepdims = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray<double>(cp.nanvar(a.CupyNDarray, axis.CupyAxis, dtype?.CupyDtype, @out?.CupyNDarray,
                    ddof, keepdims));
            }
            else
            {
                return new NDarray<double>(np.nanvar(a.NumpyNDarray, axis.NumpyAxis, dtype?.NumpyDtype, @out?.NumpyNDarray, ddof, keepdims));
            }
        }

        /// <summary>
        ///     Compute the variance along the specified axis, while ignoring NaNs.<br></br>
        ///     Returns the variance of the array elements, a measure of the spread of
        ///     a distribution.<br></br>
        ///     The variance is computed for the flattened array by
        ///     default, otherwise over the specified axis.<br></br>
        ///     For all-NaN slices or slices with zero degrees of freedom, NaN is
        ///     returned and a RuntimeWarning is raised.<br></br>
        ///     Notes
        ///     The variance is the average of the squared deviations from the mean,
        ///     i.e.,  var = mean(abs(x - x.mean())**2).<br></br>
        ///     The mean is normally calculated as x.sum() / N, where N = len(x).<br></br>
        ///     If, however, ddof is specified, the divisor N - ddof is used
        ///     instead.<br></br>
        ///     In standard statistical practice, ddof=1 provides an
        ///     unbiased estimator of the variance of a hypothetical infinite
        ///     population.<br></br>
        ///     ddof=0 provides a maximum likelihood estimate of the
        ///     variance for normally distributed variables.<br></br>
        ///     Note that for complex numbers, the absolute value is taken before
        ///     squaring, so that the result is always real and nonnegative.<br></br>
        ///     For floating-point input, the variance is computed using the same
        ///     precision the input has.<br></br>
        ///     Depending on the input data, this can cause
        ///     the results to be inaccurate, especially for float32 (see example
        ///     below).<br></br>
        ///     Specifying a higher-accuracy accumulator using the dtype
        ///     keyword can alleviate this issue.<br></br>
        ///     For this function to work on sub-classes of ndarray, they must define
        ///     sum with the kwarg keepdims
        /// </summary>
        /// <param name="a">
        ///     Array containing numbers whose variance is desired.<br></br>
        ///     If a is not an
        ///     array, a conversion is attempted.
        /// </param>
        /// <param name="dtype">
        ///     Type to use in computing the variance.<br></br>
        ///     For arrays of integer type
        ///     the default is float32; for arrays of float types it is the same as
        ///     the array type.
        /// </param>
        /// <param name="out">
        ///     Alternate output array in which to place the result.<br></br>
        ///     It must have
        ///     the same shape as the expected output, but the type is cast if
        ///     necessary.
        /// </param>
        /// <param name="ddof">
        ///     “Delta Degrees of Freedom”: the divisor used in the calculation is
        ///     N - ddof, where N represents the number of non-NaN
        ///     elements.<br></br>
        ///     By default ddof is zero.
        /// </param>
        /// <returns>
        ///     If out is None, return a new array containing the variance,
        ///     otherwise return a reference to the output array.<br></br>
        ///     If ddof is &gt;= the
        ///     number of non-NaN elements in a slice or the slice contains only
        ///     NaNs, then the result for that slice is NaN.
        /// </returns>
        public static double nanvar(this NDarray a, Dtype dtype = null, NDarray @out = null, int? ddof = 0)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return cp.nanvar(a.CupyNDarray, dtype?.CupyDtype, @out?.CupyNDarray, ddof);
            }
            else
            {
                return np.nanvar(a.NumpyNDarray, dtype?.NumpyDtype, @out?.NumpyNDarray, ddof);
            }
        }

        /// <summary>
        ///     Return Pearson product-moment correlation coefficients.<br></br>
        ///     Please refer to the documentation for cov for more detail.<br></br>
        ///     The
        ///     relationship between the correlation coefficient matrix, R, and the
        ///     covariance matrix, C, is
        ///     The values of R are between -1 and 1, inclusive.<br></br>
        ///     Notes
        ///     Due to floating point rounding the resulting array may not be Hermitian,
        ///     the diagonal elements may not be 1, and the elements may not satisfy the
        ///     inequality abs(a) &lt;= 1.<br></br>
        ///     The real and imaginary parts are clipped to the
        ///     interval [-1,  1] in an attempt to improve on that situation but is not
        ///     much help in the complex case.<br></br>
        ///     This function accepts but discards arguments bias and ddof.<br></br>
        ///     This is
        ///     for backwards compatibility with previous versions of this function.<br></br>
        ///     These
        ///     arguments had no effect on the return values of the function and can be
        ///     safely ignored in this and previous versions of Cupy.
        /// </summary>
        /// <param name="x">
        ///     A 1-D or 2-D array containing multiple variables and observations.<br></br>
        ///     Each row of x represents a variable, and each column a single
        ///     observation of all those variables.<br></br>
        ///     Also see rowvar below.
        /// </param>
        /// <param name="y">
        ///     An additional set of variables and observations.<br></br>
        ///     y has the same
        ///     shape as x.
        /// </param>
        /// <param name="rowvar">
        ///     If rowvar is True (default), then each row represents a
        ///     variable, with observations in the columns.<br></br>
        ///     Otherwise, the relationship
        ///     is transposed: each column represents a variable, while the rows
        ///     contain observations.
        /// </param>
        /// <returns>
        ///     The correlation coefficient matrix of the variables.
        /// </returns>
        public static NDarray corrcoef(this NDarray x, NDarray y = null, bool? rowvar = true)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.corrcoef(x.CupyNDarray, y?.CupyNDarray, rowvar));
            }
            else
            {
                return new NDarray(np.corrcoef(x.NumpyNDarray, y?.NumpyNDarray, rowvar));
            }
        }

        /// <summary>
        ///     Cross-correlation of two 1-dimensional sequences.<br></br>
        ///     This function computes the correlation as generally defined in signal
        ///     processing texts:
        ///     with a and v sequences being zero-padded where necessary and conj being
        ///     the conjugate.<br></br>
        ///     Notes
        ///     The definition of correlation above is not unique and sometimes correlation
        ///     may be defined differently.<br></br>
        ///     Another common definition is:
        ///     which is related to c_{av}[k] by c'_{av}[k] = c_{av}[-k].
        /// </summary>
        /// <param name="v">
        ///     Input sequences.
        /// </param>
        /// <param name="a">
        ///     Input sequences.
        /// </param>
        /// <param name="mode">
        ///     Refer to the convolve docstring.<br></br>
        ///     Note that the default
        ///     is ‘valid’, unlike convolve, which uses ‘full’.
        /// </param>
        /// <returns>
        ///     Discrete cross-correlation of a and v.
        /// </returns>
        public static NDarray correlate(this NDarray v, NDarray a, string mode = "valid")
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.correlate(v.CupyNDarray, a.CupyNDarray, mode));
            }
            else
            {
                return new NDarray(np.correlate(v.NumpyNDarray, a.NumpyNDarray, mode));
            }
        }

        /// <summary>
        ///     Estimate a covariance matrix, given data and weights.<br></br>
        ///     Covariance indicates the level to which two variables vary together.<br></br>
        ///     If we examine N-dimensional samples, ,
        ///     then the covariance matrix element  is the covariance of
        ///     and . The element  is the variance
        ///     of .
        ///     See the notes for an outline of the algorithm.<br></br>
        ///     Notes
        ///     Assume that the observations are in the columns of the observation
        ///     array m and let f = fweights and a = aweights for brevity.<br></br>
        ///     The
        ///     steps to compute the weighted covariance are as follows:
        ///     Note that when a == 1, the normalization factor
        ///     v1 / (v1**2 - ddof * v2) goes over to 1 / (cp.sum(f) - ddof)
        ///     as it should.
        /// </summary>
        /// <param name="m">
        ///     A 1-D or 2-D array containing multiple variables and observations.<br></br>
        ///     Each row of m represents a variable, and each column a single
        ///     observation of all those variables.<br></br>
        ///     Also see rowvar below.
        /// </param>
        /// <param name="y">
        ///     An additional set of variables and observations.<br></br>
        ///     y has the same form
        ///     as that of m.
        /// </param>
        /// <param name="rowvar">
        ///     If rowvar is True (default), then each row represents a
        ///     variable, with observations in the columns.<br></br>
        ///     Otherwise, the relationship
        ///     is transposed: each column represents a variable, while the rows
        ///     contain observations.
        /// </param>
        /// <param name="bias">
        ///     Default normalization (False) is by (N - 1), where N is the
        ///     number of observations given (unbiased estimate).<br></br>
        ///     If bias is True,
        ///     then normalization is by N.<br></br>
        ///     These values can be overridden by using
        ///     the keyword ddof in Cupy versions &gt;= 1.5.
        /// </param>
        /// <param name="ddof">
        ///     If not None the default value implied by bias is overridden.<br></br>
        ///     Note that ddof=1 will return the unbiased estimate, even if both
        ///     fweights and aweights are specified, and ddof=0 will return
        ///     the simple average.<br></br>
        ///     See the notes for the details.<br></br>
        ///     The default value
        ///     is None.
        /// </param>
        /// <param name="fweights">
        ///     1-D array of integer frequency weights; the number of times each
        ///     observation vector should be repeated.
        /// </param>
        /// <param name="aweights">
        ///     1-D array of observation vector weights.<br></br>
        ///     These relative weights are
        ///     typically large for observations considered “important” and smaller for
        ///     observations considered less “important”. If ddof=0 the array of
        ///     weights can be used to assign probabilities to observation vectors.
        /// </param>
        /// <returns>
        ///     The covariance matrix of the variables.
        /// </returns>
        public static NDarray cov(this NDarray m, NDarray y = null, bool? rowvar = true, bool? bias = false,
            int? ddof = null, NDarray fweights = null, NDarray aweights = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.cov(m.CupyNDarray, y?.CupyNDarray, rowvar, bias, ddof, fweights?.CupyNDarray,
                    aweights?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.cov(m.NumpyNDarray, y?.NumpyNDarray, rowvar, bias, ddof, fweights?.NumpyNDarray,
                    aweights?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Compute the histogram of a set of data.<br></br>
        ///     Notes
        ///     All but the last (righthand-most) bin is half-open.<br></br>
        ///     In other words,
        ///     if bins is:
        ///     then the first bin is [1, 2) (including 1, but excluding 2) and
        ///     the second [2, 3).<br></br>
        ///     The last bin, however, is [3, 4], which
        ///     includes 4.
        /// </summary>
        /// <param name="a">
        ///     Input data.<br></br>
        ///     The histogram is computed over the flattened array.
        /// </param>
        /// <param name="bins">
        ///     If bins is an int, it defines the number of equal-width
        ///     bins in the given range (10, by default).<br></br>
        ///     If bins is a
        ///     sequence, it defines a monotonically increasing array of bin edges,
        ///     including the rightmost edge, allowing for non-uniform bin widths.<br></br>
        ///     If bins is a string, it defines the method used to calculate the
        ///     optimal bin width, as defined by histogram_bin_edges.
        /// </param>
        /// <param name="range">
        ///     The lower and upper range of the bins.<br></br>
        ///     If not provided, range
        ///     is simply (a.min(), a.max()).<br></br>
        ///     Values outside the range are
        ///     ignored.<br></br>
        ///     The first element of the range must be less than or
        ///     equal to the second.<br></br>
        ///     range affects the automatic bin
        ///     computation as well.<br></br>
        ///     While bin width is computed to be optimal
        ///     based on the actual data within range, the bin count will fill
        ///     the entire range including portions containing no data.
        /// </param>
        /// <param name="normed">
        ///     This is equivalent to the density argument, but produces incorrect
        ///     results for unequal bin widths.<br></br>
        ///     It should not be used.
        /// </param>
        /// <param name="weights">
        ///     An array of weights, of the same shape as a.<br></br>
        ///     Each value in
        ///     a only contributes its associated weight towards the bin count
        ///     (instead of 1).<br></br>
        ///     If density is True, the weights are
        ///     normalized, so that the integral of the density over the range
        ///     remains 1.
        /// </param>
        /// <param name="density">
        ///     If False, the result will contain the number of samples in
        ///     each bin.<br></br>
        ///     If True, the result is the value of the
        ///     probability density function at the bin, normalized such that
        ///     the integral over the range is 1.<br></br>
        ///     Note that the sum of the
        ///     histogram values will not be equal to 1 unless bins of unity
        ///     width are chosen; it is not a probability mass function.<br></br>
        ///     Overrides the normed keyword if given.
        /// </param>
        /// <returns>
        ///     A tuple of:
        ///     hist
        ///     The values of the histogram. See density and weights for a
        ///     description of the possible semantics.
        ///     bin_edges
        ///     Return the bin edges (length(hist)+1).
        /// </returns>
        public static (NDarray, NDarray) histogram(this NDarray a, int? bins = null, (float, float)? range = null,
            bool? normed = null, NDarray weights = null, bool? density = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                var ret = cp.histogram(a.CupyNDarray, bins, range, normed, weights?.CupyNDarray, density);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2));
            }
            else
            {
                var ret = np.histogram(a.NumpyNDarray, bins, range, normed, weights?.NumpyNDarray, density);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2));
            }
        }

        /// <summary>
        ///     Compute the histogram of a set of data.<br></br>
        ///     Notes
        ///     All but the last (righthand-most) bin is half-open.<br></br>
        ///     In other words,
        ///     if bins is:
        ///     then the first bin is [1, 2) (including 1, but excluding 2) and
        ///     the second [2, 3).<br></br>
        ///     The last bin, however, is [3, 4], which
        ///     includes 4.
        /// </summary>
        /// <param name="a">
        ///     Input data.<br></br>
        ///     The histogram is computed over the flattened array.
        /// </param>
        /// <param name="bins">
        ///     If bins is an int, it defines the number of equal-width
        ///     bins in the given range (10, by default).<br></br>
        ///     If bins is a
        ///     sequence, it defines a monotonically increasing array of bin edges,
        ///     including the rightmost edge, allowing for non-uniform bin widths.<br></br>
        ///     If bins is a string, it defines the method used to calculate the
        ///     optimal bin width, as defined by histogram_bin_edges.
        /// </param>
        /// <param name="range">
        ///     The lower and upper range of the bins.<br></br>
        ///     If not provided, range
        ///     is simply (a.min(), a.max()).<br></br>
        ///     Values outside the range are
        ///     ignored.<br></br>
        ///     The first element of the range must be less than or
        ///     equal to the second.<br></br>
        ///     range affects the automatic bin
        ///     computation as well.<br></br>
        ///     While bin width is computed to be optimal
        ///     based on the actual data within range, the bin count will fill
        ///     the entire range including portions containing no data.
        /// </param>
        /// <param name="normed">
        ///     This is equivalent to the density argument, but produces incorrect
        ///     results for unequal bin widths.<br></br>
        ///     It should not be used.
        /// </param>
        /// <param name="weights">
        ///     An array of weights, of the same shape as a.<br></br>
        ///     Each value in
        ///     a only contributes its associated weight towards the bin count
        ///     (instead of 1).<br></br>
        ///     If density is True, the weights are
        ///     normalized, so that the integral of the density over the range
        ///     remains 1.
        /// </param>
        /// <param name="density">
        ///     If False, the result will contain the number of samples in
        ///     each bin.<br></br>
        ///     If True, the result is the value of the
        ///     probability density function at the bin, normalized such that
        ///     the integral over the range is 1.<br></br>
        ///     Note that the sum of the
        ///     histogram values will not be equal to 1 unless bins of unity
        ///     width are chosen; it is not a probability mass function.<br></br>
        ///     Overrides the normed keyword if given.
        /// </param>
        /// <returns>
        ///     A tuple of:
        ///     hist
        ///     The values of the histogram. See density and weights for a
        ///     description of the possible semantics.
        ///     bin_edges
        ///     Return the bin edges (length(hist)+1).
        /// </returns>
        public static (NDarray, NDarray) histogram(this NDarray a, NDarray bins = null, (float, float)? range = null,
            bool? normed = null, NDarray weights = null, bool? density = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                var ret = cp.histogram(a.CupyNDarray, bins.CupyNDarray, range, normed, weights?.CupyNDarray, density);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2));
            }
            else
            {
                var ret = np.histogram(a.NumpyNDarray, bins.NumpyNDarray, range, normed, weights?.NumpyNDarray, density);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2));
            }
        }

        /// <summary>
        ///     Compute the histogram of a set of data.<br></br>
        ///     Notes
        ///     All but the last (righthand-most) bin is half-open.<br></br>
        ///     In other words,
        ///     if bins is:
        ///     then the first bin is [1, 2) (including 1, but excluding 2) and
        ///     the second [2, 3).<br></br>
        ///     The last bin, however, is [3, 4], which
        ///     includes 4.
        /// </summary>
        /// <param name="a">
        ///     Input data.<br></br>
        ///     The histogram is computed over the flattened array.
        /// </param>
        /// <param name="bins">
        ///     If bins is an int, it defines the number of equal-width
        ///     bins in the given range (10, by default).<br></br>
        ///     If bins is a
        ///     sequence, it defines a monotonically increasing array of bin edges,
        ///     including the rightmost edge, allowing for non-uniform bin widths.<br></br>
        ///     If bins is a string, it defines the method used to calculate the
        ///     optimal bin width, as defined by histogram_bin_edges.
        /// </param>
        /// <param name="range">
        ///     The lower and upper range of the bins.<br></br>
        ///     If not provided, range
        ///     is simply (a.min(), a.max()).<br></br>
        ///     Values outside the range are
        ///     ignored.<br></br>
        ///     The first element of the range must be less than or
        ///     equal to the second.<br></br>
        ///     range affects the automatic bin
        ///     computation as well.<br></br>
        ///     While bin width is computed to be optimal
        ///     based on the actual data within range, the bin count will fill
        ///     the entire range including portions containing no data.
        /// </param>
        /// <param name="normed">
        ///     This is equivalent to the density argument, but produces incorrect
        ///     results for unequal bin widths.<br></br>
        ///     It should not be used.
        /// </param>
        /// <param name="weights">
        ///     An array of weights, of the same shape as a.<br></br>
        ///     Each value in
        ///     a only contributes its associated weight towards the bin count
        ///     (instead of 1).<br></br>
        ///     If density is True, the weights are
        ///     normalized, so that the integral of the density over the range
        ///     remains 1.
        /// </param>
        /// <param name="density">
        ///     If False, the result will contain the number of samples in
        ///     each bin.<br></br>
        ///     If True, the result is the value of the
        ///     probability density function at the bin, normalized such that
        ///     the integral over the range is 1.<br></br>
        ///     Note that the sum of the
        ///     histogram values will not be equal to 1 unless bins of unity
        ///     width are chosen; it is not a probability mass function.<br></br>
        ///     Overrides the normed keyword if given.
        /// </param>
        /// <returns>
        ///     A tuple of:
        ///     hist
        ///     The values of the histogram. See density and weights for a
        ///     description of the possible semantics.
        ///     bin_edges
        ///     Return the bin edges (length(hist)+1).
        /// </returns>
        public static (NDarray, NDarray) histogram(this NDarray a, List<string> bins = null,
            (float, float)? range = null, bool? normed = null, NDarray weights = null, bool? density = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                var ret = cp.histogram(a.CupyNDarray, bins, range, normed, weights?.CupyNDarray, density);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2));
            }
            else
            {
                var ret = np.histogram(a.NumpyNDarray, bins, range, normed, weights?.NumpyNDarray, density);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2));
            }
        }

        /// <summary>
        ///     Compute the bi-dimensional histogram of two data samples.<br></br>
        ///     Notes
        ///     When normed is True, then the returned histogram is the sample
        ///     density, defined such that the sum over bins of the product
        ///     bin_value * bin_area is 1.<br></br>
        ///     Please note that the histogram does not follow the Cartesian convention
        ///     where x values are on the abscissa and y values on the ordinate
        ///     axis.<br></br>
        ///     Rather, x is histogrammed along the first dimension of the
        ///     array (vertical), and y along the second dimension of the array
        ///     (horizontal).<br></br>
        ///     This ensures compatibility with histogramdd.
        /// </summary>
        /// <param name="x">
        ///     An array containing the x coordinates of the points to be
        ///     histogrammed.
        /// </param>
        /// <param name="y">
        ///     An array containing the y coordinates of the points to be
        ///     histogrammed.
        /// </param>
        /// <param name="bins">
        ///     The bin specification:
        /// </param>
        /// <param name="range">
        ///     The leftmost and rightmost edges of the bins along each dimension
        ///     (if not specified explicitly in the bins parameters):
        ///     [[xmin, xmax], [ymin, ymax]].<br></br>
        ///     All values outside of this range
        ///     will be considered outliers and not tallied in the histogram.
        /// </param>
        /// <param name="density">
        ///     If False, the default, returns the number of samples in each bin.<br></br>
        ///     If True, returns the probability density function at the bin,
        ///     bin_count / sample_count / bin_area.
        /// </param>
        /// <param name="normed">
        ///     An alias for the density argument that behaves identically.<br></br>
        ///     To avoid
        ///     confusion with the broken normed argument to histogram, density
        ///     should be preferred.
        /// </param>
        /// <param name="weights">
        ///     An array of values w_i weighing each sample (x_i, y_i).<br></br>
        ///     Weights are normalized to 1 if normed is True.<br></br>
        ///     If normed is
        ///     False, the values of the returned histogram are equal to the sum of
        ///     the weights belonging to the samples falling into each bin.
        /// </param>
        /// <returns>
        ///     A tuple of:
        ///     H
        ///     The bi-dimensional histogram of samples x and y. Values in x
        ///     are histogrammed along the first dimension and values in y are
        ///     histogrammed along the second dimension.
        ///     xedges
        ///     The bin edges along the first dimension.
        ///     yedges
        ///     The bin edges along the second dimension.
        /// </returns>
        public static (NDarray, NDarray, NDarray) histogram2d(this NDarray x, NDarray y, int? bins = null,
            (float, float)? range = null, bool? density = null, bool? normed = null, NDarray weights = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                var ret = cp.histogram2d(x.CupyNDarray, y.CupyNDarray, bins, range, density, normed, weights?.CupyNDarray);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2), new NDarray(ret.Item3));
            }
            else
            {
                var ret = np.histogram2d(x.NumpyNDarray, y.NumpyNDarray, bins, range, density, normed, weights?.NumpyNDarray);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2), new NDarray(ret.Item3));
            }
        }

        /// <summary>
        ///     Compute the bi-dimensional histogram of two data samples.<br></br>
        ///     Notes
        ///     When normed is True, then the returned histogram is the sample
        ///     density, defined such that the sum over bins of the product
        ///     bin_value * bin_area is 1.<br></br>
        ///     Please note that the histogram does not follow the Cartesian convention
        ///     where x values are on the abscissa and y values on the ordinate
        ///     axis.<br></br>
        ///     Rather, x is histogrammed along the first dimension of the
        ///     array (vertical), and y along the second dimension of the array
        ///     (horizontal).<br></br>
        ///     This ensures compatibility with histogramdd.
        /// </summary>
        /// <param name="x">
        ///     An array containing the x coordinates of the points to be
        ///     histogrammed.
        /// </param>
        /// <param name="y">
        ///     An array containing the y coordinates of the points to be
        ///     histogrammed.
        /// </param>
        /// <param name="bins">
        ///     The bin specification:
        /// </param>
        /// <param name="range">
        ///     The leftmost and rightmost edges of the bins along each dimension
        ///     (if not specified explicitly in the bins parameters):
        ///     [[xmin, xmax], [ymin, ymax]].<br></br>
        ///     All values outside of this range
        ///     will be considered outliers and not tallied in the histogram.
        /// </param>
        /// <param name="density">
        ///     If False, the default, returns the number of samples in each bin.<br></br>
        ///     If True, returns the probability density function at the bin,
        ///     bin_count / sample_count / bin_area.
        /// </param>
        /// <param name="normed">
        ///     An alias for the density argument that behaves identically.<br></br>
        ///     To avoid
        ///     confusion with the broken normed argument to histogram, density
        ///     should be preferred.
        /// </param>
        /// <param name="weights">
        ///     An array of values w_i weighing each sample (x_i, y_i).<br></br>
        ///     Weights are normalized to 1 if normed is True.<br></br>
        ///     If normed is
        ///     False, the values of the returned histogram are equal to the sum of
        ///     the weights belonging to the samples falling into each bin.
        /// </param>
        /// <returns>
        ///     A tuple of:
        ///     H
        ///     The bi-dimensional histogram of samples x and y. Values in x
        ///     are histogrammed along the first dimension and values in y are
        ///     histogrammed along the second dimension.
        ///     xedges
        ///     The bin edges along the first dimension.
        ///     yedges
        ///     The bin edges along the second dimension.
        /// </returns>
        public static (NDarray, NDarray, NDarray) histogram2d(this NDarray x, NDarray y, NDarray bins = null,
            (float, float)? range = null, bool? density = null, bool? normed = null, NDarray weights = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                var ret = cp.histogram2d(x.CupyNDarray, y.CupyNDarray, bins.CupyNDarray, range, density, normed, weights?.CupyNDarray);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2), new NDarray(ret.Item3));
            }
            else
            {
                var ret = np.histogram2d(x.NumpyNDarray, y.NumpyNDarray, bins.NumpyNDarray, range, density, normed, weights?.NumpyNDarray);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2), new NDarray(ret.Item3));
            }
        }

        /// <summary>
        ///     Compute the bi-dimensional histogram of two data samples.<br></br>
        ///     Notes
        ///     When normed is True, then the returned histogram is the sample
        ///     density, defined such that the sum over bins of the product
        ///     bin_value * bin_area is 1.<br></br>
        ///     Please note that the histogram does not follow the Cartesian convention
        ///     where x values are on the abscissa and y values on the ordinate
        ///     axis.<br></br>
        ///     Rather, x is histogrammed along the first dimension of the
        ///     array (vertical), and y along the second dimension of the array
        ///     (horizontal).<br></br>
        ///     This ensures compatibility with histogramdd.
        /// </summary>
        /// <param name="x">
        ///     An array containing the x coordinates of the points to be
        ///     histogrammed.
        /// </param>
        /// <param name="y">
        ///     An array containing the y coordinates of the points to be
        ///     histogrammed.
        /// </param>
        /// <param name="bins">
        ///     The bin specification:
        /// </param>
        /// <param name="range">
        ///     The leftmost and rightmost edges of the bins along each dimension
        ///     (if not specified explicitly in the bins parameters):
        ///     [[xmin, xmax], [ymin, ymax]].<br></br>
        ///     All values outside of this range
        ///     will be considered outliers and not tallied in the histogram.
        /// </param>
        /// <param name="density">
        ///     If False, the default, returns the number of samples in each bin.<br></br>
        ///     If True, returns the probability density function at the bin,
        ///     bin_count / sample_count / bin_area.
        /// </param>
        /// <param name="normed">
        ///     An alias for the density argument that behaves identically.<br></br>
        ///     To avoid
        ///     confusion with the broken normed argument to histogram, density
        ///     should be preferred.
        /// </param>
        /// <param name="weights">
        ///     An array of values w_i weighing each sample (x_i, y_i).<br></br>
        ///     Weights are normalized to 1 if normed is True.<br></br>
        ///     If normed is
        ///     False, the values of the returned histogram are equal to the sum of
        ///     the weights belonging to the samples falling into each bin.
        /// </param>
        /// <returns>
        ///     A tuple of:
        ///     H
        ///     The bi-dimensional histogram of samples x and y. Values in x
        ///     are histogrammed along the first dimension and values in y are
        ///     histogrammed along the second dimension.
        ///     xedges
        ///     The bin edges along the first dimension.
        ///     yedges
        ///     The bin edges along the second dimension.
        /// </returns>
        public static (NDarray, NDarray, NDarray) histogram2d(this NDarray x, NDarray y, List<string> bins = null,
            (float, float)? range = null, bool? density = null, bool? normed = null, NDarray weights = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                var ret = cp.histogram2d(x.CupyNDarray, y.CupyNDarray, bins, range, density, normed, weights?.CupyNDarray);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2), new NDarray(ret.Item3));
            }
            else
            {
                var ret = np.histogram2d(x.NumpyNDarray, y.NumpyNDarray, bins, range, density, normed, weights?.NumpyNDarray);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2), new NDarray(ret.Item3));
            }
        }

        /// <summary>
        ///     Compute the multidimensional histogram of some data.
        /// </summary>
        /// <param name="sample">
        ///     The data to be histogrammed.<br></br>
        ///     Note the unusual interpretation of sample when an array_like:
        ///     The first form should be preferred.
        /// </param>
        /// <param name="bins">
        ///     The bin specification:
        /// </param>
        /// <param name="range">
        ///     A sequence of length D, each an optional (lower, upper) tuple giving
        ///     the outer bin edges to be used if the edges are not given explicitly in
        ///     bins.<br></br>
        ///     An entry of None in the sequence results in the minimum and maximum
        ///     values being used for the corresponding dimension.<br></br>
        ///     The default, None, is equivalent to passing a tuple of D None values.
        /// </param>
        /// <param name="density">
        ///     If False, the default, returns the number of samples in each bin.<br></br>
        ///     If True, returns the probability density function at the bin,
        ///     bin_count / sample_count / bin_volume.
        /// </param>
        /// <param name="normed">
        ///     An alias for the density argument that behaves identically.<br></br>
        ///     To avoid
        ///     confusion with the broken normed argument to histogram, density
        ///     should be preferred.
        /// </param>
        /// <param name="weights">
        ///     An array of values w_i weighing each sample (x_i, y_i, z_i, …).<br></br>
        ///     Weights are normalized to 1 if normed is True.<br></br>
        ///     If normed is False,
        ///     the values of the returned histogram are equal to the sum of the
        ///     weights belonging to the samples falling into each bin.
        /// </param>
        /// <returns>
        ///     A tuple of:
        ///     H
        ///     The multidimensional histogram of sample x. See normed and weights
        ///     for the different possible semantics.
        ///     edges
        ///     A list of D arrays describing the bin edges for each dimension.
        /// </returns>
        public static (NDarray, NDarray) histogramdd(this NDarray sample, int? bins = null,
            (float, float)? range = null, bool? density = null, bool? normed = null, NDarray weights = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                var ret = cp.histogramdd(sample.CupyNDarray, bins, range, density, normed, weights?.CupyNDarray);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2));
            }
            else
            {
                var ret = np.histogramdd(sample.NumpyNDarray, bins, range, density, normed, weights?.NumpyNDarray);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2));
            }
        }

        /// <summary>
        ///     Compute the multidimensional histogram of some data.
        /// </summary>
        /// <param name="sample">
        ///     The data to be histogrammed.<br></br>
        ///     Note the unusual interpretation of sample when an array_like:
        ///     The first form should be preferred.
        /// </param>
        /// <param name="bins">
        ///     The bin specification:
        /// </param>
        /// <param name="range">
        ///     A sequence of length D, each an optional (lower, upper) tuple giving
        ///     the outer bin edges to be used if the edges are not given explicitly in
        ///     bins.<br></br>
        ///     An entry of None in the sequence results in the minimum and maximum
        ///     values being used for the corresponding dimension.<br></br>
        ///     The default, None, is equivalent to passing a tuple of D None values.
        /// </param>
        /// <param name="density">
        ///     If False, the default, returns the number of samples in each bin.<br></br>
        ///     If True, returns the probability density function at the bin,
        ///     bin_count / sample_count / bin_volume.
        /// </param>
        /// <param name="normed">
        ///     An alias for the density argument that behaves identically.<br></br>
        ///     To avoid
        ///     confusion with the broken normed argument to histogram, density
        ///     should be preferred.
        /// </param>
        /// <param name="weights">
        ///     An array of values w_i weighing each sample (x_i, y_i, z_i, …).<br></br>
        ///     Weights are normalized to 1 if normed is True.<br></br>
        ///     If normed is False,
        ///     the values of the returned histogram are equal to the sum of the
        ///     weights belonging to the samples falling into each bin.
        /// </param>
        /// <returns>
        ///     A tuple of:
        ///     H
        ///     The multidimensional histogram of sample x. See normed and weights
        ///     for the different possible semantics.
        ///     edges
        ///     A list of D arrays describing the bin edges for each dimension.
        /// </returns>
        public static (NDarray, NDarray) histogramdd(this NDarray sample, NDarray bins = null,
            (float, float)? range = null, bool? density = null, bool? normed = null, NDarray weights = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                var ret = cp.histogramdd(sample.CupyNDarray, bins.CupyNDarray, range, density, normed, weights?.CupyNDarray);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2));
            }
            else
            {
                var ret = np.histogramdd(sample.NumpyNDarray, bins.NumpyNDarray, range, density, normed, weights?.NumpyNDarray);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2));
            }
        }

        /// <summary>
        ///     Compute the multidimensional histogram of some data.
        /// </summary>
        /// <param name="sample">
        ///     The data to be histogrammed.<br></br>
        ///     Note the unusual interpretation of sample when an array_like:
        ///     The first form should be preferred.
        /// </param>
        /// <param name="bins">
        ///     The bin specification:
        /// </param>
        /// <param name="range">
        ///     A sequence of length D, each an optional (lower, upper) tuple giving
        ///     the outer bin edges to be used if the edges are not given explicitly in
        ///     bins.<br></br>
        ///     An entry of None in the sequence results in the minimum and maximum
        ///     values being used for the corresponding dimension.<br></br>
        ///     The default, None, is equivalent to passing a tuple of D None values.
        /// </param>
        /// <param name="density">
        ///     If False, the default, returns the number of samples in each bin.<br></br>
        ///     If True, returns the probability density function at the bin,
        ///     bin_count / sample_count / bin_volume.
        /// </param>
        /// <param name="normed">
        ///     An alias for the density argument that behaves identically.<br></br>
        ///     To avoid
        ///     confusion with the broken normed argument to histogram, density
        ///     should be preferred.
        /// </param>
        /// <param name="weights">
        ///     An array of values w_i weighing each sample (x_i, y_i, z_i, …).<br></br>
        ///     Weights are normalized to 1 if normed is True.<br></br>
        ///     If normed is False,
        ///     the values of the returned histogram are equal to the sum of the
        ///     weights belonging to the samples falling into each bin.
        /// </param>
        /// <returns>
        ///     A tuple of:
        ///     H
        ///     The multidimensional histogram of sample x. See normed and weights
        ///     for the different possible semantics.
        ///     edges
        ///     A list of D arrays describing the bin edges for each dimension.
        /// </returns>
        public static (NDarray, NDarray) histogramdd(this NDarray sample, List<string> bins = null,
            (float, float)? range = null, bool? density = null, bool? normed = null, NDarray weights = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                var ret = cp.histogramdd(sample.CupyNDarray, bins, range, density, normed, weights?.CupyNDarray);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2));
            }
            else
            {
                var ret = np.histogramdd(sample.NumpyNDarray, bins, range, density, normed, weights?.NumpyNDarray);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2));
            }
        }

        /// <summary>
        ///     Count number of occurrences of each value in array of non-negative ints.<br></br>
        ///     The number of bins (of size 1) is one larger than the largest value in
        ///     x.<br></br>
        ///     If minlength is specified, there will be at least this number
        ///     of bins in the output array (though it will be longer if necessary,
        ///     depending on the contents of x).<br></br>
        ///     Each bin gives the number of occurrences of its index value in x.<br></br>
        ///     If weights is specified the input array is weighted by it, i.e.<br></br>
        ///     if a
        ///     value n is found at position i, out[n] += weight[i] instead
        ///     of out[n] += 1.
        /// </summary>
        /// <param name="x">
        ///     Input array.
        /// </param>
        /// <param name="weights">
        ///     Weights, array of the same shape as x.
        /// </param>
        /// <param name="minlength">
        ///     A minimum number of bins for the output array.
        /// </param>
        /// <returns>
        ///     The result of binning the input array.<br></br>
        ///     The length of out is equal to cp.amax(x)+1.
        /// </returns>
        public static NDarray bincount(this NDarray x, NDarray weights = null, int? minlength = 0)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.bincount(x.CupyNDarray, weights?.CupyNDarray, minlength));
            }
            else
            {
                return new NDarray(np.bincount(x.NumpyNDarray, weights?.NumpyNDarray, minlength));
            }
        }

        /// <summary>
        ///     Function to calculate only the edges of the bins used by the histogram function.<br></br>
        ///     Notes
        ///     The methods to estimate the optimal number of bins are well founded
        ///     in literature, and are inspired by the choices R provides for
        ///     histogram visualisation.<br></br>
        ///     Note that having the number of bins
        ///     proportional to  is asymptotically optimal, which is
        ///     why it appears in most estimators.<br></br>
        ///     These are simply plug-in methods
        ///     that give good starting points for number of bins.<br></br>
        ///     In the equations
        ///     below,  is the binwidth and  is the number of
        ///     bins.<br></br>
        ///     All estimators that compute bin counts are recast to bin width
        ///     using the ptp of the data.<br></br>
        ///     The final bin count is obtained from
        ///     cp.round(cp.ceil(range / h)).
        /// </summary>
        /// <param name="a">
        ///     Input data.<br></br>
        ///     The histogram is computed over the flattened array.
        /// </param>
        /// <param name="bins">
        ///     If bins is an int, it defines the number of equal-width
        ///     bins in the given range (10, by default).<br></br>
        ///     If bins is a
        ///     sequence, it defines the bin edges, including the rightmost
        ///     edge, allowing for non-uniform bin widths.<br></br>
        ///     If bins is a string from the list below, histogram_bin_edges will use
        ///     the method chosen to calculate the optimal bin width and
        ///     consequently the number of bins (see Notes for more detail on
        ///     the estimators) from the data that falls within the requested
        ///     range.<br></br>
        ///     While the bin width will be optimal for the actual data
        ///     in the range, the number of bins will be computed to fill the
        ///     entire range, including the empty portions.<br></br>
        ///     For visualisation,
        ///     using the ‘auto’ option is suggested.<br></br>
        ///     Weighted data is not
        ///     supported for automated bin size selection.
        /// </param>
        /// <param name="range">
        ///     The lower and upper range of the bins.<br></br>
        ///     If not provided, range
        ///     is simply (a.min(), a.max()).<br></br>
        ///     Values outside the range are
        ///     ignored.<br></br>
        ///     The first element of the range must be less than or
        ///     equal to the second.<br></br>
        ///     range affects the automatic bin
        ///     computation as well.<br></br>
        ///     While bin width is computed to be optimal
        ///     based on the actual data within range, the bin count will fill
        ///     the entire range including portions containing no data.
        /// </param>
        /// <param name="weights">
        ///     An array of weights, of the same shape as a.<br></br>
        ///     Each value in
        ///     a only contributes its associated weight towards the bin count
        ///     (instead of 1).<br></br>
        ///     This is currently not used by any of the bin estimators,
        ///     but may be in the future.
        /// </param>
        /// <returns>
        ///     The edges to pass into histogram
        /// </returns>
        public static NDarray<float> histogram_bin_edges(this NDarray a, int? bins = null, (float, float)? range = null,
            NDarray weights = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray<float>(cp.histogram_bin_edges(a.CupyNDarray, bins, range, weights?.CupyNDarray));
            }
            else
            {
                return new NDarray<float>(np.histogram_bin_edges(a.NumpyNDarray, bins, range, weights?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Function to calculate only the edges of the bins used by the histogram function.<br></br>
        ///     Notes
        ///     The methods to estimate the optimal number of bins are well founded
        ///     in literature, and are inspired by the choices R provides for
        ///     histogram visualisation.<br></br>
        ///     Note that having the number of bins
        ///     proportional to  is asymptotically optimal, which is
        ///     why it appears in most estimators.<br></br>
        ///     These are simply plug-in methods
        ///     that give good starting points for number of bins.<br></br>
        ///     In the equations
        ///     below,  is the binwidth and  is the number of
        ///     bins.<br></br>
        ///     All estimators that compute bin counts are recast to bin width
        ///     using the ptp of the data.<br></br>
        ///     The final bin count is obtained from
        ///     cp.round(cp.ceil(range / h)).
        /// </summary>
        /// <param name="a">
        ///     Input data.<br></br>
        ///     The histogram is computed over the flattened array.
        /// </param>
        /// <param name="bins">
        ///     If bins is an int, it defines the number of equal-width
        ///     bins in the given range (10, by default).<br></br>
        ///     If bins is a
        ///     sequence, it defines the bin edges, including the rightmost
        ///     edge, allowing for non-uniform bin widths.<br></br>
        ///     If bins is a string from the list below, histogram_bin_edges will use
        ///     the method chosen to calculate the optimal bin width and
        ///     consequently the number of bins (see Notes for more detail on
        ///     the estimators) from the data that falls within the requested
        ///     range.<br></br>
        ///     While the bin width will be optimal for the actual data
        ///     in the range, the number of bins will be computed to fill the
        ///     entire range, including the empty portions.<br></br>
        ///     For visualisation,
        ///     using the ‘auto’ option is suggested.<br></br>
        ///     Weighted data is not
        ///     supported for automated bin size selection.
        /// </param>
        /// <param name="range">
        ///     The lower and upper range of the bins.<br></br>
        ///     If not provided, range
        ///     is simply (a.min(), a.max()).<br></br>
        ///     Values outside the range are
        ///     ignored.<br></br>
        ///     The first element of the range must be less than or
        ///     equal to the second.<br></br>
        ///     range affects the automatic bin
        ///     computation as well.<br></br>
        ///     While bin width is computed to be optimal
        ///     based on the actual data within range, the bin count will fill
        ///     the entire range including portions containing no data.
        /// </param>
        /// <param name="weights">
        ///     An array of weights, of the same shape as a.<br></br>
        ///     Each value in
        ///     a only contributes its associated weight towards the bin count
        ///     (instead of 1).<br></br>
        ///     This is currently not used by any of the bin estimators,
        ///     but may be in the future.
        /// </param>
        /// <returns>
        ///     The edges to pass into histogram
        /// </returns>
        public static NDarray<float> histogram_bin_edges(this NDarray a, NDarray bins = null,
            (float, float)? range = null, NDarray weights = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray<float>(cp.histogram_bin_edges(a.CupyNDarray, bins.CupyNDarray, range, weights?.CupyNDarray));
            }
            else
            {
                return new NDarray<float>(np.histogram_bin_edges(a.NumpyNDarray, bins.NumpyNDarray, range, weights?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Function to calculate only the edges of the bins used by the histogram function.<br></br>
        ///     Notes
        ///     The methods to estimate the optimal number of bins are well founded
        ///     in literature, and are inspired by the choices R provides for
        ///     histogram visualisation.<br></br>
        ///     Note that having the number of bins
        ///     proportional to  is asymptotically optimal, which is
        ///     why it appears in most estimators.<br></br>
        ///     These are simply plug-in methods
        ///     that give good starting points for number of bins.<br></br>
        ///     In the equations
        ///     below,  is the binwidth and  is the number of
        ///     bins.<br></br>
        ///     All estimators that compute bin counts are recast to bin width
        ///     using the ptp of the data.<br></br>
        ///     The final bin count is obtained from
        ///     cp.round(cp.ceil(range / h)).
        /// </summary>
        /// <param name="a">
        ///     Input data.<br></br>
        ///     The histogram is computed over the flattened array.
        /// </param>
        /// <param name="bins">
        ///     If bins is an int, it defines the number of equal-width
        ///     bins in the given range (10, by default).<br></br>
        ///     If bins is a
        ///     sequence, it defines the bin edges, including the rightmost
        ///     edge, allowing for non-uniform bin widths.<br></br>
        ///     If bins is a string from the list below, histogram_bin_edges will use
        ///     the method chosen to calculate the optimal bin width and
        ///     consequently the number of bins (see Notes for more detail on
        ///     the estimators) from the data that falls within the requested
        ///     range.<br></br>
        ///     While the bin width will be optimal for the actual data
        ///     in the range, the number of bins will be computed to fill the
        ///     entire range, including the empty portions.<br></br>
        ///     For visualisation,
        ///     using the ‘auto’ option is suggested.<br></br>
        ///     Weighted data is not
        ///     supported for automated bin size selection.
        /// </param>
        /// <param name="range">
        ///     The lower and upper range of the bins.<br></br>
        ///     If not provided, range
        ///     is simply (a.min(), a.max()).<br></br>
        ///     Values outside the range are
        ///     ignored.<br></br>
        ///     The first element of the range must be less than or
        ///     equal to the second.<br></br>
        ///     range affects the automatic bin
        ///     computation as well.<br></br>
        ///     While bin width is computed to be optimal
        ///     based on the actual data within range, the bin count will fill
        ///     the entire range including portions containing no data.
        /// </param>
        /// <param name="weights">
        ///     An array of weights, of the same shape as a.<br></br>
        ///     Each value in
        ///     a only contributes its associated weight towards the bin count
        ///     (instead of 1).<br></br>
        ///     This is currently not used by any of the bin estimators,
        ///     but may be in the future.
        /// </param>
        /// <returns>
        ///     The edges to pass into histogram
        /// </returns>
        public static NDarray<float> histogram_bin_edges(this NDarray a, List<string> bins = null,
            (float, float)? range = null, NDarray weights = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray<float>(cp.histogram_bin_edges(a.CupyNDarray, bins, range, weights?.CupyNDarray));
            }
            else
            {
                return new NDarray<float>(np.histogram_bin_edges(a.NumpyNDarray, bins, range, weights?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Return the indices of the bins to which each value in input array belongs.<br></br>
        ///     If values in x are beyond the bounds of bins, 0 or len(bins) is
        ///     returned as appropriate.<br></br>
        ///     Notes
        ///     If values in x are such that they fall outside the bin range,
        ///     attempting to index bins with the indices that digitize returns
        ///     will result in an IndexError.<br></br>
        ///     cp.digitize is  implemented in terms of cp.searchsorted.<br></br>
        ///     This means
        ///     that a binary search is used to bin the values, which scales much better
        ///     for larger number of bins than the previous linear search.<br></br>
        ///     It also removes
        ///     the requirement for the input array to be 1-dimensional.<br></br>
        ///     For monotonically _increasing_ bins, the following are equivalent:
        ///     Note that as the order of the arguments are reversed, the side must be too.<br></br>
        ///     The searchsorted call is marginally faster, as it does not do any
        ///     monotonicity checks.<br></br>
        ///     Perhaps more importantly, it supports all dtypes.
        /// </summary>
        /// <param name="x">
        ///     Input array to be binned.<br></br>
        ///     Prior to Cupy 1.10.0, this array had to
        ///     be 1-dimensional, but can now have any shape.
        /// </param>
        /// <param name="bins">
        ///     Array of bins.<br></br>
        ///     It has to be 1-dimensional and monotonic.
        /// </param>
        /// <param name="right">
        ///     Indicating whether the intervals include the right or the left bin
        ///     edge.<br></br>
        ///     Default behavior is (right==False) indicating that the interval
        ///     does not include the right edge.<br></br>
        ///     The left bin end is open in this
        ///     case, i.e., bins[i-1] &lt;= x &lt; bins[i] is the default behavior for
        ///     monotonically increasing bins.
        /// </param>
        /// <returns>
        ///     Output array of indices, of same shape as x.
        /// </returns>
        public static NDarray digitize(this NDarray x, NDarray bins, bool? right = false)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.digitize(x.CupyNDarray, bins.CupyNDarray, right));
            }
            else
            {
                return new NDarray(np.digitize(x.NumpyNDarray, bins.NumpyNDarray, right));
            }
        }
    }
}
