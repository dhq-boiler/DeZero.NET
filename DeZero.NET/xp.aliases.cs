﻿using Cupy;
using Numpy;
using System.Numerics;

namespace DeZero.NET
{
    public static partial class xp
    {
        /// <summary>
        ///     Gives a new shape to an array without changing its data.
        ///     Notes
        ///     It is not always possible to change the shape of an array without
        ///     copying the data. If you want an error to be raised when the data is copied,
        ///     you should assign the new shape to the shape attribute of the array:
        ///     The order keyword gives the index ordering both for fetching the values
        ///     from a, and then placing the values into the output array.
        ///     For example, let’s say you have an array:
        ///     You can think of reshaping as first raveling the array (using the given
        ///     index order), then inserting the elements from the raveled array into the
        ///     new array using the same kind of index ordering as was used for the
        ///     raveling.
        /// </summary>
        /// <param name="a">The array to reshape</param>
        /// <param name="newshape">
        ///     The new shape should be compatible with the original shape. If
        ///     an integer, then the result will be a 1-D array of that length.
        ///     One shape dimension can be -1. In this case, the value is
        ///     inferred from the length of the array and remaining dimensions.
        /// </param>
        /// <returns>
        ///     This will be a new view object if possible; otherwise, it will
        ///     be a copy.  Note there is no guarantee of the memory layout (C- or
        ///     Fortran- contiguous) of the returned array.
        /// </returns>
        public static NDarray reshape(NDarray a, params int[] newshape)
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                return new NDarray(cp.reshape(a.CupyNDarray, newshape));
            }
            else
            {
                return new NDarray(np.reshape(a.NumpyNDarray, newshape));
            }
        }

        /// <summary>
        ///     Calculate the absolute value element-wise.
        ///     cp.abs is a shorthand for this function.
        /// </summary>
        /// <param name="x">
        ///     Input array.
        /// </param>
        /// <param name="@out">
        ///     A location into which the result is stored. If provided, it must have
        ///     a shape that the inputs broadcast to. If not provided or None,
        ///     a freshly-allocated array is returned. A tuple (possible only as a
        ///     keyword argument) must have length equal to the number of outputs.
        /// </param>
        /// <param name="@where">
        ///     Values of True indicate to calculate the ufunc at that position, values
        ///     of False indicate to leave the value in the output alone.
        /// </param>
        /// <returns>
        ///     An ndarray containing the absolute value of
        ///     each element in x.  For complex input, a + ib, the
        ///     absolute value is .
        ///     This is a scalar if x is a scalar.
        /// </returns>
        public static NDarray abs(NDarray x, NDarray @out = null, NDarray where = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                return new NDarray(cp.abs(x.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.abs(x.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Return the minimum of an array or minimum along an axis.
        ///     Notes
        ///     NaN values are propagated, that is if at least one item is NaN, the
        ///     corresponding min value will be NaN as well. To ignore NaN values
        ///     (MATLAB behavior), please use nanmin.
        ///     Don’t use amin for element-wise comparison of 2 arrays; when
        ///     a.shape[0] is 2, minimum(a[0], a[1]) is faster than
        ///     amin(a, axis=0).
        /// </summary>
        /// <param name="a">
        ///     Input data.
        /// </param>
        /// <param name="axis">
        ///     Axis or axes along which to operate.  By default, flattened input is
        ///     used.
        ///     If this is a tuple of ints, the minimum is selected over multiple axes,
        ///     instead of a single axis or all the axes as before.
        /// </param>
        /// <param name="@out">
        ///     Alternative output array in which to place the result.  Must
        ///     be of the same shape and buffer length as the expected output.
        ///     See doc.ufuncs (Section “Output arguments”) for more details.
        /// </param>
        /// <param name="keepdims">
        ///     If this is set to True, the axes which are reduced are left
        ///     in the result as dimensions with size one. With this option,
        ///     the result will broadcast correctly against the input array.
        ///     If the default value is passed, then keepdims will not be
        ///     passed through to the amin method of sub-classes of
        ///     ndarray, however any non-default value will be.  If the
        ///     sub-class’ method does not implement keepdims any
        ///     exceptions will be raised.
        /// </param>
        /// <param name="initial">
        ///     The maximum value of an output element. Must be present to allow
        ///     computation on empty slice. See reduce for details.
        /// </param>
        /// <returns>
        ///     Minimum of a. If axis is None, the result is a scalar value.
        ///     If axis is given, the result is an array of dimension
        ///     a.ndim - 1.
        /// </returns>
        public static NDarray min(NDarray a, int[] axis = null, NDarray @out = null, bool? keepdims = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                return new NDarray(cp.min(a.CupyNDarray, axis, @out?.CupyNDarray, keepdims));
            }
            else
            {
                return new NDarray(np.min(a.NumpyNDarray, axis, @out?.NumpyNDarray, keepdims));
            }
        }

        /// <summary>
        ///     Return the maximum of an array or maximum along an axis.
        ///     Notes
        ///     NaN values are propagated, that is if at least one item is NaN, the
        ///     corresponding max value will be NaN as well. To ignore NaN values
        ///     (MATLAB behavior), please use nanmax.
        ///     Don’t use amax for element-wise comparison of 2 arrays; when
        ///     a.shape[0] is 2, maximum(a[0], a[1]) is faster than
        ///     amax(a, axis=0).
        /// </summary>
        /// <param name="a">
        ///     Input data.
        /// </param>
        /// <param name="axis">
        ///     Axis or axes along which to operate.  By default, flattened input is
        ///     used.
        ///     If this is a tuple of ints, the maximum is selected over multiple axes,
        ///     instead of a single axis or all the axes as before.
        /// </param>
        /// <param name="@out">
        ///     Alternative output array in which to place the result.  Must
        ///     be of the same shape and buffer length as the expected output.
        ///     See doc.ufuncs (Section “Output arguments”) for more details.
        /// </param>
        /// <param name="keepdims">
        ///     If this is set to True, the axes which are reduced are left
        ///     in the result as dimensions with size one. With this option,
        ///     the result will broadcast correctly against the input array.
        ///     If the default value is passed, then keepdims will not be
        ///     passed through to the amax method of sub-classes of
        ///     ndarray, however any non-default value will be.  If the
        ///     sub-class’ method does not implement keepdims any
        ///     exceptions will be raised.
        /// </param>
        /// <param name="initial">
        ///     The minimum value of an output element. Must be present to allow
        ///     computation on empty slice. See reduce for details.
        /// </param>
        /// <returns>
        ///     Maximum of a. If axis is None, the result is a scalar value.
        ///     If axis is given, the result is an array of dimension
        ///     a.ndim - 1.
        /// </returns>
        public static NDarray max(NDarray a, int[] axis = null, NDarray @out = null, bool? keepdims = null,
            ValueType initial = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                return new NDarray(cp.max(a.CupyNDarray, axis, @out?.CupyNDarray, keepdims, initial));
            }
            else
            {
                return new NDarray(np.max(a.NumpyNDarray, axis, @out?.NumpyNDarray, keepdims, initial));
            }
        }


        /// <summary>
        ///     Return a new array of given shape and type, filled with zeros.
        /// </summary>
        /// <param name="shape">
        ///     Shape of the new array, e.g., (2, 3) or 2.
        /// </param>
        /// <param name="dtype">
        ///     The desired data-type for the array, e.g., Cupy.int8.  Default is
        ///     Cupy.float64.
        /// </param>
        /// <param name="order">
        ///     Whether to store multi-dimensional data in row-major
        ///     (C-style) or column-major (Fortran-style) order in
        ///     memory.
        /// </param>
        /// <returns>
        ///     Array of zeros with the given shape, dtype, and order.
        /// </returns>
        public static NDarray zeros(params int[] shape)
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                return new NDarray(cp.zeros(shape));
            }
            else
            {
                return new NDarray(np.zeros(shape));
            }
        }


        /// <summary>
        ///     Return a new array of given shape and type, filled with ones.
        /// </summary>
        /// <param name="shape">
        ///     Shape of the new array, e.g., (2, 3) or 2.
        /// </param>
        /// <param name="dtype">
        ///     The desired data-type for the array, e.g., Cupy.int8.  Default is
        ///     Cupy.float64.
        /// </param>
        /// <param name="order">
        ///     Whether to store multi-dimensional data in row-major
        ///     (C-style) or column-major (Fortran-style) order in
        ///     memory.
        /// </param>
        /// <returns>
        ///     Array of ones with the given shape, dtype, and order.
        /// </returns>
        public static NDarray ones(params int[] shape)
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                return new NDarray(cp.ones(shape));
            }
            else
            {
                return new NDarray(np.ones(shape));
            }
        }

        /// <summary>
        ///     Create an array.
        ///     <param name="shape">
        ///         The shape of the empty ndarray
        ///     </param>
        ///     <returns>
        ///         An array object satisfying the specified requirements.
        ///     </returns>
        public static NDarray empty(params int[] shape)
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                return new NDarray(cp.empty(shape));
            }
            else
            {
                return new NDarray(np.empty(shape));
            }
        }

        public static NDarray imag(Complex val)
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                return new NDarray(cp.imag(val));
            }
            else
            {
                return new NDarray(np.imag(val));
            }
        }

        public static NDarray real(Complex val)
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                return new NDarray(cp.real(val));
            }
            else
            {
                return new NDarray(np.real(val));
            }
        }
    }
}
