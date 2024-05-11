﻿using Cupy;
using Numpy;

namespace DeZero.NET
{
    public static partial class xp
    {
        /// <summary>
        ///     Test whether all array elements along a given axis evaluate to True.<br></br>
        ///     Notes
        ///     Not a Number (NaN), positive infinity and negative infinity
        ///     evaluate to True because these are not equal to zero.
        /// </summary>
        /// <param name="a">
        ///     Input array or object that can be converted to an array.
        /// </param>
        /// <param name="axis">
        ///     Axis or axes along which a logical AND reduction is performed.<br></br>
        ///     The default (axis = None) is to perform a logical AND over all
        ///     the dimensions of the input array.<br></br>
        ///     axis may be negative, in
        ///     which case it counts from the last to the first axis.<br></br>
        ///     If this is a tuple of ints, a reduction is performed on multiple
        ///     axes, instead of a single axis or all the axes as before.
        /// </param>
        /// <param name="out">
        ///     Alternate output array in which to place the result.<br></br>
        ///     It must have the same shape as the expected output and its
        ///     type is preserved (e.g., if dtype(out) is float, the result
        ///     will consist of 0.0’s and 1.0’s).<br></br>
        ///     See doc.ufuncs (Section
        ///     “Output arguments”) for more details.
        /// </param>
        /// <param name="keepdims">
        ///     If this is set to True, the axes which are reduced are left
        ///     in the result as dimensions with size one.<br></br>
        ///     With this option,
        ///     the result will broadcast correctly against the input array.<br></br>
        ///     If the default value is passed, then keepdims will not be
        ///     passed through to the all method of sub-classes of
        ///     ndarray, however any non-default value will be.<br></br>
        ///     If the
        ///     sub-class’ method does not implement keepdims any
        ///     exceptions will be raised.
        /// </param>
        /// <returns>
        ///     A new boolean or array is returned unless out is specified,
        ///     in which case a reference to out is returned.
        /// </returns>
        public static NDarray<bool> all(this NDarray a, Axis axis, NDarray @out = null, bool? keepdims = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray<bool>(cp.all(a.CupyNDarray, axis.CupyAxis, @out?.CupyNDarray, keepdims));
            }
            else
            {
                return new NDarray<bool>(np.all(a.NumpyNDarray, axis.NumpyAxis, @out?.NumpyNDarray, keepdims));
            }
        }

        /// <summary>
        ///     Test whether all array elements along a given axis evaluate to True.<br></br>
        ///     Notes
        ///     Not a Number (NaN), positive infinity and negative infinity
        ///     evaluate to True because these are not equal to zero.
        /// </summary>
        /// <param name="a">
        ///     Input array or object that can be converted to an array.
        /// </param>
        /// <returns>
        ///     A new boolean or array is returned unless out is specified,
        ///     in which case a reference to out is returned.
        /// </returns>
        public static bool all(this NDarray a)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return cp.all(a.CupyNDarray);
            }
            else
            {
                return np.all(a.NumpyNDarray);
            }
        }

        /// <summary>
        ///     Test whether any array element along a given axis evaluates to True.<br></br>
        ///     Returns single boolean unless axis is not None
        ///     Notes
        ///     Not a Number (NaN), positive infinity and negative infinity evaluate
        ///     to True because these are not equal to zero.
        /// </summary>
        /// <param name="a">
        ///     Input array or object that can be converted to an array.
        /// </param>
        /// <param name="axis">
        ///     Axis or axes along which a logical OR reduction is performed.<br></br>
        ///     The default (axis = None) is to perform a logical OR over all
        ///     the dimensions of the input array.<br></br>
        ///     axis may be negative, in
        ///     which case it counts from the last to the first axis.<br></br>
        ///     If this is a tuple of ints, a reduction is performed on multiple
        ///     axes, instead of a single axis or all the axes as before.
        /// </param>
        /// <param name="out">
        ///     Alternate output array in which to place the result.<br></br>
        ///     It must have
        ///     the same shape as the expected output and its type is preserved
        ///     (e.g., if it is of type float, then it will remain so, returning
        ///     1.0 for True and 0.0 for False, regardless of the type of a).<br></br>
        ///     See doc.ufuncs (Section “Output arguments”) for details.
        /// </param>
        /// <param name="keepdims">
        ///     If this is set to True, the axes which are reduced are left
        ///     in the result as dimensions with size one.<br></br>
        ///     With this option,
        ///     the result will broadcast correctly against the input array.<br></br>
        ///     If the default value is passed, then keepdims will not be
        ///     passed through to the any method of sub-classes of
        ///     ndarray, however any non-default value will be.<br></br>
        ///     If the
        ///     sub-class’ method does not implement keepdims any
        ///     exceptions will be raised.
        /// </param>
        /// <returns>
        ///     A new boolean or ndarray is returned unless out is specified,
        ///     in which case a reference to out is returned.
        /// </returns>
        public static NDarray<bool> any(this NDarray a, Axis axis, NDarray @out = null, bool? keepdims = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray<bool>(cp.any(a.CupyNDarray, axis.CupyAxis, @out?.CupyNDarray, keepdims));
            }
            else
            {
                return new NDarray<bool>(np.any(a.NumpyNDarray, axis.NumpyAxis, @out?.NumpyNDarray, keepdims));
            }
        }

        /// <summary>
        ///     Test whether any array element along a given axis evaluates to True.<br></br>
        ///     Returns single boolean unless axis is not None
        ///     Notes
        ///     Not a Number (NaN), positive infinity and negative infinity evaluate
        ///     to True because these are not equal to zero.
        /// </summary>
        /// <param name="a">
        ///     Input array or object that can be converted to an array.
        /// </param>
        /// <returns>
        ///     A new boolean or ndarray is returned unless out is specified,
        ///     in which case a reference to out is returned.
        /// </returns>
        public static bool any(this NDarray a)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return cp.any(a.CupyNDarray);
            }
            else
            {
                return np.any(a.NumpyNDarray);
            }
        }

        /// <summary>
        ///     Test element-wise for finiteness (not infinity or not Not a Number).<br></br>
        ///     The result is returned as a boolean array.<br></br>
        ///     Notes
        ///     Not a Number, positive infinity and negative infinity are considered
        ///     to be non-finite.<br></br>
        ///     Cupy uses the IEEE Standard for Binary Floating-Point for Arithmetic
        ///     (IEEE 754).<br></br>
        ///     This means that Not a Number is not equivalent to infinity.<br></br>
        ///     Also that positive infinity is not equivalent to negative infinity.<br></br>
        ///     But
        ///     infinity is equivalent to positive infinity.<br></br>
        ///     Errors result if the
        ///     second argument is also supplied when x is a scalar input, or if
        ///     first and second arguments have different shapes.
        /// </summary>
        /// <param name="x">
        ///     Input values.
        /// </param>
        /// <param name="out">
        ///     A location into which the result is stored.<br></br>
        ///     If provided, it must have
        ///     a shape that the inputs broadcast to.<br></br>
        ///     If not provided or None,
        ///     a freshly-allocated array is returned.<br></br>
        ///     A tuple (possible only as a
        ///     keyword argument) must have length equal to the number of outputs.
        /// </param>
        /// <param name="where">
        ///     Values of True indicate to calculate the ufunc at that position, values
        ///     of False indicate to leave the value in the output alone.
        /// </param>
        /// <returns>
        ///     True where x is not positive infinity, negative infinity,
        ///     or NaN; false otherwise.<br></br>
        ///     This is a scalar if x is a scalar.
        /// </returns>
        public static NDarray isfinite(this NDarray x, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.isfinite(x.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.isfinite(x.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Test element-wise for positive or negative infinity.<br></br>
        ///     Returns a boolean array of the same shape as x, True where x ==
        ///     +/-inf, otherwise False.<br></br>
        ///     Notes
        ///     Cupy uses the IEEE Standard for Binary Floating-Point for Arithmetic
        ///     (IEEE 754).<br></br>
        ///     Errors result if the second argument is supplied when the first
        ///     argument is a scalar, or if the first and second arguments have
        ///     different shapes.
        /// </summary>
        /// <param name="x">
        ///     Input values
        /// </param>
        /// <param name="out">
        ///     A location into which the result is stored.<br></br>
        ///     If provided, it must have
        ///     a shape that the inputs broadcast to.<br></br>
        ///     If not provided or None,
        ///     a freshly-allocated array is returned.<br></br>
        ///     A tuple (possible only as a
        ///     keyword argument) must have length equal to the number of outputs.
        /// </param>
        /// <param name="where">
        ///     Values of True indicate to calculate the ufunc at that position, values
        ///     of False indicate to leave the value in the output alone.
        /// </param>
        /// <returns>
        ///     True where x is positive or negative infinity, false otherwise.<br></br>
        ///     This is a scalar if x is a scalar.
        /// </returns>
        public static NDarray<bool> isinf(this NDarray x, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray<bool>(cp.isinf(x.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            }
            else
            {
                return new NDarray<bool>(np.isinf(x.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Test element-wise for NaN and return result as a boolean array.<br></br>
        ///     Notes
        ///     Cupy uses the IEEE Standard for Binary Floating-Point for Arithmetic
        ///     (IEEE 754).<br></br>
        ///     This means that Not a Number is not equivalent to infinity.
        /// </summary>
        /// <param name="x">
        ///     Input array.
        /// </param>
        /// <param name="out">
        ///     A location into which the result is stored.<br></br>
        ///     If provided, it must have
        ///     a shape that the inputs broadcast to.<br></br>
        ///     If not provided or None,
        ///     a freshly-allocated array is returned.<br></br>
        ///     A tuple (possible only as a
        ///     keyword argument) must have length equal to the number of outputs.
        /// </param>
        /// <param name="where">
        ///     Values of True indicate to calculate the ufunc at that position, values
        ///     of False indicate to leave the value in the output alone.
        /// </param>
        /// <returns>
        ///     True where x is NaN, false otherwise.<br></br>
        ///     This is a scalar if x is a scalar.
        /// </returns>
        public static NDarray isnan(this NDarray x, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.isnan(x.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.isnan(x.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Test element-wise for NaT (not a time) and return result as a boolean array.
        /// </summary>
        /// <param name="x">
        ///     Input array with datetime or timedelta data type.
        /// </param>
        /// <param name="out">
        ///     A location into which the result is stored.<br></br>
        ///     If provided, it must have
        ///     a shape that the inputs broadcast to.<br></br>
        ///     If not provided or None,
        ///     a freshly-allocated array is returned.<br></br>
        ///     A tuple (possible only as a
        ///     keyword argument) must have length equal to the number of outputs.
        /// </param>
        /// <param name="where">
        ///     Values of True indicate to calculate the ufunc at that position, values
        ///     of False indicate to leave the value in the output alone.
        /// </param>
        /// <returns>
        ///     True where x is NaT, false otherwise.<br></br>
        ///     This is a scalar if x is a scalar.
        /// </returns>
        public static NDarray isnat(this NDarray x, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.isnat(x.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.isnat(x.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Test element-wise for negative infinity, return result as bool array.<br></br>
        ///     Notes
        ///     Cupy uses the IEEE Standard for Binary Floating-Point for Arithmetic
        ///     (IEEE 754).<br></br>
        ///     Errors result if the second argument is also supplied when x is a scalar
        ///     input, if first and second arguments have different shapes, or if the
        ///     first argument has complex values.
        /// </summary>
        /// <param name="x">
        ///     The input array.
        /// </param>
        /// <param name="out">
        ///     A boolean array with the same shape and type as x to store the
        ///     result.
        /// </param>
        /// <returns>
        ///     A boolean array with the same dimensions as the input.<br></br>
        ///     If second argument is not supplied then a Cupy boolean array is
        ///     returned with values True where the corresponding element of the
        ///     input is negative infinity and values False where the element of
        ///     the input is not negative infinity.<br></br>
        ///     If a second argument is supplied the result is stored there.<br></br>
        ///     If the
        ///     type of that array is a numeric type the result is represented as
        ///     zeros and ones, if the type is boolean then as False and True.<br></br>
        ///     The
        ///     return value out is then a reference to that array.
        /// </returns>
        public static NDarray isneginf(this NDarray x, NDarray @out = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.isneginf(x.CupyNDarray, @out?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.isneginf(x.NumpyNDarray, @out?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Test element-wise for positive infinity, return result as bool array.<br></br>
        ///     Notes
        ///     Cupy uses the IEEE Standard for Binary Floating-Point for Arithmetic
        ///     (IEEE 754).<br></br>
        ///     Errors result if the second argument is also supplied when x is a scalar
        ///     input, if first and second arguments have different shapes, or if the
        ///     first argument has complex values
        /// </summary>
        /// <param name="x">
        ///     The input array.
        /// </param>
        /// <param name="y">
        ///     A boolean array with the same shape as x to store the result.
        /// </param>
        /// <returns>
        ///     A boolean array with the same dimensions as the input.<br></br>
        ///     If second argument is not supplied then a boolean array is returned
        ///     with values True where the corresponding element of the input is
        ///     positive infinity and values False where the element of the input is
        ///     not positive infinity.<br></br>
        ///     If a second argument is supplied the result is stored there.<br></br>
        ///     If the
        ///     type of that array is a numeric type the result is represented as zeros
        ///     and ones, if the type is boolean then as False and True.<br></br>
        ///     The return value out is then a reference to that array.
        /// </returns>
        public static NDarray isposinf(this NDarray x, NDarray y = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.isposinf(x.CupyNDarray, y?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.isposinf(x.NumpyNDarray, y?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Returns a bool array, where True if input element is complex.<br></br>
        ///     What is tested is whether the input has a non-zero imaginary part, not if
        ///     the input type is complex.
        /// </summary>
        /// <param name="x">
        ///     Input array.
        /// </param>
        /// <returns>
        ///     Output array.
        /// </returns>
        public static NDarray iscomplex(this NDarray x)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.iscomplex(x.CupyNDarray));
            }
            else
            {
                return new NDarray(np.iscomplex(x.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Check for a complex type or an array of complex numbers.<br></br>
        ///     The type of the input is checked, not the value.<br></br>
        ///     Even if the input
        ///     has an imaginary part equal to zero, iscomplexobj evaluates to True.
        /// </summary>
        /// <param name="x">
        ///     The input can be of any type and shape.
        /// </param>
        /// <returns>
        ///     The return value, True if x is of a complex type or has at least
        ///     one complex element.
        /// </returns>
        public static bool iscomplexobj(object x)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return cp.iscomplexobj(x);
            }
            else
            {
                return np.iscomplexobj(x);
            }
        }

        /// <summary>
        ///     Returns True if the array is Fortran contiguous but not C contiguous.<br></br>
        ///     This function is obsolete and, because of changes due to relaxed stride
        ///     checking, its return value for the same array may differ for versions
        ///     of Cupy &gt;= 1.10.0 and previous versions.<br></br>
        ///     If you only want to check if an
        ///     array is Fortran contiguous use a.flags.f_contiguous instead.
        /// </summary>
        /// <param name="a">
        ///     Input array.
        /// </param>
        public static bool isfortran(this NDarray a)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return cp.isfortran(a.CupyNDarray);
            }
            else
            {
                return np.isfortran(a.NumpyNDarray);
            }
        }

        /// <summary>
        ///     Returns a bool array, where True if input element is real.<br></br>
        ///     If element has complex type with zero complex part, the return value
        ///     for that element is True.
        /// </summary>
        /// <param name="x">
        ///     Input array.
        /// </param>
        /// <returns>
        ///     Boolean array of same shape as x.
        /// </returns>
        public static NDarray isreal(this NDarray x)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.isreal(x.CupyNDarray));
            }
            else
            {
                return new NDarray(np.isreal(x.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Return True if x is a not complex type or an array of complex numbers.<br></br>
        ///     The type of the input is checked, not the value.<br></br>
        ///     So even if the input
        ///     has an imaginary part equal to zero, isrealobj evaluates to False
        ///     if the data type is complex.
        /// </summary>
        /// <param name="x">
        ///     The input can be of any type and shape.
        /// </param>
        /// <returns>
        ///     The return value, False if x is of a complex type.
        /// </returns>
        public static bool isrealobj(object x)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return cp.isrealobj(x);
            }
            else
            {
                return np.isrealobj(x);
            }
        }

        /// <summary>
        ///     Returns True if the type of num is a scalar type.<br></br>
        ///     Notes
        ///     In almost all cases cp.ndim(x) == 0 should be used instead of this
        ///     function, as that will also return true for 0d arrays.<br></br>
        ///     This is how
        ///     Cupy overloads functions in the style of the dx arguments to gradient
        ///     and the bins argument to histogram.<br></br>
        ///     Some key differences:
        /// </summary>
        /// <param name="num">
        ///     Input argument, can be of any type and shape.
        /// </param>
        /// <returns>
        ///     True if num is a scalar type, False if it is not.
        /// </returns>
        public static bool isscalar(object num)
        {
            var ndarray = num as NDarray;
            if (Gpu.Available && Gpu.Use)
            {
                try
                {
                    ndarray?.Push(ArrayMode.cp);
                    return cp.isscalar(ndarray?.CupyNDarray ?? num);
                }
                finally
                {
                    ndarray?.Pop();
                }
            }
            else
            {
                try
                {
                    ndarray?.Push(ArrayMode.np);
                    return np.isscalar(ndarray?.NumpyNDarray ?? num);
                }
                finally
                {
                    ndarray?.Pop();
                }
            }
        }

        /// <summary>
        ///     Compute the truth value of x1 AND x2 element-wise.
        /// </summary>
        /// <param name="x2">
        ///     Input arrays.<br></br>
        ///     x1 and x2 must be of the same shape.
        /// </param>
        /// <param name="x1">
        ///     Input arrays.<br></br>
        ///     x1 and x2 must be of the same shape.
        /// </param>
        /// <param name="out">
        ///     A location into which the result is stored.<br></br>
        ///     If provided, it must have
        ///     a shape that the inputs broadcast to.<br></br>
        ///     If not provided or None,
        ///     a freshly-allocated array is returned.<br></br>
        ///     A tuple (possible only as a
        ///     keyword argument) must have length equal to the number of outputs.
        /// </param>
        /// <param name="where">
        ///     Values of True indicate to calculate the ufunc at that position, values
        ///     of False indicate to leave the value in the output alone.
        /// </param>
        /// <returns>
        ///     Boolean result with the same shape as x1 and x2 of the logical
        ///     AND operation on corresponding elements of x1 and x2.
        ///     This is a scalar if both x1 and x2 are scalars.
        /// </returns>
        public static NDarray logical_and(this NDarray x2, NDarray x1, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(
                    cp.logical_and(x2.CupyNDarray, x1.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.logical_and(x2.NumpyNDarray, x1.NumpyNDarray, @out?.NumpyNDarray,
                    where?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Compute the truth value of x1 OR x2 element-wise.
        /// </summary>
        /// <param name="x2">
        ///     Logical OR is applied to the elements of x1 and x2.
        ///     They have to be of the same shape.
        /// </param>
        /// <param name="x1">
        ///     Logical OR is applied to the elements of x1 and x2.
        ///     They have to be of the same shape.
        /// </param>
        /// <param name="out">
        ///     A location into which the result is stored.<br></br>
        ///     If provided, it must have
        ///     a shape that the inputs broadcast to.<br></br>
        ///     If not provided or None,
        ///     a freshly-allocated array is returned.<br></br>
        ///     A tuple (possible only as a
        ///     keyword argument) must have length equal to the number of outputs.
        /// </param>
        /// <param name="where">
        ///     Values of True indicate to calculate the ufunc at that position, values
        ///     of False indicate to leave the value in the output alone.
        /// </param>
        /// <returns>
        ///     Boolean result with the same shape as x1 and x2 of the logical
        ///     OR operation on elements of x1 and x2.
        ///     This is a scalar if both x1 and x2 are scalars.
        /// </returns>
        public static NDarray logical_or(this NDarray x2, NDarray x1, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(
                    cp.logical_or(x2.CupyNDarray, x1.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.logical_or(x2.NumpyNDarray, x1.NumpyNDarray, @out?.NumpyNDarray,
                    where?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Compute the truth value of NOT x element-wise.
        /// </summary>
        /// <param name="x">
        ///     Logical NOT is applied to the elements of x.
        /// </param>
        /// <param name="out">
        ///     A location into which the result is stored.<br></br>
        ///     If provided, it must have
        ///     a shape that the inputs broadcast to.<br></br>
        ///     If not provided or None,
        ///     a freshly-allocated array is returned.<br></br>
        ///     A tuple (possible only as a
        ///     keyword argument) must have length equal to the number of outputs.
        /// </param>
        /// <param name="where">
        ///     Values of True indicate to calculate the ufunc at that position, values
        ///     of False indicate to leave the value in the output alone.
        /// </param>
        /// <returns>
        ///     Boolean result with the same shape as x of the NOT operation
        ///     on elements of x.<br></br>
        ///     This is a scalar if x is a scalar.
        /// </returns>
        public static NDarray<bool> logical_not(this NDarray x, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray<bool>(
                    cp.logical_not(x.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            }
            else
            {
                return new NDarray<bool>(np.logical_not(x.NumpyNDarray, @out?.NumpyNDarray,
                    where?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Compute the truth value of x1 XOR x2, element-wise.
        /// </summary>
        /// <param name="x2">
        ///     Logical XOR is applied to the elements of x1 and x2.  They must
        ///     be broadcastable to the same shape.
        /// </param>
        /// <param name="x1">
        ///     Logical XOR is applied to the elements of x1 and x2.  They must
        ///     be broadcastable to the same shape.
        /// </param>
        /// <param name="out">
        ///     A location into which the result is stored.<br></br>
        ///     If provided, it must have
        ///     a shape that the inputs broadcast to.<br></br>
        ///     If not provided or None,
        ///     a freshly-allocated array is returned.<br></br>
        ///     A tuple (possible only as a
        ///     keyword argument) must have length equal to the number of outputs.
        /// </param>
        /// <param name="where">
        ///     Values of True indicate to calculate the ufunc at that position, values
        ///     of False indicate to leave the value in the output alone.
        /// </param>
        /// <returns>
        ///     Boolean result of the logical XOR operation applied to the elements
        ///     of x1 and x2; the shape is determined by whether or not
        ///     broadcasting of one or both arrays was required.<br></br>
        ///     This is a scalar if both x1 and x2 are scalars.
        /// </returns>
        public static NDarray<bool> logical_xor(this NDarray x2, NDarray x1, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray<bool>(
                    cp.logical_xor(x2.CupyNDarray, x1.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            }
            else
            {
                return new NDarray<bool>(np.logical_xor(x2.NumpyNDarray, x1.NumpyNDarray, @out?.NumpyNDarray,
                    where?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Returns True if two arrays are element-wise equal within a tolerance.<br></br>
        ///     The tolerance values are positive, typically very small numbers.<br></br>
        ///     The
        ///     relative difference (rtol * abs(b)) and the absolute difference
        ///     atol are added together to compare against the absolute difference
        ///     between a and b.<br></br>
        ///     If either array contains one or more NaNs, False is returned.<br></br>
        ///     Infs are treated as equal if they are in the same place and of the same
        ///     sign in both arrays.<br></br>
        ///     Notes
        ///     If the following equation is element-wise True, then allclose returns
        ///     True.<br></br>
        ///     The above equation is not symmetric in a and b, so that
        ///     allclose(a, b) might be different from allclose(b, a) in
        ///     some rare cases.<br></br>
        ///     The comparison of a and b uses standard broadcasting, which
        ///     means that a and b need not have the same shape in order for
        ///     allclose(a, b) to evaluate to True.<br></br>
        ///     The same is true for
        ///     equal but not array_equal.
        /// </summary>
        /// <param name="b">
        ///     Input arrays to compare.
        /// </param>
        /// <param name="a">
        ///     Input arrays to compare.
        /// </param>
        /// <param name="rtol">
        ///     The relative tolerance parameter (see Notes).
        /// </param>
        /// <param name="atol">
        ///     The absolute tolerance parameter (see Notes).
        /// </param>
        /// <param name="equal_nan">
        ///     Whether to compare NaN’s as equal.<br></br>
        ///     If True, NaN’s in a will be
        ///     considered equal to NaN’s in b in the output array.
        /// </param>
        /// <returns>
        ///     Returns True if the two arrays are equal within the given
        ///     tolerance; False otherwise.
        /// </returns>
        public static bool allclose(this NDarray b, NDarray a, float rtol = 1e-05f, float atol = 1e-08f,
            bool equal_nan = false)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return cp.allclose(b.ToCupyNDarray, a.ToCupyNDarray, rtol, atol, equal_nan);
            }
            else
            {
                return np.allclose(b.ToNumpyNDarray, a.ToNumpyNDarray, rtol, atol, equal_nan);
            }
        }

        /// <summary>
        ///     Returns a boolean array where two arrays are element-wise equal within a
        ///     tolerance.<br></br>
        ///     The tolerance values are positive, typically very small numbers.<br></br>
        ///     The
        ///     relative difference (rtol * abs(b)) and the absolute difference
        ///     atol are added together to compare against the absolute difference
        ///     between a and b.<br></br>
        ///     Notes
        ///     For finite values, isclose uses the following equation to test whether
        ///     two floating point values are equivalent.<br></br>
        ///     Unlike the built-in math.isclose, the above equation is not symmetric
        ///     in a and b – it assumes b is the reference value – so that
        ///     isclose(a, b) might be different from isclose(b, a).<br></br>
        ///     Furthermore,
        ///     the default value of atol is not zero, and is used to determine what
        ///     small values should be considered close to zero.<br></br>
        ///     The default value is
        ///     appropriate for expected values of order unity: if the expected values
        ///     are significantly smaller than one, it can result in false positives.<br></br>
        ///     atol should be carefully selected for the use case at hand.<br></br>
        ///     A zero value
        ///     for atol will result in False if either a or b is zero.
        /// </summary>
        /// <param name="b">
        ///     Input arrays to compare.
        /// </param>
        /// <param name="a">
        ///     Input arrays to compare.
        /// </param>
        /// <param name="rtol">
        ///     The relative tolerance parameter (see Notes).
        /// </param>
        /// <param name="atol">
        ///     The absolute tolerance parameter (see Notes).
        /// </param>
        /// <param name="equal_nan">
        ///     Whether to compare NaN’s as equal.<br></br>
        ///     If True, NaN’s in a will be
        ///     considered equal to NaN’s in b in the output array.
        /// </param>
        /// <returns>
        ///     Returns a boolean array of where a and b are equal within the
        ///     given tolerance.<br></br>
        ///     If both a and b are scalars, returns a single
        ///     boolean value.
        /// </returns>
        public static NDarray isclose(this NDarray b, NDarray a, float rtol = 1e-05f, float atol = 1e-08f,
            bool equal_nan = false)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.isclose(b.ToCupyNDarray, a.ToCupyNDarray, rtol, atol, equal_nan));
            }
            else
            {
                return new NDarray(np.isclose(b.ToNumpyNDarray, a.ToNumpyNDarray, rtol, atol, equal_nan));
            }
        }

        /// <summary>
        ///     True if two arrays have the same shape and elements, False otherwise.
        /// </summary>
        /// <param name="a2">
        ///     Input arrays.
        /// </param>
        /// <param name="a1">
        ///     Input arrays.
        /// </param>
        /// <returns>
        ///     Returns True if the arrays are equal.
        /// </returns>
        public static bool array_equal(this NDarray a2, NDarray a1)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return cp.array_equal(a2.ToCupyNDarray, a1.ToCupyNDarray);
            }
            else
            {
                return np.array_equal(a2.ToNumpyNDarray, a1.ToNumpyNDarray);
            }
        }

        /// <summary>
        ///     Returns True if input arrays are shape consistent and all elements equal.<br></br>
        ///     Shape consistent means they are either the same shape, or one input array
        ///     can be broadcasted to create the same shape as the other one.
        /// </summary>
        /// <param name="a2">
        ///     Input arrays.
        /// </param>
        /// <param name="a1">
        ///     Input arrays.
        /// </param>
        /// <returns>
        ///     True if equivalent, False otherwise.
        /// </returns>
        public static bool array_equiv(this NDarray a2, NDarray a1)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return cp.array_equiv(a2.ToCupyNDarray, a1.ToCupyNDarray);
            }
            else
            {
                return np.array_equiv(a2.ToNumpyNDarray, a1.ToNumpyNDarray);
            }
        }

        /// <summary>
        ///     Return the truth value of (x1 &gt; x2) element-wise.
        /// </summary>
        /// <param name="x2">
        ///     Input arrays.<br></br>
        ///     If x1.shape != x2.shape, they must be
        ///     broadcastable to a common shape (which may be the shape of one or
        ///     the other).
        /// </param>
        /// <param name="x1">
        ///     Input arrays.<br></br>
        ///     If x1.shape != x2.shape, they must be
        ///     broadcastable to a common shape (which may be the shape of one or
        ///     the other).
        /// </param>
        /// <param name="out">
        ///     A location into which the result is stored.<br></br>
        ///     If provided, it must have
        ///     a shape that the inputs broadcast to.<br></br>
        ///     If not provided or None,
        ///     a freshly-allocated array is returned.<br></br>
        ///     A tuple (possible only as a
        ///     keyword argument) must have length equal to the number of outputs.
        /// </param>
        /// <param name="where">
        ///     Values of True indicate to calculate the ufunc at that position, values
        ///     of False indicate to leave the value in the output alone.
        /// </param>
        /// <returns>
        ///     Output array, element-wise comparison of x1 and x2.
        ///     Typically of type bool, unless dtype=object is passed.<br></br>
        ///     This is a scalar if both x1 and x2 are scalars.
        /// </returns>
        public static NDarray greater(this NDarray x2, NDarray x1, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(
                    cp.greater(x2.ToCupyNDarray, x1.ToCupyNDarray, @out?.ToCupyNDarray, where?.ToCupyNDarray));
            }
            else
            {
                return new NDarray(np.greater(x2.ToNumpyNDarray, x1.ToNumpyNDarray, @out?.ToNumpyNDarray,
                    where?.ToNumpyNDarray));
            }
        }

        /// <summary>
        ///     Return the truth value of (x1 &gt;= x2) element-wise.
        /// </summary>
        /// <param name="x2">
        ///     Input arrays.<br></br>
        ///     If x1.shape != x2.shape, they must be
        ///     broadcastable to a common shape (which may be the shape of one or
        ///     the other).
        /// </param>
        /// <param name="x1">
        ///     Input arrays.<br></br>
        ///     If x1.shape != x2.shape, they must be
        ///     broadcastable to a common shape (which may be the shape of one or
        ///     the other).
        /// </param>
        /// <param name="out">
        ///     A location into which the result is stored.<br></br>
        ///     If provided, it must have
        ///     a shape that the inputs broadcast to.<br></br>
        ///     If not provided or None,
        ///     a freshly-allocated array is returned.<br></br>
        ///     A tuple (possible only as a
        ///     keyword argument) must have length equal to the number of outputs.
        /// </param>
        /// <param name="where">
        ///     Values of True indicate to calculate the ufunc at that position, values
        ///     of False indicate to leave the value in the output alone.
        /// </param>
        /// <returns>
        ///     Output array, element-wise comparison of x1 and x2.
        ///     Typically of type bool, unless dtype=object is passed.<br></br>
        ///     This is a scalar if both x1 and x2 are scalars.
        /// </returns>
        public static NDarray<bool> greater_equal(this NDarray x2, NDarray x1, NDarray @out = null,
            NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray<bool>(
                    cp.greater_equal(x2.ToCupyNDarray, x1.ToCupyNDarray, @out?.ToCupyNDarray, where?.ToCupyNDarray));
            }
            else
            {
                return new NDarray<bool>(np.greater_equal(x2.ToNumpyNDarray, x1.ToNumpyNDarray, @out?.ToNumpyNDarray,
                    where?.ToNumpyNDarray));
            }
        }

        /// <summary>
        ///     Return the truth value of (x1 &lt; x2) element-wise.
        /// </summary>
        /// <param name="x2">
        ///     Input arrays.<br></br>
        ///     If x1.shape != x2.shape, they must be
        ///     broadcastable to a common shape (which may be the shape of one or
        ///     the other).
        /// </param>
        /// <param name="x1">
        ///     Input arrays.<br></br>
        ///     If x1.shape != x2.shape, they must be
        ///     broadcastable to a common shape (which may be the shape of one or
        ///     the other).
        /// </param>
        /// <param name="out">
        ///     A location into which the result is stored.<br></br>
        ///     If provided, it must have
        ///     a shape that the inputs broadcast to.<br></br>
        ///     If not provided or None,
        ///     a freshly-allocated array is returned.<br></br>
        ///     A tuple (possible only as a
        ///     keyword argument) must have length equal to the number of outputs.
        /// </param>
        /// <param name="where">
        ///     Values of True indicate to calculate the ufunc at that position, values
        ///     of False indicate to leave the value in the output alone.
        /// </param>
        /// <returns>
        ///     Output array, element-wise comparison of x1 and x2.
        ///     Typically of type bool, unless dtype=object is passed.<br></br>
        ///     This is a scalar if both x1 and x2 are scalars.
        /// </returns>
        public static NDarray less(this NDarray x2, NDarray x1, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(
                    cp.less(x2.ToCupyNDarray, x1.ToCupyNDarray, @out?.ToCupyNDarray, where?.ToCupyNDarray));
            }
            else
            {
                return new NDarray(np.less(x2.ToNumpyNDarray, x1.ToNumpyNDarray, @out?.ToNumpyNDarray,
                    where?.ToNumpyNDarray));
            }
        }

        /// <summary>
        ///     Return the truth value of (x1 =&lt; x2) element-wise.
        /// </summary>
        /// <param name="x2">
        ///     Input arrays.<br></br>
        ///     If x1.shape != x2.shape, they must be
        ///     broadcastable to a common shape (which may be the shape of one or
        ///     the other).
        /// </param>
        /// <param name="x1">
        ///     Input arrays.<br></br>
        ///     If x1.shape != x2.shape, they must be
        ///     broadcastable to a common shape (which may be the shape of one or
        ///     the other).
        /// </param>
        /// <param name="out">
        ///     A location into which the result is stored.<br></br>
        ///     If provided, it must have
        ///     a shape that the inputs broadcast to.<br></br>
        ///     If not provided or None,
        ///     a freshly-allocated array is returned.<br></br>
        ///     A tuple (possible only as a
        ///     keyword argument) must have length equal to the number of outputs.
        /// </param>
        /// <param name="where">
        ///     Values of True indicate to calculate the ufunc at that position, values
        ///     of False indicate to leave the value in the output alone.
        /// </param>
        /// <returns>
        ///     Output array, element-wise comparison of x1 and x2.
        ///     Typically of type bool, unless dtype=object is passed.<br></br>
        ///     This is a scalar if both x1 and x2 are scalars.
        /// </returns>
        public static NDarray less_equal(this NDarray x2, NDarray x1, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(
                    cp.less_equal(x2.ToCupyNDarray, x1.ToCupyNDarray, @out?.ToCupyNDarray, where?.ToCupyNDarray));
            }
            else
            {
                return new NDarray(np.less_equal(x2.ToNumpyNDarray, x1.ToNumpyNDarray, @out?.ToNumpyNDarray,
                    where?.ToNumpyNDarray));
            }
        }

        /// <summary>
        ///     Return (x1 == x2) element-wise.
        /// </summary>
        /// <param name="x2">
        ///     Input arrays of the same shape.
        /// </param>
        /// <param name="x1">
        ///     Input arrays of the same shape.
        /// </param>
        /// <param name="out">
        ///     A location into which the result is stored.<br></br>
        ///     If provided, it must have
        ///     a shape that the inputs broadcast to.<br></br>
        ///     If not provided or None,
        ///     a freshly-allocated array is returned.<br></br>
        ///     A tuple (possible only as a
        ///     keyword argument) must have length equal to the number of outputs.
        /// </param>
        /// <param name="where">
        ///     Values of True indicate to calculate the ufunc at that position, values
        ///     of False indicate to leave the value in the output alone.
        /// </param>
        /// <returns>
        ///     Output array, element-wise comparison of x1 and x2.
        ///     Typically of type bool, unless dtype=object is passed.<br></br>
        ///     This is a scalar if both x1 and x2 are scalars.
        /// </returns>
        public static NDarray equal(this NDarray x2, NDarray x1, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(
                    cp.equal(x2.ToCupyNDarray, x1.ToCupyNDarray, @out?.ToCupyNDarray, where?.ToCupyNDarray));
            }
            else
            {
                return new NDarray(np.equal(x2.ToNumpyNDarray, x1.ToNumpyNDarray, @out?.ToNumpyNDarray,
                    where?.ToNumpyNDarray));
            }
        }

        /// <summary>
        ///     Return (x1 != x2) element-wise.
        /// </summary>
        /// <param name="x2">
        ///     Input arrays.
        /// </param>
        /// <param name="x1">
        ///     Input arrays.
        /// </param>
        /// <param name="out">
        ///     A location into which the result is stored.<br></br>
        ///     If provided, it must have
        ///     a shape that the inputs broadcast to.<br></br>
        ///     If not provided or None,
        ///     a freshly-allocated array is returned.<br></br>
        ///     A tuple (possible only as a
        ///     keyword argument) must have length equal to the number of outputs.
        /// </param>
        /// <param name="where">
        ///     Values of True indicate to calculate the ufunc at that position, values
        ///     of False indicate to leave the value in the output alone.
        /// </param>
        /// <returns>
        ///     Output array, element-wise comparison of x1 and x2.
        ///     Typically of type bool, unless dtype=object is passed.<br></br>
        ///     This is a scalar if both x1 and x2 are scalars.
        /// </returns>
        public static NDarray not_equal(this NDarray x2, NDarray x1, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(
                    cp.not_equal(x2.ToCupyNDarray, x1.ToCupyNDarray, @out?.ToCupyNDarray, where?.ToCupyNDarray));
            }
            else
            {
                return new NDarray(np.not_equal(x2.ToNumpyNDarray, x1.ToNumpyNDarray, @out?.ToNumpyNDarray,
                    where?.ToNumpyNDarray));
            }
        }
    }
}
