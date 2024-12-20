﻿using Cupy;
using Numpy;
using Python.Runtime;

namespace DeZero.NET
{
    public static partial class xp
    {
        /// <summary>
        ///     Return the gradient of an N-dimensional array.<br></br>
        ///     The gradient is computed using second order accurate central differences
        ///     in the interior points and either first or second order accurate one-sides
        ///     (forward or backwards) differences at the boundaries.<br></br>
        ///     The returned gradient hence has the same shape as the input array.<br></br>
        ///     Notes
        ///     Assuming that  (i.e.,  has at least 3 continuous
        ///     derivatives) and let  be a non-homogeneous stepsize, we
        ///     minimize the “consistency error”  between the true gradient
        ///     and its estimate from a linear combination of the neighboring grid-points:
        ///     By substituting  and
        ///     with their Taylor series expansion, this translates into solving
        ///     the following the linear system:
        ///     The resulting approximation of  is the following:
        ///     It is worth noting that if
        ///     (i.e., data are evenly spaced)
        ///     we find the standard second order approximation:
        ///     With a similar procedure the forward/backward approximations used for
        ///     boundaries can be derived.<br></br>
        ///     References
        /// </summary>
        /// <param name="f">
        ///     An N-dimensional array containing samples of a scalar function.
        /// </param>
        /// <param name="varargs">
        ///     Spacing between f values.<br></br>
        ///     Default unitary spacing for all dimensions.<br></br>
        ///     Spacing can be specified using:
        ///     If axis is given, the number of varargs must equal the number of axis.<br></br>
        ///     Default: 1.
        /// </param>
        /// <param name="edge_order">
        ///     Gradient is calculated using N-th order accurate differences
        ///     at the boundaries.<br></br>
        ///     Default: 1.
        /// </param>
        /// <param name="axis">
        ///     Gradient is calculated only along the given axis or axis
        ///     The default (axis = None) is to calculate the gradient for all the axis
        ///     of the input array.<br></br>
        ///     axis may be negative, in which case it counts from
        ///     the last to the first axis.
        /// </param>
        /// <returns>
        ///     A set of ndarrays (or a single ndarray if there is only one dimension)
        ///     corresponding to the derivatives of f with respect to each dimension.<br></br>
        ///     Each derivative has the same shape as f.
        /// </returns>
        public static NDarray gradient(NDarray f, int? edge_order = null, Axis axis = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.gradient(f.CupyNDarray, edge_order, axis?.CupyAxis));
            }
            else
            {
                return new NDarray(np.gradient(f.NumpyNDarray, edge_order, axis?.NumpyAxis));
            }
        }

        /// <summary>
        ///     Return the gradient of an N-dimensional array.<br></br>
        ///     The gradient is computed using second order accurate central differences
        ///     in the interior points and either first or second order accurate one-sides
        ///     (forward or backwards) differences at the boundaries.<br></br>
        ///     The returned gradient hence has the same shape as the input array.<br></br>
        ///     Notes
        ///     Assuming that  (i.e.,  has at least 3 continuous
        ///     derivatives) and let  be a non-homogeneous stepsize, we
        ///     minimize the “consistency error”  between the true gradient
        ///     and its estimate from a linear combination of the neighboring grid-points:
        ///     By substituting  and
        ///     with their Taylor series expansion, this translates into solving
        ///     the following the linear system:
        ///     The resulting approximation of  is the following:
        ///     It is worth noting that if
        ///     (i.e., data are evenly spaced)
        ///     we find the standard second order approximation:
        ///     With a similar procedure the forward/backward approximations used for
        ///     boundaries can be derived.<br></br>
        ///     References
        /// </summary>
        /// <param name="f">
        ///     An N-dimensional array containing samples of a scalar function.
        /// </param>
        /// <param name="varargs">
        ///     Spacing between f values.<br></br>
        ///     Default unitary spacing for all dimensions.<br></br>
        ///     Spacing can be specified using:
        ///     If axis is given, the number of varargs must equal the number of axis.<br></br>
        ///     Default: 1.
        /// </param>
        /// <param name="edge_order">
        ///     Gradient is calculated using N-th order accurate differences
        ///     at the boundaries.<br></br>
        ///     Default: 1.
        /// </param>
        /// <param name="axis">
        ///     Gradient is calculated only along the given axis or axis
        ///     The default (axis = None) is to calculate the gradient for all the axis
        ///     of the input array.<br></br>
        ///     axis may be negative, in which case it counts from
        ///     the last to the first axis.
        /// </param>
        /// <returns>
        ///     A set of ndarrays (or a single ndarray if there is only one dimension)
        ///     corresponding to the derivatives of f with respect to each dimension.<br></br>
        ///     Each derivative has the same shape as f.
        /// </returns>
        public static NDarray gradient(NDarray f, List<double> varargs, int? edge_order = null, Axis axis = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.gradient(f.CupyNDarray, varargs, edge_order, axis?.CupyAxis));
            }
            else
            {
                return new NDarray(np.gradient(f.NumpyNDarray, varargs, edge_order, axis?.NumpyAxis));
            }
        }

        /// <summary>
        ///     Trigonometric sine, element-wise.<br></br>
        ///     Notes
        ///     The sine is one of the fundamental functions of trigonometry (the
        ///     mathematical study of triangles).<br></br>
        ///     Consider a circle of radius 1
        ///     centered on the origin.<br></br>
        ///     A ray comes in from the  axis, makes
        ///     an angle at the origin (measured counter-clockwise from that axis), and
        ///     departs from the origin.<br></br>
        ///     The  coordinate of the outgoing
        ///     ray’s intersection with the unit circle is the sine of that angle.<br></br>
        ///     It
        ///     ranges from -1 for  to +1 for   The
        ///     function has zeroes where the angle is a multiple of .
        ///     Sines of angles between  and  are negative.<br></br>
        ///     The numerous properties of the sine and related functions are included
        ///     in any standard trigonometry text.
        /// </summary>
        /// <param name="x">
        ///     Angle, in radians ( rad equals 360 degrees).
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
        ///     The sine of each element of x.<br></br>
        ///     This is a scalar if x is a scalar.
        /// </returns>
        public static NDarray sin(this NDarray x, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.sin(x.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.sin(x.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Cosine element-wise.<br></br>
        ///     Notes
        ///     If out is provided, the function writes the result into it,
        ///     and returns a reference to out.<br></br>
        ///     (See Examples)
        ///     References
        ///     M.<br></br>
        ///     Abramowitz and I.<br></br>
        ///     A.<br></br>
        ///     Stegun, Handbook of Mathematical Functions.<br></br>
        ///     New York, NY: Dover, 1972.
        /// </summary>
        /// <param name="x">
        ///     Input array in radians.
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
        ///     The corresponding cosine values.<br></br>
        ///     This is a scalar if x is a scalar.
        /// </returns>
        public static NDarray cos(this NDarray x, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.cos(x.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.cos(x.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Compute tangent element-wise.<br></br>
        ///     Equivalent to cp.sin(x)/cp.cos(x) element-wise.<br></br>
        ///     Notes
        ///     If out is provided, the function writes the result into it,
        ///     and returns a reference to out.<br></br>
        ///     (See Examples)
        ///     References
        ///     M.<br></br>
        ///     Abramowitz and I.<br></br>
        ///     A.<br></br>
        ///     Stegun, Handbook of Mathematical Functions.<br></br>
        ///     New York, NY: Dover, 1972.
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
        ///     The corresponding tangent values.<br></br>
        ///     This is a scalar if x is a scalar.
        /// </returns>
        public static NDarray tan(this NDarray x, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.tan(x.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.tan(x.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Inverse sine, element-wise.<br></br>
        ///     Notes
        ///     arcsin is a multivalued function: for each x there are infinitely
        ///     many numbers z such that .  The convention is to
        ///     return the angle z whose real part lies in [-pi/2, pi/2].<br></br>
        ///     For real-valued input data types, arcsin always returns real output.<br></br>
        ///     For each value that cannot be expressed as a real number or infinity,
        ///     it yields nan and sets the invalid floating point error flag.<br></br>
        ///     For complex-valued input, arcsin is a complex analytic function that
        ///     has, by convention, the branch cuts [-inf, -1] and [1, inf]  and is
        ///     continuous from above on the former and from below on the latter.<br></br>
        ///     The inverse sine is also known as asin or sin^{-1}.
        ///     References
        ///     Abramowitz, M.<br></br>
        ///     and Stegun, I.<br></br>
        ///     A., Handbook of Mathematical Functions,
        ///     10th printing, New York: Dover, 1964, pp.<br></br>
        ///     79ff.<br></br>
        ///     http://www.math.sfu.ca/~cbm/aands/
        /// </summary>
        /// <param name="x">
        ///     y-coordinate on the unit circle.
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
        ///     The inverse sine of each element in x, in radians and in the
        ///     closed interval [-pi/2, pi/2].<br></br>
        ///     This is a scalar if x is a scalar.
        /// </returns>
        public static NDarray arcsin(this NDarray x, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.arcsin(x.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.arcsin(x.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Trigonometric inverse cosine, element-wise.<br></br>
        ///     The inverse of cos so that, if y = cos(x), then x = arccos(y).<br></br>
        ///     Notes
        ///     arccos is a multivalued function: for each x there are infinitely
        ///     many numbers z such that cos(z) = x.<br></br>
        ///     The convention is to return
        ///     the angle z whose real part lies in [0, pi].<br></br>
        ///     For real-valued input data types, arccos always returns real output.<br></br>
        ///     For each value that cannot be expressed as a real number or infinity,
        ///     it yields nan and sets the invalid floating point error flag.<br></br>
        ///     For complex-valued input, arccos is a complex analytic function that
        ///     has branch cuts [-inf, -1] and [1, inf] and is continuous from
        ///     above on the former and from below on the latter.<br></br>
        ///     The inverse cos is also known as acos or cos^-1.
        ///     References
        ///     M.<br></br>
        ///     Abramowitz and I.A.<br></br>
        ///     Stegun, “Handbook of Mathematical Functions”,
        ///     10th printing, 1964, pp.<br></br>
        ///     79. http://www.math.sfu.ca/~cbm/aands/
        /// </summary>
        /// <param name="x">
        ///     x-coordinate on the unit circle.<br></br>
        ///     For real arguments, the domain is [-1, 1].
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
        ///     The angle of the ray intersecting the unit circle at the given
        ///     x-coordinate in radians [0, pi].<br></br>
        ///     This is a scalar if x is a scalar.
        /// </returns>
        public static NDarray arccos(this NDarray x, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.arccos(x.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.arccos(x.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Trigonometric inverse tangent, element-wise.<br></br>
        ///     The inverse of tan, so that if y = tan(x) then x = arctan(y).<br></br>
        ///     Notes
        ///     arctan is a multi-valued function: for each x there are infinitely
        ///     many numbers z such that tan(z) = x.<br></br>
        ///     The convention is to return
        ///     the angle z whose real part lies in [-pi/2, pi/2].<br></br>
        ///     For real-valued input data types, arctan always returns real output.<br></br>
        ///     For each value that cannot be expressed as a real number or infinity,
        ///     it yields nan and sets the invalid floating point error flag.<br></br>
        ///     For complex-valued input, arctan is a complex analytic function that
        ///     has [1j, infj] and [-1j, -infj] as branch cuts, and is continuous
        ///     from the left on the former and from the right on the latter.<br></br>
        ///     The inverse tangent is also known as atan or tan^{-1}.
        ///     References
        ///     Abramowitz, M.<br></br>
        ///     and Stegun, I.<br></br>
        ///     A., Handbook of Mathematical Functions,
        ///     10th printing, New York: Dover, 1964, pp.<br></br>
        ///     79.
        ///     http://www.math.sfu.ca/~cbm/aands/
        /// </summary>
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
        ///     Out has the same shape as x.<br></br>
        ///     Its real part is in
        ///     [-pi/2, pi/2] (arctan(+/-inf) returns +/-pi/2).<br></br>
        ///     This is a scalar if x is a scalar.
        /// </returns>
        public static NDarray arctan(this NDarray x, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.arctan(x.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.arctan(x.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Given the “legs” of a right triangle, return its hypotenuse.<br></br>
        ///     Equivalent to sqrt(x1**2 + x2**2), element-wise.<br></br>
        ///     If x1 or
        ///     x2 is scalar_like (i.e., unambiguously cast-able to a scalar type),
        ///     it is broadcast for use with each element of the other argument.<br></br>
        ///     (See Examples)
        /// </summary>
        /// <param name="x2">
        ///     Leg of the triangle(s).
        /// </param>
        /// <param name="x1">
        ///     Leg of the triangle(s).
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
        ///     The hypotenuse of the triangle(s).<br></br>
        ///     This is a scalar if both x1 and x2 are scalars.
        /// </returns>
        public static NDarray hypot(this NDarray x2, NDarray x1, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.hypot(x2.CupyNDarray, x1.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.hypot(x2.NumpyNDarray, x1.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Element-wise arc tangent of x1/x2 choosing the quadrant correctly.<br></br>
        ///     The quadrant (i.e., branch) is chosen so that arctan2(x1, x2) is
        ///     the signed angle in radians between the ray ending at the origin and
        ///     passing through the point (1,0), and the ray ending at the origin and
        ///     passing through the point (x2, x1).<br></br>
        ///     (Note the role reversal: the
        ///     “y-coordinate” is the first function parameter, the “x-coordinate”
        ///     is the second.)  By IEEE convention, this function is defined for
        ///     x2 = +/-0 and for either or both of x1 and x2 = +/-inf (see
        ///     Notes for specific values).<br></br>
        ///     This function is not defined for complex-valued arguments; for the
        ///     so-called argument of complex values, use angle.<br></br>
        ///     Notes
        ///     arctan2 is identical to the atan2 function of the underlying
        ///     C library.<br></br>
        ///     The following special values are defined in the C
        ///     standard: [1]
        ///     Note that +0 and -0 are distinct floating point numbers, as are +inf
        ///     and -inf.<br></br>
        ///     References
        /// </summary>
        /// <param name="x1">
        ///     y-coordinates.
        /// </param>
        /// <param name="x2">
        ///     x-coordinates.<br></br>
        ///     x2 must be broadcastable to match the shape of
        ///     x1 or vice versa.
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
        ///     Array of angles in radians, in the range [-pi, pi].<br></br>
        ///     This is a scalar if both x1 and x2 are scalars.
        /// </returns>
        public static NDarray arctan2(this NDarray x1, NDarray x2, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.arctan2(x1.CupyNDarray, x2.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.arctan2(x1.NumpyNDarray, x2.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Convert angles from radians to degrees.
        /// </summary>
        /// <param name="x">
        ///     Input array in radians.
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
        ///     The corresponding degree values; if out was supplied this is a
        ///     reference to it.<br></br>
        ///     This is a scalar if x is a scalar.
        /// </returns>
        public static NDarray degrees(this NDarray x, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.degrees(x.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.degrees(x.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Convert angles from degrees to radians.
        /// </summary>
        /// <param name="x">
        ///     Input array in degrees.
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
        ///     The corresponding radian values.<br></br>
        ///     This is a scalar if x is a scalar.
        /// </returns>
        public static NDarray radians(this NDarray x, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.radians(x.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.radians(x.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Unwrap by changing deltas between values to 2*pi complement.<br></br>
        ///     Unwrap radian phase p by changing absolute jumps greater than
        ///     discont to their 2*pi complement along the given axis.<br></br>
        ///     Notes
        ///     If the discontinuity in p is smaller than pi, but larger than
        ///     discont, no unwrapping is done because taking the 2*pi complement
        ///     would only make the discontinuity larger.
        /// </summary>
        /// <param name="p">
        ///     Input array.
        /// </param>
        /// <param name="discont">
        ///     Maximum discontinuity between values, default is pi.
        /// </param>
        /// <param name="axis">
        ///     Axis along which unwrap will operate, default is the last axis.
        /// </param>
        /// <returns>
        ///     Output array.
        /// </returns>
        public static NDarray unwrap(this NDarray p, float? discont = 3.141592653589793f, int? axis = -1)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.unwrap(p.CupyNDarray, discont, axis));
            }
            else
            {
                return new NDarray(np.unwrap(p.NumpyNDarray, discont, axis));
            }
        }

        /// <summary>
        ///     Convert angles from degrees to radians.<br></br>
        ///     Notes
        ///     deg2rad(x) is x * pi / 180.
        /// </summary>
        /// <param name="x">
        ///     Angles in degrees.
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
        ///     The corresponding angle in radians.<br></br>
        ///     This is a scalar if x is a scalar.
        /// </returns>
        public static NDarray deg2rad(this NDarray x, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.deg2rad(x.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.deg2rad(x.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Convert angles from radians to degrees.<br></br>
        ///     Notes
        ///     rad2deg(x) is 180 * x / pi.
        /// </summary>
        /// <param name="x">
        ///     Angle in radians.
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
        ///     The corresponding angle in degrees.<br></br>
        ///     This is a scalar if x is a scalar.
        /// </returns>
        public static NDarray rad2deg(this NDarray x, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.rad2deg(x.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.rad2deg(x.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Hyperbolic sine, element-wise.<br></br>
        ///     Equivalent to 1/2 * (cp.exp(x) - cp.exp(-x)) or
        ///     -1j * cp.sin(1j*x).<br></br>
        ///     Notes
        ///     If out is provided, the function writes the result into it,
        ///     and returns a reference to out.<br></br>
        ///     (See Examples)
        ///     References
        ///     M.<br></br>
        ///     Abramowitz and I.<br></br>
        ///     A.<br></br>
        ///     Stegun, Handbook of Mathematical Functions.<br></br>
        ///     New York, NY: Dover, 1972, pg.<br></br>
        ///     83.
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
        ///     The corresponding hyperbolic sine values.<br></br>
        ///     This is a scalar if x is a scalar.
        /// </returns>
        public static NDarray sinh(this NDarray x, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.sinh(x.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.sinh(x.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Hyperbolic cosine, element-wise.<br></br>
        ///     Equivalent to 1/2 * (cp.exp(x) + cp.exp(-x)) and cp.cos(1j*x).
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
        ///     Output array of same shape as x.<br></br>
        ///     This is a scalar if x is a scalar.
        /// </returns>
        public static NDarray cosh(this NDarray x, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.cosh(x.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.cosh(x.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Compute hyperbolic tangent element-wise.<br></br>
        ///     Equivalent to cp.sinh(x)/cp.cosh(x) or -1j * cp.tan(1j*x).<br></br>
        ///     Notes
        ///     If out is provided, the function writes the result into it,
        ///     and returns a reference to out.<br></br>
        ///     (See Examples)
        ///     References
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
        ///     The corresponding hyperbolic tangent values.<br></br>
        ///     This is a scalar if x is a scalar.
        /// </returns>
        public static NDarray tanh(this NDarray x, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.tanh(x.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.tanh(x.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Inverse hyperbolic sine element-wise.<br></br>
        ///     Notes
        ///     arcsinh is a multivalued function: for each x there are infinitely
        ///     many numbers z such that sinh(z) = x.<br></br>
        ///     The convention is to return the
        ///     z whose imaginary part lies in [-pi/2, pi/2].<br></br>
        ///     For real-valued input data types, arcsinh always returns real output.<br></br>
        ///     For each value that cannot be expressed as a real number or infinity, it
        ///     returns nan and sets the invalid floating point error flag.<br></br>
        ///     For complex-valued input, arccos is a complex analytical function that
        ///     has branch cuts [1j, infj] and [-1j, -infj] and is continuous from
        ///     the right on the former and from the left on the latter.<br></br>
        ///     The inverse hyperbolic sine is also known as asinh or sinh^-1.
        ///     References
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
        ///     Array of the same shape as x.<br></br>
        ///     This is a scalar if x is a scalar.
        /// </returns>
        public static NDarray arcsinh(this NDarray x, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.arcsinh(x.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.arcsinh(x.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Inverse hyperbolic cosine, element-wise.<br></br>
        ///     Notes
        ///     arccosh is a multivalued function: for each x there are infinitely
        ///     many numbers z such that cosh(z) = x.<br></br>
        ///     The convention is to return the
        ///     z whose imaginary part lies in [-pi, pi] and the real part in
        ///     [0, inf].<br></br>
        ///     For real-valued input data types, arccosh always returns real output.<br></br>
        ///     For each value that cannot be expressed as a real number or infinity, it
        ///     yields nan and sets the invalid floating point error flag.<br></br>
        ///     For complex-valued input, arccosh is a complex analytical function that
        ///     has a branch cut [-inf, 1] and is continuous from above on it.<br></br>
        ///     References
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
        ///     Array of the same shape as x.<br></br>
        ///     This is a scalar if x is a scalar.
        /// </returns>
        public static NDarray arccosh(this NDarray x, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.arccosh(x.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.arccosh(x.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Inverse hyperbolic tangent element-wise.<br></br>
        ///     Notes
        ///     arctanh is a multivalued function: for each x there are infinitely
        ///     many numbers z such that tanh(z) = x.<br></br>
        ///     The convention is to return
        ///     the z whose imaginary part lies in [-pi/2, pi/2].<br></br>
        ///     For real-valued input data types, arctanh always returns real output.<br></br>
        ///     For each value that cannot be expressed as a real number or infinity,
        ///     it yields nan and sets the invalid floating point error flag.<br></br>
        ///     For complex-valued input, arctanh is a complex analytical function
        ///     that has branch cuts [-1, -inf] and [1, inf] and is continuous from
        ///     above on the former and from below on the latter.<br></br>
        ///     The inverse hyperbolic tangent is also known as atanh or tanh^-1.
        ///     References
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
        ///     Array of the same shape as x.<br></br>
        ///     This is a scalar if x is a scalar.
        /// </returns>
        public static NDarray arctanh(this NDarray x, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.arctanh(x.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.arctanh(x.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Evenly round to the given number of decimals.<br></br>
        ///     Notes
        ///     For values exactly halfway between rounded decimal values, Cupy
        ///     rounds to the nearest even value.<br></br>
        ///     Thus 1.5 and 2.5 round to 2.0,
        ///     -0.5 and 0.5 round to 0.0, etc.<br></br>
        ///     Results may also be surprising due
        ///     to the inexact representation of decimal fractions in the IEEE
        ///     floating point standard [1] and errors introduced when scaling
        ///     by powers of ten.<br></br>
        ///     References
        /// </summary>
        /// <param name="a">
        ///     Input data.
        /// </param>
        /// <param name="decimals">
        ///     Number of decimal places to round to (default: 0).<br></br>
        ///     If
        ///     decimals is negative, it specifies the number of positions to
        ///     the left of the decimal point.
        /// </param>
        /// <param name="out">
        ///     Alternative output array in which to place the result.<br></br>
        ///     It must have
        ///     the same shape as the expected output, but the type of the output
        ///     values will be cast if necessary.<br></br>
        ///     See doc.ufuncs (Section
        ///     “Output arguments”) for details.
        /// </param>
        /// <returns>
        ///     An array of the same type as a, containing the rounded values.<br></br>
        ///     Unless out was specified, a new array is created.<br></br>
        ///     A reference to
        ///     the result is returned.<br></br>
        ///     The real and imaginary parts of complex numbers are rounded
        ///     separately.<br></br>
        ///     The result of rounding a float is a float.
        /// </returns>
        public static NDarray around(this NDarray a, int? decimals = 0, NDarray @out = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.around(a.CupyNDarray, decimals, @out?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.around(a.NumpyNDarray, decimals, @out?.NumpyNDarray));
            }
        }

        public static Cupy.NDarray round(this Cupy.NDarray a, int? decimals = 0, Cupy.NDarray @out = null)
        {
            var self = Py.Import("cupy");
            using var pyargs = ToTuple(new PyObject[] { a.PyObject });
            using var kwargs = new PyDict();
            using var decimalsPy = decimals != null ? ToPython(decimals) : null;
            using var outPy = @out != null ? ToPython(@out) : null;

            if (decimalsPy != null) kwargs["decimals"] = decimalsPy;
            if (outPy != null) kwargs["out"] = outPy;

            dynamic py = self.InvokeMethod("round", pyargs, kwargs);
            return ToCsharp<Cupy.NDarray>(py);
        }

        public static Numpy.NDarray round(this Numpy.NDarray a, int? decimals = 0, Numpy.NDarray @out = null)
        {
            var self = Py.Import("numpy");
            using var pyargs = ToTuple(new PyObject[] { a.PyObject });
            using var kwargs = new PyDict();
            using var decimalsPy = decimals != null ? ToPython(decimals) : null;
            using var outPy = @out != null ? ToPython(@out) : null;

            if (decimalsPy != null) kwargs["decimals"] = decimalsPy;
            if (outPy != null) kwargs["out"] = outPy;

            dynamic py = self.InvokeMethod("round", pyargs, kwargs);
            return ToCsharp<Numpy.NDarray>(py);
        }

        public static NDarray round(this NDarray a, int? decimals = 0, NDarray @out = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(xp.round(a.ToCupyNDarray, decimals, @out?.ToCupyNDarray));
            }
            else
            {
                return new NDarray(xp.round(a.ToNumpyNDarray, decimals, @out?.ToNumpyNDarray));
            }
        }

        /// <summary>
        ///     Round elements of the array to the nearest integer.
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
        ///     Output array is same shape and type as x.<br></br>
        ///     This is a scalar if x is a scalar.
        /// </returns>
        public static NDarray rint(this NDarray x, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.rint(x.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.rint(x.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Round to nearest integer towards zero.<br></br>
        ///     Round an array of floats element-wise to nearest integer towards zero.<br></br>
        ///     The rounded values are returned as floats.
        /// </summary>
        /// <param name="x">
        ///     An array of floats to be rounded
        /// </param>
        /// <param name="y">
        ///     Output array
        /// </param>
        /// <returns>
        ///     The array of rounded numbers
        /// </returns>
        public static NDarray fix(this NDarray x, NDarray y = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.fix(x.CupyNDarray, y?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.fix(x.NumpyNDarray, y?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Return the floor of the input, element-wise.<br></br>
        ///     The floor of the scalar x is the largest integer i, such that
        ///     i &lt;= x.<br></br>
        ///     It is often denoted as .
        ///     Notes
        ///     Some spreadsheet programs calculate the “floor-towards-zero”, in other
        ///     words floor(-2.5) == -2.  Cupy instead uses the definition of
        ///     floor where floor(-2.5) == -3.
        /// </summary>
        /// <param name="x">
        ///     Input data.
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
        ///     The floor of each element in x.<br></br>
        ///     This is a scalar if x is a scalar.
        /// </returns>
        public static NDarray floor(this NDarray x, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.floor(x.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.floor(x.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Return the ceiling of the input, element-wise.<br></br>
        ///     The ceil of the scalar x is the smallest integer i, such that
        ///     i &gt;= x.<br></br>
        ///     It is often denoted as .
        /// </summary>
        /// <param name="x">
        ///     Input data.
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
        ///     The ceiling of each element in x, with float dtype.<br></br>
        ///     This is a scalar if x is a scalar.
        /// </returns>
        public static NDarray ceil(this NDarray x, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.ceil(x.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.ceil(x.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Return the truncated value of the input, element-wise.<br></br>
        ///     The truncated value of the scalar x is the nearest integer i which
        ///     is closer to zero than x is.<br></br>
        ///     In short, the fractional part of the
        ///     signed number x is discarded.<br></br>
        ///     Notes
        /// </summary>
        /// <param name="x">
        ///     Input data.
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
        ///     The truncated value of each element in x.<br></br>
        ///     This is a scalar if x is a scalar.
        /// </returns>
        public static NDarray trunc(this NDarray x, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.trunc(x.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.trunc(x.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Return the product of array elements over a given axis.<br></br>
        ///     Notes
        ///     Arithmetic is modular when using integer types, and no error is
        ///     raised on overflow.<br></br>
        ///     That means that, on a 32-bit platform:
        ///     The product of an empty array is the neutral element 1:
        /// </summary>
        /// <param name="a">
        ///     Input data.
        /// </param>
        /// <param name="axis">
        ///     Axis or axis along which a product is performed.<br></br>
        ///     The default,
        ///     axis=None, will calculate the product of all the elements in the
        ///     input array.<br></br>
        ///     If axis is negative it counts from the last to the
        ///     first axis.<br></br>
        ///     If axis is a tuple of ints, a product is performed on all of the
        ///     axis specified in the tuple instead of a single axis or all the
        ///     axis as before.
        /// </param>
        /// <param name="dtype">
        ///     The type of the returned array, as well as of the accumulator in
        ///     which the elements are multiplied.<br></br>
        ///     The dtype of a is used by
        ///     default unless a has an integer dtype of less precision than the
        ///     default platform integer.<br></br>
        ///     In that case, if a is signed then the
        ///     platform integer is used while if a is unsigned then an unsigned
        ///     integer of the same precision as the platform integer is used.
        /// </param>
        /// <param name="out">
        ///     Alternative output array in which to place the result.<br></br>
        ///     It must have
        ///     the same shape as the expected output, but the type of the output
        ///     values will be cast if necessary.
        /// </param>
        /// <param name="keepdims">
        ///     If this is set to True, the axis which are reduced are left in the
        ///     result as dimensions with size one.<br></br>
        ///     With this option, the result
        ///     will broadcast correctly against the input array.<br></br>
        ///     If the default value is passed, then keepdims will not be
        ///     passed through to the prod method of sub-classes of
        ///     ndarray, however any non-default value will be.<br></br>
        ///     If the
        ///     sub-class’ method does not implement keepdims any
        ///     exceptions will be raised.
        /// </param>
        /// <param name="initial">
        ///     The starting value for this product.<br></br>
        ///     See reduce for details.
        /// </param>
        /// <returns>
        ///     An array shaped as a but with the specified axis removed.<br></br>
        ///     Returns a reference to out if specified.
        /// </returns>
        public static NDarray prod(this NDarray a, Axis axis = null, Dtype dtype = null, NDarray @out = null,
            bool? keepdims = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.prod(a.CupyNDarray, axis?.CupyAxis, dtype?.CupyDtype, @out?.CupyNDarray, keepdims));
            }
            else
            {
                return new NDarray(np.prod(a.NumpyNDarray, axis?.NumpyAxis, dtype?.NumpyDtype, @out?.NumpyNDarray,
                    keepdims));
            }
        }

        /// <summary>
        ///     Sum of array elements over a given axis.<br></br>
        ///     Notes
        ///     Arithmetic is modular when using integer types, and no error is
        ///     raised on overflow.<br></br>
        ///     The sum of an empty array is the neutral element 0:
        /// </summary>
        /// <param name="a">
        ///     Elements to sum.
        /// </param>
        /// <param name="axis">
        ///     Axis or axis along which a sum is performed.<br></br>
        ///     The default,
        ///     axis=None, will sum all of the elements of the input array.<br></br>
        ///     If
        ///     axis is negative it counts from the last to the first axis.<br></br>
        ///     If axis is a tuple of ints, a sum is performed on all of the axis
        ///     specified in the tuple instead of a single axis or all the axis as
        ///     before.
        /// </param>
        /// <param name="dtype">
        ///     The type of the returned array and of the accumulator in which the
        ///     elements are summed.<br></br>
        ///     The dtype of a is used by default unless a
        ///     has an integer dtype of less precision than the default platform
        ///     integer.<br></br>
        ///     In that case, if a is signed then the platform integer
        ///     is used while if a is unsigned then an unsigned integer of the
        ///     same precision as the platform integer is used.
        /// </param>
        /// <param name="out">
        ///     Alternative output array in which to place the result.<br></br>
        ///     It must have
        ///     the same shape as the expected output, but the type of the output
        ///     values will be cast if necessary.
        /// </param>
        /// <param name="keepdims">
        ///     If this is set to True, the axis which are reduced are left
        ///     in the result as dimensions with size one.<br></br>
        ///     With this option,
        ///     the result will broadcast correctly against the input array.<br></br>
        ///     If the default value is passed, then keepdims will not be
        ///     passed through to the sum method of sub-classes of
        ///     ndarray, however any non-default value will be.<br></br>
        ///     If the
        ///     sub-class’ method does not implement keepdims any
        ///     exceptions will be raised.
        /// </param>
        /// <param name="initial">
        ///     Starting value for the sum.<br></br>
        ///     See reduce for details.
        /// </param>
        /// <returns>
        ///     An array with the same shape as a, with the specified
        ///     axis removed.<br></br>
        ///     If a is a 0-d array, or if axis is None, a scalar
        ///     is returned.<br></br>
        ///     If an output array is specified, a reference to
        ///     out is returned.
        /// </returns>
        public static NDarray sum(this NDarray a, Axis axis = null, Dtype dtype = null, NDarray @out = null,
            bool? keepdims = null, ValueType initial = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.sum(a.CupyNDarray, axis?.CupyAxis, dtype?.CupyDtype, @out?.CupyNDarray, keepdims,
                    initial));
            }
            else
            {
                return new NDarray(np.sum(a.NumpyNDarray, axis?.NumpyAxis, dtype?.NumpyDtype, @out?.NumpyNDarray,
                    keepdims, initial));
            }
        }

        /// <summary>
        ///     Return the product of array elements over a given axis treating Not a
        ///     Numbers (NaNs) as ones.<br></br>
        ///     One is returned for slices that are all-NaN or empty.
        /// </summary>
        /// <param name="a">
        ///     Array containing numbers whose product is desired.<br></br>
        ///     If a is not an
        ///     array, a conversion is attempted.
        /// </param>
        /// <param name="axis">
        ///     Axis or axis along which the product is computed.<br></br>
        ///     The default is to compute
        ///     the product of the flattened array.
        /// </param>
        /// <param name="dtype">
        ///     The type of the returned array and of the accumulator in which the
        ///     elements are summed.<br></br>
        ///     By default, the dtype of a is used.<br></br>
        ///     An
        ///     exception is when a has an integer type with less precision than
        ///     the platform (u)intp.<br></br>
        ///     In that case, the default will be either
        ///     (u)int32 or (u)int64 depending on whether the platform is 32 or 64
        ///     bits.<br></br>
        ///     For inexact inputs, dtype must be inexact.
        /// </param>
        /// <param name="out">
        ///     Alternate output array in which to place the result.<br></br>
        ///     The default
        ///     is None.<br></br>
        ///     If provided, it must have the same shape as the
        ///     expected output, but the type will be cast if necessary.<br></br>
        ///     See
        ///     doc.ufuncs for details.<br></br>
        ///     The casting of NaN to integer can yield
        ///     unexpected results.
        /// </param>
        /// <param name="keepdims">
        ///     If True, the axis which are reduced are left in the result as
        ///     dimensions with size one.<br></br>
        ///     With this option, the result will
        ///     broadcast correctly against the original arr.
        /// </param>
        /// <returns>
        ///     A new array holding the result is returned unless out is
        ///     specified, in which case it is returned.
        /// </returns>
        public static NDarray nanprod(this NDarray a, Axis axis = null, Dtype dtype = null, NDarray @out = null,
            bool? keepdims = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.nanprod(a.CupyNDarray, axis?.CupyAxis, dtype?.CupyDtype, @out?.CupyNDarray,
                    keepdims));
            }
            else
            {
                return new NDarray(np.nanprod(a.NumpyNDarray, axis?.NumpyAxis, dtype?.NumpyDtype, @out?.NumpyNDarray,
                    keepdims));
            }
        }

        /// <summary>
        ///     Return the sum of array elements over a given axis treating Not a
        ///     Numbers (NaNs) as zero.<br></br>
        ///     In Cupy versions &lt;= 1.9.0 Nan is returned for slices that are all-NaN or
        ///     empty.<br></br>
        ///     In later versions zero is returned.<br></br>
        ///     Notes
        ///     If both positive and negative infinity are present, the sum will be Not
        ///     A Number (NaN).
        /// </summary>
        /// <param name="a">
        ///     Array containing numbers whose sum is desired.<br></br>
        ///     If a is not an
        ///     array, a conversion is attempted.
        /// </param>
        /// <param name="axis">
        ///     Axis or axis along which the sum is computed.<br></br>
        ///     The default is to compute the
        ///     sum of the flattened array.
        /// </param>
        /// <param name="dtype">
        ///     The type of the returned array and of the accumulator in which the
        ///     elements are summed.<br></br>
        ///     By default, the dtype of a is used.<br></br>
        ///     An
        ///     exception is when a has an integer type with less precision than
        ///     the platform (u)intp.<br></br>
        ///     In that case, the default will be either
        ///     (u)int32 or (u)int64 depending on whether the platform is 32 or 64
        ///     bits.<br></br>
        ///     For inexact inputs, dtype must be inexact.
        /// </param>
        /// <param name="out">
        ///     Alternate output array in which to place the result.<br></br>
        ///     The default
        ///     is None.<br></br>
        ///     If provided, it must have the same shape as the
        ///     expected output, but the type will be cast if necessary.<br></br>
        ///     See
        ///     doc.ufuncs for details.<br></br>
        ///     The casting of NaN to integer can yield
        ///     unexpected results.
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
        ///     A new array holding the result is returned unless out is
        ///     specified, in which it is returned.<br></br>
        ///     The result has the same
        ///     size as a, and the same shape as a if axis is not None
        ///     or a is a 1-d array.
        /// </returns>
        public static NDarray nansum(this NDarray a, Axis axis = null, Dtype dtype = null, NDarray @out = null,
            bool? keepdims = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.nansum(a.CupyNDarray, axis?.CupyAxis, dtype?.CupyDtype, @out?.CupyNDarray,
                    keepdims));
            }
            else
            {
                return new NDarray(np.nansum(a.NumpyNDarray, axis?.NumpyAxis, dtype?.NumpyDtype, @out?.NumpyNDarray,
                    keepdims));
            }
        }

        /// <summary>
        ///     Return the cumulative product of elements along a given axis.<br></br>
        ///     Notes
        ///     Arithmetic is modular when using integer types, and no error is
        ///     raised on overflow.
        /// </summary>
        /// <param name="a">
        ///     Input array.
        /// </param>
        /// <param name="axis">
        ///     Axis along which the cumulative product is computed.<br></br>
        ///     By default
        ///     the input is flattened.
        /// </param>
        /// <param name="dtype">
        ///     Type of the returned array, as well as of the accumulator in which
        ///     the elements are multiplied.<br></br>
        ///     If dtype is not specified, it
        ///     defaults to the dtype of a, unless a has an integer dtype with
        ///     a precision less than that of the default platform integer.<br></br>
        ///     In
        ///     that case, the default platform integer is used instead.
        /// </param>
        /// <param name="out">
        ///     Alternative output array in which to place the result.<br></br>
        ///     It must
        ///     have the same shape and buffer length as the expected output
        ///     but the type of the resulting values will be cast if necessary.
        /// </param>
        /// <returns>
        ///     A new array holding the result is returned unless out is
        ///     specified, in which case a reference to out is returned.
        /// </returns>
        public static NDarray cumprod(this NDarray a, int? axis = null, Dtype dtype = null, NDarray @out = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.cumprod(a.CupyNDarray, axis, dtype?.CupyDtype, @out?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.cumprod(a.NumpyNDarray, axis, dtype?.NumpyDtype, @out?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Return the cumulative sum of the elements along a given axis.<br></br>
        ///     Notes
        ///     Arithmetic is modular when using integer types, and no error is
        ///     raised on overflow.
        /// </summary>
        /// <param name="a">
        ///     Input array.
        /// </param>
        /// <param name="axis">
        ///     Axis along which the cumulative sum is computed.<br></br>
        ///     The default
        ///     (None) is to compute the cumsum over the flattened array.
        /// </param>
        /// <param name="dtype">
        ///     Type of the returned array and of the accumulator in which the
        ///     elements are summed.<br></br>
        ///     If dtype is not specified, it defaults
        ///     to the dtype of a, unless a has an integer dtype with a
        ///     precision less than that of the default platform integer.<br></br>
        ///     In
        ///     that case, the default platform integer is used.
        /// </param>
        /// <param name="out">
        ///     Alternative output array in which to place the result.<br></br>
        ///     It must
        ///     have the same shape and buffer length as the expected output
        ///     but the type will be cast if necessary.<br></br>
        ///     See doc.ufuncs
        ///     (Section “Output arguments”) for more details.
        /// </param>
        /// <returns>
        ///     A new array holding the result is returned unless out is
        ///     specified, in which case a reference to out is returned.<br></br>
        ///     The
        ///     result has the same size as a, and the same shape as a if
        ///     axis is not None or a is a 1-d array.
        /// </returns>
        public static NDarray cumsum(this NDarray a, int? axis = null, Dtype dtype = null, NDarray @out = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.cumsum(a.CupyNDarray, axis, dtype?.CupyDtype, @out?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.cumsum(a.NumpyNDarray, axis, dtype?.NumpyDtype, @out?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Return the cumulative product of array elements over a given axis treating Not a
        ///     Numbers (NaNs) as one.<br></br>
        ///     The cumulative product does not change when NaNs are
        ///     encountered and leading NaNs are replaced by ones.<br></br>
        ///     Ones are returned for slices that are all-NaN or empty.
        /// </summary>
        /// <param name="a">
        ///     Input array.
        /// </param>
        /// <param name="axis">
        ///     Axis along which the cumulative product is computed.<br></br>
        ///     By default
        ///     the input is flattened.
        /// </param>
        /// <param name="dtype">
        ///     Type of the returned array, as well as of the accumulator in which
        ///     the elements are multiplied.<br></br>
        ///     If dtype is not specified, it
        ///     defaults to the dtype of a, unless a has an integer dtype with
        ///     a precision less than that of the default platform integer.<br></br>
        ///     In
        ///     that case, the default platform integer is used instead.
        /// </param>
        /// <param name="out">
        ///     Alternative output array in which to place the result.<br></br>
        ///     It must
        ///     have the same shape and buffer length as the expected output
        ///     but the type of the resulting values will be cast if necessary.
        /// </param>
        /// <returns>
        ///     A new array holding the result is returned unless out is
        ///     specified, in which case it is returned.
        /// </returns>
        public static NDarray nancumprod(this NDarray a, int? axis = null, Dtype dtype = null, NDarray @out = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.cumprod(a.CupyNDarray, axis, dtype?.CupyDtype, @out?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.cumprod(a.NumpyNDarray, axis, dtype?.NumpyDtype, @out?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Return the cumulative sum of array elements over a given axis treating Not a
        ///     Numbers (NaNs) as zero.<br></br>
        ///     The cumulative sum does not change when NaNs are
        ///     encountered and leading NaNs are replaced by zeros.<br></br>
        ///     Zeros are returned for slices that are all-NaN or empty.
        /// </summary>
        /// <param name="a">
        ///     Input array.
        /// </param>
        /// <param name="axis">
        ///     Axis along which the cumulative sum is computed.<br></br>
        ///     The default
        ///     (None) is to compute the cumsum over the flattened array.
        /// </param>
        /// <param name="dtype">
        ///     Type of the returned array and of the accumulator in which the
        ///     elements are summed.<br></br>
        ///     If dtype is not specified, it defaults
        ///     to the dtype of a, unless a has an integer dtype with a
        ///     precision less than that of the default platform integer.<br></br>
        ///     In
        ///     that case, the default platform integer is used.
        /// </param>
        /// <param name="out">
        ///     Alternative output array in which to place the result.<br></br>
        ///     It must
        ///     have the same shape and buffer length as the expected output
        ///     but the type will be cast if necessary.<br></br>
        ///     See doc.ufuncs
        ///     (Section “Output arguments”) for more details.
        /// </param>
        /// <returns>
        ///     A new array holding the result is returned unless out is
        ///     specified, in which it is returned.<br></br>
        ///     The result has the same
        ///     size as a, and the same shape as a if axis is not None
        ///     or a is a 1-d array.
        /// </returns>
        public static NDarray nancumsum(this NDarray a, int? axis = null, Dtype dtype = null, NDarray @out = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.cumsum(a.CupyNDarray, axis, dtype?.CupyDtype, @out?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.cumsum(a.NumpyNDarray, axis, dtype?.NumpyDtype, @out?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Calculate the n-th discrete difference along the given axis.<br></br>
        ///     The first difference is given by out[n] = a[n+1] - a[n] along
        ///     the given axis, higher differences are calculated by using diff
        ///     recursively.<br></br>
        ///     Notes
        ///     Type is preserved for boolean arrays, so the result will contain
        ///     False when consecutive elements are the same and True when they
        ///     differ.<br></br>
        ///     For unsigned integer arrays, the results will also be unsigned.<br></br>
        ///     This
        ///     should not be surprising, as the result is consistent with
        ///     calculating the difference directly:
        ///     If this is not desirable, then the array should be cast to a larger
        ///     integer type first:
        /// </summary>
        /// <param name="a">
        ///     Input array
        /// </param>
        /// <param name="n">
        ///     The number of times values are differenced.<br></br>
        ///     If zero, the input
        ///     is returned as-is.
        /// </param>
        /// <param name="axis">
        ///     The axis along which the difference is taken, default is the
        ///     last axis.
        /// </param>
        /// <param name="append">
        ///     Values to prepend or append to “a” along axis prior to
        ///     performing the difference.<br></br>
        ///     Scalar values are expanded to
        ///     arrays with length 1 in the direction of axis and the shape
        ///     of the input array in along all other axis.<br></br>
        ///     Otherwise the
        ///     dimension and shape must match “a” except along axis.
        /// </param>
        /// <param name="prepend">
        ///     Values to prepend or append to “a” along axis prior to
        ///     performing the difference.<br></br>
        ///     Scalar values are expanded to
        ///     arrays with length 1 in the direction of axis and the shape
        ///     of the input array in along all other axis.<br></br>
        ///     Otherwise the
        ///     dimension and shape must match “a” except along axis.
        /// </param>
        /// <returns>
        ///     The n-th differences.<br></br>
        ///     The shape of the output is the same as a
        ///     except along axis where the dimension is smaller by n.<br></br>
        ///     The
        ///     type of the output is the same as the type of the difference
        ///     between any two elements of a.<br></br>
        ///     This is the same as the type of
        ///     a in most cases.<br></br>
        ///     A notable exception is datetime64, which
        ///     results in a timedelta64 output array.
        /// </returns>
        public static NDarray diff(this NDarray a, int? n = 1, int? axis = -1, NDarray append = null,
            NDarray prepend = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.diff(a.CupyNDarray, n, axis, append?.CupyNDarray, prepend?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.diff(a.NumpyNDarray, n, axis, append?.NumpyNDarray, prepend?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     The differences between consecutive elements of an array.<br></br>
        ///     Notes
        ///     When applied to masked arrays, this function drops the mask information
        ///     if the to_begin and/or to_end parameters are used.
        /// </summary>
        /// <param name="ary">
        ///     If necessary, will be flattened before the differences are taken.
        /// </param>
        /// <param name="to_end">
        ///     Number(s) to append at the end of the returned differences.
        /// </param>
        /// <param name="to_begin">
        ///     Number(s) to prepend at the beginning of the returned differences.
        /// </param>
        /// <returns>
        ///     The differences.<br></br>
        ///     Loosely, this is ary.flat[1:] - ary.flat[:-1].
        /// </returns>
        public static NDarray ediff1d(this NDarray ary, NDarray to_end = null, NDarray to_begin = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.ediff1d(ary.CupyNDarray, to_end?.CupyNDarray, to_begin?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.ediff1d(ary.NumpyNDarray, to_end?.NumpyNDarray, to_begin?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Return the cross product of two (arrays of) vectors.<br></br>
        ///     The cross product of a and b in  is a vector perpendicular
        ///     to both a and b.<br></br>
        ///     If a and b are arrays of vectors, the vectors
        ///     are defined by the last axis of a and b by default, and these axis
        ///     can have dimensions 2 or 3.<br></br>
        ///     Where the dimension of either a or b is
        ///     2, the third component of the input vector is assumed to be zero and the
        ///     cross product calculated accordingly.<br></br>
        ///     In cases where both input vectors
        ///     have dimension 2, the z-component of the cross product is returned.<br></br>
        ///     Notes
        ///     Supports full broadcasting of the inputs.
        /// </summary>
        /// <param name="a">
        ///     Components of the first vector(s).
        /// </param>
        /// <param name="b">
        ///     Components of the second vector(s).
        /// </param>
        /// <param name="axisa">
        ///     Axis of a that defines the vector(s).<br></br>
        ///     By default, the last axis.
        /// </param>
        /// <param name="axisb">
        ///     Axis of b that defines the vector(s).<br></br>
        ///     By default, the last axis.
        /// </param>
        /// <param name="axisc">
        ///     Axis of c containing the cross product vector(s).<br></br>
        ///     Ignored if
        ///     both input vectors have dimension 2, as the return is scalar.<br></br>
        ///     By default, the last axis.
        /// </param>
        /// <param name="axis">
        ///     If defined, the axis of a, b and c that defines the vector(s)
        ///     and cross product(s).<br></br>
        ///     Overrides axisa, axisb and axisc.
        /// </param>
        /// <returns>
        ///     Vector cross product(s).
        /// </returns>
        public static NDarray cross(this NDarray a, NDarray b, int? axisa = -1, int? axisb = -1, int? axisc = -1,
            int? axis = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.cross(a.CupyNDarray, b.CupyNDarray, axisa, axisb, axisc, axis));
            }
            else
            {
                return new NDarray(np.cross(a.NumpyNDarray, b.NumpyNDarray, axisa, axisb, axisc, axis));
            }
        }

        /// <summary>
        ///     Integrate along the given axis using the composite trapezoidal rule.<br></br>
        ///     Integrate y (x) along given axis.<br></br>
        ///     Notes
        ///     Image [2] illustrates trapezoidal rule – y-axis locations of points
        ///     will be taken from y array, by default x-axis distances between
        ///     points will be 1.0, alternatively they can be provided with x array
        ///     or with dx scalar.<br></br>
        ///     Return value will be equal to combined area under
        ///     the red lines.<br></br>
        ///     References
        /// </summary>
        /// <param name="y">
        ///     Input array to integrate.
        /// </param>
        /// <param name="x">
        ///     The sample points corresponding to the y values.<br></br>
        ///     If x is None,
        ///     the sample points are assumed to be evenly spaced dx apart.<br></br>
        ///     The
        ///     default is None.
        /// </param>
        /// <param name="dx">
        ///     The spacing between sample points when x is None.<br></br>
        ///     The default is 1.
        /// </param>
        /// <param name="axis">
        ///     The axis along which to integrate.
        /// </param>
        /// <returns>
        ///     Definite integral as approximated by trapezoidal rule.
        /// </returns>
        public static float trapz(this NDarray y, NDarray x = null, float? dx = 1.0f, int? axis = -1)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return cp.trapz(y.CupyNDarray, x?.CupyNDarray, dx, axis);
            }
            else
            {
                return np.trapz(y.NumpyNDarray, x?.NumpyNDarray, dx, axis);
            }
        }

        /// <summary>
        ///     Calculate the exponential of all elements in the input array.<br></br>
        ///     Notes
        ///     The irrational number e is also known as Euler’s number.<br></br>
        ///     It is
        ///     approximately 2.718281, and is the base of the natural logarithm,
        ///     ln (this means that, if ,
        ///     then . For real input, exp(x) is always positive.<br></br>
        ///     For complex arguments, x = a + ib, we can write
        ///     .  The first term, , is already
        ///     known (it is the real argument, described above).<br></br>
        ///     The second term,
        ///     , is , a function with
        ///     magnitude 1 and a periodic phase.<br></br>
        ///     References
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
        ///     Output array, element-wise exponential of x.<br></br>
        ///     This is a scalar if x is a scalar.
        /// </returns>
        public static NDarray exp(this NDarray x, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.exp(x.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.exp(x.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Calculate exp(x) - 1 for all elements in the array.<br></br>
        ///     Notes
        ///     This function provides greater precision than exp(x) - 1
        ///     for small values of x.
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
        ///     Element-wise exponential minus one: out = exp(x) - 1.<br></br>
        ///     This is a scalar if x is a scalar.
        /// </returns>
        public static NDarray expm1(this NDarray x, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.expm1(x.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.expm1(x.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Calculate 2**p for all p in the input array.<br></br>
        ///     Notes
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
        ///     Element-wise 2 to the power x.<br></br>
        ///     This is a scalar if x is a scalar.
        /// </returns>
        public static NDarray exp2(this NDarray x, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.exp2(x.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.exp2(x.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Natural logarithm, element-wise.<br></br>
        ///     The natural logarithm log is the inverse of the exponential function,
        ///     so that log(exp(x)) = x.<br></br>
        ///     The natural logarithm is logarithm in base
        ///     e.<br></br>
        ///     Notes
        ///     Logarithm is a multivalued function: for each x there is an infinite
        ///     number of z such that exp(z) = x.<br></br>
        ///     The convention is to return the
        ///     z whose imaginary part lies in [-pi, pi].<br></br>
        ///     For real-valued input data types, log always returns real output.<br></br>
        ///     For
        ///     each value that cannot be expressed as a real number or infinity, it
        ///     yields nan and sets the invalid floating point error flag.<br></br>
        ///     For complex-valued input, log is a complex analytical function that
        ///     has a branch cut [-inf, 0] and is continuous from above on it.<br></br>
        ///     log
        ///     handles the floating-point negative zero as an infinitesimal negative
        ///     number, conforming to the C99 standard.<br></br>
        ///     References
        /// </summary>
        /// <param name="x">
        ///     Input value.
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
        ///     The natural logarithm of x, element-wise.<br></br>
        ///     This is a scalar if x is a scalar.
        /// </returns>
        public static NDarray log(this NDarray x, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.log(x.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.log(x.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Return the base 10 logarithm of the input array, element-wise.<br></br>
        ///     Notes
        ///     Logarithm is a multivalued function: for each x there is an infinite
        ///     number of z such that 10**z = x.<br></br>
        ///     The convention is to return the
        ///     z whose imaginary part lies in [-pi, pi].<br></br>
        ///     For real-valued input data types, log10 always returns real output.<br></br>
        ///     For each value that cannot be expressed as a real number or infinity,
        ///     it yields nan and sets the invalid floating point error flag.<br></br>
        ///     For complex-valued input, log10 is a complex analytical function that
        ///     has a branch cut [-inf, 0] and is continuous from above on it.<br></br>
        ///     log10 handles the floating-point negative zero as an infinitesimal
        ///     negative number, conforming to the C99 standard.<br></br>
        ///     References
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
        ///     The logarithm to the base 10 of x, element-wise.<br></br>
        ///     NaNs are
        ///     returned where x is negative.<br></br>
        ///     This is a scalar if x is a scalar.
        /// </returns>
        public static NDarray log10(this NDarray x, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.log10(x.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.log10(x.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Base-2 logarithm of x.<br></br>
        ///     Notes
        ///     Logarithm is a multivalued function: for each x there is an infinite
        ///     number of z such that 2**z = x.<br></br>
        ///     The convention is to return the z
        ///     whose imaginary part lies in [-pi, pi].<br></br>
        ///     For real-valued input data types, log2 always returns real output.<br></br>
        ///     For each value that cannot be expressed as a real number or infinity,
        ///     it yields nan and sets the invalid floating point error flag.<br></br>
        ///     For complex-valued input, log2 is a complex analytical function that
        ///     has a branch cut [-inf, 0] and is continuous from above on it.<br></br>
        ///     log2
        ///     handles the floating-point negative zero as an infinitesimal negative
        ///     number, conforming to the C99 standard.
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
        ///     Base-2 logarithm of x.<br></br>
        ///     This is a scalar if x is a scalar.
        /// </returns>
        public static NDarray log2(this NDarray x, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.log2(x.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.log2(x.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Return the natural logarithm of one plus the input array, element-wise.<br></br>
        ///     Calculates log(1 + x).<br></br>
        ///     Notes
        ///     For real-valued input, log1p is accurate also for x so small
        ///     that 1 + x == 1 in floating-point accuracy.<br></br>
        ///     Logarithm is a multivalued function: for each x there is an infinite
        ///     number of z such that exp(z) = 1 + x.<br></br>
        ///     The convention is to return
        ///     the z whose imaginary part lies in [-pi, pi].<br></br>
        ///     For real-valued input data types, log1p always returns real output.<br></br>
        ///     For each value that cannot be expressed as a real number or infinity,
        ///     it yields nan and sets the invalid floating point error flag.<br></br>
        ///     For complex-valued input, log1p is a complex analytical function that
        ///     has a branch cut [-inf, -1] and is continuous from above on it.<br></br>
        ///     log1p handles the floating-point negative zero as an infinitesimal
        ///     negative number, conforming to the C99 standard.<br></br>
        ///     References
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
        ///     Natural logarithm of 1 + x, element-wise.<br></br>
        ///     This is a scalar if x is a scalar.
        /// </returns>
        public static NDarray log1p(this NDarray x, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.log1p(x.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.log1p(x.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Logarithm of the sum of exponentiations of the inputs.<br></br>
        ///     Calculates log(exp(x1) + exp(x2)).<br></br>
        ///     This function is useful in
        ///     statistics where the calculated probabilities of events may be so small
        ///     as to exceed the range of normal floating point numbers.<br></br>
        ///     In such cases
        ///     the logarithm of the calculated probability is stored.<br></br>
        ///     This function
        ///     allows adding probabilities stored in such a fashion.<br></br>
        ///     Notes
        /// </summary>
        /// <param name="x2">
        ///     Input values.
        /// </param>
        /// <param name="x1">
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
        ///     Logarithm of exp(x1) + exp(x2).<br></br>
        ///     This is a scalar if both x1 and x2 are scalars.
        /// </returns>
        public static NDarray logaddexp(this NDarray x2, NDarray x1, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.logaddexp(x2.CupyNDarray, x1.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.logaddexp(x2.NumpyNDarray, x1.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Logarithm of the sum of exponentiations of the inputs in base-2.
        ///     Calculates log2(2**x1 + 2**x2).<br></br>
        ///     This function is useful in machine
        ///     learning when the calculated probabilities of events may be so small as
        ///     to exceed the range of normal floating point numbers.<br></br>
        ///     In such cases
        ///     the base-2 logarithm of the calculated probability can be used instead.<br></br>
        ///     This function allows adding probabilities stored in such a fashion.<br></br>
        ///     Notes
        /// </summary>
        /// <param name="x2">
        ///     Input values.
        /// </param>
        /// <param name="x1">
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
        ///     Base-2 logarithm of 2**x1 + 2**x2.
        ///     This is a scalar if both x1 and x2 are scalars.
        /// </returns>
        public static NDarray logaddexp2(this NDarray x2, NDarray x1, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.logaddexp2(x2.CupyNDarray, x1.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.logaddexp2(x2.NumpyNDarray, x1.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Return the sinc function.<br></br>
        ///     The sinc function is .
        ///     Notes
        ///     sinc(0) is the limit value 1.<br></br>
        ///     The name sinc is short for “sine cardinal” or “sinus cardinalis”.
        ///     The sinc function is used in various signal processing applications,
        ///     including in anti-aliasing, in the construction of a Lanczos resampling
        ///     filter, and in interpolation.<br></br>
        ///     For bandlimited interpolation of discrete-time signals, the ideal
        ///     interpolation kernel is proportional to the sinc function.<br></br>
        ///     References
        /// </summary>
        /// <param name="x">
        ///     Array (possibly multi-dimensional) of values for which to to
        ///     calculate sinc(x).
        /// </param>
        /// <returns>
        ///     sinc(x), which has the same shape as the input.
        /// </returns>
        public static NDarray sinc(this NDarray x)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.sinc(x.CupyNDarray));
            }
            else
            {
                return new NDarray(np.sinc(x.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Returns element-wise True where signbit is set (less than zero).
        /// </summary>
        /// <param name="x">
        ///     The input value(s).
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
        ///     Output array, or reference to out if that was supplied.<br></br>
        ///     This is a scalar if x is a scalar.
        /// </returns>
        public static NDarray signbit(this NDarray x, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.signbit(x.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.signbit(x.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Change the sign of x1 to that of x2, element-wise.<br></br>
        ///     If both arguments are arrays or sequences, they have to be of the same
        ///     length.<br></br>
        ///     If x2 is a scalar, its sign will be copied to all elements of
        ///     x1.
        /// </summary>
        /// <param name="x1">
        ///     Values to change the sign of.
        /// </param>
        /// <param name="x2">
        ///     The sign of x2 is copied to x1.
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
        ///     The values of x1 with the sign of x2.
        ///     This is a scalar if both x1 and x2 are scalars.
        /// </returns>
        public static NDarray copysign(this NDarray x1, NDarray x2, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.copysign(x1.CupyNDarray, x2.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.copysign(x1.NumpyNDarray, x2.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Decompose the elements of x into mantissa and twos exponent.<br></br>
        ///     Returns (mantissa, exponent), where x = mantissa * 2**exponent`.
        ///     The mantissa is lies in the open interval(-1, 1), while the twos
        ///     exponent is a signed integer.<br></br>
        ///     Notes
        ///     Complex dtypes are not supported, they will raise a TypeError.
        /// </summary>
        /// <param name="x">
        ///     Array of numbers to be decomposed.
        /// </param>
        /// <param name="out1">
        ///     Output array for the mantissa.<br></br>
        ///     Must have the same shape as x.
        /// </param>
        /// <param name="out2">
        ///     Output array for the exponent.<br></br>
        ///     Must have the same shape as x.
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
        ///     A tuple of:
        ///     mantissa
        ///     Floating values between -1 and 1.
        ///     This is a scalar if x is a scalar.
        ///     exponent
        ///     Integer exponents of 2.
        ///     This is a scalar if x is a scalar.
        /// </returns>
        public static (NDarray, NDarray) frexp(this NDarray x, NDarray out1 = null, NDarray out2 = null,
            NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                var ret = cp.frexp(x.CupyNDarray, out1?.CupyNDarray, out2?.CupyNDarray, @out?.CupyNDarray,
                    where?.CupyNDarray);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2));
            }
            else
            {
                var ret = np.frexp(x.NumpyNDarray, out1?.NumpyNDarray, out2?.NumpyNDarray, @out?.NumpyNDarray,
                    where?.NumpyNDarray);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2));
            }
        }

        /// <summary>
        ///     Returns x1 * 2**x2, element-wise.<br></br>
        ///     The mantissas x1 and twos exponents x2 are used to construct
        ///     floating point numbers x1 * 2**x2.
        ///     Notes
        ///     Complex dtypes are not supported, they will raise a TypeError.<br></br>
        ///     ldexp is useful as the inverse of frexp, if used by itself it is
        ///     more clear to simply use the expression x1 * 2**x2.
        /// </summary>
        /// <param name="x1">
        ///     Array of multipliers.
        /// </param>
        /// <param name="x2">
        ///     Array of twos exponents.
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
        ///     The result of x1 * 2**x2.
        ///     This is a scalar if both x1 and x2 are scalars.
        /// </returns>
        public static NDarray ldexp(this NDarray x1, NDarray x2, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.ldexp(x1.CupyNDarray, x2.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.ldexp(x1.NumpyNDarray, x2.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Return the next floating-point value after x1 towards x2, element-wise.
        /// </summary>
        /// <param name="x1">
        ///     Values to find the next representable value of.
        /// </param>
        /// <param name="x2">
        ///     The direction where to look for the next representable value of x1.
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
        ///     The next representable values of x1 in the direction of x2.
        ///     This is a scalar if both x1 and x2 are scalars.
        /// </returns>
        public static NDarray nextafter(this NDarray x1, NDarray x2, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.nextafter(x1.CupyNDarray, x2.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.nextafter(x1.NumpyNDarray, x2.NumpyNDarray, @out?.NumpyNDarray,
                    where?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Return the distance between x and the nearest adjacent number.<br></br>
        ///     Notes
        ///     It can be considered as a generalization of EPS:
        ///     spacing(cp.float64(1)) == cp.finfo(cp.float64).eps, and there
        ///     should not be any representable number between x + spacing(x) and
        ///     x for any finite x.<br></br>
        ///     Spacing of +- inf and NaN is NaN.
        /// </summary>
        /// <param name="x">
        ///     Values to find the spacing of.
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
        ///     The spacing of values of x.<br></br>
        ///     This is a scalar if x is a scalar.
        /// </returns>
        public static NDarray spacing(this NDarray x, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.spacing(x.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.spacing(x.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Returns the lowest common multiple of |x1| and |x2|
        /// </summary>
        /// <param name="x2">
        ///     Arrays of values
        /// </param>
        /// <param name="x1">
        ///     Arrays of values
        /// </param>
        /// <returns>
        ///     The lowest common multiple of the absolute value of the inputs
        ///     This is a scalar if both x1 and x2 are scalars.
        /// </returns>
        public static NDarray lcm(this NDarray x2, NDarray x1)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.lcm(x2.CupyNDarray, x1.CupyNDarray));
            }
            else
            {
                return new NDarray(np.lcm(x2.NumpyNDarray, x1.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Returns the greatest common divisor of |x1| and |x2|
        /// </summary>
        /// <param name="x2">
        ///     Arrays of values
        /// </param>
        /// <param name="x1">
        ///     Arrays of values
        /// </param>
        /// <returns>
        ///     The greatest common divisor of the absolute value of the inputs
        ///     This is a scalar if both x1 and x2 are scalars.
        /// </returns>
        public static NDarray gcd(this NDarray x2, NDarray x1)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.gcd(x2.CupyNDarray, x1.CupyNDarray));
            }
            else
            {
                return new NDarray(np.gcd(x2.NumpyNDarray, x1.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Add arguments element-wise.<br></br>
        ///     Notes
        ///     Equivalent to x1 + x2 in terms of array broadcasting.
        /// </summary>
        /// <param name="x2">
        ///     The arrays to be added.<br></br>
        ///     If x1.shape != x2.shape, they must be
        ///     broadcastable to a common shape (which may be the shape of one or
        ///     the other).
        /// </param>
        /// <param name="x1">
        ///     The arrays to be added.<br></br>
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
        ///     The sum of x1 and x2, element-wise.<br></br>
        ///     This is a scalar if both x1 and x2 are scalars.
        /// </returns>
        public static NDarray add(this NDarray x2, NDarray x1, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.add(x2.ToCupyNDarray, x1.ToCupyNDarray, @out?.ToCupyNDarray, where?.ToCupyNDarray));
            }
            else
            {
                return new NDarray(np.add(x2.ToNumpyNDarray, x1.ToNumpyNDarray, @out?.ToNumpyNDarray, where?.ToNumpyNDarray));
            }
        }

        public static void add_at(this NDarray a, NDarray[] indeces, NDarray b)
        {
            if (Gpu.Available && Gpu.Use)
            {
                throw new NotSupportedException();
            }
            else
            {
                using dynamic np = Py.Import("numpy");
                using dynamic add = np.add;
                using var pyargs = ToTuple(new object[]
                {
                    a.ToNumpyNDarray.PyObject,
                    indeces.Select(x => x.ToNumpyNDarray.PyObject).ToArray(),
                    b.ToNumpyNDarray.PyObject
                });
                using var kwargs = new PyDict();
                using dynamic py = add.InvokeMethod("at", pyargs, kwargs);
            }
        }

        public static void scatter_add(this NDarray a, NDarray[] slices, NDarray b)
        {
            if (Gpu.Available && Gpu.Use)
            {
                using var __self__ = Py.Import("cupyx");
                using var pyargs = ToTuple(new object[]
                {
                    a.ToCupyNDarray.PyObject,
                    slices.Select(x => x.CupyNDarray.PyObject).ToArray(),
                    b.ToCupyNDarray.PyObject
                });
                using var kwargs = new PyDict();
                using dynamic py = __self__.InvokeMethod("scatter_add", pyargs, kwargs);
            }
            else
            {
                using var __self__ = Py.Import("numpy");
                using var pyargs = ToTuple(new object[]
                {
                    a.ToNumpyNDarray.PyObject,
                    slices.Select(x => x.NumpyNDarray.PyObject).ToArray(),
                    b.ToNumpyNDarray.PyObject
                });
                using var kwargs = new PyDict();
                using dynamic py = __self__.InvokeMethod("scatter_add", pyargs, kwargs);
            }
        }

        private static Cupy.NDarray ToNdarray(Cupy.Models.Slice slice, int len)
        {
            var start = slice.Start ?? 0;
            var stop = slice.Stop ?? (len - 1);
            var step = slice.Step;
            if (start <= stop)
            {
                return cp.array(Enumerable.Range(start, stop + 1).Select((num, Index) => new { Number = num, Group = Index })
                    .GroupBy(x => x.Group)
                    .SelectMany(g => g.Skip(step - 1).Take(1)).Select(x => x.Number).ToArray()).astype(cp.int32);
            }
            else
            {
                var arr1 = Enumerable.Range(0, stop + 1).Select((num, Index) => new {Number = num, Group = Index})
                    .GroupBy(x => x.Group)
                    .SelectMany(g => g.Skip(step - 1).Take(1)).Select(x => x.Number).ToArray();
                var arr2 = Enumerable.Range(start, len - start).Select((num, Index) => new { Number = num, Group = Index })
                    .GroupBy(x => x.Group)
                    .SelectMany(g => g.Skip(step - 1).Take(1)).Select(x => x.Number).ToArray();
                return cp.concatenate([cp.array(arr2), cp.array(arr1)]).astype(cp.int32);
            }
        }

        private static Numpy.NDarray ToNdarray(Numpy.Models.Slice slice, int len)
        {
            var start = slice.Start ?? 0;
            var stop = slice.Stop ?? (len - 1);
            var step = slice.Step;
            if (start <= stop)
            {
                return np.array(Enumerable.Range(start, stop + 1).Select((num, Index) => new { Number = num, Group = Index })
                    .GroupBy(x => x.Group)
                    .SelectMany(g => g.Skip(step - 1).Take(1)).Select(x => x.Number).ToArray()).astype(np.int32);
            }
            else
            {
                var arr1 = Enumerable.Range(0, stop + 1).Select((num, Index) => new { Number = num, Group = Index })
                    .GroupBy(x => x.Group)
                    .SelectMany(g => g.Skip(step - 1).Take(1)).Select(x => x.Number).ToArray();
                var arr2 = Enumerable.Range(start, len - start).Select((num, Index) => new { Number = num, Group = Index })
                    .GroupBy(x => x.Group)
                    .SelectMany(g => g.Skip(step - 1).Take(1)).Select(x => x.Number).ToArray();
                return np.concatenate([np.array(arr2), np.array(arr1)]).astype(np.int32);
            }
        }

        /// <summary>
        ///     Return the reciprocal of the argument, element-wise.<br></br>
        ///     Calculates 1/x.<br></br>
        ///     Notes
        ///     For integer arguments with absolute value larger than 1 the result is
        ///     always zero because of the way Python handles integer division.<br></br>
        ///     For
        ///     integer zero the result is an overflow.
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
        ///     Return array.<br></br>
        ///     This is a scalar if x is a scalar.
        /// </returns>
        public static NDarray reciprocal(this NDarray x, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.reciprocal(x.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.reciprocal(x.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Numerical positive, element-wise.<br></br>
        ///     Notes
        ///     Equivalent to x.copy(), but only defined for types that support
        ///     arithmetic.
        /// </summary>
        /// <param name="x">
        ///     Input array.
        /// </param>
        /// <returns>
        ///     Returned array or scalar: y = +x.<br></br>
        ///     This is a scalar if x is a scalar.
        /// </returns>
        public static NDarray positive(this NDarray x)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.positive(x.CupyNDarray));
            }
            else
            {
                return new NDarray(np.positive(x.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Numerical negative, element-wise.
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
        ///     Returned array or scalar: y = -x.<br></br>
        ///     This is a scalar if x is a scalar.
        /// </returns>
        public static NDarray negative(this NDarray x, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.negative(x.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.negative(x.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Multiply arguments element-wise.<br></br>
        ///     Notes
        ///     Equivalent to x1 * x2 in terms of array broadcasting.
        /// </summary>
        /// <param name="x2">
        ///     Input arrays to be multiplied.
        /// </param>
        /// <param name="x1">
        ///     Input arrays to be multiplied.
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
        ///     The product of x1 and x2, element-wise.<br></br>
        ///     This is a scalar if both x1 and x2 are scalars.
        /// </returns>
        public static NDarray multiply(this NDarray x2, NDarray x1, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.multiply(x2.ToCupyNDarray, x1.ToCupyNDarray, @out?.ToCupyNDarray,
                    where?.ToCupyNDarray));
            }
            else
            {
                return new NDarray(np.multiply(x2.ToNumpyNDarray, x1.ToNumpyNDarray, @out?.ToNumpyNDarray,
                    where?.ToNumpyNDarray));
            }
        }

        /// <summary>
        ///     Returns a true division of the inputs, element-wise.<br></br>
        ///     Instead of the Python traditional ‘floor division’, this returns a true
        ///     division.<br></br>
        ///     True division adjusts the output type to present the best
        ///     answer, regardless of input types.<br></br>
        ///     Notes
        ///     The floor division operator // was added in Python 2.2 making
        ///     // and / equivalent operators.<br></br>
        ///     The default floor division
        ///     operation of / can be replaced by true division with from
        ///     __future__ import division.<br></br>
        ///     In Python 3.0, // is the floor division operator and / the
        ///     true division operator.<br></br>
        ///     The true_divide(x1, x2) function is
        ///     equivalent to true division in Python.
        /// </summary>
        /// <param name="x1">
        ///     Dividend array.
        /// </param>
        /// <param name="x2">
        ///     Divisor array.
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
        ///     This is a scalar if both x1 and x2 are scalars.
        /// </returns>
        public static NDarray divide(this NDarray x1, NDarray x2, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.divide(x1.ToCupyNDarray, x2.ToCupyNDarray, @out?.ToCupyNDarray,
                    where?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.divide(x1.ToNumpyNDarray, x2.ToNumpyNDarray, @out?.ToNumpyNDarray,
                    where?.ToNumpyNDarray));
            }
        }

        /// <summary>
        ///     First array elements raised to powers from second array, element-wise.<br></br>
        ///     Raise each base in x1 to the positionally-corresponding power in
        ///     x2.  x1 and x2 must be broadcastable to the same shape.<br></br>
        ///     Note that an
        ///     integer type raised to a negative integer power will raise a ValueError.
        /// </summary>
        /// <param name="x1">
        ///     The bases.
        /// </param>
        /// <param name="x2">
        ///     The exponents.
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
        ///     The bases in x1 raised to the exponents in x2.
        ///     This is a scalar if both x1 and x2 are scalars.
        /// </returns>
        public static NDarray power(this NDarray x1, NDarray x2, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.power(x1.ToCupyNDarray, x2.ToCupyNDarray, @out?.ToCupyNDarray, where?.ToCupyNDarray));
            }
            else
            {
                return new NDarray(np.power(x1.ToNumpyNDarray, x2.ToNumpyNDarray, @out?.ToNumpyNDarray, where?.ToNumpyNDarray));
            }
        }

        /// <summary>
        ///     Subtract arguments, element-wise.<br></br>
        ///     Notes
        ///     Equivalent to x1 - x2 in terms of array broadcasting.
        /// </summary>
        /// <param name="x2">
        ///     The arrays to be subtracted from each other.
        /// </param>
        /// <param name="x1">
        ///     The arrays to be subtracted from each other.
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
        ///     The difference of x1 and x2, element-wise.<br></br>
        ///     This is a scalar if both x1 and x2 are scalars.
        /// </returns>
        public static NDarray subtract(this NDarray x2, NDarray x1, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.subtract(x2.ToCupyNDarray, x1.ToCupyNDarray, @out?.ToCupyNDarray, where?.ToCupyNDarray));
            }
            else
            {
                return new NDarray(np.subtract(x2.ToNumpyNDarray, x1.ToNumpyNDarray, @out?.ToNumpyNDarray,
                        where?.ToNumpyNDarray));
            }
        }

        /// <summary>
        ///     Returns a true division of the inputs, element-wise.<br></br>
        ///     Instead of the Python traditional ‘floor division’, this returns a true
        ///     division.<br></br>
        ///     True division adjusts the output type to present the best
        ///     answer, regardless of input types.<br></br>
        ///     Notes
        ///     The floor division operator // was added in Python 2.2 making
        ///     // and / equivalent operators.<br></br>
        ///     The default floor division
        ///     operation of / can be replaced by true division with from
        ///     __future__ import division.<br></br>
        ///     In Python 3.0, // is the floor division operator and / the
        ///     true division operator.<br></br>
        ///     The true_divide(x1, x2) function is
        ///     equivalent to true division in Python.
        /// </summary>
        /// <param name="x1">
        ///     Dividend array.
        /// </param>
        /// <param name="x2">
        ///     Divisor array.
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
        ///     This is a scalar if both x1 and x2 are scalars.
        /// </returns>
        public static NDarray true_divide(this NDarray x1, NDarray x2, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(
                    cp.true_divide(x1.CupyNDarray, x2.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.true_divide(x1.NumpyNDarray, x2.NumpyNDarray, @out?.NumpyNDarray,
                    where?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Return the largest integer smaller or equal to the division of the inputs.<br></br>
        ///     It is equivalent to the Python // operator and pairs with the
        ///     Python % (remainder), function so that b = a % b + b * (a // b)
        ///     up to roundoff.
        /// </summary>
        /// <param name="x1">
        ///     Numerator.
        /// </param>
        /// <param name="x2">
        ///     Denominator.
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
        ///     y = floor(x1/x2)
        ///     This is a scalar if both x1 and x2 are scalars.
        /// </returns>
        public static NDarray floor_divide(this NDarray x1, NDarray x2, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.floor_divide(x1.CupyNDarray, x2.CupyNDarray, @out?.CupyNDarray,
                    where?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.floor_divide(x1.NumpyNDarray, x2.NumpyNDarray, @out?.NumpyNDarray,
                    where?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     First array elements raised to powers from second array, element-wise.<br></br>
        ///     Raise each base in x1 to the positionally-corresponding power in x2.
        ///     x1 and x2 must be broadcastable to the same shape.<br></br>
        ///     This differs from
        ///     the power function in that integers, float16, and float32  are promoted to
        ///     floats with a minimum precision of float64 so that the result is always
        ///     inexact.<br></br>
        ///     The intent is that the function will return a usable result for
        ///     negative powers and seldom overflow for positive powers.
        /// </summary>
        /// <param name="x1">
        ///     The bases.
        /// </param>
        /// <param name="x2">
        ///     The exponents.
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
        ///     The bases in x1 raised to the exponents in x2.
        ///     This is a scalar if both x1 and x2 are scalars.
        /// </returns>
        public static NDarray float_power(this NDarray x1, NDarray x2, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(
                    cp.float_power(x1.CupyNDarray, x2.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.float_power(x1.NumpyNDarray, x2.NumpyNDarray, @out?.NumpyNDarray,
                    where?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Return the element-wise remainder of division.<br></br>
        ///     This is the Cupy implementation of the C library function fmod, the
        ///     remainder has the same sign as the dividend x1. It is equivalent to
        ///     the Matlab(TM) rem function and should not be confused with the
        ///     Python modulus operator x1 % x2.
        ///     Notes
        ///     The result of the modulo operation for negative dividend and divisors
        ///     is bound by conventions.<br></br>
        ///     For fmod, the sign of result is the sign of
        ///     the dividend, while for remainder the sign of the result is the sign
        ///     of the divisor.<br></br>
        ///     The fmod function is equivalent to the Matlab(TM)
        ///     rem function.
        /// </summary>
        /// <param name="x1">
        ///     Dividend.
        /// </param>
        /// <param name="x2">
        ///     Divisor.
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
        ///     The remainder of the division of x1 by x2.
        ///     This is a scalar if both x1 and x2 are scalars.
        /// </returns>
        public static NDarray fmod(this NDarray x1, NDarray x2, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(
                    cp.fmod(x1.CupyNDarray, x2.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.fmod(x1.NumpyNDarray, x2.NumpyNDarray, @out?.NumpyNDarray,
                    where?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Return element-wise remainder of division.<br></br>
        ///     Computes the remainder complementary to the floor_divide function.<br></br>
        ///     It is
        ///     equivalent to the Python modulus operator``x1 % x2`` and has the same sign
        ///     as the divisor x2. The MATLAB function equivalent to cp.remainder
        ///     is mod.<br></br>
        ///     Notes
        ///     Returns 0 when x2 is 0 and both x1 and x2 are (arrays of)
        ///     integers.
        /// </summary>
        /// <param name="x1">
        ///     Dividend array.
        /// </param>
        /// <param name="x2">
        ///     Divisor array.
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
        ///     The element-wise remainder of the quotient floor_divide(x1, x2).<br></br>
        ///     This is a scalar if both x1 and x2 are scalars.
        /// </returns>
        public static NDarray mod(this NDarray x1, NDarray x2, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(
                    cp.mod(x1.CupyNDarray, x2.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.mod(x1.NumpyNDarray, x2.NumpyNDarray, @out?.NumpyNDarray,
                    where?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Return the fractional and integral parts of an array, element-wise.<br></br>
        ///     The fractional and integral parts are negative if the given number is
        ///     negative.<br></br>
        ///     Notes
        ///     For integer input the return values are floats.
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
        ///     A tuple of:
        ///     y1
        ///     Fractional part of x.
        ///     This is a scalar if x is a scalar.
        ///     y2
        ///     Integral part of x.
        ///     This is a scalar if x is a scalar.
        /// </returns>
        public static (NDarray, NDarray) modf(this NDarray x, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                var ret = cp.modf(x.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2));
            }
            else
            {
                var ret = np.modf(x.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2));
            }
        }

        /// <summary>
        ///     Return element-wise remainder of division.<br></br>
        ///     Computes the remainder complementary to the floor_divide function.<br></br>
        ///     It is
        ///     equivalent to the Python modulus operator``x1 % x2`` and has the same sign
        ///     as the divisor x2. The MATLAB function equivalent to cp.remainder
        ///     is mod.<br></br>
        ///     Notes
        ///     Returns 0 when x2 is 0 and both x1 and x2 are (arrays of)
        ///     integers.
        /// </summary>
        /// <param name="x1">
        ///     Dividend array.
        /// </param>
        /// <param name="x2">
        ///     Divisor array.
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
        ///     The element-wise remainder of the quotient floor_divide(x1, x2).<br></br>
        ///     This is a scalar if both x1 and x2 are scalars.
        /// </returns>
        public static NDarray remainder(this NDarray x1, NDarray x2, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(
                    cp.remainder(x1.CupyNDarray, x2.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.remainder(x1.NumpyNDarray, x2.NumpyNDarray, @out?.NumpyNDarray,
                    where?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Return element-wise quotient and remainder simultaneously.<br></br>
        ///     cp.divmod(x, y) is equivalent to (x // y, x % y), but faster
        ///     because it avoids redundant work.<br></br>
        ///     It is used to implement the Python
        ///     built-in function divmod on Cupy arrays.
        /// </summary>
        /// <param name="x1">
        ///     Dividend array.
        /// </param>
        /// <param name="x2">
        ///     Divisor array.
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
        ///     A tuple of:
        ///     out1
        ///     Element-wise quotient resulting from floor division.
        ///     This is a scalar if both x1 and x2 are scalars.
        ///     out2
        ///     Element-wise remainder from floor division.
        ///     This is a scalar if both x1 and x2 are scalars.
        /// </returns>
        public static (NDarray, NDarray) divmod(this NDarray x1, NDarray x2, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                var ret = cp.divmod(x1.CupyNDarray, x2.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2));
            }
            else
            {
                var ret = np.divmod(x1.NumpyNDarray, x2.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2));
            }
        }

        /// <summary>
        ///     Return the angle of the complex argument.
        /// </summary>
        /// <param name="z">
        ///     A complex number or sequence of complex numbers.
        /// </param>
        /// <param name="deg">
        ///     Return angle in degrees if True, radians if False (default).
        /// </param>
        /// <returns>
        ///     The counterclockwise angle from the positive real axis on
        ///     the complex plane, with dtype as Cupy.float64.
        /// </returns>
        public static NDarray angle(this NDarray z, bool? deg = false)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.angle(z.CupyNDarray, deg));
            }
            else
            {
                return new NDarray(np.angle(z.NumpyNDarray, deg));
            }
        }

        /// <summary>
        ///     Return the real part of the complex argument.
        /// </summary>
        /// <param name="val">
        ///     Input array.
        /// </param>
        /// <returns>
        ///     The real component of the complex argument.<br></br>
        ///     If val is real, the type
        ///     of val is used for the output.<br></br>
        ///     If val has complex elements, the
        ///     returned type is float.
        /// </returns>
        public static NDarray real(this NDarray val)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.real(val.CupyNDarray));
            }
            else
            {
                return new NDarray(np.real(val.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Return the imaginary part of the complex argument.
        /// </summary>
        /// <param name="val">
        ///     Input array.
        /// </param>
        /// <returns>
        ///     The imaginary component of the complex argument.<br></br>
        ///     If val is real,
        ///     the type of val is used for the output.<br></br>
        ///     If val has complex
        ///     elements, the returned type is float.
        /// </returns>
        public static NDarray imag(this NDarray val)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.imag(val.CupyNDarray));
            }
            else
            {
                return new NDarray(np.imag(val.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Return the complex conjugate, element-wise.<br></br>
        ///     The complex conjugate of a complex number is obtained by changing the
        ///     sign of its imaginary part.
        /// </summary>
        /// <param name="x">
        ///     Input value.
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
        ///     The complex conjugate of x, with same dtype as y.<br></br>
        ///     This is a scalar if x is a scalar.
        /// </returns>
        public static NDarray conj(this NDarray x, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.conj(x.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.conj(x.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Returns the discrete, linear convolution of two one-dimensional sequences.<br></br>
        ///     The convolution operator is often seen in signal processing, where it
        ///     models the effect of a linear time-invariant system on a signal [1].<br></br>
        ///     In
        ///     probability theory, the sum of two independent random variables is
        ///     distributed according to the convolution of their individual
        ///     distributions.<br></br>
        ///     If v is longer than a, the arrays are swapped before computation.<br></br>
        ///     Notes
        ///     The discrete convolution operation is defined as
        ///     It can be shown that a convolution  in time/space
        ///     is equivalent to the multiplication  in the Fourier
        ///     domain, after appropriate padding (padding is necessary to prevent
        ///     circular convolution).<br></br>
        ///     Since multiplication is more efficient (faster)
        ///     than convolution, the function scipy.signal.fftconvolve exploits the
        ///     FFT to calculate the convolution of large data-sets.<br></br>
        ///     References
        /// </summary>
        /// <param name="a">
        ///     First one-dimensional input array.
        /// </param>
        /// <param name="v">
        ///     Second one-dimensional input array.
        /// </param>
        /// <returns>
        ///     Discrete, linear convolution of a and v.
        /// </returns>
        public static NDarray convolve(this NDarray a, NDarray v, string mode = "full")
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.convolve(a.CupyNDarray, v.CupyNDarray, mode));
            }
            else
            {
                return new NDarray(np.convolve(a.NumpyNDarray, v.NumpyNDarray, mode));
            }
        }

        /// <summary>
        ///     Clip (limit) the values in an array.<br></br>
        ///     Given an interval, values outside the interval are clipped to
        ///     the interval edges.<br></br>
        ///     For example, if an interval of [0, 1]
        ///     is specified, values smaller than 0 become 0, and values larger
        ///     than 1 become 1.
        /// </summary>
        /// <param name="a">
        ///     Array containing elements to clip.
        /// </param>
        /// <param name="a_min">
        ///     Minimum value.<br></br>
        ///     If None, clipping is not performed on lower
        ///     interval edge.<br></br>
        ///     Not more than one of a_min and a_max may be
        ///     None.
        /// </param>
        /// <param name="a_max">
        ///     Maximum value.<br></br>
        ///     If None, clipping is not performed on upper
        ///     interval edge.<br></br>
        ///     Not more than one of a_min and a_max may be
        ///     None.<br></br>
        ///     If a_min or a_max are array_like, then the three
        ///     arrays will be broadcasted to match their shapes.
        /// </param>
        /// <param name="out">
        ///     The results will be placed in this array.<br></br>
        ///     It may be the input
        ///     array for in-place clipping.<br></br>
        ///     out must be of the right shape
        ///     to hold the output.<br></br>
        ///     Its type is preserved.
        /// </param>
        /// <returns>
        ///     An array with the elements of a, but where values
        ///     &lt; a_min are replaced with a_min, and those &gt; a_max
        ///     with a_max.
        /// </returns>
        public static NDarray clip(this NDarray a, NDarray a_min, NDarray a_max, NDarray @out = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.clip(a.CupyNDarray, a_min.CupyNDarray, a_max.CupyNDarray, @out?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.clip(a.NumpyNDarray, a_min.NumpyNDarray, a_max.NumpyNDarray, @out?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Return the non-negative square-root of an array, element-wise.<br></br>
        ///     Notes
        ///     sqrt has–consistent with common convention–as its branch cut the
        ///     real “interval” [-inf, 0), and is continuous from above on it.<br></br>
        ///     A branch cut is a curve in the complex plane across which a given
        ///     complex function fails to be continuous.
        /// </summary>
        /// <param name="x">
        ///     The values whose square-roots are required.
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
        ///     An array of the same shape as x, containing the positive
        ///     square-root of each element in x.<br></br>
        ///     If any element in x is
        ///     complex, a complex array is returned (and the square-roots of
        ///     negative reals are calculated).<br></br>
        ///     If all of the elements in x
        ///     are real, so is y, with negative elements returning nan.<br></br>
        ///     If out was provided, y is a reference to it.<br></br>
        ///     This is a scalar if x is a scalar.
        /// </returns>
        public static NDarray sqrt(this NDarray x, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.sqrt(x.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.sqrt(x.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Return the cube-root of an array, element-wise.
        /// </summary>
        /// <param name="x">
        ///     The values whose cube-roots are required.
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
        ///     An array of the same shape as x, containing the cube
        ///     cube-root of each element in x.<br></br>
        ///     If out was provided, y is a reference to it.<br></br>
        ///     This is a scalar if x is a scalar.
        /// </returns>
        public static NDarray cbrt(this NDarray x, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.cbrt(x.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.cbrt(x.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Return the element-wise square of the input.
        /// </summary>
        /// <param name="x">
        ///     Input data.
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
        ///     Element-wise x*x, of the same shape and dtype as x.<br></br>
        ///     This is a scalar if x is a scalar.
        /// </returns>
        public static NDarray square(this NDarray x, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.square(x.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.square(x.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Calculate the absolute value element-wise.<br></br>
        ///     cp.abs is a shorthand for this function.
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
        ///     An ndarray containing the absolute value of
        ///     each element in x.<br></br>
        ///     For complex input, a + ib, the
        ///     absolute value is .
        ///     This is a scalar if x is a scalar.
        /// </returns>
        public static NDarray absolute(this NDarray x, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.absolute(x.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.absolute(x.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Compute the absolute values element-wise.<br></br>
        ///     This function returns the absolute values (positive magnitude) of the
        ///     data in x.<br></br>
        ///     Complex values are not handled, use absolute to find the
        ///     absolute values of complex data.
        /// </summary>
        /// <param name="x">
        ///     The array of numbers for which the absolute values are required.<br></br>
        ///     If
        ///     x is a scalar, the result y will also be a scalar.
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
        ///     The absolute values of x, the returned values are always floats.<br></br>
        ///     This is a scalar if x is a scalar.
        /// </returns>
        public static NDarray fabs(this NDarray x, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.fabs(x.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.fabs(x.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Returns an element-wise indication of the sign of a number.<br></br>
        ///     The sign function returns -1 if x &lt; 0, 0 if x==0, 1 if x &gt; 0.<br></br>
        ///     nan
        ///     is returned for nan inputs.<br></br>
        ///     For complex inputs, the sign function returns
        ///     sign(x.real) + 0j if x.real != 0 else sign(x.imag) + 0j.<br></br>
        ///     complex(nan, 0) is returned for complex nan inputs.<br></br>
        ///     Notes
        ///     There is more than one definition of sign in common use for complex
        ///     numbers.<br></br>
        ///     The definition used here is equivalent to
        ///     which is different from a common alternative, .
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
        ///     The sign of x.<br></br>
        ///     This is a scalar if x is a scalar.
        /// </returns>
        public static NDarray sign(this NDarray x, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.sign(x.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.sign(x.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Compute the Heaviside step function.<br></br>
        ///     The Heaviside step function is defined as:
        ///     where x2 is often taken to be 0.5, but 0 and 1 are also sometimes used.<br></br>
        ///     Notes
        ///     References
        /// </summary>
        /// <param name="x1">
        ///     Input values.
        /// </param>
        /// <param name="x2">
        ///     The value of the function when x1 is 0.
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
        ///     The output array, element-wise Heaviside step function of x1.
        ///     This is a scalar if both x1 and x2 are scalars.
        /// </returns>
        public static NDarray heaviside(this NDarray x1, NDarray x2, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.heaviside(x1.CupyNDarray, x2.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.heaviside(x1.NumpyNDarray, x2.NumpyNDarray, @out?.NumpyNDarray,
                    where?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Element-wise maximum of array elements.<br></br>
        ///     Compare two arrays and returns a new array containing the element-wise
        ///     maxima.<br></br>
        ///     If one of the elements being compared is a NaN, then that
        ///     element is returned.<br></br>
        ///     If both elements are NaNs then the first is
        ///     returned.<br></br>
        ///     The latter distinction is important for complex NaNs, which
        ///     are defined as at least one of the real or imaginary parts being a NaN.<br></br>
        ///     The net effect is that NaNs are propagated.<br></br>
        ///     Notes
        ///     The maximum is equivalent to cp.where(x1 &gt;= x2, x1, x2) when
        ///     neither x1 nor x2 are nans, but it is faster and does proper
        ///     broadcasting.
        /// </summary>
        /// <param name="x2">
        ///     The arrays holding the elements to be compared.<br></br>
        ///     They must have
        ///     the same shape, or shapes that can be broadcast to a single shape.
        /// </param>
        /// <param name="x1">
        ///     The arrays holding the elements to be compared.<br></br>
        ///     They must have
        ///     the same shape, or shapes that can be broadcast to a single shape.
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
        ///     The maximum of x1 and x2, element-wise.<br></br>
        ///     This is a scalar if both x1 and x2 are scalars.
        /// </returns>
        public static NDarray maximum(this NDarray x2, NDarray x1, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.maximum(x2.CupyNDarray, x1.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            }
            else
            {
                return new NDarray(
                    np.maximum(x2.NumpyNDarray, x1.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Element-wise minimum of array elements.<br></br>
        ///     Compare two arrays and returns a new array containing the element-wise
        ///     minima.<br></br>
        ///     If one of the elements being compared is a NaN, then that
        ///     element is returned.<br></br>
        ///     If both elements are NaNs then the first is
        ///     returned.<br></br>
        ///     The latter distinction is important for complex NaNs, which
        ///     are defined as at least one of the real or imaginary parts being a NaN.<br></br>
        ///     The net effect is that NaNs are propagated.<br></br>
        ///     Notes
        ///     The minimum is equivalent to cp.where(x1 &lt;= x2, x1, x2) when
        ///     neither x1 nor x2 are NaNs, but it is faster and does proper
        ///     broadcasting.
        /// </summary>
        /// <param name="x2">
        ///     The arrays holding the elements to be compared.<br></br>
        ///     They must have
        ///     the same shape, or shapes that can be broadcast to a single shape.
        /// </param>
        /// <param name="x1">
        ///     The arrays holding the elements to be compared.<br></br>
        ///     They must have
        ///     the same shape, or shapes that can be broadcast to a single shape.
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
        ///     The minimum of x1 and x2, element-wise.<br></br>
        ///     This is a scalar if both x1 and x2 are scalars.
        /// </returns>
        public static NDarray minimum(this NDarray x2, NDarray x1, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.minimum(x2.CupyNDarray, x1.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            }
            else
            {
                return new NDarray(
                    np.minimum(x2.NumpyNDarray, x1.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Element-wise maximum of array elements.<br></br>
        ///     Compare two arrays and returns a new array containing the element-wise
        ///     maxima.<br></br>
        ///     If one of the elements being compared is a NaN, then the
        ///     non-nan element is returned.<br></br>
        ///     If both elements are NaNs then the first
        ///     is returned.<br></br>
        ///     The latter distinction is important for complex NaNs,
        ///     which are defined as at least one of the real or imaginary parts being
        ///     a NaN.<br></br>
        ///     The net effect is that NaNs are ignored when possible.<br></br>
        ///     Notes
        ///     The fmax is equivalent to cp.where(x1 &gt;= x2, x1, x2) when neither
        ///     x1 nor x2 are NaNs, but it is faster and does proper broadcasting.
        /// </summary>
        /// <param name="x2">
        ///     The arrays holding the elements to be compared.<br></br>
        ///     They must have
        ///     the same shape.
        /// </param>
        /// <param name="x1">
        ///     The arrays holding the elements to be compared.<br></br>
        ///     They must have
        ///     the same shape.
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
        ///     The maximum of x1 and x2, element-wise.<br></br>
        ///     This is a scalar if both x1 and x2 are scalars.
        /// </returns>
        public static NDarray fmax(this NDarray x2, NDarray x1, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.fmax(x2.CupyNDarray, x1.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.fmax(x2.NumpyNDarray, x1.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Element-wise minimum of array elements.<br></br>
        ///     Compare two arrays and returns a new array containing the element-wise
        ///     minima.<br></br>
        ///     If one of the elements being compared is a NaN, then the
        ///     non-nan element is returned.<br></br>
        ///     If both elements are NaNs then the first
        ///     is returned.<br></br>
        ///     The latter distinction is important for complex NaNs,
        ///     which are defined as at least one of the real or imaginary parts being
        ///     a NaN.<br></br>
        ///     The net effect is that NaNs are ignored when possible.<br></br>
        ///     Notes
        ///     The fmin is equivalent to cp.where(x1 &lt;= x2, x1, x2) when neither
        ///     x1 nor x2 are NaNs, but it is faster and does proper broadcasting.
        /// </summary>
        /// <param name="x2">
        ///     The arrays holding the elements to be compared.<br></br>
        ///     They must have
        ///     the same shape.
        /// </param>
        /// <param name="x1">
        ///     The arrays holding the elements to be compared.<br></br>
        ///     They must have
        ///     the same shape.
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
        ///     The minimum of x1 and x2, element-wise.<br></br>
        ///     This is a scalar if both x1 and x2 are scalars.
        /// </returns>
        public static NDarray fmin(this NDarray x2, NDarray x1, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.fmin(x2.CupyNDarray, x1.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.fmin(x2.NumpyNDarray, x1.NumpyNDarray, @out?.NumpyNDarray, where?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Replace NaN with zero and infinity with large finite numbers.<br></br>
        ///     If x is inexact, NaN is replaced by zero, and infinity and -infinity
        ///     replaced by the respectively largest and most negative finite floating
        ///     point values representable by x.dtype.<br></br>
        ///     For complex dtypes, the above is applied to each of the real and
        ///     imaginary components of x separately.<br></br>
        ///     If x is not inexact, then no replacements are made.<br></br>
        ///     Notes
        ///     Cupy uses the IEEE Standard for Binary Floating-Point for Arithmetic
        ///     (IEEE 754).<br></br>
        ///     This means that Not a Number is not equivalent to infinity.
        /// </summary>
        /// <param name="x">
        ///     Input data.
        /// </param>
        /// <param name="copy">
        ///     Whether to create a copy of x (True) or to replace values
        ///     in-place (False).<br></br>
        ///     The in-place operation only occurs if
        ///     casting to an array does not require a copy.<br></br>
        ///     Default is True.
        /// </param>
        /// <returns>
        ///     x, with the non-finite values replaced.<br></br>
        ///     If copy is False, this may
        ///     be x itself.
        /// </returns>
        public static NDarray nan_to_num(this NDarray x, bool? copy = true)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.nan_to_num(x.CupyNDarray, copy));
            }
            else
            {
                return new NDarray(np.nan_to_num(x.NumpyNDarray, copy));
            }
        }

        /// <summary>
        ///     If complex input returns a real array if complex parts are close to zero.<br></br>
        ///     “Close to zero” is defined as tol * (machine epsilon of the type for
        ///     a).<br></br>
        ///     Notes
        ///     Machine epsilon varies from machine to machine and between data types
        ///     but Python floats on most platforms have a machine epsilon equal to
        ///     2.2204460492503131e-16.  You can use ‘cp.finfo(float).eps’ to print
        ///     out the machine epsilon for floats.
        /// </summary>
        /// <param name="a">
        ///     Input array.
        /// </param>
        /// <param name="tol">
        ///     Tolerance in machine epsilons for the complex part of the elements
        ///     in the array.
        /// </param>
        /// <returns>
        ///     If a is real, the type of a is used for the output.<br></br>
        ///     If a
        ///     has complex elements, the returned type is float.
        /// </returns>
        public static NDarray real_if_close(this NDarray a, float tol = 100)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.real_if_close(a.CupyNDarray, tol));
            }
            else
            {
                return new NDarray(np.real_if_close(a.NumpyNDarray, tol));
            }
        }

        /*
        /// <summary>
        ///	One-dimensional linear interpolation.<br></br>
        ///	
        ///	Returns the one-dimensional piecewise linear interpolant to a function
        ///	with given discrete data points (xp, fp), evaluated at x.<br></br>
        ///	
        ///	Notes
        ///	
        ///	Does not check that the x-coordinate sequence xp is increasing.<br></br>
        ///	
        ///	If xp is not increasing, the results are nonsense.<br></br>
        ///	
        ///	A simple check for increasing is:
        /// </summary>
        /// <param name="x">
        ///	The x-coordinates at which to evaluate the interpolated values.
        /// </param>
        /// <param name="xp">
        ///	The x-coordinates of the data points, must be increasing if argument
        ///	period is not specified.<br></br>
        ///	Otherwise, xp is internally sorted after
        ///	normalizing the periodic boundaries with xp = xp % period.
        /// </param>
        /// <param name="fp">
        ///	The y-coordinates of the data points, same length as xp.
        /// </param>
        /// <param name="left">
        ///	Value to return for x &lt; xp[0], default is fp[0].
        /// </param>
        /// <param name="right">
        ///	Value to return for x &gt; xp[-1], default is fp[-1].
        /// </param>
        /// <param name="period">
        ///	A period for the x-coordinates.<br></br>
        ///	This parameter allows the proper
        ///	interpolation of angular x-coordinates.<br></br>
        ///	Parameters left and right
        ///	are ignored if period is specified.
        /// </param>
        /// <returns>
        ///	The interpolated values, same shape as x.
        /// </returns>
        public static float or complex (corresponding to fp) or ndarray interp(this NDarray x, 1-D sequence of floats xp, 1-D sequence of float or complex fp, optional float or complex corresponding to fp left = null, optional float or complex corresponding to fp right = null, None or float period = null)
        {
            //auto-generated code, do not change
            var __self__=self;
            var pyargs=ToTuple(new object[]
            {
                x,
                xp,
                fp,
            });
            var kwargs=new PyDict();
            if (left!=null) kwargs["left"]=ToPython(left);
            if (right!=null) kwargs["right"]=ToPython(right);
            if (period!=null) kwargs["period"]=ToPython(period);
            dynamic py = __self__.InvokeMethod("interp", pyargs, kwargs);
            return ToCsharp<float or complex (corresponding to fp) or ndarray>(py);
        }
        */
    }
}
