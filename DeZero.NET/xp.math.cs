using Cupy;
using Numpy;

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
        ///     If axis is given, the number of varargs must equal the number of axes.<br></br>
        ///     Default: 1.
        /// </param>
        /// <param name="edge_order">
        ///     Gradient is calculated using N-th order accurate differences
        ///     at the boundaries.<br></br>
        ///     Default: 1.
        /// </param>
        /// <param name="axis">
        ///     Gradient is calculated only along the given axis or axes
        ///     The default (axis = None) is to calculate the gradient for all the axes
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
            if (Core.GpuAvailable && Core.UseGpu)
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
        ///     If axis is given, the number of varargs must equal the number of axes.<br></br>
        ///     Default: 1.
        /// </param>
        /// <param name="edge_order">
        ///     Gradient is calculated using N-th order accurate differences
        ///     at the boundaries.<br></br>
        ///     Default: 1.
        /// </param>
        /// <param name="axis">
        ///     Gradient is calculated only along the given axis or axes
        ///     The default (axis = None) is to calculate the gradient for all the axes
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
            if (Core.GpuAvailable && Core.UseGpu)
            {
                return new NDarray(cp.gradient(f.CupyNDarray, varargs, edge_order, axis?.CupyAxis));
            }
            else
            {
                return new NDarray(np.gradient(f.NumpyNDarray, varargs, edge_order, axis?.NumpyAxis));
            }
        }
    }
}
