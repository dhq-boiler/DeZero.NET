using Cupy;
using Numpy;

namespace DeZero.NET
{
    public static partial class xp
    {
        /// <summary>
        ///     Copies values from one array to another, broadcasting as necessary.<br></br>
        ///     Raises a TypeError if the casting rule is violated, and if
        ///     where is provided, it selects which elements to copy.
        /// </summary>
        /// <param name="dst">
        ///     The array into which values are copied.
        /// </param>
        /// <param name="src">
        ///     The array from which values are copied.
        /// </param>
        /// <param name="casting">
        ///     Controls what kind of data casting may occur when copying.
        /// </param>
        /// <param name="where">
        ///     A boolean array which is broadcasted to match the dimensions
        ///     of dst, and selects elements to copy from src to dst
        ///     wherever it contains the value True.
        /// </param>
        public static void copyto(NDarray dst, NDarray src, string casting = "same_kind", NDarray where = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                cp.copyto(dst.CupyNDarray, src.CupyNDarray, casting, where?.CupyNDarray);
            }
            else
            {
                np.copyto(dst.NumpyNDarray, src.NumpyNDarray, casting, where?.NumpyNDarray);
            }
        }

        /// <summary>
        ///     Copies values from one array to another, broadcasting as necessary.<br></br>
        ///     Raises a TypeError if the casting rule is violated, and if
        ///     where is provided, it selects which elements to copy.
        /// </summary>
        /// <param name="dst">
        ///     The array into which values are copied.
        /// </param>
        /// <param name="src">
        ///     The array from which values are copied.
        /// </param>
        /// <param name="casting">
        ///     Controls what kind of data casting may occur when copying.
        /// </param>
        /// <param name="where">
        ///     A boolean array which is broadcasted to match the dimensions
        ///     of dst, and selects elements to copy from src to dst
        ///     wherever it contains the value True.
        /// </param>
        public static void copyto(NDarray dst, NDarray src, string casting = "same_kind", bool[] where = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                cp.copyto(dst.CupyNDarray, src.CupyNDarray, casting, where);
            }
            else
            {
                np.copyto(dst.NumpyNDarray, src.NumpyNDarray, casting, where);
            }
        }

        /// <summary>
        ///     Gives a new shape to an array without changing its data.<br></br>
        ///     Notes
        ///     It is not always possible to change the shape of an array without
        ///     copying the data.<br></br>
        ///     If you want an error to be raised when the data is copied,
        ///     you should assign the new shape to the shape attribute of the array:
        ///     The order keyword gives the index ordering both for fetching the values
        ///     from a, and then placing the values into the output array.<br></br>
        ///     For example, let’s say you have an array:
        ///     You can think of reshaping as first raveling the array (using the given
        ///     index order), then inserting the elements from the raveled array into the
        ///     new array using the same kind of index ordering as was used for the
        ///     raveling.
        /// </summary>
        /// <param name="a">
        ///     Array to be reshaped.
        /// </param>
        /// <param name="newshape">
        ///     The new shape should be compatible with the original shape.<br></br>
        ///     If
        ///     an integer, then the result will be a 1-D array of that length.<br></br>
        ///     One shape dimension can be -1. In this case, the value is
        ///     inferred from the length of the array and remaining dimensions.
        /// </param>
        /// <param name="order">
        ///     Read the elements of a using this index order, and place the
        ///     elements into the reshaped array using this index order.<br></br>
        ///     ‘C’
        ///     means to read / write the elements using C-like index order,
        ///     with the last axis index changing fastest, back to the first
        ///     axis index changing slowest.<br></br>
        ///     ‘F’ means to read / write the
        ///     elements using Fortran-like index order, with the first index
        ///     changing fastest, and the last index changing slowest.<br></br>
        ///     Note that
        ///     the ‘C’ and ‘F’ options take no account of the memory layout of
        ///     the underlying array, and only refer to the order of indexing.<br></br>
        ///     ‘A’ means to read / write the elements in Fortran-like index
        ///     order if a is Fortran contiguous in memory, C-like order
        ///     otherwise.
        /// </param>
        /// <returns>
        ///     This will be a new view object if possible; otherwise, it will
        ///     be a copy.<br></br>
        ///     Note there is no guarantee of the memory layout (C- or
        ///     Fortran- contiguous) of the returned array.
        /// </returns>
        public static NDarray reshape(this NDarray a, Shape newshape, string order = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                return new NDarray(cp.reshape(a.CupyNDarray, newshape.CupyShape, order));
            }
            else
            {
                return new NDarray(np.reshape(a.NumpyNDarray, newshape.NumpyShape, order));
            }
        }

        /// <summary>
        ///     Return a contiguous flattened array.<br></br>
        ///     A 1-D array, containing the elements of the input, is returned.<br></br>
        ///     A copy is
        ///     made only if needed.<br></br>
        ///     As of Cupy 1.10, the returned array will have the same type as the input
        ///     array.<br></br>
        ///     (for example, a masked array will be returned for a masked array
        ///     input)
        ///     Notes
        ///     In row-major, C-style order, in two dimensions, the row index
        ///     varies the slowest, and the column index the quickest.<br></br>
        ///     This can
        ///     be generalized to multiple dimensions, where row-major order
        ///     implies that the index along the first axis varies slowest, and
        ///     the index along the last quickest.<br></br>
        ///     The opposite holds for
        ///     column-major, Fortran-style index ordering.<br></br>
        ///     When a view is desired in as many cases as possible, arr.reshape(-1)
        ///     may be preferable.
        /// </summary>
        /// <param name="a">
        ///     Input array.<br></br>
        ///     The elements in a are read in the order specified by
        ///     order, and packed as a 1-D array.
        /// </param>
        /// <param name="order">
        ///     The elements of a are read using this index order.<br></br>
        ///     ‘C’ means
        ///     to index the elements in row-major, C-style order,
        ///     with the last axis index changing fastest, back to the first
        ///     axis index changing slowest.<br></br>
        ///     ‘F’ means to index the elements
        ///     in column-major, Fortran-style order, with the
        ///     first index changing fastest, and the last index changing
        ///     slowest.<br></br>
        ///     Note that the ‘C’ and ‘F’ options take no account of
        ///     the memory layout of the underlying array, and only refer to
        ///     the order of axis indexing.<br></br>
        ///     ‘A’ means to read the elements in
        ///     Fortran-like index order if a is Fortran contiguous in
        ///     memory, C-like order otherwise.<br></br>
        ///     ‘K’ means to read the
        ///     elements in the order they occur in memory, except for
        ///     reversing the data when strides are negative.<br></br>
        ///     By default, ‘C’
        ///     index order is used.
        /// </param>
        /// <returns>
        ///     y is an array of the same subtype as a, with shape (a.size,).<br></br>
        ///     Note that matrices are special cased for backward compatibility, if a
        ///     is a matrix, then y is a 1-D ndarray.
        /// </returns>
        public static NDarray ravel(this NDarray a, string order = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                return new NDarray(cp.ravel(a.CupyNDarray, order));
            }
            else
            {
                return new NDarray(np.ravel(a.NumpyNDarray, order));
            }
        }

        /// <summary>
        ///     Return a copy of the array collapsed into one dimension.
        /// </summary>
        /// <param name="order">
        ///     ‘C’ means to flatten in row-major (C-style) order.<br></br>
        ///     ‘F’ means to flatten in column-major (Fortran-
        ///     style) order.<br></br>
        ///     ‘A’ means to flatten in column-major
        ///     order if a is Fortran contiguous in memory,
        ///     row-major order otherwise.<br></br>
        ///     ‘K’ means to flatten
        ///     a in the order the elements occur in memory.<br></br>
        ///     The default is ‘C’.
        /// </param>
        /// <returns>
        ///     A copy of the input array, flattened to one dimension.
        /// </returns>
        public static NDarray flatten(string order = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                return new NDarray(cp.flatten(order));
            }
            else
            {
                return new NDarray(np.flatten(order));
            }
        }

        /// <summary>
        ///     Move axes of an array to new positions.<br></br>
        ///     Other axes remain in their original order.
        /// </summary>
        /// <param name="a">
        ///     The array whose axes should be reordered.
        /// </param>
        /// <param name="source">
        ///     Original positions of the axes to move.<br></br>
        ///     These must be unique.
        /// </param>
        /// <param name="destination">
        ///     Destination positions for each of the original axes.<br></br>
        ///     These must also be
        ///     unique.
        /// </param>
        /// <returns>
        ///     Array with moved axes.<br></br>
        ///     This array is a view of the input array.
        /// </returns>
        public static NDarray moveaxis(this NDarray a, int[] source, int[] destination)
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                return new NDarray(cp.moveaxis(a.CupyNDarray, source, destination));
            }
            else
            {
                return new NDarray(np.moveaxis(a.NumpyNDarray, source, destination));
            }
        }

        /// <summary>
        ///     Roll the specified axis backwards, until it lies in a given position.<br></br>
        ///     This function continues to be supported for backward compatibility, but you
        ///     should prefer moveaxis.<br></br>
        ///     The moveaxis function was added in Cupy
        ///     1.11.
        /// </summary>
        /// <param name="a">
        ///     Input array.
        /// </param>
        /// <param name="axis">
        ///     The axis to roll backwards.<br></br>
        ///     The positions of the other axes do not
        ///     change relative to one another.
        /// </param>
        /// <param name="start">
        ///     The axis is rolled until it lies before this position.<br></br>
        ///     The default,
        ///     0, results in a “complete” roll.
        /// </param>
        /// <returns>
        ///     For Cupy &gt;= 1.10.0 a view of a is always returned.<br></br>
        ///     For earlier
        ///     Cupy versions a view of a is returned only if the order of the
        ///     axes is changed, otherwise the input array is returned.
        /// </returns>
        public static NDarray rollaxis(this NDarray a, int axis, int? start = 0)
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                return new NDarray(cp.rollaxis(a.CupyNDarray, axis, start));
            }
            else
            {
                return new NDarray(np.rollaxis(a.NumpyNDarray, axis, start));
            }
        }

        /// <summary>
        ///     Interchange two axes of an array.
        /// </summary>
        /// <param name="a">
        ///     Input array.
        /// </param>
        /// <param name="axis1">
        ///     First axis.
        /// </param>
        /// <param name="axis2">
        ///     Second axis.
        /// </param>
        /// <returns>
        ///     For Cupy &gt;= 1.10.0, if a is an ndarray, then a view of a is
        ///     returned; otherwise a new array is created.<br></br>
        ///     For earlier Cupy
        ///     versions a view of a is returned only if the order of the
        ///     axes is changed, otherwise the input array is returned.
        /// </returns>
        public static NDarray swapaxes(this NDarray a, int axis1, int axis2)
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                return new NDarray(cp.swapaxes(a.CupyNDarray, axis1, axis2));
            }
            else
            {
                return new NDarray(np.swapaxes(a.NumpyNDarray, axis1, axis2));
            }
        }

        /// <summary>
        ///     Permute the dimensions of an array.<br></br>
        ///     Notes
        ///     Use transpose(a, argsort(axes)) to invert the transposition of tensors
        ///     when using the axes keyword argument.<br></br>
        ///     Transposing a 1-D array returns an unchanged view of the original array.
        /// </summary>
        /// <param name="a">
        ///     Input array.
        /// </param>
        /// <param name="axes">
        ///     By default, reverse the dimensions, otherwise permute the axes
        ///     according to the values given.
        /// </param>
        /// <returns>
        ///     a with its axes permuted.<br></br>
        ///     A view is returned whenever
        ///     possible.
        /// </returns>
        public static NDarray transpose(this NDarray a, int[] axes = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                return new NDarray(cp.transpose(a.CupyNDarray, axes));
            }
            else
            {
                return new NDarray(np.transpose(a.NumpyNDarray, axes));
            }
        }

        /// <summary>
        ///     Permute the dimensions of an array.<br></br>
        ///     Notes
        ///     Use transpose(a, argsort(axes)) to invert the transposition of tensors
        ///     when using the axes keyword argument.<br></br>
        ///     Transposing a 1-D array returns an unchanged view of the original array.
        /// </summary>
        /// <param name="a">
        ///     Input array.
        /// </param>
        /// <param name="axes">
        ///     By default, reverse the dimensions, otherwise permute the axes
        ///     according to the values given.
        /// </param>
        /// <returns>
        ///     a with its axes permuted.<br></br>
        ///     A view is returned whenever
        ///     possible.
        /// </returns>
        public static NDarray transpose(this NDarray[] a, int[] axes = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                return new NDarray(cp.transpose(a.Select(x => x.CupyNDarray).ToArray(), axes));
            }
            else
            {
                return new NDarray(np.transpose(a.Select(x => x.NumpyNDarray).ToArray(), axes));
            }
        }

        /// <summary>
        ///     Convert inputs to arrays with at least one dimension.<br></br>
        ///     Scalar inputs are converted to 1-dimensional arrays, whilst
        ///     higher-dimensional inputs are preserved.
        /// </summary>
        /// <param name="arys">
        ///     One or more input arrays.
        /// </param>
        /// <returns>
        ///     An array, or list of arrays, each with a.ndim &gt;= 1.<br></br>
        ///     Copies are made only if necessary.
        /// </returns>
        public static NDarray atleast_1d(params NDarray[] arys)
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                return new NDarray(cp.atleast_1d(arys.Select(x => x.CupyNDarray).ToArray()));
            }
            else
            {
                return new NDarray(np.atleast_1d(arys.Select(x => x.NumpyNDarray).ToArray()));
            }
        }

        /// <summary>
        ///     View inputs as arrays with at least two dimensions.
        /// </summary>
        /// <param name="arys">
        ///     One or more array-like sequences.<br></br>
        ///     Non-array inputs are converted
        ///     to arrays.<br></br>
        ///     Arrays that already have two or more dimensions are
        ///     preserved.
        /// </param>
        /// <returns>
        ///     An array, or list of arrays, each with a.ndim &gt;= 2.<br></br>
        ///     Copies are avoided where possible, and views with two or more
        ///     dimensions are returned.
        /// </returns>
        public static NDarray atleast_2d(params NDarray[] arys)
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                return new NDarray(cp.atleast_2d(arys.Select(x => x.CupyNDarray).ToArray()));
            }
            else
            {
                return new NDarray(np.atleast_2d(arys.Select(x => x.NumpyNDarray).ToArray()));
            }
        }

        /// <summary>
        ///     View inputs as arrays with at least three dimensions.
        /// </summary>
        /// <param name="arys">
        ///     One or more array-like sequences.<br></br>
        ///     Non-array inputs are converted to
        ///     arrays.<br></br>
        ///     Arrays that already have three or more dimensions are
        ///     preserved.
        /// </param>
        /// <returns>
        ///     An array, or list of arrays, each with a.ndim &gt;= 3.<br></br>
        ///     Copies are
        ///     avoided where possible, and views with three or more dimensions are
        ///     returned.<br></br>
        ///     For example, a 1-D array of shape (N,) becomes a view
        ///     of shape (1, N, 1), and a 2-D array of shape (M, N) becomes a
        ///     view of shape (M, N, 1).
        /// </returns>
        public static NDarray atleast_3d(params NDarray[] arys)
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                return new NDarray(cp.atleast_3d(arys.Select(x => x.CupyNDarray).ToArray()));
            }
            else
            {
                return new NDarray(np.atleast_3d(arys.Select(x => x.NumpyNDarray).ToArray()));
            }
        }

        /// <summary>
        ///     Produce an object that mimics broadcasting.
        /// </summary>
        /// <param name="in2">
        ///     Input parameters.
        /// </param>
        /// <param name="in1">
        ///     Input parameters.
        /// </param>
        /// <returns>
        ///     Broadcast the input parameters against one another, and
        ///     return an object that encapsulates the result.<br></br>
        ///     Amongst others, it has shape and nd properties, and
        ///     may be used as an iterator.
        /// </returns>
        public static NDarray broadcast(this NDarray in2, NDarray in1)
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                return new NDarray(cp.broadcast(in2.CupyNDarray, in1.CupyNDarray));
            }
            else
            {
                return new NDarray(np.broadcast(in2.NumpyNDarray, in1.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Broadcast an array to a new shape.<br></br>
        ///     Notes
        /// </summary>
        /// <param name="array">
        ///     The array to broadcast.
        /// </param>
        /// <param name="shape">
        ///     The shape of the desired array.
        /// </param>
        /// <param name="subok">
        ///     If True, then sub-classes will be passed-through, otherwise
        ///     the returned array will be forced to be a base-class array (default).
        /// </param>
        /// <returns>
        ///     A readonly view on the original array with the given shape.<br></br>
        ///     It is
        ///     typically not contiguous.<br></br>
        ///     Furthermore, more than one element of a
        ///     broadcasted array may refer to a single memory location.
        /// </returns>
        public static NDarray broadcast_to(this NDarray array, Shape shape, bool? subok = false)
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                return new NDarray(cp.broadcast_to(array.CupyNDarray, shape.CupyShape, subok));
            }
            else
            {
                return new NDarray(np.broadcast_to(array.NumpyNDarray, shape.NumpyShape, subok));
            }
        }

        /// <summary>
        ///     Broadcast any number of arrays against each other.
        /// </summary>
        /// <param name="args">
        ///     The arrays to broadcast.
        /// </param>
        /// <param name="subok">
        ///     If True, then sub-classes will be passed-through, otherwise
        ///     the returned arrays will be forced to be a base-class array (default).
        /// </param>
        /// <returns>
        ///     These arrays are views on the original arrays.<br></br>
        ///     They are typically
        ///     not contiguous.<br></br>
        ///     Furthermore, more than one element of a
        ///     broadcasted array may refer to a single memory location.<br></br>
        ///     If you
        ///     need to write to the arrays, make copies first.
        /// </returns>
        public static NDarray[] broadcast_arrays(NDarray[] args, bool? subok = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                return cp.broadcast_arrays(args.Select(x => x.CupyNDarray).ToArray(), subok).Select(x => new NDarray(x)).ToArray();
            }
            else
            {
                return np.broadcast_arrays(args.Select(x => x.NumpyNDarray).ToArray(), subok).Select(x => new NDarray(x)).ToArray();
            }
        }

        /// <summary>
        ///     Expand the shape of an array.<br></br>
        ///     Insert a new axis that will appear at the axis position in the expanded
        ///     array shape.
        /// </summary>
        /// <param name="a">
        ///     Input array.
        /// </param>
        /// <param name="axis">
        ///     Position in the expanded axes where the new axis is placed.
        /// </param>
        /// <returns>
        ///     Output array.<br></br>
        ///     The number of dimensions is one greater than that of
        ///     the input array.
        /// </returns>
        public static NDarray expand_dims(this NDarray a, int axis)
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                return new NDarray(cp.expand_dims(a.CupyNDarray, axis));
            }
            else
            {
                return new NDarray(np.expand_dims(a.NumpyNDarray, axis));
            }
        }

        /// <summary>
        ///     Remove single-dimensional entries from the shape of an array.
        /// </summary>
        /// <param name="a">
        ///     Input data.
        /// </param>
        /// <param name="axis">
        ///     Selects a subset of the single-dimensional entries in the
        ///     shape.<br></br>
        ///     If an axis is selected with shape entry greater than
        ///     one, an error is raised.
        /// </param>
        /// <returns>
        ///     The input array, but with all or a subset of the
        ///     dimensions of length 1 removed.<br></br>
        ///     This is always a itself
        ///     or a view into a.
        /// </returns>
        public static NDarray squeeze(this NDarray a, Axis axis = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                return new NDarray(cp.squeeze(a.CupyNDarray, axis?.CupyAxis));
            }
            else
            {
                return new NDarray(np.squeeze(a.NumpyNDarray, axis?.NumpyAxis));
            }
        }

        /// <summary>
        ///     Return an array converted to a float type.
        /// </summary>
        /// <param name="a">
        ///     The input array.
        /// </param>
        /// <param name="dtype">
        ///     Float type code to coerce input array a.<br></br>
        ///     If dtype is one of the
        ///     ‘int’ dtypes, it is replaced with float64.
        /// </param>
        /// <returns>
        ///     The input a as a float ndarray.
        /// </returns>
        public static NDarray asfarray(this NDarray a, Dtype dtype = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                return new NDarray(cp.asfarray(a.CupyNDarray, dtype?.CupyDtype));
            }
            else
            {
                return new NDarray(np.asfarray(a.NumpyNDarray, dtype?.NumpyDtype));
            }
        }

        /// <summary>
        ///     Return an array (ndim &gt;= 1) laid out in Fortran order in memory.
        /// </summary>
        /// <param name="a">
        ///     Input array.
        /// </param>
        /// <param name="dtype">
        ///     By default, the data-type is inferred from the input data.
        /// </param>
        /// <returns>
        ///     The input a in Fortran, or column-major, order.
        /// </returns>
        public static NDarray asfortranarray(this NDarray a, Dtype dtype = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                return new NDarray(cp.asfortranarray(a.CupyNDarray, dtype?.CupyDtype));
            }
            else
            {
                return new NDarray(np.asfortranarray(a.NumpyNDarray, dtype?.NumpyDtype));
            }
        }

        /// <summary>
        ///     Convert the input to an array, checking for NaNs or Infs.
        /// </summary>
        /// <param name="a">
        ///     Input data, in any form that can be converted to an array.<br></br>
        ///     This
        ///     includes lists, lists of tuples, tuples, tuples of tuples, tuples
        ///     of lists and ndarrays.<br></br>
        ///     Success requires no NaNs or Infs.
        /// </param>
        /// <param name="dtype">
        ///     By default, the data-type is inferred from the input data.
        /// </param>
        /// <param name="order">
        ///     Whether to use row-major (C-style) or
        ///     column-major (Fortran-style) memory representation.<br></br>
        ///     Defaults to ‘C’.
        /// </param>
        /// <returns>
        ///     Array interpretation of a.<br></br>
        ///     No copy is performed if the input
        ///     is already an ndarray.<br></br>
        ///     If a is a subclass of ndarray, a base
        ///     class ndarray is returned.
        /// </returns>
        public static NDarray asarray_chkfinite(this NDarray a, Dtype dtype = null, string order = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                return new NDarray(cp.asarray_chkfinite(a.CupyNDarray, dtype?.CupyDtype, order));
            }
            else
            {
                return new NDarray(np.asarray_chkfinite(a.NumpyNDarray, dtype?.NumpyDtype, order));
            }
        }

        /// <summary>
        ///     Return an ndarray of the provided type that satisfies requirements.<br></br>
        ///     This function is useful to be sure that an array with the correct flags
        ///     is returned for passing to compiled code (perhaps through ctypes).<br></br>
        ///     Notes
        ///     The returned array will be guaranteed to have the listed requirements
        ///     by making a copy if needed.
        /// </summary>
        /// <param name="a">
        ///     The object to be converted to a type-and-requirement-satisfying array.
        /// </param>
        /// <param name="dtype">
        ///     The required data-type.<br></br>
        ///     If None preserve the current dtype.<br></br>
        ///     If your
        ///     application requires the data to be in native byteorder, include
        ///     a byteorder specification as a part of the dtype specification.
        /// </param>
        /// <param name="requirements">
        ///     The requirements list can be any of the following
        /// </param>
        public static NDarray require(this NDarray a, Dtype dtype, string[] requirements = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                return new NDarray(cp.require(a.CupyNDarray, dtype.CupyDtype, requirements));
            }
            else
            {
                return new NDarray(np.require(a.NumpyNDarray, dtype.NumpyDtype, requirements));
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
        public static NDarray concatenate(NDarray[] arys, int? axis = 0, NDarray @out = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                return new NDarray(cp.concatenate(arys.Select(x => x.CupyNDarray).ToArray(), axis, @out?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.concatenate(arys.Select(x => x.NumpyNDarray).ToArray(), axis,
                    @out?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Join a sequence of arrays along a new axis.<br></br>
        ///     The axis parameter specifies the index of the new axis in the dimensions
        ///     of the result.<br></br>
        ///     For example, if axis=0 it will be the first dimension
        ///     and if axis=-1 it will be the last dimension.
        /// </summary>
        /// <param name="arrays">
        ///     Each array must have the same shape.
        /// </param>
        /// <param name="axis">
        ///     The axis in the result array along which the input arrays are stacked.
        /// </param>
        /// <param name="out">
        ///     If provided, the destination to place the result.<br></br>
        ///     The shape must be
        ///     correct, matching that of what stack would have returned if no
        ///     out argument were specified.
        /// </param>
        /// <returns>
        ///     The stacked array has one more dimension than the input arrays.
        /// </returns>
        public static NDarray stack(NDarray[] arrays, int? axis = 0, NDarray @out = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                return new NDarray(cp.stack(arrays.Select(x => x.CupyNDarray).ToArray(), axis, @out?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.stack(arrays.Select(x => x.NumpyNDarray).ToArray(), axis,
                    @out?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Stack arrays in sequence depth wise (along third axis).<br></br>
        ///     This is equivalent to concatenation along the third axis after 2-D arrays
        ///     of shape (M,N) have been reshaped to (M,N,1) and 1-D arrays of shape
        ///     (N,) have been reshaped to (1,N,1).<br></br>
        ///     Rebuilds arrays divided by
        ///     dsplit.<br></br>
        ///     This function makes most sense for arrays with up to 3 dimensions.<br></br>
        ///     For
        ///     instance, for pixel-data with a height (first axis), width (second axis),
        ///     and r/g/b channels (third axis).<br></br>
        ///     The functions concatenate, stack and
        ///     block provide more general stacking and concatenation operations.
        /// </summary>
        /// <param name="tup">
        ///     The arrays must have the same shape along all but the third axis.<br></br>
        ///     1-D or 2-D arrays must have the same shape.
        /// </param>
        /// <returns>
        ///     The array formed by stacking the given arrays, will be at least 3-D.
        /// </returns>
        public static NDarray dstack(params NDarray[] tup)
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                return new NDarray(cp.dstack(tup.Select(x => x.CupyNDarray).ToArray()));
            }
            else
            {
                return new NDarray(np.dstack(tup.Select(x => x.NumpyNDarray).ToArray()));
            }
        }

        /// <summary>
        ///     Stack arrays in sequence horizontally (column wise).<br></br>
        ///     This is equivalent to concatenation along the second axis, except for 1-D
        ///     arrays where it concatenates along the first axis.<br></br>
        ///     Rebuilds arrays divided
        ///     by hsplit.<br></br>
        ///     This function makes most sense for arrays with up to 3 dimensions.<br></br>
        ///     For
        ///     instance, for pixel-data with a height (first axis), width (second axis),
        ///     and r/g/b channels (third axis).<br></br>
        ///     The functions concatenate, stack and
        ///     block provide more general stacking and concatenation operations.
        /// </summary>
        /// <param name="tup">
        ///     The arrays must have the same shape along all but the second axis,
        ///     except 1-D arrays which can be any length.
        /// </param>
        /// <returns>
        ///     The array formed by stacking the given arrays.
        /// </returns>
        public static NDarray hstack(params NDarray[] tup)
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                return new NDarray(cp.hstack(tup.Select(x => x.CupyNDarray).ToArray()));
            }
            else
            {
                return new NDarray(np.hstack(tup.Select(x => x.NumpyNDarray).ToArray()));
            }
        }

        /// <summary>
        ///     Stack arrays in sequence vertically (row wise).<br></br>
        ///     This is equivalent to concatenation along the first axis after 1-D arrays
        ///     of shape (N,) have been reshaped to (1,N).<br></br>
        ///     Rebuilds arrays divided by
        ///     vsplit.<br></br>
        ///     This function makes most sense for arrays with up to 3 dimensions.<br></br>
        ///     For
        ///     instance, for pixel-data with a height (first axis), width (second axis),
        ///     and r/g/b channels (third axis).<br></br>
        ///     The functions concatenate, stack and
        ///     block provide more general stacking and concatenation operations.
        /// </summary>
        /// <param name="tup">
        ///     The arrays must have the same shape along all but the first axis.<br></br>
        ///     1-D arrays must have the same length.
        /// </param>
        /// <returns>
        ///     The array formed by stacking the given arrays, will be at least 2-D.
        /// </returns>
        public static NDarray vstack(params NDarray[] tup)
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                return new NDarray(cp.vstack(tup.Select(x => x.CupyNDarray).ToArray()));
            }
            else
            {
                return new NDarray(np.vstack(tup.Select(x => x.NumpyNDarray).ToArray()));
            }
        }

        /*
        /// <summary>
        ///	Assemble an nd-array from nested lists of blocks.<br></br>
        ///	
        ///	Blocks in the innermost lists are concatenated (see concatenate) along
        ///	the last dimension (-1), then these are concatenated along the
        ///	second-last dimension (-2), and so on until the outermost list is reached.<br></br>
        ///	
        ///	Blocks can be of any dimension, but will not be broadcasted using the normal
        ///	rules.<br></br>
        ///	 Instead, leading axes of size 1 are inserted, to make block.ndim
        ///	the same for all blocks.<br></br>
        ///	 This is primarily useful for working with scalars,
        ///	and means that code like cp.block([v, 1]) is valid, where
        ///	v.ndim == 1.<br></br>
        ///	
        ///	When the nested list is two levels deep, this allows block matrices to be
        ///	constructed from their components.<br></br>
        ///	
        ///	Notes
        ///	
        ///	When called with only scalars, cp.block is equivalent to an ndarray
        ///	call.<br></br>
        ///	 So cp.block([[1, 2], [3, 4]]) is equivalent to
        ///	cp.array([[1, 2], [3, 4]]).<br></br>
        ///	
        ///	This function does not enforce that the blocks lie on a fixed grid.<br></br>
        ///	
        ///	cp.block([[a, b], [c, d]]) is not restricted to arrays of the form:
        ///	
        ///	But is also allowed to produce, for some a, b, c, d:
        ///	
        ///	Since concatenation happens along the last axis first, block is _not_
        ///	capable of producing the following directly:
        ///	
        ///	Matlab’s “square bracket stacking”, [A, B, ...; p, q, ...], is
        ///	equivalent to cp.block([[A, B, ...], [p, q, ...]]).
        /// </summary>
        /// <param name="arrays">
        ///	If passed a single ndarray or scalar (a nested list of depth 0), this
        ///	is returned unmodified (and not copied).<br></br>
        ///	
        ///	Elements shapes must match along the appropriate axes (without
        ///	broadcasting), but leading 1s will be prepended to the shape as
        ///	necessary to make the dimensions match.
        /// </param>
        /// <returns>
        ///	The array assembled from the given blocks.<br></br>
        ///	
        ///	The dimensionality of the output is equal to the greatest of:
        ///	* the dimensionality of all the inputs
        ///	* the depth to which the input list is nested
        /// </returns>
        public static NDarray block(nested list of array_like or scalars (but not tuples) arrays)
        {
            //auto-generated code, do not change
            var __self__=self;
            var pyargs=ToTuple(new object[]
            {
                arrays,
            });
            var kwargs=new PyDict();
            dynamic py = __self__.InvokeMethod("block", pyargs, kwargs);
            return ToCsharp<NDarray>(py);
        }
        */

        /// <summary>
        ///     Split an array into multiple sub-arrays.
        /// </summary>
        /// <param name="ary">
        ///     Array to be divided into sub-arrays.
        /// </param>
        /// <param name="indices_or_sections">
        ///     If indices_or_sections is an integer, N, the array will be divided
        ///     into N equal arrays along axis.<br></br>
        ///     If such a split is not possible,
        ///     an error is raised.<br></br>
        ///     If indices_or_sections is a 1-D array of sorted integers, the entries
        ///     indicate where along axis the array is split.<br></br>
        ///     For example,
        ///     [2, 3] would, for axis=0, result in
        ///     If an index exceeds the dimension of the array along axis,
        ///     an empty sub-array is returned correspondingly.
        /// </param>
        /// <param name="axis">
        ///     The axis along which to split, default is 0.
        /// </param>
        /// <returns>
        ///     A list of sub-arrays.
        /// </returns>
        public static NDarray[] split(this NDarray ary, int[] indices_or_sections, int? axis = 0)
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                return cp.split(ary.CupyNDarray, indices_or_sections, axis).Select(x => new NDarray(x)).ToArray();
            }
            else
            {
                return np.split(ary.NumpyNDarray, indices_or_sections, axis).Select(x => new NDarray(x)).ToArray();
            }
        }

        /// <summary>
        ///     Split an array into multiple sub-arrays.
        /// </summary>
        /// <param name="ary">
        ///     Array to be divided into sub-arrays.
        /// </param>
        /// <param name="indices_or_sections">
        ///     If indices_or_sections is an integer, N, the array will be divided
        ///     into N equal arrays along axis.<br></br>
        ///     If such a split is not possible,
        ///     an error is raised.<br></br>
        ///     If indices_or_sections is a 1-D array of sorted integers, the entries
        ///     indicate where along axis the array is split.<br></br>
        ///     For example,
        ///     [2, 3] would, for axis=0, result in
        ///     If an index exceeds the dimension of the array along axis,
        ///     an empty sub-array is returned correspondingly.
        /// </param>
        /// <param name="axis">
        ///     The axis along which to split, default is 0.
        /// </param>
        /// <returns>
        ///     A list of sub-arrays.
        /// </returns>
        public static NDarray[] split(this NDarray ary, int indices_or_sections, int? axis = 0)
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                return cp.split(ary.CupyNDarray, indices_or_sections, axis).Select(x => new NDarray(x)).ToArray();
            }
            else
            {
                return np.split(ary.NumpyNDarray, indices_or_sections, axis).Select(x => new NDarray(x)).ToArray();
            }
        }

        /// <summary>
        ///     Construct an array by repeating A the number of times given by reps.<br></br>
        ///     If reps has length d, the result will have dimension of
        ///     max(d, A.ndim).<br></br>
        ///     If A.ndim &lt; d, A is promoted to be d-dimensional by prepending new
        ///     axes.<br></br>
        ///     So a shape (3,) array is promoted to (1, 3) for 2-D replication,
        ///     or shape (1, 1, 3) for 3-D replication.<br></br>
        ///     If this is not the desired
        ///     behavior, promote A to d-dimensions manually before calling this
        ///     function.<br></br>
        ///     If A.ndim &gt; d, reps is promoted to A.ndim by pre-pending 1’s to it.<br></br>
        ///     Thus for an A of shape (2, 3, 4, 5), a reps of (2, 2) is treated as
        ///     (1, 1, 2, 2).<br></br>
        ///     Note : Although tile may be used for broadcasting, it is strongly
        ///     recommended to use Cupy’s broadcasting operations and functions.
        /// </summary>
        /// <param name="A">
        ///     The input array.
        /// </param>
        /// <param name="reps">
        ///     The number of repetitions of A along each axis.
        /// </param>
        /// <returns>
        ///     The tiled output array.
        /// </returns>
        public static NDarray tile(this NDarray A, NDarray reps)
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                return new NDarray(cp.tile(A.CupyNDarray, reps.CupyNDarray));
            }
            else
            {
                return new NDarray(np.tile(A.NumpyNDarray, reps.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Repeat elements of an array.
        /// </summary>
        /// <param name="a">
        ///     Input array.
        /// </param>
        /// <param name="repeats">
        ///     The number of repetitions for each element.<br></br>
        ///     repeats is broadcasted
        ///     to fit the shape of the given axis.
        /// </param>
        /// <param name="axis">
        ///     The axis along which to repeat values.<br></br>
        ///     By default, use the
        ///     flattened input array, and return a flat output array.
        /// </param>
        /// <returns>
        ///     Output array which has the same shape as a, except along
        ///     the given axis.
        /// </returns>
        public static NDarray repeat(this NDarray a, int[] repeats, int? axis = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                return new NDarray(cp.repeat(a.CupyNDarray, repeats, axis));
            }
            else
            {
                return new NDarray(np.repeat(a.NumpyNDarray, repeats, axis));
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
        public static NDarray delete(this NDarray arr, Slice obj, int? axis = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                return new NDarray(cp.delete(arr.CupyNDarray, obj.CupySlice, axis));
            }
            else
            {
                return new NDarray(np.delete(arr.NumpyNDarray, obj.NumpySlice, axis));
            }
        }

        /// <summary>
        ///     Append values to the end of an array.
        /// </summary>
        /// <param name="arr">
        ///     Values are appended to a copy of this array.
        /// </param>
        /// <param name="values">
        ///     These values are appended to a copy of arr.<br></br>
        ///     It must be of the
        ///     correct shape (the same shape as arr, excluding axis).<br></br>
        ///     If
        ///     axis is not specified, values can be any shape and will be
        ///     flattened before use.
        /// </param>
        /// <param name="axis">
        ///     The axis along which values are appended.<br></br>
        ///     If axis is not
        ///     given, both arr and values are flattened before use.
        /// </param>
        /// <returns>
        ///     A copy of arr with values appended to axis.<br></br>
        ///     Note that
        ///     append does not occur in-place: a new array is allocated and
        ///     filled.<br></br>
        ///     If axis is None, out is a flattened array.
        /// </returns>
        public static NDarray append(this NDarray arr, NDarray values, int? axis = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                return new NDarray(cp.append(arr.CupyNDarray, values.CupyNDarray, axis));
            }
            else
            {
                return new NDarray(np.append(arr.NumpyNDarray, values.NumpyNDarray, axis));
            }
        }

        /// <summary>
        ///     Trim the leading and/or trailing zeros from a 1-D array or sequence.
        /// </summary>
        /// <param name="filt">
        ///     Input array.
        /// </param>
        /// <param name="trim">
        ///     A string with ‘f’ representing trim from front and ‘b’ to trim from
        ///     back.<br></br>
        ///     Default is ‘fb’, trim zeros from both front and back of the
        ///     array.
        /// </param>
        /// <returns>
        ///     The result of trimming the input.<br></br>
        ///     The input data type is preserved.
        /// </returns>
        public static NDarray trim_zeros(this NDarray filt, string trim = "fb")
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                return new NDarray(cp.trim_zeros(filt.CupyNDarray, trim));
            }
            else
            {
                return new NDarray(np.trim_zeros(filt.NumpyNDarray, trim));
            }
        }

        /// <summary>
        ///     Find the unique elements of an array.<br></br>
        ///     Returns the sorted unique elements of an array.<br></br>
        ///     There are three optional
        ///     outputs in addition to the unique elements:
        ///     Notes
        ///     When an axis is specified the subarrays indexed by the axis are sorted.<br></br>
        ///     This is done by making the specified axis the first dimension of the array
        ///     and then flattening the subarrays in C order.<br></br>
        ///     The flattened subarrays are
        ///     then viewed as a structured type with each element given a label, with the
        ///     effect that we end up with a 1-D array of structured types that can be
        ///     treated in the same way as any other 1-D array.<br></br>
        ///     The result is that the
        ///     flattened subarrays are sorted in lexicographic order starting with the
        ///     first element.
        /// </summary>
        /// <param name="ar">
        ///     Input array.<br></br>
        ///     Unless axis is specified, this will be flattened if it
        ///     is not already 1-D.
        /// </param>
        /// <param name="axis">
        ///     The axis to operate on.<br></br>
        ///     If None, ar will be flattened.<br></br>
        ///     If an integer,
        ///     the subarrays indexed by the given axis will be flattened and treated
        ///     as the elements of a 1-D array with the dimension of the given axis,
        ///     see the notes for more details.<br></br>
        ///     Object arrays or structured arrays
        ///     that contain objects are not supported if the axis kwarg is used.<br></br>
        ///     The
        ///     default is None.
        /// </param>
        /// <returns>
        ///     The sorted unique values.
        /// </returns>
        public static NDarray unique(NDarray ar, int? axis = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                return new NDarray(cp.unique(ar.CupyNDarray, axis));
            }
            else
            {
                return new NDarray(np.unique(ar.NumpyNDarray, axis));
            }
        }

        /// <summary>
        ///     Find the unique elements of an array.<br></br>
        ///     Returns the sorted unique elements of an array.<br></br>
        ///     There are three optional
        ///     outputs in addition to the unique elements:
        ///     Notes
        ///     When an axis is specified the subarrays indexed by the axis are sorted.<br></br>
        ///     This is done by making the specified axis the first dimension of the array
        ///     and then flattening the subarrays in C order.<br></br>
        ///     The flattened subarrays are
        ///     then viewed as a structured type with each element given a label, with the
        ///     effect that we end up with a 1-D array of structured types that can be
        ///     treated in the same way as any other 1-D array.<br></br>
        ///     The result is that the
        ///     flattened subarrays are sorted in lexicographic order starting with the
        ///     first element.
        /// </summary>
        /// <param name="ar">
        ///     Input array.<br></br>
        ///     Unless axis is specified, this will be flattened if it
        ///     is not already 1-D.
        /// </param>
        /// <param name="return_index">
        ///     If True, also return the indices of ar (along the specified axis,
        ///     if provided, or in the flattened array) that result in the unique array.
        /// </param>
        /// <param name="return_inverse">
        ///     If True, also return the indices of the unique array (for the specified
        ///     axis, if provided) that can be used to reconstruct ar.
        /// </param>
        /// <param name="return_counts">
        ///     If True, also return the number of times each unique item appears
        ///     in ar.
        /// </param>
        /// <param name="axis">
        ///     The axis to operate on.<br></br>
        ///     If None, ar will be flattened.<br></br>
        ///     If an integer,
        ///     the subarrays indexed by the given axis will be flattened and treated
        ///     as the elements of a 1-D array with the dimension of the given axis,
        ///     see the notes for more details.<br></br>
        ///     Object arrays or structured arrays
        ///     that contain objects are not supported if the axis kwarg is used.<br></br>
        ///     The
        ///     default is None.
        /// </param>
        /// <returns>
        ///     The sorted unique values.
        /// </returns>
        public static NDarray[] unique(this NDarray ar, bool return_index, bool return_inverse, bool return_counts,
            int? axis = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                return cp.unique(ar.CupyNDarray, return_index, return_inverse, return_counts, axis)
                    .Select(x => new NDarray(x)).ToArray();
            }
            else
            {
                return np.unique(ar.NumpyNDarray, return_index, return_inverse, return_counts, axis)
                    .Select(x => new NDarray(x)).ToArray();
            }
        }

        /// <summary>
        ///     Reverse the order of elements in an array along the given axis.<br></br>
        ///     The shape of the array is preserved, but the elements are reordered.<br></br>
        ///     Notes
        ///     flip(m, 0) is equivalent to flipud(m).<br></br>
        ///     flip(m, 1) is equivalent to fliplr(m).<br></br>
        ///     flip(m, n) corresponds to m[...,::-1,...] with ::-1 at position n.<br></br>
        ///     flip(m) corresponds to m[::-1,::-1,...,::-1] with ::-1 at all
        ///     positions.<br></br>
        ///     flip(m, (0, 1)) corresponds to m[::-1,::-1,...] with ::-1 at
        ///     position 0 and position 1.
        /// </summary>
        /// <param name="m">
        ///     Input array.
        /// </param>
        /// <param name="axis">
        ///     Axis or axes along which to flip over.<br></br>
        ///     The default,
        ///     axis=None, will flip over all of the axes of the input array.<br></br>
        ///     If axis is negative it counts from the last to the first axis.<br></br>
        ///     If axis is a tuple of ints, flipping is performed on all of the axes
        ///     specified in the tuple.
        /// </param>
        /// <returns>
        ///     A view of m with the entries of axis reversed.<br></br>
        ///     Since a view is
        ///     returned, this operation is done in constant time.
        /// </returns>
        public static NDarray flip(this NDarray m, Axis axis = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                return new NDarray(cp.flip(m.CupyNDarray, axis?.CupyAxis));
            }
            else
            {
                return new NDarray(np.flip(m.NumpyNDarray, axis?.NumpyAxis));
            }
        }

        /// <summary>
        ///     Flip array in the left/right direction.<br></br>
        ///     Flip the entries in each row in the left/right direction.<br></br>
        ///     Columns are preserved, but appear in a different order than before.<br></br>
        ///     Notes
        ///     Equivalent to m[:,::-1].<br></br>
        ///     Requires the array to be at least 2-D.
        /// </summary>
        /// <param name="m">
        ///     Input array, must be at least 2-D.
        /// </param>
        /// <returns>
        ///     A view of m with the columns reversed.<br></br>
        ///     Since a view
        ///     is returned, this operation is .
        /// </returns>
        public static NDarray fliplr(this NDarray m)
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                return new NDarray(cp.fliplr(m.CupyNDarray));
            }
            else
            {
                return new NDarray(np.fliplr(m.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Flip array in the up/down direction.<br></br>
        ///     Flip the entries in each column in the up/down direction.<br></br>
        ///     Rows are preserved, but appear in a different order than before.<br></br>
        ///     Notes
        ///     Equivalent to m[::-1,...].<br></br>
        ///     Does not require the array to be two-dimensional.
        /// </summary>
        /// <param name="m">
        ///     Input array.
        /// </param>
        /// <returns>
        ///     A view of m with the rows reversed.<br></br>
        ///     Since a view is
        ///     returned, this operation is .
        /// </returns>
        public static NDarray flipud(this NDarray m)
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                return new NDarray(cp.flipud(m.CupyNDarray));
            }
            else
            {
                return new NDarray(np.flipud(m.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Roll array elements along a given axis.<br></br>
        ///     Elements that roll beyond the last position are re-introduced at
        ///     the first.<br></br>
        ///     Notes
        ///     Supports rolling over multiple dimensions simultaneously.
        /// </summary>
        /// <param name="a">
        ///     Input array.
        /// </param>
        /// <param name="shift">
        ///     The number of places by which elements are shifted.<br></br>
        ///     If a tuple,
        ///     then axis must be a tuple of the same size, and each of the
        ///     given axes is shifted by the corresponding number.<br></br>
        ///     If an int
        ///     while axis is a tuple of ints, then the same value is used for
        ///     all given axes.
        /// </param>
        /// <param name="axis">
        ///     Axis or axes along which elements are shifted.<br></br>
        ///     By default, the
        ///     array is flattened before shifting, after which the original
        ///     shape is restored.
        /// </param>
        /// <returns>
        ///     Output array, with the same shape as a.
        /// </returns>
        public static NDarray roll(this NDarray a, int[] shift, Axis axis = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                return new NDarray(cp.roll(a.CupyNDarray, shift, axis?.CupyAxis));
            }
            else
            {
                return new NDarray(np.roll(a.NumpyNDarray, shift, axis?.NumpyAxis));
            }
        }

        /// <summary>
        ///     Rotate an array by 90 degrees in the plane specified by axes.<br></br>
        ///     Rotation direction is from the first towards the second axis.<br></br>
        ///     Notes
        ///     rot90(m, k=1, axes=(1,0)) is the reverse of rot90(m, k=1, axes=(0,1))
        ///     rot90(m, k=1, axes=(1,0)) is equivalent to rot90(m, k=-1, axes=(0,1))
        /// </summary>
        /// <param name="m">
        ///     Array of two or more dimensions.
        /// </param>
        /// <param name="k">
        ///     Number of times the array is rotated by 90 degrees.
        /// </param>
        /// <param name="axes">
        ///     The array is rotated in the plane defined by the axes.<br></br>
        ///     Axes must be different.
        /// </param>
        /// <returns>
        ///     A rotated view of m.
        /// </returns>
        public static NDarray rot90(this NDarray m, int k = 1, int[] axes = null)
        {
            if (Core.GpuAvailable && Core.UseGpu)
            {
                return new NDarray(cp.rot90(m.CupyNDarray, k, axes));
            }
            else
            {
                return new NDarray(np.rot90(m.NumpyNDarray, k, axes));
            }
        }
    }
}
