﻿using Cupy;
using Numpy;

namespace DeZero.NET
{
    public static partial class xp
    {
        /// <summary>
        ///     Translates slice objects to concatenation along the first axis.<br></br>
        ///     This is a simple way to build up arrays quickly.<br></br>
        ///     There are two use cases.<br></br>
        ///     If slice notation is used, the syntax start:stop:step is equivalent
        ///     to cp.arange(start, stop, step) inside of the brackets.<br></br>
        ///     However, if
        ///     step is an imaginary number (i.e.<br></br>
        ///     100j) then its integer portion is
        ///     interpreted as a number-of-points desired and the start and stop are
        ///     inclusive.<br></br>
        ///     In other words start:stop:stepj is interpreted as
        ///     cp.linspace(start, stop, step, endpoint=1) inside of the brackets.<br></br>
        ///     After expansion of slice notation, all comma separated sequences are
        ///     concatenated together.<br></br>
        ///     Optional character strings placed as the first element of the index
        ///     expression can be used to change the output.<br></br>
        ///     The strings ‘r’ or ‘c’ result
        ///     in matrix output.<br></br>
        ///     If the result is 1-D and ‘r’ is specified a 1 x N (row)
        ///     matrix is produced.<br></br>
        ///     If the result is 1-D and ‘c’ is specified, then a N x 1
        ///     (column) matrix is produced.<br></br>
        ///     If the result is 2-D then both provide the
        ///     same matrix result.<br></br>
        ///     A string integer specifies which axis to stack multiple comma separated
        ///     arrays along.<br></br>
        ///     A string of two comma-separated integers allows indication
        ///     of the minimum number of dimensions to force each entry into as the
        ///     second integer (the axis to concatenate along is still the first integer).<br></br>
        ///     A string with three comma-separated integers allows specification of the
        ///     axis to concatenate along, the minimum number of dimensions to force the
        ///     entries to, and which axis should contain the start of the arrays which
        ///     are less than the specified number of dimensions.<br></br>
        ///     In other words the third
        ///     integer allows you to specify where the 1’s should be placed in the shape
        ///     of the arrays that have their shapes upgraded.<br></br>
        ///     By default, they are placed
        ///     in the front of the shape tuple.<br></br>
        ///     The third argument allows you to specify
        ///     where the start of the array should be instead.<br></br>
        ///     Thus, a third argument of
        ///     ‘0’ would place the 1’s at the end of the array shape.<br></br>
        ///     Negative integers
        ///     specify where in the new shape tuple the last dimension of upgraded arrays
        ///     should be placed, so the default is ‘-1’.
        /// </summary>
        public static void r_()
        {
            if (Gpu.Available && Gpu.Use)
            {
                cp.r_();
            }
            else
            {
                np.r_();
            }
        }

        /// <summary>
        ///     A nicer way to build up index tuples for arrays.<br></br>
        ///     For any index combination, including slicing and axis insertion,
        ///     a[indices] is the same as a[cp.index_exp[indices]] for any
        ///     array a.<br></br>
        ///     However, cp.index_exp[indices] can be used anywhere
        ///     in Python code and returns a tuple of slice objects that can be
        ///     used in the construction of complex index expressions.<br></br>
        ///     Notes
        ///     You can do all this with slice() plus a few special objects,
        ///     but there’s a lot to remember and this version is simpler because
        ///     it uses the standard array indexing syntax.
        /// </summary>
        /// <param name="maketuple">
        ///     If True, always returns a tuple.
        /// </param>
        public static void s_(bool maketuple)
        {
            if (Gpu.Available && Gpu.Use)
            {
                cp.s_(maketuple);
            }
            else
            {
                np.s_(maketuple);
            }
        }

        /// <summary>
        ///     Return the indices of the elements that are non-zero.<br></br>
        ///     Returns a tuple of arrays, one for each dimension of a,
        ///     containing the indices of the non-zero elements in that
        ///     dimension.<br></br>
        ///     The values in a are always tested and returned in
        ///     row-major, C-style order.<br></br>
        ///     The corresponding non-zero
        ///     values can be obtained with:
        ///     To group the indices by element, rather than dimension, use:
        ///     The result of this is always a 2-D array, with a row for
        ///     each non-zero element.
        /// </summary>
        /// <param name="a">
        ///     Input array.
        /// </param>
        /// <returns>
        ///     Indices of elements that are non-zero.
        /// </returns>
        public static NDarray[] nonzero(this NDarray a)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return cp.nonzero(a.CupyNDarray).Select(x => new NDarray(x)).ToArray();
            }
            else
            {
                return np.nonzero(a.NumpyNDarray).Select(x => new NDarray(x)).ToArray();
            }
        }

        /// <summary>
        ///     Return elements chosen from x or y depending on condition.<br></br>
        ///     Notes
        ///     If all the arrays are 1-D, where is equivalent to:
        /// </summary>
        /// <param name="condition">
        ///     Where True, yield x, otherwise yield y.
        /// </param>
        /// <param name="y">
        ///     Values from which to choose.<br></br>
        ///     x, y and condition need to be
        ///     broadcastable to some shape.
        /// </param>
        /// <param name="x">
        ///     Values from which to choose.<br></br>
        ///     x, y and condition need to be
        ///     broadcastable to some shape.
        /// </param>
        /// <returns>
        ///     An array with elements from x where condition is True, and elements
        ///     from y elsewhere.
        /// </returns>
        public static NDarray where(this NDarray condition, NDarray y, NDarray x)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.where(condition.CupyNDarray, y.CupyNDarray, x.CupyNDarray));
            }
            else
            {
                return new NDarray(np.where(condition.NumpyNDarray, y.NumpyNDarray, x.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Return elements chosen from x or y depending on condition.<br></br>
        ///     Notes
        ///     If all the arrays are 1-D, where is equivalent to:
        /// </summary>
        /// <param name="condition">
        ///     Where True, yield x, otherwise yield y.
        /// </param>
        /// <returns>
        ///     An array with elements from x where condition is True, and elements
        ///     from y elsewhere.
        /// </returns>
        public static NDarray[] where(this NDarray condition)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return cp.where(condition.CupyNDarray).Select(x => new NDarray(x)).ToArray();
            }
            else
            {
                return np.where(condition.NumpyNDarray).Select(x => new NDarray(x)).ToArray();
            }
        }

        /// <summary>
        ///     Return an array representing the indices of a grid.<br></br>
        ///     Compute an array where the subarrays contain index values 0,1,…
        ///     varying only along the corresponding axis.<br></br>
        ///     Notes
        ///     The output shape is obtained by prepending the number of dimensions
        ///     in front of the tuple of dimensions, i.e.<br></br>
        ///     if dimensions is a tuple
        ///     (r0, ..., rN-1) of length N, the output shape is
        ///     (N,r0,...,rN-1).<br></br>
        ///     The subarrays grid[k] contains the N-D array of indices along the
        ///     k-th axis.<br></br>
        ///     Explicitly:
        /// </summary>
        /// <param name="dimensions">
        ///     The shape of the grid.
        /// </param>
        /// <param name="dtype">
        ///     Data type of the result.
        /// </param>
        /// <returns>
        ///     The array of grid indices,
        ///     grid.shape = (len(dimensions),) + tuple(dimensions).
        /// </returns>
        public static NDarray indices(int[] dimensions, Dtype dtype = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.indices(dimensions, dtype?.CupyDtype));
            }
            else
            {
                return new NDarray(np.indices(dimensions, dtype?.NumpyDtype));
            }
        }

        /// <summary>
        ///     Construct an open mesh from multiple sequences.<br></br>
        ///     This function takes N 1-D sequences and returns N outputs with N
        ///     dimensions each, such that the shape is 1 in all but one dimension
        ///     and the dimension with the non-unit shape value cycles through all
        ///     N dimensions.<br></br>
        ///     Using ix_ one can quickly construct index arrays that will index
        ///     the cross product.<br></br>
        ///     a[cp.ix_([1,3],[2,5])] returns the array
        ///     [[a[1,2] a[1,5]], [a[3,2] a[3,5]]].
        /// </summary>
        /// <param name="args">
        ///     Each sequence should be of integer or boolean type.<br></br>
        ///     Boolean sequences will be interpreted as boolean masks for the
        ///     corresponding dimension (equivalent to passing in
        ///     cp.nonzero(boolean_sequence)).
        /// </param>
        /// <returns>
        ///     N arrays with N dimensions each, with N the number of input
        ///     sequences.<br></br>
        ///     Together these arrays form an open mesh.
        /// </returns>
        public static NDarray[] ix_(params NDarray[] args)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return cp.ix_(args.Select(x => x.CupyNDarray).ToArray()).Select(x => new NDarray(x)).ToArray();
            }
            else
            {
                return np.ix_(args.Select(x => x.NumpyNDarray).ToArray()).Select(x => new NDarray(x)).ToArray();
            }
        }

        /*
        /// <summary>
        ///	Converts a tuple of index arrays into an array of flat
        ///	indices, applying boundary modes to the multi-index.<br></br>
        ///	
        ///	Notes
        /// </summary>
        /// <param name="multi_index">
        ///	A tuple of integer arrays, one array for each dimension.
        /// </param>
        /// <param name="dims">
        ///	The shape of array into which the indices from multi_index apply.
        /// </param>
        /// <param name="mode">
        ///	Specifies how out-of-bounds indices are handled.<br></br>
        ///	Can specify
        ///	either one mode or a tuple of modes, one mode per index.<br></br>
        ///	
        ///	In ‘clip’ mode, a negative index which would normally
        ///	wrap will clip to 0 instead.
        /// </param>
        /// <param name="order">
        ///	Determines whether the multi-index should be viewed as
        ///	indexing in row-major (C-style) or column-major
        ///	(Fortran-style) order.
        /// </param>
        /// <returns>
        ///	An array of indices into the flattened version of an array
        ///	of dimensions dims.
        /// </returns>
        public static NDarray ravel_multi_index(tuple of array_like multi_index, tuple of ints dims, string mode = "raise", string order = null)
        {
            //auto-generated code, do not change
            var __self__=self;
            var pyargs=ToTuple(new object[]
            {
                multi_index,
                dims,
            });
            var kwargs=new PyDict();
            if (mode!="raise") kwargs["mode"]=ToPython(mode);
            if (order!=null) kwargs["order"]=ToPython(order);
            dynamic py = __self__.InvokeMethod("ravel_multi_index", pyargs, kwargs);
            return ToCsharp<NDarray>(py);
        }
        */

        /// <summary>
        ///     Converts a flat index or array of flat indices into a tuple
        ///     of coordinate arrays.
        /// </summary>
        /// <param name="indices">
        ///     An integer array whose elements are indices into the flattened
        ///     version of an array of dimensions shape.<br></br>
        ///     Before version 1.6.0,
        ///     this function accepted just one index value.
        /// </param>
        /// <param name="shape">
        ///     The shape of the array to use for unraveling indices.
        /// </param>
        /// <param name="order">
        ///     Determines whether the indices should be viewed as indexing in
        ///     row-major (C-style) or column-major (Fortran-style) order.
        /// </param>
        /// <returns>
        ///     Each array in the tuple has the same shape as the indices
        ///     array.
        /// </returns>
        public static NDarray[] unravel_index(this NDarray indices, Shape shape, string order = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return cp.unravel_index(indices.CupyNDarray, shape.CupyShape, order).Select(x => new NDarray(x))
                    .ToArray();
            }
            else
            {
                return np.unravel_index(indices.NumpyNDarray, shape.NumpyShape, order).Select(x => new NDarray(x))
                    .ToArray();
            }
        }

        /// <summary>
        ///     Return the indices to access the main diagonal of an array.<br></br>
        ///     This returns a tuple of indices that can be used to access the main
        ///     diagonal of an array a with a.ndim &gt;= 2 dimensions and shape
        ///     (n, n, …, n).<br></br>
        ///     For a.ndim = 2 this is the usual diagonal, for
        ///     a.ndim &gt; 2 this is the set of indices to access a[i, i, ..., i]
        ///     for i = [0..n-1].<br></br>
        ///     Notes
        /// </summary>
        /// <param name="n">
        ///     The size, along each dimension, of the arrays for which the returned
        ///     indices can be used.
        /// </param>
        /// <param name="ndim">
        ///     The number of dimensions.
        /// </param>
        public static void diag_indices(int n, int? ndim = 2)
        {
            if (Gpu.Available && Gpu.Use)
            {
                cp.diag_indices(n, ndim);
            }
            else
            {
                np.diag_indices(n, ndim);
            }
        }

        /// <summary>
        ///     Return the indices to access the main diagonal of an n-dimensional array.<br></br>
        ///     See diag_indices for full details.<br></br>
        ///     Notes
        /// </summary>
        public static void diag_indices_from(this NDarray arr)
        {
            if (Gpu.Available && Gpu.Use)
            {
                cp.diag_indices_from(arr.CupyNDarray);
            }
            else
            {
                np.diag_indices_from(arr.NumpyNDarray);
            }
        }

        /// <summary>
        ///     Return the indices to access (n, n) arrays, given a masking function.<br></br>
        ///     Assume mask_func is a function that, for a square array a of size
        ///     (n, n) with a possible offset argument k, when called as
        ///     mask_func(a, k) returns a new array with zeros in certain locations
        ///     (functions like triu or tril do precisely this).<br></br>
        ///     Then this function
        ///     returns the indices where the non-zero values would be located.<br></br>
        ///     Notes
        /// </summary>
        /// <param name="n">
        ///     The returned indices will be valid to access arrays of shape (n, n).
        /// </param>
        /// <param name="mask_func">
        ///     A function whose call signature is similar to that of triu, tril.<br></br>
        ///     That is, mask_func(x, k) returns a boolean array, shaped like x.<br></br>
        ///     k is an optional argument to the function.
        /// </param>
        /// <param name="k">
        ///     An optional argument which is passed through to mask_func.<br></br>
        ///     Functions
        ///     like triu, tril take a second argument that is interpreted as an
        ///     offset.
        /// </param>
        /// <returns>
        ///     The n arrays of indices corresponding to the locations where
        ///     mask_func(cp.ones((n, n)), k) is True.
        /// </returns>
        public static NDarray[] mask_indices(int n, Delegate mask_func, int k = 0)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return cp.mask_indices(n, mask_func, k).Select(x => new NDarray(x)).ToArray();
            }
            else
            {
                return np.mask_indices(n, mask_func, k).Select(x => new NDarray(x)).ToArray();
            }
        }

        /// <summary>
        ///     Return the indices for the lower-triangle of an (n, m) array.<br></br>
        ///     Notes
        /// </summary>
        /// <param name="n">
        ///     The row dimension of the arrays for which the returned
        ///     indices will be valid.
        /// </param>
        /// <param name="k">
        ///     Diagonal offset (see tril for details).
        /// </param>
        /// <param name="m">
        ///     The column dimension of the arrays for which the returned
        ///     arrays will be valid.<br></br>
        ///     By default m is taken equal to n.
        /// </param>
        /// <returns>
        ///     The indices for the triangle.<br></br>
        ///     The returned tuple contains two arrays,
        ///     each with the indices along one dimension of the array.
        /// </returns>
        public static NDarray[] tril_indices(int n, int? k = 0, int? m = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return cp.tril_indices(n, k, m).Select(x => new NDarray(x)).ToArray();
            }
            else
            {
                return np.tril_indices(n, k, m).Select(x => new NDarray(x)).ToArray();
            }
        }

        /// <summary>
        ///     Return the indices for the lower-triangle of arr.<br></br>
        ///     See tril_indices for full details.<br></br>
        ///     Notes
        /// </summary>
        /// <param name="arr">
        ///     The indices will be valid for square arrays whose dimensions are
        ///     the same as arr.
        /// </param>
        /// <param name="k">
        ///     Diagonal offset (see tril for details).
        /// </param>
        public static void tril_indices_from(this NDarray arr, int? k = 0)
        {
            if (Gpu.Available && Gpu.Use)
            {
                cp.tril_indices_from(arr.CupyNDarray, k);
            }
            else
            {
                np.tril_indices_from(arr.NumpyNDarray, k);
            }
        }

        /// <summary>
        ///     Return the indices for the upper-triangle of an (n, m) array.<br></br>
        ///     Notes
        /// </summary>
        /// <param name="n">
        ///     The size of the arrays for which the returned indices will
        ///     be valid.
        /// </param>
        /// <param name="k">
        ///     Diagonal offset (see triu for details).
        /// </param>
        /// <param name="m">
        ///     The column dimension of the arrays for which the returned
        ///     arrays will be valid.<br></br>
        ///     By default m is taken equal to n.
        /// </param>
        /// <returns>
        ///     The indices for the triangle.<br></br>
        ///     The returned tuple contains two arrays,
        ///     each with the indices along one dimension of the array.<br></br>
        ///     Can be used
        ///     to slice a ndarray of shape(n, n).
        /// </returns>
        public static NDarray[] triu_indices(int n, int? k = 0, int? m = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return cp.triu_indices(n, k, m).Select(x => new NDarray(x)).ToArray();
            }
            else
            {
                return np.triu_indices(n, k, m).Select(x => new NDarray(x)).ToArray();
            }
        }

        /// <summary>
        ///     Return the indices for the upper-triangle of arr.<br></br>
        ///     See triu_indices for full details.<br></br>
        ///     Notes
        /// </summary>
        /// <param name="arr">
        ///     The indices will be valid for square arrays.
        /// </param>
        /// <param name="k">
        ///     Diagonal offset (see triu for details).
        /// </param>
        /// <returns>
        ///     Indices for the upper-triangle of arr.
        /// </returns>
        public static NDarray[] triu_indices_from(this NDarray arr, int? k = 0)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return cp.triu_indices_from(arr.CupyNDarray, k).Select(x => new NDarray(x)).ToArray();
            }
            else
            {
                return np.triu_indices_from(arr.NumpyNDarray, k).Select(x => new NDarray(x)).ToArray();
            }
        }

        /// <summary>
        ///     Take elements from an array along an axis.<br></br>
        ///     When axis is not None, this function does the same thing as “fancy”
        ///     indexing (indexing arrays using arrays); however, it can be easier to use
        ///     if you need elements along a given axis.<br></br>
        ///     A call such as
        ///     cp.take(arr, indices, axis=3) is equivalent to
        ///     arr[:,:,:,indices,...].<br></br>
        ///     Explained without fancy indexing, this is equivalent to the following use
        ///     of ndindex, which sets each of ii, jj, and kk to a tuple of
        ///     indices:
        ///     Notes
        ///     By eliminating the inner loop in the description above, and using s_ to
        ///     build simple slice objects, take can be expressed  in terms of applying
        ///     fancy indexing to each 1-d slice:
        ///     For this reason, it is equivalent to (but faster than) the following use
        ///     of apply_along_axis:
        /// </summary>
        /// <param name="a">
        ///     The source array.
        /// </param>
        /// <param name="indices">
        ///     The indices of the values to extract.<br></br>
        ///     Also allow scalars for indices.
        /// </param>
        /// <param name="axis">
        ///     The axis over which to select values.<br></br>
        ///     By default, the flattened
        ///     input array is used.
        /// </param>
        /// <param name="out">
        ///     If provided, the result will be placed in this array.<br></br>
        ///     It should
        ///     be of the appropriate shape and dtype.
        /// </param>
        /// <param name="mode">
        ///     Specifies how out-of-bounds indices will behave.<br></br>
        ///     ‘clip’ mode means that all indices that are too large are replaced
        ///     by the index that addresses the last element along that axis.<br></br>
        ///     Note
        ///     that this disables indexing with negative numbers.
        /// </param>
        /// <returns>
        ///     The returned array has the same type as a.
        /// </returns>
        public static NDarray take(NDarray[] a, NDarray[] indices, int? axis = null, NDarray @out = null,
            string mode = "raise")
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.take(a.Select(x => x.CupyNDarray).ToArray(),
                    indices.Select(x => x.CupyNDarray).ToArray(), axis, @out?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.take(a.Select(x => x.NumpyNDarray).ToArray(),
                    indices.Select(x => x.NumpyNDarray).ToArray(), axis, @out?.NumpyNDarray, mode));
            }
        }

        /// <summary>
        ///     Take values from the input array by matching 1d index and data slices.<br></br>
        ///     This iterates over matching 1d slices oriented along the specified axis in
        ///     the index and data arrays, and uses the former to look up values in the
        ///     latter.<br></br>
        ///     These slices can be different lengths.<br></br>
        ///     Functions returning an index along an axis, like argsort and
        ///     argpartition, produce suitable indices for this function.<br></br>
        ///     Notes
        ///     This is equivalent to (but faster than) the following use of ndindex and
        ///     s_, which sets each of ii and kk to a tuple of indices:
        ///     Equivalently, eliminating the inner loop, the last two lines would be:
        /// </summary>
        /// <param name="arr">
        ///     Source array
        /// </param>
        /// <param name="indices">
        ///     Indices to take along each 1d slice of arr.<br></br>
        ///     This must match the
        ///     dimension of arr, but dimensions Ni and Nj only need to broadcast
        ///     against arr.
        /// </param>
        /// <param name="axis">
        ///     The axis to take 1d slices along.<br></br>
        ///     If axis is None, the input array is
        ///     treated as if it had first been flattened to 1d, for consistency with
        ///     sort and argsort.
        /// </param>
        public static NDarray take_along_axis(this NDarray arr, NDarray indices, int? axis = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.take_along_axis(arr.CupyNDarray, indices.CupyNDarray, axis));
            }
            else
            {
                return new NDarray(np.take_along_axis(arr.NumpyNDarray, indices.NumpyNDarray, axis));
            }
        }

        /// <summary>
        ///     Construct an array from an index array and a set of arrays to choose from.<br></br>
        ///     First of all, if confused or uncertain, definitely look at the Examples -
        ///     in its full generality, this function is less simple than it might
        ///     seem from the following code description (below ndi =
        ///     Cupy.lib.index_tricks):
        ///     cp.choose(a,c) == cp.array([c[a[I]][I] for I in ndi.ndindex(a.shape)]).<br></br>
        ///     But this omits some subtleties.<br></br>
        ///     Here is a fully general summary:
        ///     Given an “index” array (a) of integers and a sequence of n arrays
        ///     (choices), a and each choice array are first broadcast, as necessary,
        ///     to arrays of a common shape; calling these Ba and Bchoices[i], i =
        ///     0,…,n-1 we have that, necessarily, Ba.shape == Bchoices[i].shape
        ///     for each i.<br></br>
        ///     Then, a new array with shape Ba.shape is created as
        ///     follows:
        ///     Notes
        ///     To reduce the chance of misinterpretation, even though the following
        ///     “abuse” is nominally supported, choices should neither be, nor be
        ///     thought of as, a single array, i.e., the outermost sequence-like container
        ///     should be either a list or a tuple.
        /// </summary>
        /// <param name="a">
        ///     This array must contain integers in [0, n-1], where n is the number
        ///     of choices, unless mode=wrap or mode=clip, in which cases any
        ///     integers are permissible.
        /// </param>
        /// <param name="choices">
        ///     Choice arrays.<br></br>
        ///     a and all of the choices must be broadcastable to the
        ///     same shape.<br></br>
        ///     If choices is itself an array (not recommended), then
        ///     its outermost dimension (i.e., the one corresponding to
        ///     choices.shape[0]) is taken as defining the “sequence”.
        /// </param>
        /// <param name="out">
        ///     If provided, the result will be inserted into this array.<br></br>
        ///     It should
        ///     be of the appropriate shape and dtype.
        /// </param>
        /// <param name="mode">
        ///     Specifies how indices outside [0, n-1] will be treated:
        /// </param>
        /// <returns>
        ///     The merged result.
        /// </returns>
        public static NDarray choose(NDarray<int> a, NDarray[] choices, NDarray @out = null, string mode = "raise")
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.choose(a.CupyNDarray, choices.Select(x => x.CupyNDarray).ToArray(),
                    @out?.CupyNDarray, mode));
            }
            else
            {
                return new NDarray(np.choose(a.NumpyNDarray, choices.Select(x => x.NumpyNDarray).ToArray(),
                    @out?.NumpyNDarray, mode));
            }
        }

        /// <summary>
        ///     Return selected slices of an array along given axis.<br></br>
        ///     When working along a given axis, a slice along that axis is returned in
        ///     output for each index where condition evaluates to True.<br></br>
        ///     When
        ///     working on a 1-D array, compress is equivalent to extract.
        /// </summary>
        /// <param name="condition">
        ///     Array that selects which entries to return.<br></br>
        ///     If len(condition)
        ///     is less than the size of a along the given axis, then output is
        ///     truncated to the length of the condition array.
        /// </param>
        /// <param name="a">
        ///     Array from which to extract a part.
        /// </param>
        /// <param name="axis">
        ///     Axis along which to take slices.<br></br>
        ///     If None (default), work on the
        ///     flattened array.
        /// </param>
        /// <param name="out">
        ///     Output array.<br></br>
        ///     Its type is preserved and it must be of the right
        ///     shape to hold the output.
        /// </param>
        /// <returns>
        ///     A copy of a without the slices along axis for which condition
        ///     is false.
        /// </returns>
        public static NDarray compress(NDarray<bool> condition, NDarray a, int? axis = null, NDarray @out = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.compress(condition.CupyNDarray, a.CupyNDarray, axis, @out?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.compress(condition.NumpyNDarray, a.NumpyNDarray, axis, @out?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Return specified diagonals.<br></br>
        ///     If a is 2-D, returns the diagonal of a with the given offset,
        ///     i.e., the collection of elements of the form a[i, i+offset].<br></br>
        ///     If
        ///     a has more than two dimensions, then the axes specified by axis1
        ///     and axis2 are used to determine the 2-D sub-array whose diagonal is
        ///     returned.<br></br>
        ///     The shape of the resulting array can be determined by
        ///     removing axis1 and axis2 and appending an index to the right equal
        ///     to the size of the resulting diagonals.<br></br>
        ///     In versions of Cupy prior to 1.7, this function always returned a new,
        ///     independent array containing a copy of the values in the diagonal.<br></br>
        ///     In Cupy 1.7 and 1.8, it continues to return a copy of the diagonal,
        ///     but depending on this fact is deprecated.<br></br>
        ///     Writing to the resulting
        ///     array continues to work as it used to, but a FutureWarning is issued.<br></br>
        ///     Starting in Cupy 1.9 it returns a read-only view on the original array.<br></br>
        ///     Attempting to write to the resulting array will produce an error.<br></br>
        ///     In some future release, it will return a read/write view and writing to
        ///     the returned array will alter your original array.<br></br>
        ///     The returned array
        ///     will have the same type as the input array.<br></br>
        ///     If you don’t write to the array returned by this function, then you can
        ///     just ignore all of the above.<br></br>
        ///     If you depend on the current behavior, then we suggest copying the
        ///     returned array explicitly, i.e., use cp.diagonal(a).copy() instead
        ///     of just cp.diagonal(a).<br></br>
        ///     This will work with both past and future
        ///     versions of Cupy.
        /// </summary>
        /// <param name="a">
        ///     Array from which the diagonals are taken.
        /// </param>
        /// <param name="offset">
        ///     Offset of the diagonal from the main diagonal.<br></br>
        ///     Can be positive or
        ///     negative.<br></br>
        ///     Defaults to main diagonal (0).
        /// </param>
        /// <param name="axis1">
        ///     Axis to be used as the first axis of the 2-D sub-arrays from which
        ///     the diagonals should be taken.<br></br>
        ///     Defaults to first axis (0).
        /// </param>
        /// <param name="axis2">
        ///     Axis to be used as the second axis of the 2-D sub-arrays from
        ///     which the diagonals should be taken.<br></br>
        ///     Defaults to second axis (1).
        /// </param>
        /// <returns>
        ///     If a is 2-D, then a 1-D array containing the diagonal and of the
        ///     same type as a is returned unless a is a matrix, in which case
        ///     a 1-D array rather than a (2-D) matrix is returned in order to
        ///     maintain backward compatibility.<br></br>
        ///     If a.ndim &gt; 2, then the dimensions specified by axis1 and axis2
        ///     are removed, and a new axis inserted at the end corresponding to the
        ///     diagonal.
        /// </returns>
        public static NDarray diagonal(this NDarray a, int? offset = 0, int? axis1 = 0, int? axis2 = 1)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.diagonal(a.CupyNDarray, offset, axis1, axis2));
            }
            else
            {
                return new NDarray(np.diagonal(a.NumpyNDarray, offset, axis1, axis2));
            }
        }

        /// <summary>
        ///     Return an array drawn from elements in choicelist, depending on conditions.
        /// </summary>
        /// <param name="condlist">
        ///     The list of conditions which determine from which array in choicelist
        ///     the output elements are taken.<br></br>
        ///     When multiple conditions are satisfied,
        ///     the first one encountered in condlist is used.
        /// </param>
        /// <param name="choicelist">
        ///     The list of arrays from which the output elements are taken.<br></br>
        ///     It has
        ///     to be of the same length as condlist.
        /// </param>
        /// <param name="default">
        ///     The element inserted in output when all conditions evaluate to False.
        /// </param>
        /// <returns>
        ///     The output at position m is the m-th element of the array in
        ///     choicelist where the m-th element of the corresponding array in
        ///     condlist is True.
        /// </returns>
        public static NDarray select(NDarray<bool>[] condlist, NDarray[] choicelist, object @default = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.select(condlist.Select(x => x.CupyNDarray).ToArray(),
                    choicelist.Select(x => x.CupyNDarray).ToArray(), @default));
            }
            else
            {
                return new NDarray(np.select(condlist.Select(x => x.NumpyNDarray).ToArray(),
                    choicelist.Select(x => x.NumpyNDarray).ToArray(), @default));
            }
        }

        /// <summary>
        ///     Change elements of an array based on conditional and input values.<br></br>
        ///     Similar to cp.copyto(arr, vals, where=mask), the difference is that
        ///     place uses the first N elements of vals, where N is the number of
        ///     True values in mask, while copyto uses the elements where mask
        ///     is True.<br></br>
        ///     Note that extract does the exact opposite of place.
        /// </summary>
        /// <param name="arr">
        ///     Array to put data into.
        /// </param>
        /// <param name="mask">
        ///     Boolean mask array.<br></br>
        ///     Must have the same size as a.
        /// </param>
        /// <param name="vals">
        ///     Values to put into a.<br></br>
        ///     Only the first N elements are used, where
        ///     N is the number of True values in mask.<br></br>
        ///     If vals is smaller
        ///     than N, it will be repeated, and if elements of a are to be masked,
        ///     this sequence must be non-empty.
        /// </param>
        public static void place(this NDarray arr, NDarray mask, NDarray vals)
        {
            if (Gpu.Available && Gpu.Use)
            {
                cp.place(arr.CupyNDarray, mask.CupyNDarray, vals.CupyNDarray);
            }
            else
            {
                np.place(arr.NumpyNDarray, mask.NumpyNDarray, vals.NumpyNDarray);
            }
        }

        /// <summary>
        ///     Replaces specified elements of an array with given values.<br></br>
        ///     The indexing works on the flattened target array.<br></br>
        ///     put is roughly
        ///     equivalent to:
        /// </summary>
        /// <param name="a">
        ///     Target array.
        /// </param>
        /// <param name="ind">
        ///     Target indices, interpreted as integers.
        /// </param>
        /// <param name="v">
        ///     Values to place in a at target indices.<br></br>
        ///     If v is shorter than
        ///     ind it will be repeated as necessary.
        /// </param>
        /// <param name="mode">
        ///     Specifies how out-of-bounds indices will behave.<br></br>
        ///     ‘clip’ mode means that all indices that are too large are replaced
        ///     by the index that addresses the last element along that axis.<br></br>
        ///     Note
        ///     that this disables indexing with negative numbers.
        /// </param>
        public static void put(this NDarray a, NDarray ind, NDarray v, string mode = "raise")
        {
            if (Gpu.Available && Gpu.Use)
            {
                cp.put(a.CupyNDarray, ind.CupyNDarray, v.CupyNDarray, mode);
            }
            else
            {
                np.put(a.NumpyNDarray, ind.NumpyNDarray, v.NumpyNDarray, mode);
            }
        }

        /// <summary>
        ///     Put values into the destination array by matching 1d index and data slices.<br></br>
        ///     This iterates over matching 1d slices oriented along the specified axis in
        ///     the index and data arrays, and uses the former to place values into the
        ///     latter.<br></br>
        ///     These slices can be different lengths.<br></br>
        ///     Functions returning an index along an axis, like argsort and
        ///     argpartition, produce suitable indices for this function.<br></br>
        ///     Notes
        ///     This is equivalent to (but faster than) the following use of ndindex and
        ///     s_, which sets each of ii and kk to a tuple of indices:
        ///     Equivalently, eliminating the inner loop, the last two lines would be:
        /// </summary>
        /// <param name="arr">
        ///     Destination array.
        /// </param>
        /// <param name="indices">
        ///     Indices to change along each 1d slice of arr.<br></br>
        ///     This must match the
        ///     dimension of arr, but dimensions in Ni and Nj may be 1 to broadcast
        ///     against arr.
        /// </param>
        /// <param name="values">
        ///     values to insert at those indices.<br></br>
        ///     Its shape and dimension are
        ///     broadcast to match that of indices.
        /// </param>
        /// <param name="axis">
        ///     The axis to take 1d slices along.<br></br>
        ///     If axis is None, the destination
        ///     array is treated as if a flattened 1d view had been created of it.
        /// </param>
        public static void put_along_axis(this NDarray arr, NDarray indices, NDarray[] values, int axis)
        {
            if (Gpu.Available && Gpu.Use)
            {
                cp.put_along_axis(arr.CupyNDarray, indices.CupyNDarray, values.Select(x => x.CupyNDarray).ToArray(), axis);
            }
            else
            {
                np.put_along_axis(arr.NumpyNDarray, indices.NumpyNDarray, values.Select(x => x.NumpyNDarray).ToArray(), axis);
            }
        }

        /// <summary>
        ///     Changes elements of an array based on conditional and input values.<br></br>
        ///     Sets a.flat[n] = values[n] for each n where mask.flat[n]==True.<br></br>
        ///     If values is not the same size as a and mask then it will repeat.<br></br>
        ///     This gives behavior different from a[mask] = values.
        /// </summary>
        /// <param name="a">
        ///     Target array.
        /// </param>
        /// <param name="mask">
        ///     Boolean mask array.<br></br>
        ///     It has to be the same shape as a.
        /// </param>
        /// <param name="values">
        ///     Values to put into a where mask is True.<br></br>
        ///     If values is smaller
        ///     than a it will be repeated.
        /// </param>
        public static void putmask(this NDarray a, NDarray mask, NDarray values)
        {
            if (Gpu.Available && Gpu.Use)
            {
                cp.putmask(a.CupyNDarray, mask.CupyNDarray, values.CupyNDarray);
            }
            else
            {
                np.putmask(a.NumpyNDarray, mask.NumpyNDarray, values.NumpyNDarray);
            }
        }

        /// <summary>
        ///     Fill the main diagonal of the given array of any dimensionality.<br></br>
        ///     For an array a with a.ndim &gt;= 2, the diagonal is the list of
        ///     locations with indices a[i, ..., i] all identical.<br></br>
        ///     This function
        ///     modifies the input array in-place, it does not return a value.<br></br>
        ///     Notes
        ///     This functionality can be obtained via diag_indices, but internally
        ///     this version uses a much faster implementation that never constructs the
        ///     indices and uses simple slicing.
        /// </summary>
        /// <param name="a">
        ///     Array whose diagonal is to be filled, it gets modified in-place.
        /// </param>
        /// <param name="val">
        ///     Value to be written on the diagonal, its type must be compatible with
        ///     that of the array a.
        /// </param>
        /// <param name="wrap">
        ///     For tall matrices in Cupy version up to 1.6.2, the
        ///     diagonal “wrapped” after N columns.<br></br>
        ///     You can have this behavior
        ///     with this option.<br></br>
        ///     This affects only tall matrices.
        /// </param>
        public static void fill_diagonal(this NDarray a, ValueType val, bool wrap = false)
        {
            if (Gpu.Available && Gpu.Use)
            {
                cp.fill_diagonal(a.CupyNDarray, val, wrap);
            }
            else
            {
                np.fill_diagonal(a.NumpyNDarray, val, wrap);
            }
        }

        /*
        /// <summary>
        ///	Efficient multi-dimensional iterator object to iterate over arrays.<br></br>
        ///	
        ///	To get started using this object, see the
        ///	introductory guide to array iteration.<br></br>
        ///	
        ///	Notes
        ///	
        ///	nditer supersedes flatiter.<br></br>
        ///	  The iterator implementation behind
        ///	nditer is also exposed by the Cupy C API.<br></br>
        ///	
        ///	The Python exposure supplies two iteration interfaces, one which follows
        ///	the Python iterator protocol, and another which mirrors the C-style
        ///	do-while pattern.<br></br>
        ///	  The native Python approach is better in most cases, but
        ///	if you need the iterator’s coordinates or index, use the C-style pattern.
        /// </summary>
        /// <param name="op">
        ///	The array(s) to iterate over.
        /// </param>
        /// <param name="flags">
        ///	Flags to control the behavior of the iterator.
        /// </param>
        /// <param name="op_flags">
        ///	This is a list of flags for each operand.<br></br>
        ///	At minimum, one of
        ///	“readonly”, “readwrite”, or “writeonly” must be specified.
        /// </param>
        /// <param name="op_dtypes">
        ///	The required data type(s) of the operands.<br></br>
        ///	If copying or buffering
        ///	is enabled, the data will be converted to/from their original types.
        /// </param>
        /// <param name="order">
        ///	Controls the iteration order.<br></br>
        ///	‘C’ means C order, ‘F’ means
        ///	Fortran order, ‘A’ means ‘F’ order if all the arrays are Fortran
        ///	contiguous, ‘C’ order otherwise, and ‘K’ means as close to the
        ///	order the array elements appear in memory as possible.<br></br>
        ///	This also
        ///	affects the element memory order of “allocate” operands, as they
        ///	are allocated to be compatible with iteration order.<br></br>
        ///	
        ///	Default is ‘K’.
        /// </param>
        /// <param name="casting">
        ///	Controls what kind of data casting may occur when making a copy
        ///	or buffering.<br></br>
        ///	Setting this to ‘unsafe’ is not recommended,
        ///	as it can adversely affect accumulations.
        /// </param>
        /// <param name="op_axes">
        ///	If provided, is a list of ints or None for each operands.<br></br>
        ///	
        ///	The list of axes for an operand is a mapping from the dimensions
        ///	of the iterator to the dimensions of the operand.<br></br>
        ///	A value of
        ///	-1 can be placed for entries, causing that dimension to be
        ///	treated as “newaxis”.
        /// </param>
        /// <param name="itershape">
        ///	The desired shape of the iterator.<br></br>
        ///	This allows “allocate” operands
        ///	with a dimension mapped by op_axes not corresponding to a dimension
        ///	of a different operand to get a value not equal to 1 for that
        ///	dimension.
        /// </param>
        /// <param name="buffersize">
        ///	When buffering is enabled, controls the size of the temporary
        ///	buffers.<br></br>
        ///	Set to 0 for the default value.
        /// </param>
        public static void nditer(this NDarray op, string[] flags = null, list of list of str op_flags = null, dtype or tuple of dtype(s) op_dtypes = null, string order = null, string casting = null, list of list of ints op_axes = null, tuple of ints itershape = null, int? buffersize = null)
        {
            //auto-generated code, do not change
            var __self__=self;
            var pyargs=ToTuple(new object[]
            {
                op,
            });
            var kwargs=new PyDict();
            if (flags!=null) kwargs["flags"]=ToPython(flags);
            if (op_flags!=null) kwargs["op_flags"]=ToPython(op_flags);
            if (op_dtypes!=null) kwargs["op_dtypes"]=ToPython(op_dtypes);
            if (order!=null) kwargs["order"]=ToPython(order);
            if (casting!=null) kwargs["casting"]=ToPython(casting);
            if (op_axes!=null) kwargs["op_axes"]=ToPython(op_axes);
            if (itershape!=null) kwargs["itershape"]=ToPython(itershape);
            if (buffersize!=null) kwargs["buffersize"]=ToPython(buffersize);
            dynamic py = __self__.InvokeMethod("nditer", pyargs, kwargs);
        }
        */

        /// <summary>
        ///     Multidimensional index iterator.<br></br>
        ///     Return an iterator yielding pairs of array coordinates and values.
        /// </summary>
        /// <param name="arr">
        ///     Input array.
        /// </param>
        public static void ndenumerate(this NDarray arr)
        {
            if (Gpu.Available && Gpu.Use)
            {
                cp.ndenumerate(arr.CupyNDarray);
            }
            else
            {
                np.ndenumerate(arr.NumpyNDarray);
            }
        }

        /// <summary>
        ///     An N-dimensional iterator object to index arrays.<br></br>
        ///     Given the shape of an array, an ndindex instance iterates over
        ///     the N-dimensional index of the array.<br></br>
        ///     At each iteration a tuple
        ///     of indices is returned, the last dimension is iterated over first.
        /// </summary>
        /// <param name="args">
        ///     The size of each dimension of the array.
        /// </param>
        public static void ndindex(params int[] args)
        {
            if (Gpu.Available && Gpu.Use)
            {
                cp.ndindex(args);
            }
            else
            {
                np.ndindex(args);
            }
        }

        /*
        /// <summary>
        ///	Create nditers for use in nested loops
        ///	
        ///	Create a tuple of nditer objects which iterate in nested loops over
        ///	different axes of the op argument.<br></br>
        ///	 The first iterator is used in the
        ///	outermost loop, the last in the innermost loop.<br></br>
        ///	 Advancing one will change
        ///	the subsequent iterators to point at its new element.
        /// </summary>
        /// <param name="op">
        ///	The array(s) to iterate over.
        /// </param>
        /// <param name="axes">
        ///	Each item is used as an “op_axes” argument to an nditer
        /// </param>
        /// <returns>
        ///	An nditer for each item in axes, outermost first
        /// </returns>
        public static tuple of nditer nested_iters(this NDarray op, int[] axes = null)
        {
            //auto-generated code, do not change
            var __self__=self;
            var pyargs=ToTuple(new object[]
            {
                op,
            });
            var kwargs=new PyDict();
            if (axes!=null) kwargs["axes"]=ToPython(axes);
            dynamic py = __self__.InvokeMethod("nested_iters", pyargs, kwargs);
            return ToCsharp<tuple of nditer>(py);
        }
        */

        /// <summary>
        ///     Flat iterator object to iterate over arrays.<br></br>
        ///     A flatiter iterator is returned by x.flat for any array x.<br></br>
        ///     It allows iterating over the array as if it were a 1-D array,
        ///     either in a for-loop or by calling its next method.<br></br>
        ///     Iteration is done in row-major, C-style order (the last
        ///     index varying the fastest).<br></br>
        ///     The iterator can also be indexed using
        ///     basic slicing or advanced indexing.<br></br>
        ///     Notes
        ///     A flatiter iterator can not be constructed directly from Python code
        ///     by calling the flatiter constructor.
        /// </summary>
        public static void flatiter()
        {
            if (Gpu.Available && Gpu.Use)
            {
                cp.flatiter();
            }
            else
            {
                np.flatiter();
            }
        }

        public static partial class lib
        {
            public static class stride_tricks
            {
                /// <summary>
                ///     Create a view into the array with the given shape and strides.<br></br>
                ///     Notes
                ///     as_strided creates a view into the array given the exact strides
                ///     and shape.<br></br>
                ///     This means it manipulates the internal data structure of
                ///     ndarray and, if done incorrectly, the array elements can point to
                ///     invalid memory and can corrupt results or crash your program.<br></br>
                ///     It is advisable to always use the original x.strides when
                ///     calculating new strides to avoid reliance on a contiguous memory
                ///     layout.<br></br>
                ///     Furthermore, arrays created with this function often contain self
                ///     overlapping memory, so that two elements are identical.<br></br>
                ///     Vectorized write operations on such arrays will typically be
                ///     unpredictable.<br></br>
                ///     They may even give different results for small, large,
                ///     or transposed arrays.<br></br>
                ///     Since writing to these arrays has to be tested and done with great
                ///     care, you may want to use writeable=False to avoid accidental write
                ///     operations.<br></br>
                ///     For these reasons it is advisable to avoid as_strided when
                ///     possible.
                /// </summary>
                /// <param name="x">
                ///     Array to create a new.
                /// </param>
                /// <param name="shape">
                ///     The shape of the new array.<br></br>
                ///     Defaults to x.shape.
                /// </param>
                /// <param name="strides">
                ///     The strides of the new array.<br></br>
                ///     Defaults to x.strides.
                /// </param>
                /// <param name="subok">
                ///     If True, subclasses are preserved.
                /// </param>
                /// <param name="writeable">
                ///     If set to False, the returned array will always be readonly.<br></br>
                ///     Otherwise it will be writable if the original array was.<br></br>
                ///     It
                ///     is advisable to set this to False if possible (see Notes).
                /// </param>
                public static NDarray as_strided(NDarray x, Shape shape = null, int[] strides = null,
                    bool? subok = false, bool? writeable = true)
                {
                    if (Gpu.Available && Gpu.Use)
                    {
                        return new NDarray(cp.lib.stride_tricks.as_strided(x.CupyNDarray, shape?.CupyShape, strides,
                            subok, writeable));
                    }
                    else
                    {
                        return new NDarray(np.lib.stride_tricks.as_strided(x.NumpyNDarray, shape?.NumpyShape, strides,
                            subok, writeable));
                    }
                }
            }
        }

        public static partial class lib
        {
            /// <summary>
            ///     Buffered iterator for big arrays.<br></br>
            ///     Arrayterator creates a buffered iterator for reading big arrays in small
            ///     contiguous blocks.<br></br>
            ///     The class is useful for objects stored in the
            ///     file system.<br></br>
            ///     It allows iteration over the object without reading
            ///     everything in memory; instead, small blocks are read and iterated over.<br></br>
            ///     Arrayterator can be used with any object that supports multidimensional
            ///     slices.<br></br>
            ///     This includes Cupy arrays, but also variables from
            ///     Scientific.IO.NetCDF or pynetcdf for example.<br></br>
            ///     Notes
            ///     The algorithm works by first finding a “running dimension”, along which
            ///     the blocks will be extracted.<br></br>
            ///     Given an array of dimensions
            ///     (d1, d2, ..., dn), e.g.<br></br>
            ///     if buf_size is smaller than d1, the
            ///     first dimension will be used.<br></br>
            ///     If, on the other hand,
            ///     d1 &lt; buf_size &lt; d1*d2 the second dimension will be used, and so on.<br></br>
            ///     Blocks are extracted along this dimension, and when the last block is
            ///     returned the process continues from the next dimension, until all
            ///     elements have been read.
            /// </summary>
            /// <param name="var">
            ///     The object to iterate over.
            /// </param>
            /// <param name="buf_size">
            ///     The buffer size.<br></br>
            ///     If buf_size is supplied, the maximum amount of
            ///     data that will be read into memory is buf_size elements.<br></br>
            ///     Default is None, which will read as many element as possible
            ///     into memory.
            /// </param>
            public static void Arrayterator(NDarray var, int? buf_size = null)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    cp.lib.Arrayterator(var.CupyNDarray, buf_size);
                }
                else
                {
                    np.lib.Arrayterator(var.NumpyNDarray, buf_size);
                }
            }
        }
    }
}
