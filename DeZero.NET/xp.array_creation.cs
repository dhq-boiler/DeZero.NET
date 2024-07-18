﻿using Cupy;
using Numpy;
using System.Collections;

namespace DeZero.NET
{
    public static partial class xp
    {
        /// <summary>
        ///     Return a new array of given shape and type, without initializing entries.<br></br>
        ///     Notes
        ///     empty, unlike zeros, does not set the array values to zero,
        ///     and may therefore be marginally faster.<br></br>
        ///     On the other hand, it requires
        ///     the user to manually set all the values in the array, and should be
        ///     used with caution.
        /// </summary>
        /// <param name="shape">
        ///     Shape of the empty array, e.g., (2, 3) or 2.
        /// </param>
        /// <param name="dtype">
        ///     Desired output data-type for the array, e.g, Cupy.int8. Default is
        ///     Cupy.float64.
        /// </param>
        /// <param name="order">
        ///     Whether to store multi-dimensional data in row-major
        ///     (C-style) or column-major (Fortran-style) order in
        ///     memory.
        /// </param>
        /// <returns>
        ///     Array of uninitialized (arbitrary) data of the given shape, dtype, and
        ///     order.<br></br>
        ///     Object arrays will be initialized to None.
        /// </returns>
        public static NDarray empty(Shape shape, Dtype dtype = null, string order = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.empty(shape.CupyShape, dtype?.CupyDtype, order));
            }
            else
            {
                return new NDarray(np.empty(shape.NumpyShape, dtype?.NumpyDtype, order));
            }
        }

        /// <summary>
        ///     Return a new array with the same shape and type as a given array.<br></br>
        ///     Notes
        ///     This function does not initialize the returned array; to do that use
        ///     zeros_like or ones_like instead.<br></br>
        ///     It may be marginally faster than
        ///     the functions that do set the array values.
        /// </summary>
        /// <param name="prototype">
        ///     The shape and data-type of prototype define these same attributes
        ///     of the returned array.
        /// </param>
        /// <param name="dtype">
        ///     Overrides the data type of the result.
        /// </param>
        /// <param name="order">
        ///     Overrides the memory layout of the result.<br></br>
        ///     ‘C’ means C-order,
        ///     ‘F’ means F-order, ‘A’ means ‘F’ if prototype is Fortran
        ///     contiguous, ‘C’ otherwise.<br></br>
        ///     ‘K’ means match the layout of prototype
        ///     as closely as possible.
        /// </param>
        /// <param name="subok">
        ///     If True, then the newly created array will use the sub-class
        ///     type of ‘a’, otherwise it will be a base-class array.<br></br>
        ///     Defaults
        ///     to True.
        /// </param>
        /// <returns>
        ///     Array of uninitialized (arbitrary) data with the same
        ///     shape and type as prototype.
        /// </returns>
        public static NDarray empty_like(NDarray prototype, Dtype dtype = null, string order = null, bool? subok = true)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.empty_like(prototype.CupyNDarray, dtype?.CupyDtype, order, subok));
            }
            else
            {
                return new NDarray(np.empty_like(prototype.NumpyNDarray, dtype?.NumpyDtype, order, subok));
            }
        }

        /// <summary>
        ///     Return a new array with the same shape and type as a given array.<br></br>
        ///     Notes
        ///     This function does not initialize the returned array; to do that use
        ///     zeros_like or ones_like instead.<br></br>
        ///     It may be marginally faster than
        ///     the functions that do set the array values.
        /// </summary>
        /// <param name="prototype">
        ///     The shape and data-type of prototype define these same attributes
        ///     of the returned array.
        /// </param>
        /// <param name="dtype">
        ///     Overrides the data type of the result.
        /// </param>
        /// <param name="order">
        ///     Overrides the memory layout of the result.<br></br>
        ///     ‘C’ means C-order,
        ///     ‘F’ means F-order, ‘A’ means ‘F’ if prototype is Fortran
        ///     contiguous, ‘C’ otherwise.<br></br>
        ///     ‘K’ means match the layout of prototype
        ///     as closely as possible.
        /// </param>
        /// <param name="subok">
        ///     If True, then the newly created array will use the sub-class
        ///     type of ‘a’, otherwise it will be a base-class array.<br></br>
        ///     Defaults
        ///     to True.
        /// </param>
        /// <returns>
        ///     Array of uninitialized (arbitrary) data with the same
        ///     shape and type as prototype.
        /// </returns>
        public static NDarray<T> empty_like<T>(T[] prototype, Dtype dtype = null, string order = null,
            bool? subok = true) where T : struct
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray<T>(cp.empty_like<T>(prototype, dtype?.CupyDtype, order, subok));
            }
            else
            {
                return new NDarray<T>(np.empty_like<T>(prototype, dtype?.NumpyDtype, order, subok));
            }
        }

        /// <summary>
        ///     Return a new array with the same shape and type as a given array.<br></br>
        ///     Notes
        ///     This function does not initialize the returned array; to do that use
        ///     zeros_like or ones_like instead.<br></br>
        ///     It may be marginally faster than
        ///     the functions that do set the array values.
        /// </summary>
        /// <param name="prototype">
        ///     The shape and data-type of prototype define these same attributes
        ///     of the returned array.
        /// </param>
        /// <param name="dtype">
        ///     Overrides the data type of the result.
        /// </param>
        /// <param name="order">
        ///     Overrides the memory layout of the result.<br></br>
        ///     ‘C’ means C-order,
        ///     ‘F’ means F-order, ‘A’ means ‘F’ if prototype is Fortran
        ///     contiguous, ‘C’ otherwise.<br></br>
        ///     ‘K’ means match the layout of prototype
        ///     as closely as possible.
        /// </param>
        /// <param name="subok">
        ///     If True, then the newly created array will use the sub-class
        ///     type of ‘a’, otherwise it will be a base-class array.<br></br>
        ///     Defaults
        ///     to True.
        /// </param>
        /// <returns>
        ///     Array of uninitialized (arbitrary) data with the same
        ///     shape and type as prototype.
        /// </returns>
        public static NDarray<T> empty_like<T>(T[,] prototype, Dtype dtype = null, string order = null,
            bool? subok = true) where T : struct
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray<T>(cp.empty_like<T>(prototype, dtype?.CupyDtype, order, subok));
            }
            else
            {
                return new NDarray<T>(np.empty_like<T>(prototype, dtype?.NumpyDtype, order, subok));
            }
        }

        /// <summary>
        ///     Return a 2-D array with ones on the diagonal and zeros elsewhere.
        /// </summary>
        /// <param name="N">
        ///     Number of rows in the output.
        /// </param>
        /// <param name="M">
        ///     Number of columns in the output.<br></br>
        ///     If None, defaults to N.
        /// </param>
        /// <param name="k">
        ///     Index of the diagonal: 0 (the default) refers to the main diagonal,
        ///     a positive value refers to an upper diagonal, and a negative value
        ///     to a lower diagonal.
        /// </param>
        /// <param name="dtype">
        ///     Data-type of the returned array.
        /// </param>
        /// <param name="order">
        ///     Whether the output should be stored in row-major (C-style) or
        ///     column-major (Fortran-style) order in memory.
        /// </param>
        /// <returns>
        ///     An array where all elements are equal to zero, except for the k-th
        ///     diagonal, whose values are equal to one.
        /// </returns>
        public static NDarray eye(int N, int? M = null, int? k = 0, Dtype dtype = null, string order = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.eye(N, M, k, dtype?.CupyDtype, order));
            }
            else
            {
                return new NDarray(np.eye(N, M, k, dtype?.NumpyDtype, order));
            }
        }

        /// <summary>
        ///     Return the identity array.<br></br>
        ///     The identity array is a square array with ones on
        ///     the main diagonal.
        /// </summary>
        /// <param name="n">
        ///     Number of rows (and columns) in n x n output.
        /// </param>
        /// <param name="dtype">
        ///     Data-type of the output.<br></br>
        ///     Defaults to float.
        /// </param>
        /// <returns>
        ///     n x n array with its main diagonal set to one,
        ///     and all other elements 0.
        /// </returns>
        public static NDarray identity(int n, Dtype dtype = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.identity(n, dtype?.CupyDtype));
            }
            else
            {
                return new NDarray(np.identity(n, dtype?.NumpyDtype));
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
        public static NDarray ones(Shape shape, Dtype dtype = null, string order = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.ones(shape.CupyShape, dtype?.CupyDtype, order));
            }
            else
            {
                return new NDarray(np.ones(shape.NumpyShape, dtype?.NumpyDtype, order));
            }
        }

        /// <summary>
        ///     Return an array of ones with the same shape and type as a given array.
        /// </summary>
        /// <param name="a">
        ///     The shape and data-type of a define these same attributes of
        ///     the returned array.
        /// </param>
        /// <param name="dtype">
        ///     Overrides the data type of the result.
        /// </param>
        /// <param name="order">
        ///     Overrides the memory layout of the result.<br></br>
        ///     ‘C’ means C-order,
        ///     ‘F’ means F-order, ‘A’ means ‘F’ if a is Fortran contiguous,
        ///     ‘C’ otherwise.<br></br>
        ///     ‘K’ means match the layout of a as closely
        ///     as possible.
        /// </param>
        /// <param name="subok">
        ///     If True, then the newly created array will use the sub-class
        ///     type of ‘a’, otherwise it will be a base-class array.<br></br>
        ///     Defaults
        ///     to True.
        /// </param>
        /// <returns>
        ///     Array of ones with the same shape and type as a.
        /// </returns>
        public static NDarray ones_like(NDarray a, Dtype dtype = null, string order = null, bool? subok = true)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.ones_like(a.CupyNDarray, dtype?.CupyDtype, order, subok));
            }
            else
            {
                return new NDarray(np.ones_like(a.NumpyNDarray, dtype?.NumpyDtype, order, subok));
            }
        }

        /// <summary>
        ///     Return an array of ones with the same shape and type as a given array.
        /// </summary>
        /// <param name="a">
        ///     The shape and data-type of a define these same attributes of
        ///     the returned array.
        /// </param>
        /// <param name="dtype">
        ///     Overrides the data type of the result.
        /// </param>
        /// <param name="order">
        ///     Overrides the memory layout of the result.<br></br>
        ///     ‘C’ means C-order,
        ///     ‘F’ means F-order, ‘A’ means ‘F’ if a is Fortran contiguous,
        ///     ‘C’ otherwise.<br></br>
        ///     ‘K’ means match the layout of a as closely
        ///     as possible.
        /// </param>
        /// <param name="subok">
        ///     If True, then the newly created array will use the sub-class
        ///     type of ‘a’, otherwise it will be a base-class array.<br></br>
        ///     Defaults
        ///     to True.
        /// </param>
        /// <returns>
        ///     Array of ones with the same shape and type as a.
        /// </returns>
        public static NDarray<T> ones_like<T>(T[] a, Dtype dtype = null, string order = null, bool? subok = true) where T : struct
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray<T>(cp.ones_like<T>(a, dtype?.CupyDtype, order, subok));
            }
            else
            {
                return new NDarray<T>(np.ones_like<T>(a, dtype?.NumpyDtype, order, subok));
            }
        }

        /// <summary>
        ///     Return an array of ones with the same shape and type as a given array.
        /// </summary>
        /// <param name="a">
        ///     The shape and data-type of a define these same attributes of
        ///     the returned array.
        /// </param>
        /// <param name="dtype">
        ///     Overrides the data type of the result.
        /// </param>
        /// <param name="order">
        ///     Overrides the memory layout of the result.<br></br>
        ///     ‘C’ means C-order,
        ///     ‘F’ means F-order, ‘A’ means ‘F’ if a is Fortran contiguous,
        ///     ‘C’ otherwise.<br></br>
        ///     ‘K’ means match the layout of a as closely
        ///     as possible.
        /// </param>
        /// <param name="subok">
        ///     If True, then the newly created array will use the sub-class
        ///     type of ‘a’, otherwise it will be a base-class array.<br></br>
        ///     Defaults
        ///     to True.
        /// </param>
        /// <returns>
        ///     Array of ones with the same shape and type as a.
        /// </returns>
        public static NDarray<T> ones_like<T>(T[,] a, Dtype dtype = null, string order = null, bool? subok = true) where T : struct
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray<T>(cp.ones_like<T>(a, dtype?.CupyDtype, order, subok));
            }
            else
            {
                return new NDarray<T>(np.ones_like<T>(a, dtype?.NumpyDtype, order, subok));
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
        public static NDarray zeros(Shape shape, Dtype dtype = null, string order = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.zeros(shape.CupyShape, dtype?.CupyDtype, order));
            }
            else
            {
                return new NDarray(np.zeros(shape.NumpyShape, dtype?.NumpyDtype, order));
            }
        }

        /// <summary>
        ///     Return an array of zeros with the same shape and type as a given array.
        /// </summary>
        /// <param name="a">
        ///     The shape and data-type of a define these same attributes of
        ///     the returned array.
        /// </param>
        /// <param name="dtype">
        ///     Overrides the data type of the result.
        /// </param>
        /// <param name="order">
        ///     Overrides the memory layout of the result.<br></br>
        ///     ‘C’ means C-order,
        ///     ‘F’ means F-order, ‘A’ means ‘F’ if a is Fortran contiguous,
        ///     ‘C’ otherwise.<br></br>
        ///     ‘K’ means match the layout of a as closely
        ///     as possible.
        /// </param>
        /// <param name="subok">
        ///     If True, then the newly created array will use the sub-class
        ///     type of ‘a’, otherwise it will be a base-class array.<br></br>
        ///     Defaults
        ///     to True.
        /// </param>
        /// <returns>
        ///     Array of zeros with the same shape and type as a.
        /// </returns>
        public static NDarray zeros_like(NDarray a, Dtype dtype = null, string order = null, bool? subok = true)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.zeros_like(a.CupyNDarray, dtype?.CupyDtype, order, subok));
            }
            else
            {
                return new NDarray(np.zeros_like(a.NumpyNDarray, dtype?.NumpyDtype, order, subok));
            }
        }

        /// <summary>
        ///     Return an array of zeros with the same shape and type as a given array.
        /// </summary>
        /// <param name="a">
        ///     The shape and data-type of a define these same attributes of
        ///     the returned array.
        /// </param>
        /// <param name="dtype">
        ///     Overrides the data type of the result.
        /// </param>
        /// <param name="order">
        ///     Overrides the memory layout of the result.<br></br>
        ///     ‘C’ means C-order,
        ///     ‘F’ means F-order, ‘A’ means ‘F’ if a is Fortran contiguous,
        ///     ‘C’ otherwise.<br></br>
        ///     ‘K’ means match the layout of a as closely
        ///     as possible.
        /// </param>
        /// <param name="subok">
        ///     If True, then the newly created array will use the sub-class
        ///     type of ‘a’, otherwise it will be a base-class array.<br></br>
        ///     Defaults
        ///     to True.
        /// </param>
        /// <returns>
        ///     Array of zeros with the same shape and type as a.
        /// </returns>
        public static NDarray<T> zeros_like<T>(T[] a, Dtype dtype = null, string order = null, bool? subok = true) where T : struct
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray<T>(cp.zeros_like<T>(a, dtype?.CupyDtype, order, subok));
            }
            else
            {
                return new NDarray<T>(np.zeros_like<T>(a, dtype?.NumpyDtype, order, subok));
            }
        }

        /// <summary>
        ///     Return an array of zeros with the same shape and type as a given array.
        /// </summary>
        /// <param name="a">
        ///     The shape and data-type of a define these same attributes of
        ///     the returned array.
        /// </param>
        /// <param name="dtype">
        ///     Overrides the data type of the result.
        /// </param>
        /// <param name="order">
        ///     Overrides the memory layout of the result.<br></br>
        ///     ‘C’ means C-order,
        ///     ‘F’ means F-order, ‘A’ means ‘F’ if a is Fortran contiguous,
        ///     ‘C’ otherwise.<br></br>
        ///     ‘K’ means match the layout of a as closely
        ///     as possible.
        /// </param>
        /// <param name="subok">
        ///     If True, then the newly created array will use the sub-class
        ///     type of ‘a’, otherwise it will be a base-class array.<br></br>
        ///     Defaults
        ///     to True.
        /// </param>
        /// <returns>
        ///     Array of zeros with the same shape and type as a.
        /// </returns>
        public static NDarray<T> zeros_like<T>(T[,] a, Dtype dtype = null, string order = null, bool? subok = true) where T : struct
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray<T>(cp.zeros_like<T>(a, dtype?.CupyDtype, order, subok));
            }
            else
            {
                return new NDarray<T>(np.zeros_like<T>(a, dtype?.NumpyDtype, order, subok));
            }
        }

        /// <summary>
        ///     Return a new array of given shape and type, filled with fill_value.
        /// </summary>
        /// <param name="shape">
        ///     Shape of the new array, e.g., (2, 3) or 2.
        /// </param>
        /// <param name="fill_value">
        ///     Fill value.
        /// </param>
        /// <param name="order">
        ///     Whether to store multidimensional data in C- or Fortran-contiguous
        ///     (row- or column-wise) order in memory.
        /// </param>
        /// <returns>
        ///     Array of fill_value with the given shape, dtype, and order.
        /// </returns>
        public static NDarray full(Shape shape, ValueType fill_value, Dtype dtype = null, string order = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.full(shape.CupyShape, fill_value, dtype?.CupyDtype, order));
            }
            else
            {
                return new NDarray(np.full(shape.NumpyShape, fill_value, dtype?.NumpyDtype, order));
            }
        }

        /// <summary>
        ///     Return a full array with the same shape and type as a given array.
        /// </summary>
        /// <param name="a">
        ///     The shape and data-type of a define these same attributes of
        ///     the returned array.
        /// </param>
        /// <param name="fill_value">
        ///     Fill value.
        /// </param>
        /// <param name="dtype">
        ///     Overrides the data type of the result.
        /// </param>
        /// <param name="order">
        ///     Overrides the memory layout of the result.<br></br>
        ///     ‘C’ means C-order,
        ///     ‘F’ means F-order, ‘A’ means ‘F’ if a is Fortran contiguous,
        ///     ‘C’ otherwise.<br></br>
        ///     ‘K’ means match the layout of a as closely
        ///     as possible.
        /// </param>
        /// <param name="subok">
        ///     If True, then the newly created array will use the sub-class
        ///     type of ‘a’, otherwise it will be a base-class array.<br></br>
        ///     Defaults
        ///     to True.
        /// </param>
        /// <returns>
        ///     Array of fill_value with the same shape and type as a.
        /// </returns>
        public static NDarray full_like(NDarray a, ValueType fill_value, Dtype dtype = null, string order = null,
            bool? subok = true)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.full_like(a.CupyNDarray, fill_value, dtype?.CupyDtype, order, subok));
            }
            else
            {
                return new NDarray(np.full_like(a.NumpyNDarray, fill_value, dtype?.NumpyDtype, order, subok));
            }
        }

        /// <summary>
        ///     Return a full array with the same shape and type as a given array.
        /// </summary>
        /// <param name="a">
        ///     The shape and data-type of a define these same attributes of
        ///     the returned array.
        /// </param>
        /// <param name="fill_value">
        ///     Fill value.
        /// </param>
        /// <param name="dtype">
        ///     Overrides the data type of the result.
        /// </param>
        /// <param name="order">
        ///     Overrides the memory layout of the result.<br></br>
        ///     ‘C’ means C-order,
        ///     ‘F’ means F-order, ‘A’ means ‘F’ if a is Fortran contiguous,
        ///     ‘C’ otherwise.<br></br>
        ///     ‘K’ means match the layout of a as closely
        ///     as possible.
        /// </param>
        /// <param name="subok">
        ///     If True, then the newly created array will use the sub-class
        ///     type of ‘a’, otherwise it will be a base-class array.<br></br>
        ///     Defaults
        ///     to True.
        /// </param>
        /// <returns>
        ///     Array of fill_value with the same shape and type as a.
        /// </returns>
        public static NDarray<T> full_like<T>(T[] a, ValueType fill_value, Dtype dtype = null, string order = null,
            bool? subok = true) where T : struct
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray<T>(cp.full_like<T>(a, fill_value, dtype?.CupyDtype, order, subok));
            }
            else
            {
                return new NDarray<T>(np.full_like<T>(a, fill_value, dtype?.NumpyDtype, order, subok));
            }
        }

        /// <summary>
        ///     Return a full array with the same shape and type as a given array.
        /// </summary>
        /// <param name="a">
        ///     The shape and data-type of a define these same attributes of
        ///     the returned array.
        /// </param>
        /// <param name="fill_value">
        ///     Fill value.
        /// </param>
        /// <param name="dtype">
        ///     Overrides the data type of the result.
        /// </param>
        /// <param name="order">
        ///     Overrides the memory layout of the result.<br></br>
        ///     ‘C’ means C-order,
        ///     ‘F’ means F-order, ‘A’ means ‘F’ if a is Fortran contiguous,
        ///     ‘C’ otherwise.<br></br>
        ///     ‘K’ means match the layout of a as closely
        ///     as possible.
        /// </param>
        /// <param name="subok">
        ///     If True, then the newly created array will use the sub-class
        ///     type of ‘a’, otherwise it will be a base-class array.<br></br>
        ///     Defaults
        ///     to True.
        /// </param>
        /// <returns>
        ///     Array of fill_value with the same shape and type as a.
        /// </returns>
        public static NDarray<T> full_like<T>(T[,] a, ValueType fill_value, Dtype dtype = null, string order = null,
            bool? subok = true) where T : struct
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray<T>(cp.full_like<T>(a, fill_value, dtype?.CupyDtype, order, subok));
            }
            else
            {
                return new NDarray<T>(np.full_like<T>(a, fill_value, dtype?.NumpyDtype, order, subok));
            }
        }

        /// <summary>
        ///     Convert the input to an array.
        /// </summary>
        /// <param name="a">
        ///     Input data, in any form that can be converted to an array.<br></br>
        ///     This
        ///     includes lists, lists of tuples, tuples, tuples of tuples, tuples
        ///     of lists and ndarrays.
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
        ///     is already an ndarray with matching dtype and order.<br></br>
        ///     If a is a
        ///     subclass of ndarray, a base class ndarray is returned.
        /// </returns>
        public static NDarray asarray(NDarray a, Dtype dtype = null, string order = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.asarray(a.CupyNDarray, dtype?.CupyDtype, order));
            }
            else
            {
                return new NDarray(np.asarray(a.NumpyNDarray, dtype?.NumpyDtype, order));
            }
        }

        /// <summary>
        ///     Convert the input to an array.
        /// </summary>
        /// <param name="a">
        ///     Input data, in any form that can be converted to an array.<br></br>
        ///     This
        ///     includes lists, lists of tuples, tuples, tuples of tuples, tuples
        ///     of lists and ndarrays.
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
        ///     is already an ndarray with matching dtype and order.<br></br>
        ///     If a is a
        ///     subclass of ndarray, a base class ndarray is returned.
        /// </returns>
        public static NDarray<T> asarray<T>(T[] a, Dtype dtype = null, string order = null) where T : struct
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray<T>(cp.asarray<T>(a, dtype?.CupyDtype, order));
            }
            else
            {
                return new NDarray<T>(np.asarray<T>(a, dtype?.NumpyDtype, order));
            }
        }

        /// <summary>
        ///     Convert the input to an array.
        /// </summary>
        /// <param name="a">
        ///     Input data, in any form that can be converted to an array.<br></br>
        ///     This
        ///     includes lists, lists of tuples, tuples, tuples of tuples, tuples
        ///     of lists and ndarrays.
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
        ///     is already an ndarray with matching dtype and order.<br></br>
        ///     If a is a
        ///     subclass of ndarray, a base class ndarray is returned.
        /// </returns>
        public static NDarray<T> asarray<T>(T[,] a, Dtype dtype = null, string order = null) where T : struct
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray<T>(cp.asarray<T>(a, dtype?.CupyDtype, order));
            }
            else
            {
                return new NDarray<T>(np.asarray<T>(a, dtype?.NumpyDtype, order));
            }
        }

        /// <summary>
        ///     Convert the input to an ndarray, but pass ndarray subclasses through.
        /// </summary>
        /// <param name="a">
        ///     Input data, in any form that can be converted to an array.<br></br>
        ///     This
        ///     includes scalars, lists, lists of tuples, tuples, tuples of tuples,
        ///     tuples of lists, and ndarrays.
        /// </param>
        /// <param name="dtype">
        ///     By default, the data-type is inferred from the input data.
        /// </param>
        /// <param name="order">
        ///     Whether to use row-major (C-style) or column-major
        ///     (Fortran-style) memory representation.<br></br>
        ///     Defaults to ‘C’.
        /// </param>
        /// <returns>
        ///     Array interpretation of a.<br></br>
        ///     If a is an ndarray or a subclass
        ///     of ndarray, it is returned as-is and no copy is performed.
        /// </returns>
        public static NDarray asanyarray(NDarray a, Dtype dtype = null, string order = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.asanyarray(a.CupyNDarray, dtype?.CupyDtype, order));
            }
            else
            {
                return new NDarray(np.asanyarray(a.NumpyNDarray, dtype?.NumpyDtype, order));
            }
        }

        /// <summary>
        ///     Convert the input to an ndarray, but pass ndarray subclasses through.
        /// </summary>
        /// <param name="a">
        ///     Input data, in any form that can be converted to an array.<br></br>
        ///     This
        ///     includes scalars, lists, lists of tuples, tuples, tuples of tuples,
        ///     tuples of lists, and ndarrays.
        /// </param>
        /// <param name="dtype">
        ///     By default, the data-type is inferred from the input data.
        /// </param>
        /// <param name="order">
        ///     Whether to use row-major (C-style) or column-major
        ///     (Fortran-style) memory representation.<br></br>
        ///     Defaults to ‘C’.
        /// </param>
        /// <returns>
        ///     Array interpretation of a.<br></br>
        ///     If a is an ndarray or a subclass
        ///     of ndarray, it is returned as-is and no copy is performed.
        /// </returns>
        public static NDarray<T> asanyarray<T>(T[] a, Dtype dtype = null, string order = null) where T : struct
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray<T>(cp.asanyarray(a, dtype?.CupyDtype, order));
            }
            else
            {
                return new NDarray<T>(np.asanyarray(a, dtype?.NumpyDtype, order));
            }
        }

        /// <summary>
        ///     Convert the input to an ndarray, but pass ndarray subclasses through.
        /// </summary>
        /// <param name="a">
        ///     Input data, in any form that can be converted to an array.<br></br>
        ///     This
        ///     includes scalars, lists, lists of tuples, tuples, tuples of tuples,
        ///     tuples of lists, and ndarrays.
        /// </param>
        /// <param name="dtype">
        ///     By default, the data-type is inferred from the input data.
        /// </param>
        /// <param name="order">
        ///     Whether to use row-major (C-style) or column-major
        ///     (Fortran-style) memory representation.<br></br>
        ///     Defaults to ‘C’.
        /// </param>
        /// <returns>
        ///     Array interpretation of a.<br></br>
        ///     If a is an ndarray or a subclass
        ///     of ndarray, it is returned as-is and no copy is performed.
        /// </returns>
        public static NDarray<T> asanyarray<T>(T[,] a, Dtype dtype = null, string order = null) where T : struct
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray<T>(cp.asanyarray<T>(a, dtype?.CupyDtype, order));
            }
            else
            {
                return new NDarray<T>(np.asanyarray<T>(a, dtype?.NumpyDtype, order));
            }
        }

        /// <summary>
        ///     Return a contiguous array (ndim &gt;= 1) in memory (C order).
        /// </summary>
        /// <param name="a">
        ///     Input array.
        /// </param>
        /// <param name="dtype">
        ///     Data-type of returned array.
        /// </param>
        /// <returns>
        ///     Contiguous array of same shape and content as a, with type dtype
        ///     if specified.
        /// </returns>
        public static NDarray ascontiguousarray(NDarray a, Dtype dtype = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.ascontiguousarray(a.CupyNDarray, dtype?.CupyDtype));
            }
            else
            {
                return new NDarray(np.ascontiguousarray(a.NumpyNDarray, dtype?.NumpyDtype));
            }
        }

        /// <summary>
        ///     Return a contiguous array (ndim &gt;= 1) in memory (C order).
        /// </summary>
        /// <param name="a">
        ///     Input array.
        /// </param>
        /// <param name="dtype">
        ///     Data-type of returned array.
        /// </param>
        /// <returns>
        ///     Contiguous array of same shape and content as a, with type dtype
        ///     if specified.
        /// </returns>
        public static NDarray<T> ascontiguousarray<T>(T[] a, Dtype dtype = null) where T : struct
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray<T>(cp.ascontiguousarray<T>(a, dtype?.CupyDtype));
            }
            else
            {
                return new NDarray<T>(np.ascontiguousarray<T>(a, dtype?.NumpyDtype));
            }
        }

        /// <summary>
        ///     Return a contiguous array (ndim &gt;= 1) in memory (C order).
        /// </summary>
        /// <param name="a">
        ///     Input array.
        /// </param>
        /// <param name="dtype">
        ///     Data-type of returned array.
        /// </param>
        /// <returns>
        ///     Contiguous array of same shape and content as a, with type dtype
        ///     if specified.
        /// </returns>
        public static NDarray<T> ascontiguousarray<T>(T[,] a, Dtype dtype = null) where T : struct
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray<T>(cp.ascontiguousarray<T>(a, dtype?.CupyDtype));
            }
            else
            {
                return new NDarray<T>(np.ascontiguousarray<T>(a, dtype?.NumpyDtype));
            }
        }

        /// <summary>
        ///     Interpret the input as a matrix.<br></br>
        ///     Unlike matrix, asmatrix does not make a copy if the input is already
        ///     a matrix or an ndarray.<br></br>
        ///     Equivalent to matrix(data, copy=False).
        /// </summary>
        /// <param name="data">
        ///     Input data.
        /// </param>
        /// <param name="dtype">
        ///     Data-type of the output matrix.
        /// </param>
        /// <returns>
        ///     data interpreted as a matrix.
        /// </returns>
        public static Matrix asmatrix(NDarray data, Dtype dtype)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new Matrix(cp.asmatrix(data.CupyNDarray, dtype.CupyDtype));
            }
            else
            {
                return new Matrix(np.asmatrix(data.NumpyNDarray, dtype.NumpyDtype));
            }
        }

        /// <summary>
        ///     Interpret the input as a matrix.<br></br>
        ///     Unlike matrix, asmatrix does not make a copy if the input is already
        ///     a matrix or an ndarray.<br></br>
        ///     Equivalent to matrix(data, copy=False).
        /// </summary>
        /// <param name="data">
        ///     Input data.
        /// </param>
        /// <param name="dtype">
        ///     Data-type of the output matrix.
        /// </param>
        /// <returns>
        ///     data interpreted as a matrix.
        /// </returns>
        public static Matrix asmatrix<T>(T[] data, Dtype dtype)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new Matrix(cp.asmatrix<T>(data, dtype?.CupyDtype));
            }
            else
            {
                return new Matrix(np.asmatrix<T>(data, dtype?.NumpyDtype));
            }
        }

        /// <summary>
        ///     Interpret the input as a matrix.<br></br>
        ///     Unlike matrix, asmatrix does not make a copy if the input is already
        ///     a matrix or an ndarray.<br></br>
        ///     Equivalent to matrix(data, copy=False).
        /// </summary>
        /// <param name="data">
        ///     Input data.
        /// </param>
        /// <param name="dtype">
        ///     Data-type of the output matrix.
        /// </param>
        /// <returns>
        ///     data interpreted as a matrix.
        /// </returns>
        public static Matrix asmatrix<T>(T[,] data, Dtype dtype)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new Matrix(cp.asmatrix<T>(data, dtype.CupyDtype));
            }
            else
            {
                return new Matrix(np.asmatrix<T>(data, dtype.NumpyDtype));
            }
        }

        /// <summary>
        ///     Return an array copy of the given object.<br></br>
        ///     Notes
        ///     This is equivalent to:
        /// </summary>
        /// <param name="a">
        ///     Input data.
        /// </param>
        /// <param name="order">
        ///     Controls the memory layout of the copy.<br></br>
        ///     ‘C’ means C-order,
        ///     ‘F’ means F-order, ‘A’ means ‘F’ if a is Fortran contiguous,
        ///     ‘C’ otherwise.<br></br>
        ///     ‘K’ means match the layout of a as closely
        ///     as possible.<br></br>
        ///     (Note that this function and ndarray.copy are very
        ///     similar, but have different default values for their order=
        ///     arguments.)
        /// </param>
        /// <returns>
        ///     Array interpretation of a.
        /// </returns>
        public static NDarray copy(NDarray a, string order = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.copy(a.CupyNDarray, order));
            }
            else
            {
                return new NDarray(np.copy(a.NumpyNDarray, order));
            }
        }

        /// <summary>
        ///     Return an array copy of the given object.<br></br>
        ///     Notes
        ///     This is equivalent to:
        /// </summary>
        /// <param name="a">
        ///     Input data.
        /// </param>
        /// <param name="order">
        ///     Controls the memory layout of the copy.<br></br>
        ///     ‘C’ means C-order,
        ///     ‘F’ means F-order, ‘A’ means ‘F’ if a is Fortran contiguous,
        ///     ‘C’ otherwise.<br></br>
        ///     ‘K’ means match the layout of a as closely
        ///     as possible.<br></br>
        ///     (Note that this function and ndarray.copy are very
        ///     similar, but have different default values for their order=
        ///     arguments.)
        /// </param>
        /// <returns>
        ///     Array interpretation of a.
        /// </returns>
        public static NDarray<T> copy<T>(T[] a, string order = null) where T : struct
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray<T>(cp.copy(a, order));
            }
            else
            {
                return new NDarray<T>(np.copy(a, order));
            }
        }

        /// <summary>
        ///     Return an array copy of the given object.<br></br>
        ///     Notes
        ///     This is equivalent to:
        /// </summary>
        /// <param name="a">
        ///     Input data.
        /// </param>
        /// <param name="order">
        ///     Controls the memory layout of the copy.<br></br>
        ///     ‘C’ means C-order,
        ///     ‘F’ means F-order, ‘A’ means ‘F’ if a is Fortran contiguous,
        ///     ‘C’ otherwise.<br></br>
        ///     ‘K’ means match the layout of a as closely
        ///     as possible.<br></br>
        ///     (Note that this function and ndarray.copy are very
        ///     similar, but have different default values for their order=
        ///     arguments.)
        /// </param>
        /// <returns>
        ///     Array interpretation of a.
        /// </returns>
        public static NDarray<T> copy<T>(T[,] a, string order = null) where T : struct
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray<T>(cp.copy(a, order));
            }
            else
            {
                return new NDarray<T>(np.copy(a, order));
            }
        }

        /*
        /// <summary>
        ///	Interpret a buffer as a 1-dimensional array.<br></br>
        ///	
        ///	Notes
        ///	
        ///	If the buffer has data that is not in machine byte-order, this should
        ///	be specified as part of the data-type, e.g.:
        ///	
        ///	The data of the resulting array will not be byteswapped, but will be
        ///	interpreted correctly.
        /// </summary>
        /// <param name="buffer">
        ///	An object that exposes the buffer interface.
        /// </param>
        /// <param name="dtype">
        ///	Data-type of the returned array; default: float.
        /// </param>
        /// <param name="count">
        ///	Number of items to read.<br></br>
        ///	-1 means all data in the buffer.
        /// </param>
        /// <param name="offset">
        ///	Start reading the buffer from this offset (in bytes); default: 0.
        /// </param>
        public static void frombuffer(buffer_like buffer, Dtype dtype = null, int? count = -1, int? offset = 0)
        {
            //auto-generated code, do not change
            var __self__=self;
            var pyargs=ToTuple(new object[]
            {
                buffer,
            });
            var kwargs=new PyDict();
            if (dtype!=null) kwargs["dtype"]=ToPython(dtype);
            if (count!=-1) kwargs["count"]=ToPython(count);
            if (offset!=0) kwargs["offset"]=ToPython(offset);
            dynamic py = __self__.InvokeMethod("frombuffer", pyargs, kwargs);
        }
        */

        /// <summary>
        ///     Construct an array from data in a text or binary file.<br></br>
        ///     A highly efficient way of reading binary data with a known data-type,
        ///     as well as parsing simply formatted text files.<br></br>
        ///     Data written using the
        ///     tofile method can be read using this function.<br></br>
        ///     Notes
        ///     Do not rely on the combination of tofile and fromfile for
        ///     data storage, as the binary files generated are are not platform
        ///     independent.<br></br>
        ///     In particular, no byte-order or data-type information is
        ///     saved.<br></br>
        ///     Data can be stored in the platform independent .npy format
        ///     using save and load instead.
        /// </summary>
        /// <param name="file">
        ///     Open file object or filename.
        /// </param>
        /// <param name="dtype">
        ///     Data type of the returned array.<br></br>
        ///     For binary files, it is used to determine the size and byte-order
        ///     of the items in the file.
        /// </param>
        /// <param name="count">
        ///     Number of items to read.<br></br>
        ///     -1 means all items (i.e., the complete
        ///     file).
        /// </param>
        /// <param name="sep">
        ///     Separator between items if file is a text file.<br></br>
        ///     Empty (“”) separator means the file should be treated as binary.<br></br>
        ///     Spaces (” “) in the separator match zero or more whitespace characters.<br></br>
        ///     A separator consisting only of spaces must match at least one
        ///     whitespace.
        /// </param>
        public static NDarray fromfile(string file, Dtype dtype = null, int count = -1, string sep = "")
        {
            //auto-generated code, do not change
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.fromfile(file, dtype?.CupyDtype, count, sep));
            }
            else
            {
                return new NDarray(np.fromfile(file, dtype?.NumpyDtype, count, sep));
            }
        }

        /// <summary>
        ///     Construct an array by executing a function over each coordinate.<br></br>
        ///     The resulting array therefore has a value fn(x, y, z) at
        ///     coordinate (x, y, z).<br></br>
        ///     Notes
        ///     Keywords other than dtype are passed to function.
        /// </summary>
        /// <param name="function">
        ///     The function is called with N parameters, where N is the rank of
        ///     shape.<br></br>
        ///     Each parameter represents the coordinates of the array
        ///     varying along a specific axis.<br></br>
        ///     For example, if shape
        ///     were (2, 2), then the parameters would be
        ///     array([[0, 0], [1, 1]]) and array([[0, 1], [0, 1]])
        /// </param>
        /// <param name="shape">
        ///     Shape of the output array, which also determines the shape of
        ///     the coordinate arrays passed to function.
        /// </param>
        /// <param name="dtype">
        ///     Data-type of the coordinate arrays passed to function.<br></br>
        ///     By default, dtype is float.
        /// </param>
        /// <returns>
        ///     The result of the call to function is passed back directly.<br></br>
        ///     Therefore the shape of fromfunction is completely determined by
        ///     function.<br></br>
        ///     If function returns a scalar value, the shape of
        ///     fromfunction would not match the shape parameter.
        /// </returns>
        public static object fromfunction(Delegate function, Shape shape, Dtype dtype = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return cp.fromfunction(function, shape.CupyShape, dtype?.CupyDtype);
            }
            else
            {
                return np.fromfunction(function, shape.NumpyShape, dtype?.NumpyDtype);
            }
        }

        /// <summary>
        ///     Create a new 1-dimensional array from an iterable object.<br></br>
        ///     Notes
        ///     Specify count to improve performance.<br></br>
        ///     It allows fromiter to
        ///     pre-allocate the output array, instead of resizing it on demand.
        /// </summary>
        /// <param name="iterable">
        ///     An iterable object providing data for the array.
        /// </param>
        /// <param name="dtype">
        ///     The data-type of the returned array.
        /// </param>
        /// <param name="count">
        ///     The number of items to read from iterable.<br></br>
        ///     The default is -1,
        ///     which means all data is read.
        /// </param>
        /// <returns>
        ///     The output array.
        /// </returns>
        public static NDarray<T> fromiter<T>(IEnumerable<T> iterable, Dtype dtype, int? count = -1) where T : struct
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray<T>(cp.fromiter(iterable, dtype.CupyDtype, count));
            }
            else
            {
                return new NDarray<T>(np.fromiter(iterable, dtype.NumpyDtype, count));
            }
        }

        /// <summary>
        ///     A new 1-D array initialized from text data in a string.
        /// </summary>
        /// <param name="string">
        ///     A string containing the data.
        /// </param>
        /// <param name="dtype">
        ///     The data type of the array; default: float.<br></br>
        ///     For binary input data,
        ///     the data must be in exactly this format.
        /// </param>
        /// <param name="count">
        ///     Read this number of dtype elements from the data.<br></br>
        ///     If this is
        ///     negative (the default), the count will be determined from the
        ///     length of the data.
        /// </param>
        /// <param name="sep">
        ///     The string separating numbers in the data; extra whitespace between
        ///     elements is also ignored.
        /// </param>
        /// <returns>
        ///     The constructed array.
        /// </returns>
        public static NDarray fromstring(string @string, Dtype dtype = null, int? count = -1, string sep = "")
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.fromstring(@string, dtype?.CupyDtype, count, sep));
            }
            else
            {
                return new NDarray(np.fromstring(@string, dtype?.NumpyDtype, count, sep));
            }
        }

        /// <summary>
        ///     Load data from a text file.<br></br>
        ///     Each row in the text file must have the same number of values.<br></br>
        ///     Notes
        ///     This function aims to be a fast reader for simply formatted files.<br></br>
        ///     The
        ///     genfromtxt function provides more sophisticated handling of, e.g.,
        ///     lines with missing values.<br></br>
        ///     The strings produced by the Python float.hex method can be used as
        ///     input for floats.
        /// </summary>
        /// <param name="fname">
        ///     File, filename, or generator to read.<br></br>
        ///     If the filename extension is
        ///     .gz or .bz2, the file is first decompressed.<br></br>
        ///     Note that
        ///     generators should return byte strings for Python 3k.
        /// </param>
        /// <param name="dtype">
        ///     Data-type of the resulting array; default: float.<br></br>
        ///     If this is a
        ///     structured data-type, the resulting array will be 1-dimensional, and
        ///     each row will be interpreted as an element of the array.<br></br>
        ///     In this
        ///     case, the number of columns used must match the number of fields in
        ///     the data-type.
        /// </param>
        /// <param name="comments">
        ///     The characters or list of characters used to indicate the start of a
        ///     comment.<br></br>
        ///     None implies no comments.<br></br>
        ///     For backwards compatibility, byte
        ///     strings will be decoded as ‘latin1’. The default is ‘#’.
        /// </param>
        /// <param name="delimiter">
        ///     The string used to separate values.<br></br>
        ///     For backwards compatibility, byte
        ///     strings will be decoded as ‘latin1’. The default is whitespace.
        /// </param>
        /// <param name="converters">
        ///     A dictionary mapping column number to a function that will parse the
        ///     column string into the desired value.<br></br>
        ///     E.g., if column 0 is a date
        ///     string: converters = {0: datestr2num}.  Converters can also be
        ///     used to provide a default value for missing data (but see also
        ///     genfromtxt): converters = {3: lambda s: float(s.strip() or 0)}.
        ///     Default: None.
        /// </param>
        /// <param name="skiprows">
        ///     Skip the first skiprows lines; default: 0.
        /// </param>
        /// <param name="usecols">
        ///     Which columns to read, with 0 being the first.<br></br>
        ///     For example,
        ///     usecols = (1,4,5) will extract the 2nd, 5th and 6th columns.<br></br>
        ///     The default, None, results in all columns being read.
        /// </param>
        /// <param name="unpack">
        ///     If True, the returned array is transposed, so that arguments may be
        ///     unpacked using x, y, z = loadtxt(...).<br></br>
        ///     When used with a structured
        ///     data-type, arrays are returned for each field.<br></br>
        ///     Default is False.
        /// </param>
        /// <param name="ndmin">
        ///     The returned array will have at least ndmin dimensions.<br></br>
        ///     Otherwise mono-dimensional axis will be squeezed.<br></br>
        ///     Legal values: 0 (default), 1 or 2.
        /// </param>
        /// <param name="encoding">
        ///     Encoding used to decode the inputfile.<br></br>
        ///     Does not apply to input streams.<br></br>
        ///     The special value ‘bytes’ enables backward compatibility workarounds
        ///     that ensures you receive byte arrays as results if possible and passes
        ///     ‘latin1’ encoded strings to converters.<br></br>
        ///     Override this value to receive
        ///     unicode arrays and pass strings as input to converters.<br></br>
        ///     If set to None
        ///     the system default is used.<br></br>
        ///     The default value is ‘bytes’.
        /// </param>
        /// <param name="max_rows">
        ///     Read max_rows lines of content after skiprows lines.<br></br>
        ///     The default
        ///     is to read all the lines.
        /// </param>
        /// <returns>
        ///     Data read from the text file.
        /// </returns>
        public static NDarray loadtxt(string fname, Dtype dtype = null, string[] comments = null,
            string delimiter = null, Hashtable converters = null, int? skiprows = 0, int[] usecols = null,
            bool? unpack = false, int? ndmin = 0, string encoding = "bytes", int? max_rows = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.loadtxt(fname, dtype?.CupyDtype, comments, delimiter, converters, skiprows,
                    usecols, unpack, ndmin, encoding, max_rows));
            }
            else
            {
                return new NDarray(np.loadtxt(fname, dtype?.NumpyDtype, comments, delimiter, converters, skiprows,
                    usecols, unpack, ndmin, encoding, max_rows));
            }
        }

        /// <summary>
        ///     Provides a convenient view on arrays of string and unicode values.<br></br>
        ///     Versus a regular Cupy array of type str or unicode, this
        ///     class adds the following functionality:
        ///     chararrays should be created using Cupy.char.array or
        ///     Cupy.char.asarray, rather than this constructor directly.<br></br>
        ///     This constructor creates the array, using buffer (with offset
        ///     and strides) if it is not None.<br></br>
        ///     If buffer is None, then
        ///     constructs a new array with strides in “C order”, unless both
        ///     len(shape) &gt;= 2 and order='Fortran', in which case strides
        ///     is in “Fortran order”.
        /// </summary>
        /// <param name="shape">
        ///     Shape of the array.
        /// </param>
        /// <param name="itemsize">
        ///     Length of each array element, in number of characters.<br></br>
        ///     Default is 1.
        /// </param>
        /// <param name="unicode">
        ///     Are the array elements of type unicode (True) or string (False).<br></br>
        ///     Default is False.
        /// </param>
        /// <param name="buffer">
        ///     Memory address of the start of the array data.<br></br>
        ///     Default is None,
        ///     in which case a new array is created.
        /// </param>
        /// <param name="offset">
        ///     Fixed stride displacement from the beginning of an axis?
        ///     Default is 0.<br></br>
        ///     Needs to be &gt;=0.
        /// </param>
        /// <param name="strides">
        ///     Strides for the array (see ndarray.strides for full description).<br></br>
        ///     Default is None.
        /// </param>
        /// <param name="order">
        ///     The order in which the array data is stored in memory: ‘C’ -&gt;
        ///     “row major” order (the default), ‘F’ -&gt; “column major”
        ///     (Fortran) order.
        /// </param>
        public static void chararray(Shape shape, int? itemsize = null, bool? unicode = null, int? buffer = null,
            int? offset = null, int[] strides = null, string order = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                cp.chararray(shape.CupyShape, itemsize, unicode, buffer, offset, strides, order);
            }
            else
            {
                np.chararray(shape.NumpyShape, itemsize, unicode, buffer, offset, strides, order);
            }
        }

        /// <summary>
        ///     Return evenly spaced values within a given interval.<br></br>
        ///     Values are generated within the half-open interval [start, stop)
        ///     (in other words, the interval including start but excluding stop).<br></br>
        ///     For integer arguments the function is equivalent to the Python built-in
        ///     range function, but returns an ndarray rather than a list.<br></br>
        ///     When using a non-integer step, such as 0.1, the results will often not
        ///     be consistent.<br></br>
        ///     It is better to use Cupy.linspace for these cases.
        /// </summary>
        /// <param name="start">
        ///     Start of interval.<br></br>
        ///     The interval includes this value.<br></br>
        ///     The default
        ///     start value is 0.
        /// </param>
        /// <param name="stop">
        ///     End of interval.<br></br>
        ///     The interval does not include this value, except
        ///     in some cases where step is not an integer and floating point
        ///     round-off affects the length of out.
        /// </param>
        /// <param name="step">
        ///     Spacing between values.<br></br>
        ///     For any output out, this is the distance
        ///     between two adjacent values, out[i+1] - out[i].<br></br>
        ///     The default
        ///     step size is 1.<br></br>
        ///     If step is specified as a position argument,
        ///     start must also be given.
        /// </param>
        /// <param name="dtype">
        ///     The type of the output array.<br></br>
        ///     If dtype is not given, infer the data
        ///     type from the other input arguments.
        /// </param>
        /// <returns>
        ///     Array of evenly spaced values.<br></br>
        ///     For floating point arguments, the length of the result is
        ///     ceil((stop - start)/step).<br></br>
        ///     Because of floating point overflow,
        ///     this rule may result in the last element of out being greater
        ///     than stop.
        /// </returns>
        public static NDarray arange(byte start, byte stop, byte step = 1, Dtype dtype = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.arange(start, stop, step, dtype?.CupyDtype));
            }
            else
            {
                return new NDarray(np.arange(start, stop, step, dtype?.NumpyDtype));
            }
        }

        /// <summary>
        ///     Return evenly spaced values within a given interval.<br></br>
        ///     Values are generated within the half-open interval [start, stop)
        ///     (in other words, the interval including start but excluding stop).<br></br>
        ///     For integer arguments the function is equivalent to the Python built-in
        ///     range function, but returns an ndarray rather than a list.<br></br>
        ///     When using a non-integer step, such as 0.1, the results will often not
        ///     be consistent.<br></br>
        ///     It is better to use Cupy.linspace for these cases.
        /// </summary>
        /// <param name="stop">
        ///     End of interval.<br></br>
        ///     The interval does not include this value, except
        ///     in some cases where step is not an integer and floating point
        ///     round-off affects the length of out.
        /// </param>
        /// <param name="step">
        ///     Spacing between values.<br></br>
        ///     For any output out, this is the distance
        ///     between two adjacent values, out[i+1] - out[i].<br></br>
        ///     The default
        ///     step size is 1.<br></br>
        ///     If step is specified as a position argument,
        ///     start must also be given.
        /// </param>
        /// <param name="dtype">
        ///     The type of the output array.<br></br>
        ///     If dtype is not given, infer the data
        ///     type from the other input arguments.
        /// </param>
        /// <returns>
        ///     Array of evenly spaced values.<br></br>
        ///     For floating point arguments, the length of the result is
        ///     ceil((stop - start)/step).<br></br>
        ///     Because of floating point overflow,
        ///     this rule may result in the last element of out being greater
        ///     than stop.
        /// </returns>
        public static NDarray arange(byte stop, byte step = 1, Dtype dtype = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.arange(stop, step, dtype?.CupyDtype));
            }
            else
            {
                return new NDarray(np.arange(stop, step, dtype?.NumpyDtype));
            }
        }

        /// <summary>
        ///     Return evenly spaced values within a given interval.<br></br>
        ///     Values are generated within the half-open interval [start, stop)
        ///     (in other words, the interval including start but excluding stop).<br></br>
        ///     For integer arguments the function is equivalent to the Python built-in
        ///     range function, but returns an ndarray rather than a list.<br></br>
        ///     When using a non-integer step, such as 0.1, the results will often not
        ///     be consistent.<br></br>
        ///     It is better to use Cupy.linspace for these cases.
        /// </summary>
        /// <param name="start">
        ///     Start of interval.<br></br>
        ///     The interval includes this value.<br></br>
        ///     The default
        ///     start value is 0.
        /// </param>
        /// <param name="stop">
        ///     End of interval.<br></br>
        ///     The interval does not include this value, except
        ///     in some cases where step is not an integer and floating point
        ///     round-off affects the length of out.
        /// </param>
        /// <param name="step">
        ///     Spacing between values.<br></br>
        ///     For any output out, this is the distance
        ///     between two adjacent values, out[i+1] - out[i].<br></br>
        ///     The default
        ///     step size is 1.<br></br>
        ///     If step is specified as a position argument,
        ///     start must also be given.
        /// </param>
        /// <param name="dtype">
        ///     The type of the output array.<br></br>
        ///     If dtype is not given, infer the data
        ///     type from the other input arguments.
        /// </param>
        /// <returns>
        ///     Array of evenly spaced values.<br></br>
        ///     For floating point arguments, the length of the result is
        ///     ceil((stop - start)/step).<br></br>
        ///     Because of floating point overflow,
        ///     this rule may result in the last element of out being greater
        ///     than stop.
        /// </returns>
        public static NDarray arange(short start, short stop, short step = 1, Dtype dtype = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.arange(start, stop, step, dtype?.CupyDtype));
            }
            else
            {
                return new NDarray(np.arange(start, stop, step, dtype?.NumpyDtype));
            }
        }

        /// <summary>
        ///     Return evenly spaced values within a given interval.<br></br>
        ///     Values are generated within the half-open interval [start, stop)
        ///     (in other words, the interval including start but excluding stop).<br></br>
        ///     For integer arguments the function is equivalent to the Python built-in
        ///     range function, but returns an ndarray rather than a list.<br></br>
        ///     When using a non-integer step, such as 0.1, the results will often not
        ///     be consistent.<br></br>
        ///     It is better to use Cupy.linspace for these cases.
        /// </summary>
        /// <param name="stop">
        ///     End of interval.<br></br>
        ///     The interval does not include this value, except
        ///     in some cases where step is not an integer and floating point
        ///     round-off affects the length of out.
        /// </param>
        /// <param name="step">
        ///     Spacing between values.<br></br>
        ///     For any output out, this is the distance
        ///     between two adjacent values, out[i+1] - out[i].<br></br>
        ///     The default
        ///     step size is 1.<br></br>
        ///     If step is specified as a position argument,
        ///     start must also be given.
        /// </param>
        /// <param name="dtype">
        ///     The type of the output array.<br></br>
        ///     If dtype is not given, infer the data
        ///     type from the other input arguments.
        /// </param>
        /// <returns>
        ///     Array of evenly spaced values.<br></br>
        ///     For floating point arguments, the length of the result is
        ///     ceil((stop - start)/step).<br></br>
        ///     Because of floating point overflow,
        ///     this rule may result in the last element of out being greater
        ///     than stop.
        /// </returns>
        public static NDarray arange(short stop, short step = 1, Dtype dtype = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.arange(stop, step, dtype?.CupyDtype));
            }
            else
            {
                return new NDarray(np.arange(stop, step, dtype?.NumpyDtype));
            }
        }

        /// <summary>
        ///     Return evenly spaced values within a given interval.<br></br>
        ///     Values are generated within the half-open interval [start, stop)
        ///     (in other words, the interval including start but excluding stop).<br></br>
        ///     For integer arguments the function is equivalent to the Python built-in
        ///     range function, but returns an ndarray rather than a list.<br></br>
        ///     When using a non-integer step, such as 0.1, the results will often not
        ///     be consistent.<br></br>
        ///     It is better to use Cupy.linspace for these cases.
        /// </summary>
        /// <param name="start">
        ///     Start of interval.<br></br>
        ///     The interval includes this value.<br></br>
        ///     The default
        ///     start value is 0.
        /// </param>
        /// <param name="stop">
        ///     End of interval.<br></br>
        ///     The interval does not include this value, except
        ///     in some cases where step is not an integer and floating point
        ///     round-off affects the length of out.
        /// </param>
        /// <param name="step">
        ///     Spacing between values.<br></br>
        ///     For any output out, this is the distance
        ///     between two adjacent values, out[i+1] - out[i].<br></br>
        ///     The default
        ///     step size is 1.<br></br>
        ///     If step is specified as a position argument,
        ///     start must also be given.
        /// </param>
        /// <param name="dtype">
        ///     The type of the output array.<br></br>
        ///     If dtype is not given, infer the data
        ///     type from the other input arguments.
        /// </param>
        /// <returns>
        ///     Array of evenly spaced values.<br></br>
        ///     For floating point arguments, the length of the result is
        ///     ceil((stop - start)/step).<br></br>
        ///     Because of floating point overflow,
        ///     this rule may result in the last element of out being greater
        ///     than stop.
        /// </returns>
        public static NDarray arange(int start, int stop, int step = 1, Dtype dtype = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.arange(start, stop, step, dtype?.CupyDtype));
            }
            else
            {
                return new NDarray(np.arange(start, stop, step, dtype?.NumpyDtype));
            }
        }

        /// <summary>
        ///     Return evenly spaced values within a given interval.<br></br>
        ///     Values are generated within the half-open interval [start, stop)
        ///     (in other words, the interval including start but excluding stop).<br></br>
        ///     For integer arguments the function is equivalent to the Python built-in
        ///     range function, but returns an ndarray rather than a list.<br></br>
        ///     When using a non-integer step, such as 0.1, the results will often not
        ///     be consistent.<br></br>
        ///     It is better to use Cupy.linspace for these cases.
        /// </summary>
        /// <param name="stop">
        ///     End of interval.<br></br>
        ///     The interval does not include this value, except
        ///     in some cases where step is not an integer and floating point
        ///     round-off affects the length of out.
        /// </param>
        /// <param name="step">
        ///     Spacing between values.<br></br>
        ///     For any output out, this is the distance
        ///     between two adjacent values, out[i+1] - out[i].<br></br>
        ///     The default
        ///     step size is 1.<br></br>
        ///     If step is specified as a position argument,
        ///     start must also be given.
        /// </param>
        /// <param name="dtype">
        ///     The type of the output array.<br></br>
        ///     If dtype is not given, infer the data
        ///     type from the other input arguments.
        /// </param>
        /// <returns>
        ///     Array of evenly spaced values.<br></br>
        ///     For floating point arguments, the length of the result is
        ///     ceil((stop - start)/step).<br></br>
        ///     Because of floating point overflow,
        ///     this rule may result in the last element of out being greater
        ///     than stop.
        /// </returns>
        public static NDarray arange(int stop, int step = 1, Dtype dtype = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.arange(stop, step, dtype?.CupyDtype));
            }
            else
            {
                return new NDarray(np.arange(stop, step, dtype?.NumpyDtype));
            }
        }

        /// <summary>
        ///     Return evenly spaced values within a given interval.<br></br>
        ///     Values are generated within the half-open interval [start, stop)
        ///     (in other words, the interval including start but excluding stop).<br></br>
        ///     For integer arguments the function is equivalent to the Python built-in
        ///     range function, but returns an ndarray rather than a list.<br></br>
        ///     When using a non-integer step, such as 0.1, the results will often not
        ///     be consistent.<br></br>
        ///     It is better to use Cupy.linspace for these cases.
        /// </summary>
        /// <param name="start">
        ///     Start of interval.<br></br>
        ///     The interval includes this value.<br></br>
        ///     The default
        ///     start value is 0.
        /// </param>
        /// <param name="stop">
        ///     End of interval.<br></br>
        ///     The interval does not include this value, except
        ///     in some cases where step is not an integer and floating point
        ///     round-off affects the length of out.
        /// </param>
        /// <param name="step">
        ///     Spacing between values.<br></br>
        ///     For any output out, this is the distance
        ///     between two adjacent values, out[i+1] - out[i].<br></br>
        ///     The default
        ///     step size is 1.<br></br>
        ///     If step is specified as a position argument,
        ///     start must also be given.
        /// </param>
        /// <param name="dtype">
        ///     The type of the output array.<br></br>
        ///     If dtype is not given, infer the data
        ///     type from the other input arguments.
        /// </param>
        /// <returns>
        ///     Array of evenly spaced values.<br></br>
        ///     For floating point arguments, the length of the result is
        ///     ceil((stop - start)/step).<br></br>
        ///     Because of floating point overflow,
        ///     this rule may result in the last element of out being greater
        ///     than stop.
        /// </returns>
        public static NDarray arange(long start, long stop, long step = 1, Dtype dtype = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.arange(start, stop, step, dtype?.CupyDtype));
            }
            else
            {
                return new NDarray(np.arange(start, stop, step, dtype?.NumpyDtype));
            }
        }

        /// <summary>
        ///     Return evenly spaced values within a given interval.<br></br>
        ///     Values are generated within the half-open interval [start, stop)
        ///     (in other words, the interval including start but excluding stop).<br></br>
        ///     For integer arguments the function is equivalent to the Python built-in
        ///     range function, but returns an ndarray rather than a list.<br></br>
        ///     When using a non-integer step, such as 0.1, the results will often not
        ///     be consistent.<br></br>
        ///     It is better to use Cupy.linspace for these cases.
        /// </summary>
        /// <param name="stop">
        ///     End of interval.<br></br>
        ///     The interval does not include this value, except
        ///     in some cases where step is not an integer and floating point
        ///     round-off affects the length of out.
        /// </param>
        /// <param name="step">
        ///     Spacing between values.<br></br>
        ///     For any output out, this is the distance
        ///     between two adjacent values, out[i+1] - out[i].<br></br>
        ///     The default
        ///     step size is 1.<br></br>
        ///     If step is specified as a position argument,
        ///     start must also be given.
        /// </param>
        /// <param name="dtype">
        ///     The type of the output array.<br></br>
        ///     If dtype is not given, infer the data
        ///     type from the other input arguments.
        /// </param>
        /// <returns>
        ///     Array of evenly spaced values.<br></br>
        ///     For floating point arguments, the length of the result is
        ///     ceil((stop - start)/step).<br></br>
        ///     Because of floating point overflow,
        ///     this rule may result in the last element of out being greater
        ///     than stop.
        /// </returns>
        public static NDarray arange(long stop, long step = 1, Dtype dtype = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.arange(stop, step, dtype?.CupyDtype));
            }
            else
            {
                return new NDarray(np.arange(stop, step, dtype?.NumpyDtype));
            }
        }

        /// <summary>
        ///     Return evenly spaced values within a given interval.<br></br>
        ///     Values are generated within the half-open interval [start, stop)
        ///     (in other words, the interval including start but excluding stop).<br></br>
        ///     For integer arguments the function is equivalent to the Python built-in
        ///     range function, but returns an ndarray rather than a list.<br></br>
        ///     When using a non-integer step, such as 0.1, the results will often not
        ///     be consistent.<br></br>
        ///     It is better to use Cupy.linspace for these cases.
        /// </summary>
        /// <param name="start">
        ///     Start of interval.<br></br>
        ///     The interval includes this value.<br></br>
        ///     The default
        ///     start value is 0.
        /// </param>
        /// <param name="stop">
        ///     End of interval.<br></br>
        ///     The interval does not include this value, except
        ///     in some cases where step is not an integer and floating point
        ///     round-off affects the length of out.
        /// </param>
        /// <param name="step">
        ///     Spacing between values.<br></br>
        ///     For any output out, this is the distance
        ///     between two adjacent values, out[i+1] - out[i].<br></br>
        ///     The default
        ///     step size is 1.<br></br>
        ///     If step is specified as a position argument,
        ///     start must also be given.
        /// </param>
        /// <param name="dtype">
        ///     The type of the output array.<br></br>
        ///     If dtype is not given, infer the data
        ///     type from the other input arguments.
        /// </param>
        /// <returns>
        ///     Array of evenly spaced values.<br></br>
        ///     For floating point arguments, the length of the result is
        ///     ceil((stop - start)/step).<br></br>
        ///     Because of floating point overflow,
        ///     this rule may result in the last element of out being greater
        ///     than stop.
        /// </returns>
        public static NDarray arange(float start, float stop, float step = 1, Dtype dtype = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.arange(start, stop, step, dtype?.CupyDtype));
            }
            else
            {
                return new NDarray(np.arange(start, stop, step, dtype?.NumpyDtype));
            }
        }

        /// <summary>
        ///     Return evenly spaced values within a given interval.<br></br>
        ///     Values are generated within the half-open interval [start, stop)
        ///     (in other words, the interval including start but excluding stop).<br></br>
        ///     For integer arguments the function is equivalent to the Python built-in
        ///     range function, but returns an ndarray rather than a list.<br></br>
        ///     When using a non-integer step, such as 0.1, the results will often not
        ///     be consistent.<br></br>
        ///     It is better to use Cupy.linspace for these cases.
        /// </summary>
        /// <param name="stop">
        ///     End of interval.<br></br>
        ///     The interval does not include this value, except
        ///     in some cases where step is not an integer and floating point
        ///     round-off affects the length of out.
        /// </param>
        /// <param name="step">
        ///     Spacing between values.<br></br>
        ///     For any output out, this is the distance
        ///     between two adjacent values, out[i+1] - out[i].<br></br>
        ///     The default
        ///     step size is 1.<br></br>
        ///     If step is specified as a position argument,
        ///     start must also be given.
        /// </param>
        /// <param name="dtype">
        ///     The type of the output array.<br></br>
        ///     If dtype is not given, infer the data
        ///     type from the other input arguments.
        /// </param>
        /// <returns>
        ///     Array of evenly spaced values.<br></br>
        ///     For floating point arguments, the length of the result is
        ///     ceil((stop - start)/step).<br></br>
        ///     Because of floating point overflow,
        ///     this rule may result in the last element of out being greater
        ///     than stop.
        /// </returns>
        public static NDarray arange(float stop, float step = 1, Dtype dtype = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.arange(stop, step, dtype?.CupyDtype));
            }
            else
            {
                return new NDarray(np.arange(stop, step, dtype?.NumpyDtype));
            }
        }

        /// <summary>
        ///     Return evenly spaced values within a given interval.<br></br>
        ///     Values are generated within the half-open interval [start, stop)
        ///     (in other words, the interval including start but excluding stop).<br></br>
        ///     For integer arguments the function is equivalent to the Python built-in
        ///     range function, but returns an ndarray rather than a list.<br></br>
        ///     When using a non-integer step, such as 0.1, the results will often not
        ///     be consistent.<br></br>
        ///     It is better to use Cupy.linspace for these cases.
        /// </summary>
        /// <param name="start">
        ///     Start of interval.<br></br>
        ///     The interval includes this value.<br></br>
        ///     The default
        ///     start value is 0.
        /// </param>
        /// <param name="stop">
        ///     End of interval.<br></br>
        ///     The interval does not include this value, except
        ///     in some cases where step is not an integer and floating point
        ///     round-off affects the length of out.
        /// </param>
        /// <param name="step">
        ///     Spacing between values.<br></br>
        ///     For any output out, this is the distance
        ///     between two adjacent values, out[i+1] - out[i].<br></br>
        ///     The default
        ///     step size is 1.<br></br>
        ///     If step is specified as a position argument,
        ///     start must also be given.
        /// </param>
        /// <param name="dtype">
        ///     The type of the output array.<br></br>
        ///     If dtype is not given, infer the data
        ///     type from the other input arguments.
        /// </param>
        /// <returns>
        ///     Array of evenly spaced values.<br></br>
        ///     For floating point arguments, the length of the result is
        ///     ceil((stop - start)/step).<br></br>
        ///     Because of floating point overflow,
        ///     this rule may result in the last element of out being greater
        ///     than stop.
        /// </returns>
        public static NDarray arange(double start, double stop, double step = 1, Dtype dtype = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.arange(start, stop, step, dtype?.CupyDtype));
            }
            else
            {
                return new NDarray(np.arange(start, stop, step, dtype?.NumpyDtype));
            }
        }

        /// <summary>
        ///     Return evenly spaced values within a given interval.<br></br>
        ///     Values are generated within the half-open interval [start, stop)
        ///     (in other words, the interval including start but excluding stop).<br></br>
        ///     For integer arguments the function is equivalent to the Python built-in
        ///     range function, but returns an ndarray rather than a list.<br></br>
        ///     When using a non-integer step, such as 0.1, the results will often not
        ///     be consistent.<br></br>
        ///     It is better to use Cupy.linspace for these cases.
        /// </summary>
        /// <param name="stop">
        ///     End of interval.<br></br>
        ///     The interval does not include this value, except
        ///     in some cases where step is not an integer and floating point
        ///     round-off affects the length of out.
        /// </param>
        /// <param name="step">
        ///     Spacing between values.<br></br>
        ///     For any output out, this is the distance
        ///     between two adjacent values, out[i+1] - out[i].<br></br>
        ///     The default
        ///     step size is 1.<br></br>
        ///     If step is specified as a position argument,
        ///     start must also be given.
        /// </param>
        /// <param name="dtype">
        ///     The type of the output array.<br></br>
        ///     If dtype is not given, infer the data
        ///     type from the other input arguments.
        /// </param>
        /// <returns>
        ///     Array of evenly spaced values.<br></br>
        ///     For floating point arguments, the length of the result is
        ///     ceil((stop - start)/step).<br></br>
        ///     Because of floating point overflow,
        ///     this rule may result in the last element of out being greater
        ///     than stop.
        /// </returns>
        public static NDarray arange(double stop, double step = 1, Dtype dtype = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.arange(stop, step, dtype?.CupyDtype));
            }
            else
            {
                return new NDarray(np.arange(stop, step, dtype?.NumpyDtype));
            }
        }

        /// <summary>
        ///     Return evenly spaced numbers over a specified interval.<br></br>
        ///     Returns num evenly spaced samples, calculated over the
        ///     interval [start, stop].<br></br>
        ///     The endpoint of the interval can optionally be excluded.
        /// </summary>
        /// <param name="start">
        ///     The starting value of the sequence.
        /// </param>
        /// <param name="stop">
        ///     The end value of the sequence, unless endpoint is set to False.<br></br>
        ///     In that case, the sequence consists of all but the last of num + 1
        ///     evenly spaced samples, so that stop is excluded.<br></br>
        ///     Note that the step
        ///     size changes when endpoint is False.
        /// </param>
        /// <param name="num">
        ///     Number of samples to generate.<br></br>
        ///     Default is 50. Must be non-negative.
        /// </param>
        /// <param name="endpoint">
        ///     If True, stop is the last sample.<br></br>
        ///     Otherwise, it is not included.<br></br>
        ///     Default is True.
        /// </param>
        /// <param name="dtype">
        ///     The type of the output array.<br></br>
        ///     If dtype is not given, infer the data
        ///     type from the other input arguments.
        /// </param>
        /// <param name="axis">
        ///     The axis in the result to store the samples.<br></br>
        ///     Relevant only if start
        ///     or stop are array-like.<br></br>
        ///     By default (0), the samples will be along a
        ///     new axis inserted at the beginning.<br></br>
        ///     Use -1 to get an axis at the end.
        /// </param>
        /// <returns>
        ///     A tuple of:
        ///     samples
        ///     There are num equally spaced samples in the closed interval
        ///     [start, stop] or the half-open interval [start, stop)
        ///     (depending on whether endpoint is True or False).
        ///     step
        ///     Only returned if retstep is True
        ///     Size of spacing between samples.
        /// </returns>
        public static (NDarray, float) linspace(NDarray start, NDarray stop, int? num = 50, bool? endpoint = true,
            Dtype dtype = null, int? axis = 0)
        {
            if (Gpu.Available && Gpu.Use)
            {
                var ret = cp.linspace(start.CupyNDarray, stop.CupyNDarray, num, endpoint, dtype?.CupyDtype, axis);
                return (new NDarray(ret.Item1), ret.Item2);
            }
            else
            {
                var ret = np.linspace(start.NumpyNDarray, stop.NumpyNDarray, num, endpoint, dtype?.NumpyDtype, axis);
                return (new NDarray(ret.Item1), ret.Item2);
            }
        }

        /// <summary>
        ///     Return evenly spaced numbers over a specified interval.<br></br>
        ///     Returns num evenly spaced samples, calculated over the
        ///     interval [start, stop].<br></br>
        ///     The endpoint of the interval can optionally be excluded.
        /// </summary>
        /// <param name="start">
        ///     The starting value of the sequence.
        /// </param>
        /// <param name="stop">
        ///     The end value of the sequence, unless endpoint is set to False.<br></br>
        ///     In that case, the sequence consists of all but the last of num + 1
        ///     evenly spaced samples, so that stop is excluded.<br></br>
        ///     Note that the step
        ///     size changes when endpoint is False.
        /// </param>
        /// <param name="num">
        ///     Number of samples to generate.<br></br>
        ///     Default is 50. Must be non-negative.
        /// </param>
        /// <param name="endpoint">
        ///     If True, stop is the last sample.<br></br>
        ///     Otherwise, it is not included.<br></br>
        ///     Default is True.
        /// </param>
        /// <param name="dtype">
        ///     The type of the output array.<br></br>
        ///     If dtype is not given, infer the data
        ///     type from the other input arguments.
        /// </param>
        /// <param name="axis">
        ///     The axis in the result to store the samples.<br></br>
        ///     Relevant only if start
        ///     or stop are array-like.<br></br>
        ///     By default (0), the samples will be along a
        ///     new axis inserted at the beginning.<br></br>
        ///     Use -1 to get an axis at the end.
        /// </param>
        /// <returns>
        ///     There are num equally spaced samples in the closed interval
        ///     [start, stop] or the half-open interval [start, stop)
        ///     (depending on whether endpoint is True or False).
        /// </returns>
        public static NDarray linspace(double start, double stop, int? num = 50, bool? endpoint = true,
            Dtype dtype = null, int? axis = 0)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.linspace(start, stop, num, endpoint, dtype?.CupyDtype, axis));
            }
            else
            {
                return new NDarray(np.linspace(start, stop, num, endpoint, dtype?.NumpyDtype, axis));
            }
        }

        /// <summary>
        ///     Return numbers spaced evenly on a log scale.<br></br>
        ///     In linear space, the sequence starts at base ** start
        ///     (base to the power of start) and ends with base ** stop
        ///     (see endpoint below).<br></br>
        ///     Notes
        ///     Logspace is equivalent to the code
        /// </summary>
        /// <param name="start">
        ///     base ** start is the starting value of the sequence.
        /// </param>
        /// <param name="stop">
        ///     base ** stop is the final value of the sequence, unless endpoint
        ///     is False.<br></br>
        ///     In that case, num + 1 values are spaced over the
        ///     interval in log-space, of which all but the last (a sequence of
        ///     length num) are returned.
        /// </param>
        /// <param name="num">
        ///     Number of samples to generate.<br></br>
        ///     Default is 50.
        /// </param>
        /// <param name="endpoint">
        ///     If true, stop is the last sample.<br></br>
        ///     Otherwise, it is not included.<br></br>
        ///     Default is True.
        /// </param>
        /// <param name="base">
        ///     The base of the log space.<br></br>
        ///     The step size between the elements in
        ///     ln(samples) / ln(base) (or log_base(samples)) is uniform.<br></br>
        ///     Default is 10.0.
        /// </param>
        /// <param name="dtype">
        ///     The type of the output array.<br></br>
        ///     If dtype is not given, infer the data
        ///     type from the other input arguments.
        /// </param>
        /// <param name="axis">
        ///     The axis in the result to store the samples.<br></br>
        ///     Relevant only if start
        ///     or stop are array-like.<br></br>
        ///     By default (0), the samples will be along a
        ///     new axis inserted at the beginning.<br></br>
        ///     Use -1 to get an axis at the end.
        /// </param>
        /// <returns>
        ///     num samples, equally spaced on a log scale.
        /// </returns>
        public static NDarray logspace(NDarray start, NDarray stop, int? num = 50, bool? endpoint = true,
            float? @base = 10.0f, Dtype dtype = null, int? axis = 0)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.logspace(start.CupyNDarray, stop.CupyNDarray, num, endpoint, @base,
                    dtype?.CupyDtype, axis));
            }
            else
            {
                return new NDarray(np.logspace(start.NumpyNDarray, stop.NumpyNDarray, num, endpoint, @base,
                    dtype?.NumpyDtype, axis));
            }
        }

        /// <summary>
        ///     Return numbers spaced evenly on a log scale (a geometric progression).<br></br>
        ///     This is similar to logspace, but with endpoints specified directly.<br></br>
        ///     Each output sample is a constant multiple of the previous.<br></br>
        ///     Notes
        ///     If the inputs or dtype are complex, the output will follow a logarithmic
        ///     spiral in the complex plane.<br></br>
        ///     (There are an infinite number of spirals
        ///     passing through two points; the output will follow the shortest such path.)
        /// </summary>
        /// <param name="start">
        ///     The starting value of the sequence.
        /// </param>
        /// <param name="stop">
        ///     The final value of the sequence, unless endpoint is False.<br></br>
        ///     In that case, num + 1 values are spaced over the
        ///     interval in log-space, of which all but the last (a sequence of
        ///     length num) are returned.
        /// </param>
        /// <param name="num">
        ///     Number of samples to generate.<br></br>
        ///     Default is 50.
        /// </param>
        /// <param name="endpoint">
        ///     If true, stop is the last sample.<br></br>
        ///     Otherwise, it is not included.<br></br>
        ///     Default is True.
        /// </param>
        /// <param name="dtype">
        ///     The type of the output array.<br></br>
        ///     If dtype is not given, infer the data
        ///     type from the other input arguments.
        /// </param>
        /// <param name="axis">
        ///     The axis in the result to store the samples.<br></br>
        ///     Relevant only if start
        ///     or stop are array-like.<br></br>
        ///     By default (0), the samples will be along a
        ///     new axis inserted at the beginning.<br></br>
        ///     Use -1 to get an axis at the end.
        /// </param>
        /// <returns>
        ///     num samples, equally spaced on a log scale.
        /// </returns>
        public static NDarray geomspace(NDarray start, NDarray stop, int? num = 50, bool? endpoint = true,
            Dtype dtype = null, int? axis = 0)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.geomspace(start.CupyNDarray, stop.CupyNDarray, num, endpoint, dtype?.CupyDtype,
                    axis));
            }
            else
            {
                return new NDarray(np.geomspace(start.NumpyNDarray, stop.NumpyNDarray, num, endpoint, dtype?.NumpyDtype,
                    axis));
            }
        }

        /// <summary>
        ///     nd_grid instance which returns a dense multi-dimensional “meshgrid”.
        ///     An instance of Cupy.lib.index_tricks.nd_grid which returns an dense
        ///     (or fleshed out) mesh-grid when indexed, so that each returned argument
        ///     has the same shape.<br></br>
        ///     The dimensions and number of the output arrays are
        ///     equal to the number of indexing dimensions.<br></br>
        ///     If the step length is not a
        ///     complex number, then the stop is not inclusive.<br></br>
        ///     However, if the step length is a complex number (e.g.<br></br>
        ///     5j), then
        ///     the integer part of its magnitude is interpreted as specifying the
        ///     number of points to create between the start and stop values, where
        ///     the stop value is inclusive.
        /// </summary>
        public static NDarray[] mgrid()
        {
            if (Gpu.Available && Gpu.Use)
            {
                return cp.mgrid().Select(x => new NDarray(x)).ToArray();
            }
            else
            {
                return np.mgrid().Select(x => new NDarray(x)).ToArray();
            }
        }

        /// <summary>
        ///     nd_grid instance which returns an open multi-dimensional “meshgrid”.
        ///     An instance of Cupy.lib.index_tricks.nd_grid which returns an open
        ///     (i.e.<br></br>
        ///     not fleshed out) mesh-grid when indexed, so that only one dimension
        ///     of each returned array is greater than 1.<br></br>
        ///     The dimension and number of the
        ///     output arrays are equal to the number of indexing dimensions.<br></br>
        ///     If the step
        ///     length is not a complex number, then the stop is not inclusive.<br></br>
        ///     However, if the step length is a complex number (e.g.<br></br>
        ///     5j), then
        ///     the integer part of its magnitude is interpreted as specifying the
        ///     number of points to create between the start and stop values, where
        ///     the stop value is inclusive.
        /// </summary>
        public static NDarray[] ogrid()
        {
            if (Gpu.Available && Gpu.Use)
            {
                return cp.ogrid().Select(x => new NDarray(x)).ToArray();
            }
            else
            {
                return np.ogrid().Select(x => new NDarray(x)).ToArray();
            }
        }

        /// <summary>
        ///     Extract a diagonal or construct a diagonal array.<br></br>
        ///     See the more detailed documentation for Cupy.diagonal if you use this
        ///     function to extract a diagonal and wish to write to the resulting array;
        ///     whether it returns a copy or a view depends on what version of Cupy you
        ///     are using.
        /// </summary>
        /// <param name="v">
        ///     If v is a 2-D array, return a copy of its k-th diagonal.<br></br>
        ///     If v is a 1-D array, return a 2-D array with v on the k-th
        ///     diagonal.
        /// </param>
        /// <param name="k">
        ///     Diagonal in question.<br></br>
        ///     The default is 0.<br></br>
        ///     Use k&gt;0 for diagonals
        ///     above the main diagonal, and k&lt;0 for diagonals below the main
        ///     diagonal.
        /// </param>
        /// <returns>
        ///     The extracted diagonal or constructed diagonal array.
        /// </returns>
        public static NDarray diag(NDarray v, int? k = 0)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.diag(v.CupyNDarray, k));
            }
            else
            {
                return new NDarray(np.diag(v.NumpyNDarray, k));
            }
        }

        /// <summary>
        ///     Extract a diagonal or construct a diagonal array.<br></br>
        ///     See the more detailed documentation for Cupy.diagonal if you use this
        ///     function to extract a diagonal and wish to write to the resulting array;
        ///     whether it returns a copy or a view depends on what version of Cupy you
        ///     are using.
        /// </summary>
        /// <param name="v">
        ///     If v is a 2-D array, return a copy of its k-th diagonal.<br></br>
        ///     If v is a 1-D array, return a 2-D array with v on the k-th
        ///     diagonal.
        /// </param>
        /// <param name="k">
        ///     Diagonal in question.<br></br>
        ///     The default is 0.<br></br>
        ///     Use k&gt;0 for diagonals
        ///     above the main diagonal, and k&lt;0 for diagonals below the main
        ///     diagonal.
        /// </param>
        /// <returns>
        ///     The extracted diagonal or constructed diagonal array.
        /// </returns>
        public static NDarray<T> diag<T>(T[] v, int? k = 0) where T : struct
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray<T>(cp.diag(v, k));
            }
            else
            {
                return new NDarray<T>(np.diag(v, k));
            }
        }

        /// <summary>
        ///     Extract a diagonal or construct a diagonal array.<br></br>
        ///     See the more detailed documentation for Cupy.diagonal if you use this
        ///     function to extract a diagonal and wish to write to the resulting array;
        ///     whether it returns a copy or a view depends on what version of Cupy you
        ///     are using.
        /// </summary>
        /// <param name="v">
        ///     If v is a 2-D array, return a copy of its k-th diagonal.<br></br>
        ///     If v is a 1-D array, return a 2-D array with v on the k-th
        ///     diagonal.
        /// </param>
        /// <param name="k">
        ///     Diagonal in question.<br></br>
        ///     The default is 0.<br></br>
        ///     Use k&gt;0 for diagonals
        ///     above the main diagonal, and k&lt;0 for diagonals below the main
        ///     diagonal.
        /// </param>
        /// <returns>
        ///     The extracted diagonal or constructed diagonal array.
        /// </returns>
        public static NDarray<T> diag<T>(T[,] v, int? k = 0) where T : struct
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray<T>(cp.diag(v, k));
            }
            else
            {
                return new NDarray<T>(np.diag(v, k));
            }
        }

        /// <summary>
        ///     Create a two-dimensional array with the flattened input as a diagonal.
        /// </summary>
        /// <param name="v">
        ///     Input data, which is flattened and set as the k-th
        ///     diagonal of the output.
        /// </param>
        /// <param name="k">
        ///     Diagonal to set; 0, the default, corresponds to the “main” diagonal,
        ///     a positive (negative) k giving the number of the diagonal above
        ///     (below) the main.
        /// </param>
        /// <returns>
        ///     The 2-D output array.
        /// </returns>
        public static NDarray diagflat(NDarray v, int? k = 0)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.diagflat(v.CupyNDarray, k));
            }
            else
            {
                return new NDarray(np.diagflat(v.NumpyNDarray, k));
            }
        }

        /// <summary>
        ///     Create a two-dimensional array with the flattened input as a diagonal.
        /// </summary>
        /// <param name="v">
        ///     Input data, which is flattened and set as the k-th
        ///     diagonal of the output.
        /// </param>
        /// <param name="k">
        ///     Diagonal to set; 0, the default, corresponds to the “main” diagonal,
        ///     a positive (negative) k giving the number of the diagonal above
        ///     (below) the main.
        /// </param>
        /// <returns>
        ///     The 2-D output array.
        /// </returns>
        public static NDarray<T> diagflat<T>(T[] v, int? k = 0) where T : struct
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray<T>(cp.diagflat(v, k));
            }
            else
            {
                return new NDarray<T>(np.diagflat(v, k));
            }
        }

        /// <summary>
        ///     Create a two-dimensional array with the flattened input as a diagonal.
        /// </summary>
        /// <param name="v">
        ///     Input data, which is flattened and set as the k-th
        ///     diagonal of the output.
        /// </param>
        /// <param name="k">
        ///     Diagonal to set; 0, the default, corresponds to the “main” diagonal,
        ///     a positive (negative) k giving the number of the diagonal above
        ///     (below) the main.
        /// </param>
        /// <returns>
        ///     The 2-D output array.
        /// </returns>
        public static NDarray<T> diagflat<T>(T[,] v, int? k = 0) where T : struct
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray<T>(cp.diagflat(v, k));
            }
            else
            {
                return new NDarray<T>(np.diagflat(v, k));
            }
        }

        /// <summary>
        ///     An array with ones at and below the given diagonal and zeros elsewhere.
        /// </summary>
        /// <param name="N">
        ///     Number of rows in the array.
        /// </param>
        /// <param name="M">
        ///     Number of columns in the array.<br></br>
        ///     By default, M is taken equal to N.
        /// </param>
        /// <param name="k">
        ///     The sub-diagonal at and below which the array is filled.<br></br>
        ///     k = 0 is the main diagonal, while k &lt; 0 is below it,
        ///     and k &gt; 0 is above.<br></br>
        ///     The default is 0.
        /// </param>
        /// <param name="dtype">
        ///     Data type of the returned array.<br></br>
        ///     The default is float.
        /// </param>
        /// <returns>
        ///     Array with its lower triangle filled with ones and zero elsewhere;
        ///     in other words T[i,j] == 1 for i &lt;= j + k, 0 otherwise.
        /// </returns>
        public static NDarray tri(int N, int? M = null, int? k = 0, Dtype dtype = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.tri(N, M, k, dtype?.CupyDtype));
            }
            else
            {
                return new NDarray(np.tri(N, M, k, dtype?.NumpyDtype));
            }
        }

        /// <summary>
        ///     Lower triangle of an array.<br></br>
        ///     Return a copy of an array with elements above the k-th diagonal zeroed.
        /// </summary>
        /// <param name="m">
        ///     Input array.
        /// </param>
        /// <param name="k">
        ///     Diagonal above which to zero elements.<br></br>
        ///     k = 0 (the default) is the
        ///     main diagonal, k &lt; 0 is below it and k &gt; 0 is above.
        /// </param>
        /// <returns>
        ///     Lower triangle of m, of same shape and data-type as m.
        /// </returns>
        public static NDarray tril(NDarray m, int? k = 0)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.tril(m.CupyNDarray, k));
            }
            else
            {
                return new NDarray(np.tril(m.NumpyNDarray, k));
            }
        }

        /// <summary>
        ///     Lower triangle of an array.<br></br>
        ///     Return a copy of an array with elements above the k-th diagonal zeroed.
        /// </summary>
        /// <param name="m">
        ///     Input array.
        /// </param>
        /// <param name="k">
        ///     Diagonal above which to zero elements.<br></br>
        ///     k = 0 (the default) is the
        ///     main diagonal, k &lt; 0 is below it and k &gt; 0 is above.
        /// </param>
        /// <returns>
        ///     Lower triangle of m, of same shape and data-type as m.
        /// </returns>
        public static NDarray<T> tril<T>(T[] m, int? k = 0) where T : struct
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray<T>(cp.tril(m, k));
            }
            else
            {
                return new NDarray<T>(np.tril(m, k));
            }
        }

        /// <summary>
        ///     Lower triangle of an array.<br></br>
        ///     Return a copy of an array with elements above the k-th diagonal zeroed.
        /// </summary>
        /// <param name="m">
        ///     Input array.
        /// </param>
        /// <param name="k">
        ///     Diagonal above which to zero elements.<br></br>
        ///     k = 0 (the default) is the
        ///     main diagonal, k &lt; 0 is below it and k &gt; 0 is above.
        /// </param>
        /// <returns>
        ///     Lower triangle of m, of same shape and data-type as m.
        /// </returns>
        public static NDarray<T> tril<T>(T[,] m, int? k = 0) where T : struct
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray<T>(cp.tril(m, k));
            }
            else
            {
                return new NDarray<T>(np.tril(m, k));
            }
        }

        /// <summary>
        ///     Generate a Vandermonde matrix.<br></br>
        ///     The columns of the output matrix are powers of the input vector.<br></br>
        ///     The
        ///     order of the powers is determined by the increasing boolean argument.<br></br>
        ///     Specifically, when increasing is False, the i-th output column is
        ///     the input vector raised element-wise to the power of N - i - 1.<br></br>
        ///     Such
        ///     a matrix with a geometric progression in each row is named for Alexandre-
        ///     Theophile Vandermonde.
        /// </summary>
        /// <param name="x">
        ///     1-D input array.
        /// </param>
        /// <param name="N">
        ///     Number of columns in the output.<br></br>
        ///     If N is not specified, a square
        ///     array is returned (N = len(x)).
        /// </param>
        /// <param name="increasing">
        ///     Order of the powers of the columns.<br></br>
        ///     If True, the powers increase
        ///     from left to right, if False (the default) they are reversed.
        /// </param>
        /// <returns>
        ///     Vandermonde matrix.<br></br>
        ///     If increasing is False, the first column is
        ///     x^(N-1), the second x^(N-2) and so forth.<br></br>
        ///     If increasing is
        ///     True, the columns are x^0, x^1, ..., x^(N-1).
        /// </returns>
        public static NDarray vander(NDarray x, int? N = null, bool? increasing = false)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.vander(x.CupyNDarray, N, increasing));
            }
            else
            {
                return new NDarray(np.vander(x.NumpyNDarray, N, increasing));
            }
        }

        /// <summary>
        ///     Generate a Vandermonde matrix.<br></br>
        ///     The columns of the output matrix are powers of the input vector.<br></br>
        ///     The
        ///     order of the powers is determined by the increasing boolean argument.<br></br>
        ///     Specifically, when increasing is False, the i-th output column is
        ///     the input vector raised element-wise to the power of N - i - 1.<br></br>
        ///     Such
        ///     a matrix with a geometric progression in each row is named for Alexandre-
        ///     Theophile Vandermonde.
        /// </summary>
        /// <param name="x">
        ///     1-D input array.
        /// </param>
        /// <param name="N">
        ///     Number of columns in the output.<br></br>
        ///     If N is not specified, a square
        ///     array is returned (N = len(x)).
        /// </param>
        /// <param name="increasing">
        ///     Order of the powers of the columns.<br></br>
        ///     If True, the powers increase
        ///     from left to right, if False (the default) they are reversed.
        /// </param>
        /// <returns>
        ///     Vandermonde matrix.<br></br>
        ///     If increasing is False, the first column is
        ///     x^(N-1), the second x^(N-2) and so forth.<br></br>
        ///     If increasing is
        ///     True, the columns are x^0, x^1, ..., x^(N-1).
        /// </returns>
        public static NDarray<T> vander<T>(T[] x, int? N = null, bool? increasing = false) where T : struct
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray<T>(cp.vander<T>(x, N, increasing));
            }
            else
            {
                return new NDarray<T>(np.vander<T>(x, N, increasing));
            }
        }

        /// <summary>
        ///     Generate a Vandermonde matrix.<br></br>
        ///     The columns of the output matrix are powers of the input vector.<br></br>
        ///     The
        ///     order of the powers is determined by the increasing boolean argument.<br></br>
        ///     Specifically, when increasing is False, the i-th output column is
        ///     the input vector raised element-wise to the power of N - i - 1.<br></br>
        ///     Such
        ///     a matrix with a geometric progression in each row is named for Alexandre-
        ///     Theophile Vandermonde.
        /// </summary>
        /// <param name="x">
        ///     1-D input array.
        /// </param>
        /// <param name="N">
        ///     Number of columns in the output.<br></br>
        ///     If N is not specified, a square
        ///     array is returned (N = len(x)).
        /// </param>
        /// <param name="increasing">
        ///     Order of the powers of the columns.<br></br>
        ///     If True, the powers increase
        ///     from left to right, if False (the default) they are reversed.
        /// </param>
        /// <returns>
        ///     Vandermonde matrix.<br></br>
        ///     If increasing is False, the first column is
        ///     x^(N-1), the second x^(N-2) and so forth.<br></br>
        ///     If increasing is
        ///     True, the columns are x^0, x^1, ..., x^(N-1).
        /// </returns>
        public static NDarray<T> vander<T>(T[,] x, int? N = null, bool? increasing = false) where T : struct
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray<T>(cp.vander<T>(x, N, increasing));
            }
            else
            {
                return new NDarray<T>(np.vander<T>(x, N, increasing));
            }
        }

        public static partial class core
        {
            public static partial class defchararray
            {
                /// <summary>
                ///     Create a chararray.<br></br>
                ///     Versus a regular Cupy array of type str or unicode, this
                ///     class adds the following functionality:
                /// </summary>
                /// <param name="itemsize">
                ///     itemsize is the number of characters per scalar in the
                ///     resulting array.<br></br>
                ///     If itemsize is None, and obj is an
                ///     object array or a Python list, the itemsize will be
                ///     automatically determined.<br></br>
                ///     If itemsize is provided and obj
                ///     is of type str or unicode, then the obj string will be
                ///     chunked into itemsize pieces.
                /// </param>
                /// <param name="copy">
                ///     If true (default), then the object is copied.<br></br>
                ///     Otherwise, a copy
                ///     will only be made if __array__ returns a copy, if obj is a
                ///     nested sequence, or if a copy is needed to satisfy any of the other
                ///     requirements (itemsize, unicode, order, etc.).
                /// </param>
                /// <param name="unicode">
                ///     When true, the resulting chararray can contain Unicode
                ///     characters, when false only 8-bit characters.<br></br>
                ///     If unicode is
                ///     None and obj is one of the following:
                ///     then the unicode setting of the output array will be
                ///     automatically determined.
                /// </param>
                /// <param name="order">
                ///     Specify the order of the array.<br></br>
                ///     If order is ‘C’ (default), then the
                ///     array will be in C-contiguous order (last-index varies the
                ///     fastest).<br></br>
                ///     If order is ‘F’, then the returned array
                ///     will be in Fortran-contiguous order (first-index varies the
                ///     fastest).<br></br>
                ///     If order is ‘A’, then the returned array may
                ///     be in any order (either C-, Fortran-contiguous, or even
                ///     discontiguous).
                /// </param>
                public static void array(string[] obj, int? itemsize = null, bool? copy = true, bool? unicode = null,
                    string order = null)
                {
                    if (Gpu.Available && Gpu.Use)
                    {
                        cp.core.defchararray.array(obj, itemsize, copy, unicode, order);
                    }
                    else
                    {
                        np.core.defchararray.array(obj, itemsize, copy, unicode, order);
                    }
                }
            }
        }

        public static partial class core
        {
            public static partial class defchararray
            {
                /// <summary>
                ///     Convert the input to a chararray, copying the data only if
                ///     necessary.<br></br>
                ///     Versus a regular Cupy array of type str or unicode, this
                ///     class adds the following functionality:
                /// </summary>
                /// <param name="itemsize">
                ///     itemsize is the number of characters per scalar in the
                ///     resulting array.<br></br>
                ///     If itemsize is None, and obj is an
                ///     object array or a Python list, the itemsize will be
                ///     automatically determined.<br></br>
                ///     If itemsize is provided and obj
                ///     is of type str or unicode, then the obj string will be
                ///     chunked into itemsize pieces.
                /// </param>
                /// <param name="unicode">
                ///     When true, the resulting chararray can contain Unicode
                ///     characters, when false only 8-bit characters.<br></br>
                ///     If unicode is
                ///     None and obj is one of the following:
                ///     then the unicode setting of the output array will be
                ///     automatically determined.
                /// </param>
                /// <param name="order">
                ///     Specify the order of the array.<br></br>
                ///     If order is ‘C’ (default), then the
                ///     array will be in C-contiguous order (last-index varies the
                ///     fastest).<br></br>
                ///     If order is ‘F’, then the returned array
                ///     will be in Fortran-contiguous order (first-index varies the
                ///     fastest).
                /// </param>
                public static void asarray(string[] obj, int? itemsize = null, bool? unicode = null,
                    string order = null)
                {
                    if (Gpu.Available && Gpu.Use)
                    {
                        cp.core.defchararray.asarray(obj, itemsize, unicode, order);
                    }
                    else
                    {
                        np.core.defchararray.asarray(obj, itemsize, unicode, order);
                    }
                }
            }
        }

        /*
        /// <summary>
        ///	Interpret the input as a matrix.<br></br>
        ///	
        ///	Unlike matrix, asmatrix does not make a copy if the input is already
        ///	a matrix or an ndarray.<br></br>
        ///	  Equivalent to matrix(data, copy=False).
        /// </summary>
        /// <param name="data">
        ///	Input data.
        /// </param>
        /// <param name="dtype">
        ///	Data-type of the output matrix.
        /// </param>
        /// <returns>
        ///	data interpreted as a matrix.
        /// </returns>
        public static Matrix mat(NDarray data, Dtype dtype)
        {
            //auto-generated code, do not change
            var __self__=self;
            var pyargs=ToTuple(new object[]
            {
                data,
                dtype,
            });
            var kwargs=new PyDict();
            dynamic py = __self__.InvokeMethod("mat", pyargs, kwargs);
            return ToCsharp<Matrix>(py);
        }
        */

        /*
        /// <summary>
        ///	Interpret the input as a matrix.<br></br>
        ///	
        ///	Unlike matrix, asmatrix does not make a copy if the input is already
        ///	a matrix or an ndarray.<br></br>
        ///	  Equivalent to matrix(data, copy=False).
        /// </summary>
        /// <param name="data">
        ///	Input data.
        /// </param>
        /// <param name="dtype">
        ///	Data-type of the output matrix.
        /// </param>
        /// <returns>
        ///	data interpreted as a matrix.
        /// </returns>
        public static Matrix mat<T>(T[] data, Dtype dtype)
        {
            //auto-generated code, do not change
            var __self__=self;
            var pyargs=ToTuple(new object[]
            {
                SharpToSharp<NDarray>(data),
                dtype,
            });
            var kwargs=new PyDict();
            dynamic py = __self__.InvokeMethod("mat", pyargs, kwargs);
            return ToCsharp<Matrix>(py);
        }
        */

        /*
        /// <summary>
        ///	Interpret the input as a matrix.<br></br>
        ///	
        ///	Unlike matrix, asmatrix does not make a copy if the input is already
        ///	a matrix or an ndarray.<br></br>
        ///	  Equivalent to matrix(data, copy=False).
        /// </summary>
        /// <param name="data">
        ///	Input data.
        /// </param>
        /// <param name="dtype">
        ///	Data-type of the output matrix.
        /// </param>
        /// <returns>
        ///	data interpreted as a matrix.
        /// </returns>
        public static Matrix mat<T>(T[,] data, Dtype dtype)
        {
            //auto-generated code, do not change
            var __self__=self;
            var pyargs=ToTuple(new object[]
            {
                SharpToSharp<NDarray>(data),
                dtype,
            });
            var kwargs=new PyDict();
            dynamic py = __self__.InvokeMethod("mat", pyargs, kwargs);
            return ToCsharp<Matrix>(py);
        }
        */

        /*
        /// <summary>
        ///	Build a matrix object from a string, nested sequence, or array.
        /// </summary>
        /// <param name="obj">
        ///	Input data.<br></br>
        ///	If a string, variables in the current scope may be
        ///	referenced by name.
        /// </param>
        /// <param name="ldict">
        ///	A dictionary that replaces local operands in current frame.<br></br>
        ///	
        ///	Ignored if obj is not a string or gdict is None.
        /// </param>
        /// <param name="gdict">
        ///	A dictionary that replaces global operands in current frame.<br></br>
        ///	
        ///	Ignored if obj is not a string.
        /// </param>
        /// <returns>
        ///	Returns a matrix object, which is a specialized 2-D array.
        /// </returns>
        public static Matrix bmat(string obj, Hashtable ldict = null, Hashtable gdict = null)
        {
            //auto-generated code, do not change
            var __self__=self;
            var pyargs=ToTuple(new object[]
            {
                obj,
            });
            var kwargs=new PyDict();
            if (ldict!=null) kwargs["ldict"]=ToPython(ldict);
            if (gdict!=null) kwargs["gdict"]=ToPython(gdict);
            dynamic py = __self__.InvokeMethod("bmat", pyargs, kwargs);
            return ToCsharp<Matrix>(py);
        }
        */

        /*
        /// <summary>
        ///	Build a matrix object from a string, nested sequence, or array.
        /// </summary>
        /// <param name="obj">
        ///	Input data.<br></br>
        ///	If a string, variables in the current scope may be
        ///	referenced by name.
        /// </param>
        /// <param name="ldict">
        ///	A dictionary that replaces local operands in current frame.<br></br>
        ///	
        ///	Ignored if obj is not a string or gdict is None.
        /// </param>
        /// <param name="gdict">
        ///	A dictionary that replaces global operands in current frame.<br></br>
        ///	
        ///	Ignored if obj is not a string.
        /// </param>
        /// <returns>
        ///	Returns a matrix object, which is a specialized 2-D array.
        /// </returns>
        public static Matrix<T> bmat<T>(T[] obj, Hashtable ldict = null, Hashtable gdict = null)
        {
            //auto-generated code, do not change
            var __self__=self;
            var pyargs=ToTuple(new object[]
            {
                SharpToSharp<NDarray>(obj),
            });
            var kwargs=new PyDict();
            if (ldict!=null) kwargs["ldict"]=ToPython(ldict);
            if (gdict!=null) kwargs["gdict"]=ToPython(gdict);
            dynamic py = __self__.InvokeMethod("bmat", pyargs, kwargs);
            return ToCsharp<Matrix<T>>(py);
        }
        */
    }
}
