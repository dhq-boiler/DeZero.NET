using Cupy;
using Numpy;
using Python.Runtime;


namespace DeZero.NET
{
    public static partial class xp
    {
        #region util
        private static T[,] ConvertToRectangularArray<T>(T[][] jaggedArray)
        {
            int rows = jaggedArray.Length;
            int cols = jaggedArray.Max(subArray => subArray.Length);

            T[,] rectangularArray = new T[rows, cols];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < jaggedArray[i].Length; j++)
                {
                    rectangularArray[i, j] = jaggedArray[i][j];
                }
            }

            return rectangularArray;
        }

        private static T[,,] ConvertToRectangularArray<T>(T[][][] jaggedArray)
        {
            int dim1 = jaggedArray.Length;
            int dim2 = jaggedArray.Max(subArray => subArray.Length);
            int dim3 = jaggedArray.Max(subArray => subArray.Max(subSubArray => subSubArray.Length));

            T[,,] rectangularArray = new T[dim1, dim2, dim3];

            for (int i = 0; i < dim1; i++)
            {
                for (int j = 0; j < jaggedArray[i].Length; j++)
                {
                    for (int k = 0; k < jaggedArray[i][j].Length; k++)
                    {
                        rectangularArray[i, j, k] = jaggedArray[i][j][k];
                    }
                }
            }

            return rectangularArray;
        }

        private static T[,,,] ConvertToRectangularArray<T>(T[][][][] jaggedArray)
        {
            int dim1 = jaggedArray.Length;
            int dim2 = jaggedArray.Max(subArray => subArray.Length);
            int dim3 = jaggedArray.Max(subArray => subArray.Max(subSubArray => subSubArray.Length));
            int dim4 = jaggedArray.Max(subArray => subArray.Max(subSubArray => subSubArray.Max(subSubSubArray => subSubSubArray.Length)));

            T[,,,] rectangularArray = new T[dim1, dim2, dim3, dim4];

            for (int i = 0; i < dim1; i++)
            {
                for (int j = 0; j < jaggedArray[i].Length; j++)
                {
                    for (int k = 0; k < jaggedArray[i][j].Length; k++)
                    {
                        for (int l = 0; l < jaggedArray[i][j][k].Length; l++)
                        {
                            rectangularArray[i, j, k, l] = jaggedArray[i][j][k][l];
                        }
                    }
                }
            }

            return rectangularArray;
        }

        private static T[,,,,] ConvertToRectangularArray<T>(T[][][][][] jaggedArray)
        {
            int dim1 = jaggedArray.Length;
            int dim2 = jaggedArray.Max(subArray => subArray.Length);
            int dim3 = jaggedArray.Max(subArray => subArray.Max(subSubArray => subSubArray.Length));
            int dim4 = jaggedArray.Max(subArray => subArray.Max(subSubArray => subSubArray.Max(subSubSubArray => subSubSubArray.Length)));
            int dim5 = jaggedArray.Max(subArray => subArray.Max(subSubArray => subSubArray.Max(subSubSubArray => subSubSubArray.Max(subSubSubSubArray => subSubSubSubArray.Length))));

            T[,,,,] rectangularArray = new T[dim1, dim2, dim3, dim4, dim5];

            for (int i = 0; i < dim1; i++)
            {
                for (int j = 0; j < jaggedArray[i].Length; j++)
                {
                    for (int k = 0; k < jaggedArray[i][j].Length; k++)
                    {
                        for (int l = 0; l < jaggedArray[i][j][k].Length; l++)
                        {
                            for (int m = 0; m < jaggedArray[i][j][k][l].Length; m++)
                            {
                                rectangularArray[i, j, k, l, m] = jaggedArray[i][j][k][l][m];
                            }
                        }
                    }
                }
            }

            return rectangularArray;
        }

        private static T[,,,,,] ConvertToRectangularArray<T>(T[][][][][][] jaggedArray)
        {
            int dim1 = jaggedArray.Length;
            int dim2 = jaggedArray.Max(subArray => subArray.Length);
            int dim3 = jaggedArray.Max(subArray => subArray.Max(subSubArray => subSubArray.Length));
            int dim4 = jaggedArray.Max(subArray => subArray.Max(subSubArray => subSubArray.Max(subSubSubArray => subSubSubArray.Length)));
            int dim5 = jaggedArray.Max(subArray => subArray.Max(subSubArray => subSubArray.Max(subSubSubArray => subSubSubArray.Max(subSubSubSubArray => subSubSubSubArray.Length))));
            int dim6 = jaggedArray.Max(subArray => subArray.Max(subSubArray => subSubArray.Max(subSubSubArray => subSubSubArray.Max(subSubSubSubArray => subSubSubSubArray.Max(subSubSubSubSubArray => subSubSubSubSubArray.Length)))));

            T[,,,,,] rectangularArray = new T[dim1, dim2, dim3, dim4, dim5, dim6];

            for (int i = 0; i < dim1; i++)
            {
                for (int j = 0; j < jaggedArray[i].Length; j++)
                {
                    for (int k = 0; k < jaggedArray[i][j].Length; k++)
                    {
                        for (int l = 0; l < jaggedArray[i][j][k].Length; l++)
                        {
                            for (int m = 0; m < jaggedArray[i][j][k][l].Length; m++)
                            {
                                for (int n = 0; n < jaggedArray[i][j][k][l][m].Length; n++)
                                {
                                    rectangularArray[i, j, k, l, m, n] = jaggedArray[i][j][k][l][m][n];
                                }
                            }
                        }
                    }
                }
            }

            return rectangularArray;
        }

        #endregion

        public static NDarray array<T>(T data) where T : struct
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray<T>(cp.array<T>(data));
            }
            else
            {
                return new NDarray<T>(np.array<T>(data));
            }
        }

        public static NDarray array<T>(T data, Dtype dtype) where T : struct
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.array<T>(data).astype(dtype.CupyDtype));
            }
            else
            {
                return new NDarray(np.array<T>(data).astype(dtype.NumpyDtype));
            }
        }

        public static NDarray array<T>(T[] data) where T : struct
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray<T>(cp.array<T>(data));
            }
            else
            {
                return new NDarray<T>(np.array<T>(data));
            }
        }

        public static NDarray array<T>(T[] data, Dtype dtype) where T : struct
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.array<T>(data, dtype: dtype.CupyDtype));
            }
            else
            {
                return new NDarray(np.array<T>(data, dtype: dtype.NumpyDtype));
            }
        }

        public static NDarray array<T>(T[][] data) where T : struct
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.array<T>(data));
            }
            else
            {
                return new NDarray<T>(np.array<T>(ConvertToRectangularArray<T>(data)));
            }
        }

        public static NDarray array<T>(T[][] data, Dtype dtype) where T : struct
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.array<T>(ConvertToRectangularArray<T>(data), dtype: dtype.CupyDtype));
            }
            else
            {
                return new NDarray<T>(np.array<T>(ConvertToRectangularArray<T>(data), dtype: dtype.NumpyDtype));
            }
        }

        private static NDarray array<T>(T[,] data) where T : struct
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.array<T>(data));
            }
            else
            {
                return new NDarray(np.array<T>(data));
            }
        }

        private static NDarray array<T>(T[,] data, Dtype dtype) where T : struct
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.array<T>(data, dtype: dtype.CupyDtype));
            }
            else
            {
                return new NDarray(np.array<T>(data, dtype: dtype.NumpyDtype));
            }
        }

        public static NDarray array<T>(T[][][] data) where T : struct
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.array<T>(ConvertToRectangularArray(data)));
            }
            else
            {
                return new NDarray(np.array<T>(ConvertToRectangularArray(data)));
            }
        }

        public static NDarray array<T>(T[][][] data, Dtype dtype) where T : struct
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.array<T>(ConvertToRectangularArray(data), dtype: dtype.CupyDtype));
            }
            else
            {
                return new NDarray(np.array<T>(ConvertToRectangularArray(data), dtype: dtype.NumpyDtype));
            }
        }

        private static NDarray array<T>(T[,,] data) where T : struct
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.array<T>(data));
            }
            else
            {
                return new NDarray(np.array<T>(data));
            }
        }

        private static NDarray array<T>(T[,,] data, Dtype dtype) where T : struct
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.array<T>(data, dtype: dtype.CupyDtype));
            }
            else
            {
                return new NDarray(np.array<T>(data, dtype: dtype.NumpyDtype));
            }
        }

        public static NDarray array<T>(T[][][][] data) where T : struct
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.array<T>(ConvertToRectangularArray(data)));
            }
            else
            {
                return new NDarray(np.array<T>(ConvertToRectangularArray(data)));
            }
        }

        public static NDarray array<T>(T[][][][] data, Dtype dtype) where T : struct
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.array<T>(ConvertToRectangularArray(data), dtype: dtype.CupyDtype));
            }
            else
            {
                return new NDarray(np.array<T>(ConvertToRectangularArray(data), dtype: dtype.NumpyDtype));
            }
        }

        private static NDarray array<T>(T[,,,] data) where T : struct
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.array<T>(data));
            }
            else
            {
                return new NDarray(np.array<T>(data));
            }
        }

        private static NDarray array<T>(T[,,,] data, Dtype dtype) where T : struct
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.array<T>(data, dtype: dtype.CupyDtype));
            }
            else
            {
                return new NDarray(np.array<T>(data, dtype: dtype.NumpyDtype));
            }
        }

        public static NDarray array<T>(T[][][][][] data) where T : struct
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.array<T>(ConvertToRectangularArray(data)));
            }
            else
            {
                return new NDarray(np.array<T>(ConvertToRectangularArray(data)));
            }
        }

        public static NDarray array<T>(T[][][][][] data, Dtype dtype) where T : struct
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.array<T>(ConvertToRectangularArray(data), dtype: dtype.CupyDtype));
            }
            else
            {
                return new NDarray(np.array<T>(ConvertToRectangularArray(data), dtype: dtype.NumpyDtype));
            }
        }

        private static NDarray array<T>(T[,,,,] data, Dtype dtype) where T : struct
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.array<T>(data, dtype: dtype.CupyDtype));
            }
            else
            {
                return new NDarray(np.array<T>(data, dtype: dtype.NumpyDtype));
            }
        }

        public static NDarray array<T>(T[][][][][][] data) where T : struct
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.array<T>(ConvertToRectangularArray<T>(data)));
            }
            else
            {
                return new NDarray(np.array<T>(ConvertToRectangularArray<T>(data)));
            }
        }

        public static NDarray array<T>(T[][][][][][] data, Dtype dtype) where T : struct
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.array<T>(ConvertToRectangularArray<T>(data), dtype: dtype.CupyDtype));
            }
            else
            {
                return new NDarray(np.array<T>(ConvertToRectangularArray<T>(data), dtype: dtype.NumpyDtype));
            }
        }

        private static NDarray array<T>(T[,,,,,] data, Dtype dtype) where T : struct
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.array<T>(data, dtype: dtype.CupyDtype));
            }
            else
            {
                return new NDarray(np.array<T>(data, dtype: dtype.NumpyDtype));
            }
        }

        public static NDarray array(NDarray @object, Dtype dtype = null, bool? copy = null, string order = null,
            bool? subok = null, int? ndmin = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.array(@object.CupyNDarray, dtype?.CupyDtype, copy, order, subok, ndmin));
            }
            else
            {
                return new NDarray(np.array(@object.NumpyNDarray, dtype?.NumpyDtype, copy, order, subok, ndmin));
            }
        }

        public static NDarray<T> array<T>(T[] @object, Dtype dtype = null, bool? copy = null, string order = null,
            bool? subok = null, int? ndmin = null) where T : struct
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray<T>(cp.array<T>(@object, dtype?.CupyDtype, copy, order, subok, ndmin));
            }
            else
            {
                return new NDarray<T>(np.array<T>(@object, dtype?.NumpyDtype, copy, order, subok, ndmin));
            }
        }

        public static NDarray<T> array<T>(T[,] @object, Dtype dtype = null, bool? copy = null, string order = null,
            bool? subok = null, int? ndmin = null) where T : struct
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray<T>(cp.array<T>(@object, dtype?.CupyDtype, copy, order, subok, ndmin));
            }
            else
            {
                return new NDarray<T>(np.array<T>(@object, dtype?.NumpyDtype, copy, order, subok, ndmin));
            }
        }

        public static NDarray<T> array<T>(T[,,] data, Dtype dtype = null, bool? copy = null, string order = null,
            bool? subok = null, int? ndmin = null) where T : struct
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray<T>(cp.array<T>(data, dtype?.CupyDtype, copy, order, subok, ndmin));
            }
            else
            {
                return new NDarray<T>(np.array<T>(data, dtype?.NumpyDtype, copy, order, subok, ndmin));
            }
        }

        public static NDarray<T> array<T>(T[,,,] data, Dtype dtype = null, bool? copy = null, string order = null,
            bool? subok = null, int? ndmin = null) where T : struct
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray<T>(cp.array<T>(data, dtype?.CupyDtype, copy, order, subok, ndmin));
            }
            else
            {
                return new NDarray<T>(np.array<T>(data, dtype?.NumpyDtype, copy, order, subok, ndmin));
            }
        }

        public static NDarray<T> array<T>(T[,,,,] data, Dtype dtype = null, bool? copy = null, string order = null,
            bool? subok = null, int? ndmin = null) where T : struct
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray<T>(cp.array<T>(data, dtype?.CupyDtype, copy, order, subok, ndmin));
            }
            else
            {
                return new NDarray<T>(np.array<T>(data, dtype?.NumpyDtype, copy, order, subok, ndmin));
            }
        }

        public static NDarray<T> array<T>(T[,,,,,] data, Dtype dtype = null, bool? copy = null, string order = null,
            bool? subok = null, int? ndmin = null) where T : struct
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray<T>(cp.array<T>(data, dtype?.CupyDtype, copy, order, subok, ndmin));
            }
            else
            {
                return new NDarray<T>(np.array<T>(data, dtype?.NumpyDtype, copy, order, subok, ndmin));
            }
        }

        public static NDarray array(string[] strings, int? itemsize = null, bool? copy = null, bool? unicode = null,
            string order = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.array(strings, itemsize, copy, unicode, order));
            }
            else
            {
                return new NDarray(np.array(strings, itemsize, copy, unicode, order));
            }
        }

        public static NDarray array<T>(NDarray<T>[] arrays, Dtype dtype = null, bool? copy = null, string order = null,
            bool? subok = null, int? ndmin = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.array<T>(arrays.Select(x => x.CupyNDarray).ToArray(), dtype?.CupyDtype, copy, order, subok, ndmin));
            }
            else
            {
                return new NDarray(np.array<T>(arrays.Select(x => x.NumpyNDarray).ToArray(), dtype?.NumpyDtype, copy, order, subok, ndmin));
            }
        }

        public static NDarray array<T>(IEnumerable<NDarray<T>> arrays, Dtype dtype = null, bool? copy = null,
            string order = null, bool? subok = null, int? ndmin = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.array<T>(arrays.Select(x => x.CupyNDarray).ToArray(), dtype?.CupyDtype, copy, order, subok, ndmin));
            }
            else
            {
                return new NDarray(np.array<T>(arrays.Select(x => x.NumpyNDarray).ToArray(), dtype?.NumpyDtype, copy, order, subok, ndmin));
            }
        }

        public static NDarray array(List<NDarray> arrays, Dtype dtype = null, bool? copy = null, string order = null,
            bool? subok = null, int? ndmin = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.array(arrays.Select(x => x.CupyNDarray).ToArray(), dtype?.CupyDtype, copy, order, subok, ndmin));
            }
            else
            {
                return new NDarray(np.array(arrays.Select(x => x.NumpyNDarray).ToArray(), dtype?.NumpyDtype, copy, order, subok, ndmin));
            }
        }

        public static NDarray array(NDarray[] arrays, Dtype dtype = null, bool? copy = null, string order = null,
            bool? subok = null, int? ndmin = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.array(arrays.Select(x => x.CupyNDarray).ToArray(), dtype?.CupyDtype, copy, order, subok, ndmin));
            }
            else
            {
                return new NDarray(np.array(arrays.Select(x => x.NumpyNDarray).ToArray(), dtype?.NumpyDtype, copy, order, subok, ndmin));
            }
        }

        public static NDarray array(IEnumerable<NDarray> arrays, Dtype dtype = null, bool? copy = null,
            string order = null, bool? subok = null, int? ndmin = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.array(arrays.Select(x => x.CupyNDarray).ToArray(), dtype?.CupyDtype, copy, order, subok, ndmin));
            }
            else
            {
                return new NDarray(np.array(arrays.Select(x => x.NumpyNDarray).ToArray(), dtype?.NumpyDtype, copy, order, subok, ndmin));
            }
        }

        public static NDarray asarray(ValueType scalar, Dtype dtype = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.asarray(scalar, dtype?.CupyDtype));
            }
            else
            {
                return new NDarray(np.asarray(scalar, dtype?.NumpyDtype));
            }
        }

        /// <summary>
        ///     Convert an array of size 1 to its scalar equivalent.
        /// </summary>
        /// <returns>
        ///     Scalar representation of a. The output data type is the same type
        ///     returned by the input’s item method.
        /// </returns>
        public static T asscalar<T>(NDarray a)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return cp.asscalar<T>(a.CupyNDarray);
            }
            else
            {
                return np.asscalar<T>(a.NumpyNDarray);
            }
        }
    }
}
