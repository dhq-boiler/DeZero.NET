using DeZero.NET;
using Python.Runtime;

namespace DeZero.NET.Extensions
{
    public static class NDArrayExtensions
    {
        public static NDarray Slice(this NDarray array, Slice[] slices)
        {
            if (slices.Length != array.ndim)
            {
                throw new ArgumentException("Number of slices must match the number of dimensions in the array");
            }

            int[] start = new int[array.ndim];
            int[] stop = new int[array.ndim];
            int[] step = new int[array.ndim];

            for (int i = 0; i < array.ndim; i++)
            {
                using var shape = array.shape;
                start[i] = slices[i].Start ?? 0;
                stop[i] = slices[i].Stop ?? shape[i];
                step[i] = slices[i].Step;

                // Handle negative indices
                if (start[i] < 0) start[i] += shape[i];
                if (stop[i] < 0) stop[i] += shape[i];

                // Ensure start and stop are within bounds
                start[i] = Math.Max(0, Math.Min(start[i], shape[i]));
                stop[i] = Math.Max(0, Math.Min(stop[i], shape[i]));
            }

            // Calculate the shape of the sliced array
            int[] newShape = new int[array.ndim];
            for (int i = 0; i < array.ndim; i++)
            {
                newShape[i] = (stop[i] - start[i] + step[i] - 1) / step[i];
            }

            using var array_dtype = array.dtype;
            // Create a new array with the calculated shape
            NDarray result = xp.empty(newShape, array_dtype);

            // Copy the data
            CopySlicedData(array, result, start, stop, step);

            return result;
        }

        public static void SetSlice(this NDarray array, Slice[] slices, NDarray value)
        {
            if (slices.Length != array.ndim)
            {
                throw new ArgumentException("Number of slices must match the number of dimensions in the array");
            }

            int[] start = new int[array.ndim];
            int[] stop = new int[array.ndim];
            int[] step = new int[array.ndim];

            // Calculate the shape of the sliced region
            int[] slicedShape = new int[array.ndim];
            for (int i = 0; i < array.ndim; i++)
            {
                using var shape = array.shape;
                start[i] = slices[i].Start ?? 0;
                stop[i] = slices[i].Stop ?? shape[i];
                step[i] = slices[i].Step;

                // Handle negative indices
                if (start[i] < 0) start[i] += shape[i];
                if (stop[i] < 0) stop[i] += shape[i];

                // Ensure start and stop are within bounds
                start[i] = Math.Max(0, Math.Min(start[i], shape[i]));
                stop[i] = Math.Max(0, Math.Min(stop[i], shape[i]));

                // Calculate shape of the slice
                slicedShape[i] = (stop[i] - start[i] + step[i] - 1) / step[i];
            }

            // If value is a scalar (0-dim array) or single value, broadcast it to the slice shape
            if (value.ndim == 0 || (value.ndim == 1 && value.size == 1))
            {
                var scalar = Convert(value.PyObject); // Get the scalar value
                using var broadcasted = xp.full(slicedShape, scalar, array.dtype);
                SetSlicedData(array, broadcasted, start, stop, step);
            }
            else
            {
                // Original behavior for non-scalar values
                SetSlicedData(array, value, start, stop, step);
            }
        }

        private static ValueType Convert(dynamic pyobj)
        {
            if (int.TryParse(pyobj.ToString(), out int intres))
            {
                return intres;
            }

            if (float.TryParse(pyobj.ToString(), out float floatres))
            {
                return floatres;
            }

            if (double.TryParse(pyobj.ToString(), out double doubleres))
            {
                return doubleres;
            }

            throw new ArgumentException("Unsupported type");
        }

        public static void SetSlice(this NDarray array, Slice[] slices, float value)
        {
            using var valueArray = xp.array(value);
            SetSlice(array, slices, valueArray);
        }

        public static void SetSlice(this NDarray array, Slice[] slices, double value)
        {
            using var valueArray = xp.array(value);
            SetSlice(array, slices, valueArray);
        }

        public static void SetSlice(this NDarray array, Slice[] slices, int value)
        {
            using var valueArray = xp.array(value);
            SetSlice(array, slices, valueArray);
        }

        private static void CopySlicedData(NDarray source, NDarray destination, int[] start, int[] stop, int[] step)
        {
            int[] sourceIndices = new int[source.ndim];
            int[] destIndices = new int[destination.ndim];

            CopySlicedDataRecursive(source, destination, start, stop, step, sourceIndices, destIndices, 0);
        }

        private static void SetSlicedData(NDarray destination, NDarray source, int[] start, int[] stop, int[] step)
        {
            int[] sourceIndices = new int[source.ndim];
            int[] destIndices = new int[destination.ndim];

            SetSlicedDataRecursive(destination, source, start, stop, step, destIndices, sourceIndices, 0);
        }

        private static void CopySlicedDataRecursive(NDarray source, NDarray destination, int[] start, int[] stop, int[] step, int[] sourceIndices, int[] destIndices, int dim)
        {
            if (dim == source.ndim)
            {
                destination[destIndices] = source[sourceIndices];
                return;
            }

            for (int i = start[dim], j = 0; i < stop[dim]; i += step[dim], j++)
            {
                sourceIndices[dim] = i;
                destIndices[dim] = j;
                CopySlicedDataRecursive(source, destination, start, stop, step, sourceIndices, destIndices, dim + 1);
            }
        }

        private static void SetSlicedDataRecursive(NDarray destination, NDarray source, int[] start, int[] stop, int[] step, int[] destIndices, int[] sourceIndices, int dim)
        {
            if (dim == destination.ndim)
            {
                destination[destIndices] = source[sourceIndices];
                return;
            }

            for (int i = start[dim], j = 0; i < stop[dim]; i += step[dim], j++)
            {
                if (destIndices.Length > 0)
                {
                    destIndices[dim] = i;
                }

                if (sourceIndices.Length > 0)
                {
                    sourceIndices[dim] = j;
                }

                SetSlicedDataRecursive(destination, source, start, stop, step, destIndices, sourceIndices, dim + 1);
            }
        }
    }
}
