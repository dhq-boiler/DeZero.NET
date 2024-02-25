using Cupy;
using Numpy;
using Python.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeZero.NET
{
    public static partial class xp
    {
        /// <summary>
        ///     Test whether each element of a 1-D array is also present in a second array.<br></br>
        ///     Returns a boolean array the same length as ar1 that is True
        ///     where an element of ar1 is in ar2 and False otherwise.<br></br>
        ///     We recommend using isin instead of in1d for new code.<br></br>
        ///     Notes
        ///     in1d can be considered as an element-wise function version of the
        ///     python keyword in, for 1-D sequences.<br></br>
        ///     in1d(a, b) is roughly
        ///     equivalent to cp.array([item in b for item in a]).<br></br>
        ///     However, this idea fails if ar2 is a set, or similar (non-sequence)
        ///     container:  As ar2 is converted to an array, in those cases
        ///     asarray(ar2) is an object array rather than the expected array of
        ///     contained values.
        /// </summary>
        /// <param name="ar1">
        ///     Input array.
        /// </param>
        /// <param name="ar2">
        ///     The values against which to test each value of ar1.
        /// </param>
        /// <param name="assume_unique">
        ///     If True, the input arrays are both assumed to be unique, which
        ///     can speed up the calculation.<br></br>
        ///     Default is False.
        /// </param>
        /// <param name="invert">
        ///     If True, the values in the returned array are inverted (that is,
        ///     False where an element of ar1 is in ar2 and True otherwise).<br></br>
        ///     Default is False.<br></br>
        ///     cp.in1d(a, b, invert=True) is equivalent
        ///     to (but is faster than) cp.invert(in1d(a, b)).
        /// </param>
        /// <returns>
        ///     The values ar1[in1d] are in ar2.
        /// </returns>
        public static NDarray in1d(this NDarray ar1, NDarray ar2, bool? assume_unique = false, bool? invert = false)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.in1d(ar1.CupyNDarray, ar2.CupyNDarray, assume_unique, invert));
            }
            else
            {
                return new NDarray(np.in1d(ar1.NumpyNDarray, ar2.NumpyNDarray, assume_unique, invert));
            }
        }

        /// <summary>
        ///     Find the intersection of two arrays.<br></br>
        ///     Return the sorted, unique values that are in both of the input arrays.
        /// </summary>
        /// <param name="ar2">
        ///     Input arrays.<br></br>
        ///     Will be flattened if not already 1D.
        /// </param>
        /// <param name="ar1">
        ///     Input arrays.<br></br>
        ///     Will be flattened if not already 1D.
        /// </param>
        /// <param name="assume_unique">
        ///     If True, the input arrays are both assumed to be unique, which
        ///     can speed up the calculation.<br></br>
        ///     Default is False.
        /// </param>
        /// <param name="return_indices">
        ///     If True, the indices which correspond to the intersection of the two
        ///     arrays are returned.<br></br>
        ///     The first instance of a value is used if there are
        ///     multiple.<br></br>
        ///     Default is False.
        /// </param>
        /// <returns>
        ///     A tuple of:
        ///     intersect1d
        ///     Sorted 1D array of common and unique elements.
        ///     comm1
        ///     The indices of the first occurrences of the common values in ar1.
        ///     Only provided if return_indices is True.
        ///     comm2
        ///     The indices of the first occurrences of the common values in ar2.
        ///     Only provided if return_indices is True.
        /// </returns>
        public static (NDarray, NDarray, NDarray) intersect1d(this NDarray ar2, NDarray ar1, bool assume_unique = false,
            bool return_indices = false)
        {
            if (Gpu.Available && Gpu.Use)
            {
                var ret = cp.intersect1d(ar2.CupyNDarray, ar1.CupyNDarray, assume_unique, return_indices);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2), new NDarray(ret.Item3));
            }
            else
            {
                var ret = np.intersect1d(ar2.NumpyNDarray, ar1.NumpyNDarray, assume_unique, return_indices);
                return (new NDarray(ret.Item1), new NDarray(ret.Item2), new NDarray(ret.Item3));
            }
        }

        /// <summary>
        ///     Calculates element in test_elements, broadcasting over element only.<br></br>
        ///     Returns a boolean array of the same shape as element that is True
        ///     where an element of element is in test_elements and False otherwise.<br></br>
        ///     Notes
        ///     isin is an element-wise function version of the python keyword in.<br></br>
        ///     isin(a, b) is roughly equivalent to
        ///     cp.array([item in b for item in a]) if a and b are 1-D sequences.<br></br>
        ///     element and test_elements are converted to arrays if they are not
        ///     already.<br></br>
        ///     If test_elements is a set (or other non-sequence collection)
        ///     it will be converted to an object array with one element, rather than an
        ///     array of the values contained in test_elements.<br></br>
        ///     This is a consequence
        ///     of the array constructor’s way of handling non-sequence collections.<br></br>
        ///     Converting the set to a list usually gives the desired behavior.
        /// </summary>
        /// <param name="element">
        ///     Input array.
        /// </param>
        /// <param name="test_elements">
        ///     The values against which to test each value of element.<br></br>
        ///     This argument is flattened if it is an array or array_like.<br></br>
        ///     See notes for behavior with non-array-like parameters.
        /// </param>
        /// <param name="assume_unique">
        ///     If True, the input arrays are both assumed to be unique, which
        ///     can speed up the calculation.<br></br>
        ///     Default is False.
        /// </param>
        /// <param name="invert">
        ///     If True, the values in the returned array are inverted, as if
        ///     calculating element not in test_elements.<br></br>
        ///     Default is False.<br></br>
        ///     cp.isin(a, b, invert=True) is equivalent to (but faster
        ///     than) cp.invert(cp.isin(a, b)).
        /// </param>
        /// <returns>
        ///     Has the same shape as element.<br></br>
        ///     The values element[isin]
        ///     are in test_elements.
        /// </returns>
        public static NDarray isin(this NDarray element, NDarray test_elements, bool? assume_unique = false,
            bool? invert = false)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.isin(element.CupyNDarray, test_elements.CupyNDarray, assume_unique, invert));
            }
            else
            {
                return new NDarray(np.isin(element.NumpyNDarray, test_elements.NumpyNDarray, assume_unique, invert));
            }
        }

        /// <summary>
        ///     Find the set difference of two arrays.<br></br>
        ///     Return the unique values in ar1 that are not in ar2.
        /// </summary>
        /// <param name="ar1">
        ///     Input array.
        /// </param>
        /// <param name="ar2">
        ///     Input comparison array.
        /// </param>
        /// <param name="assume_unique">
        ///     If True, the input arrays are both assumed to be unique, which
        ///     can speed up the calculation.<br></br>
        ///     Default is False.
        /// </param>
        /// <returns>
        ///     1D array of values in ar1 that are not in ar2. The result
        ///     is sorted when assume_unique=False, but otherwise only sorted
        ///     if the input is sorted.
        /// </returns>
        public static NDarray setdiff1d(this NDarray ar1, NDarray ar2, bool assume_unique = false)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.setdiff1d(ar1.CupyNDarray, ar2.CupyNDarray, assume_unique));
            }
            else
            {
                return new NDarray(np.setdiff1d(ar1.NumpyNDarray, ar2.NumpyNDarray, assume_unique));
            }
        }

        /// <summary>
        ///     Find the set exclusive-or of two arrays.<br></br>
        ///     Return the sorted, unique values that are in only one (not both) of the
        ///     input arrays.
        /// </summary>
        /// <param name="ar2">
        ///     Input arrays.
        /// </param>
        /// <param name="ar1">
        ///     Input arrays.
        /// </param>
        /// <param name="assume_unique">
        ///     If True, the input arrays are both assumed to be unique, which
        ///     can speed up the calculation.<br></br>
        ///     Default is False.
        /// </param>
        /// <returns>
        ///     Sorted 1D array of unique values that are in only one of the input
        ///     arrays.
        /// </returns>
        public static NDarray setxor1d(this NDarray ar2, NDarray ar1, bool assume_unique = false)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.setxor1d(ar2.CupyNDarray, ar1.CupyNDarray, assume_unique));
            }
            else
            {
                return new NDarray(np.setxor1d(ar2.NumpyNDarray, ar1.NumpyNDarray, assume_unique));
            }
        }

        /// <summary>
        ///     Find the union of two arrays.<br></br>
        ///     Return the unique, sorted array of values that are in either of the two
        ///     input arrays.
        /// </summary>
        /// <param name="ar2">
        ///     Input arrays.<br></br>
        ///     They are flattened if they are not already 1D.
        /// </param>
        /// <param name="ar1">
        ///     Input arrays.<br></br>
        ///     They are flattened if they are not already 1D.
        /// </param>
        /// <returns>
        ///     Unique, sorted union of the input arrays.
        /// </returns>
        public static NDarray union1d(this NDarray ar2, NDarray ar1)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.union1d(ar2.CupyNDarray, ar1.CupyNDarray));
            }
            else
            {
                return new NDarray(np.union1d(ar2.NumpyNDarray, ar1.NumpyNDarray));
            }
        }
    }
}
