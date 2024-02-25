using Cupy;
using Numpy;

namespace DeZero.NET
{
    public static partial class xp
    {
        /// <summary>
        ///     Compute the bit-wise AND of two arrays element-wise.<br></br>
        ///     Computes the bit-wise AND of the underlying binary representation of
        ///     the integers in the input arrays.<br></br>
        ///     This ufunc implements the C/Python
        ///     operator &amp;.
        /// </summary>
        /// <param name="x2">
        ///     Only integer and boolean types are handled.
        /// </param>
        /// <param name="x1">
        ///     Only integer and boolean types are handled.
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
        ///     Result.<br></br>
        ///     This is a scalar if both x1 and x2 are scalars.
        /// </returns>
        public static NDarray bitwise_and(this NDarray x2, NDarray x1, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(
                    cp.bitwise_and(x2.CupyNDarray, x1.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.bitwise_and(x2.NumpyNDarray, x1.NumpyNDarray, @out?.NumpyNDarray,
                    where?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Compute the bit-wise OR of two arrays element-wise.<br></br>
        ///     Computes the bit-wise OR of the underlying binary representation of
        ///     the integers in the input arrays.<br></br>
        ///     This ufunc implements the C/Python
        ///     operator |.
        /// </summary>
        /// <param name="x2">
        ///     Only integer and boolean types are handled.
        /// </param>
        /// <param name="x1">
        ///     Only integer and boolean types are handled.
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
        ///     Result.<br></br>
        ///     This is a scalar if both x1 and x2 are scalars.
        /// </returns>
        public static NDarray bitwise_or(this NDarray x2, NDarray x1, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(
                    cp.bitwise_or(x2.CupyNDarray, x1.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.bitwise_or(x2.NumpyNDarray, x1.NumpyNDarray, @out?.NumpyNDarray,
                    where?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Compute the bit-wise XOR of two arrays element-wise.<br></br>
        ///     Computes the bit-wise XOR of the underlying binary representation of
        ///     the integers in the input arrays.<br></br>
        ///     This ufunc implements the C/Python
        ///     operator ^.
        /// </summary>
        /// <param name="x2">
        ///     Only integer and boolean types are handled.
        /// </param>
        /// <param name="x1">
        ///     Only integer and boolean types are handled.
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
        ///     Result.<br></br>
        ///     This is a scalar if both x1 and x2 are scalars.
        /// </returns>
        public static NDarray bitwise_xor(this NDarray x2, NDarray x1, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(
                    cp.bitwise_xor(x2.CupyNDarray, x1.CupyNDarray, @out?.CupyNDarray, where?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.bitwise_xor(x2.NumpyNDarray, x1.NumpyNDarray, @out?.NumpyNDarray,
                    where?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Compute bit-wise inversion, or bit-wise NOT, element-wise.<br></br>
        ///     Computes the bit-wise NOT of the underlying binary representation of
        ///     the integers in the input arrays.<br></br>
        ///     This ufunc implements the C/Python
        ///     operator ~.
        ///     For signed integer inputs, the two’s complement is returned.<br></br>
        ///     In a
        ///     two’s-complement system negative numbers are represented by the two’s
        ///     complement of the absolute value.<br></br>
        ///     This is the most common method of
        ///     representing signed integers on computers [1].<br></br>
        ///     A N-bit
        ///     two’s-complement system can represent every integer in the range
        ///     to .
        ///     Notes
        ///     bitwise_not is an alias for invert:
        ///     References
        /// </summary>
        /// <param name="x">
        ///     Only integer and boolean types are handled.
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
        ///     Result.<br></br>
        ///     This is a scalar if x is a scalar.
        /// </returns>
        public static NDarray invert(this NDarray x, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(
                    cp.invert(x.CupyNDarray,  @out?.CupyNDarray, where?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.invert(x.NumpyNDarray, @out?.NumpyNDarray,
                    where?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Shift the bits of an integer to the left.<br></br>
        ///     Bits are shifted to the left by appending x2 0s at the right of x1.
        ///     Since the internal representation of numbers is in binary format, this
        ///     operation is equivalent to multiplying x1 by 2**x2.
        /// </summary>
        /// <param name="x1">
        ///     Input values.
        /// </param>
        /// <param name="x2">
        ///     Number of zeros to append to x1. Has to be non-negative.
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
        ///     Return x1 with bits shifted x2 times to the left.<br></br>
        ///     This is a scalar if both x1 and x2 are scalars.
        /// </returns>
        public static NDarray<int> left_shift(NDarray<int> x1, NDarray<int> x2, NDarray @out = null,
            NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray<int>(cp.left_shift(x1.CupyNDarray, x2.CupyNDarray, @out?.CupyNDarray,
                    where?.CupyNDarray));
            }
            else
            {
                return new NDarray<int>(np.left_shift(x1.NumpyNDarray, x2.NumpyNDarray, @out?.NumpyNDarray,
                    where?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Shift the bits of an integer to the right.<br></br>
        ///     Bits are shifted to the right x2.  Because the internal
        ///     representation of numbers is in binary format, this operation is
        ///     equivalent to dividing x1 by 2**x2.
        /// </summary>
        /// <param name="x1">
        ///     Input values.
        /// </param>
        /// <param name="x2">
        ///     Number of bits to remove at the right of x1.
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
        ///     Return x1 with bits shifted x2 times to the right.<br></br>
        ///     This is a scalar if both x1 and x2 are scalars.
        /// </returns>
        public static NDarray right_shift(this NDarray x1, NDarray x2, NDarray @out = null, NDarray where = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.right_shift(x1.CupyNDarray, x2.CupyNDarray, @out?.CupyNDarray,
                    where?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.right_shift(x1.NumpyNDarray, x2.NumpyNDarray, @out?.NumpyNDarray,
                    where?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Packs the elements of a binary-valued array into bits in a uint8 array.<br></br>
        ///     The result is padded to full bytes by inserting zero bits at the end.
        /// </summary>
        /// <param name="myarray">
        ///     An array of integers or booleans whose elements should be packed to
        ///     bits.
        /// </param>
        /// <param name="axis">
        ///     The dimension over which bit-packing is done.<br></br>
        ///     None implies packing the flattened array.
        /// </param>
        /// <returns>
        ///     Array of type uint8 whose elements represent bits corresponding to the
        ///     logical (0 or nonzero) value of the input elements.<br></br>
        ///     The shape of
        ///     packed has the same number of dimensions as the input (unless axis
        ///     is None, in which case the output is 1-D).
        /// </returns>
        public static NDarray packbits(this NDarray myarray, int? axis = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.packbits(myarray.CupyNDarray, axis));
            }
            else
            {
                return new NDarray(np.packbits(myarray.NumpyNDarray, axis));
            }
        }

        /// <summary>
        ///     Unpacks elements of a uint8 array into a binary-valued output array.<br></br>
        ///     Each element of myarray represents a bit-field that should be unpacked
        ///     into a binary-valued output array.<br></br>
        ///     The shape of the output array is either
        ///     1-D (if axis is None) or the same shape as the input array with unpacking
        ///     done along the axis specified.
        /// </summary>
        /// <param name="myarray">
        ///     Input array.
        /// </param>
        /// <param name="axis">
        ///     The dimension over which bit-unpacking is done.<br></br>
        ///     None implies unpacking the flattened array.
        /// </param>
        /// <returns>
        ///     The elements are binary-valued (0 or 1).
        /// </returns>
        public static NDarray unpackbits(this NDarray myarray, int? axis = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.unpackbits(myarray.CupyNDarray, axis));
            }
            else
            {
                return new NDarray(np.unpackbits(myarray.NumpyNDarray, axis));
            }
        }

        /// <summary>
        ///     Return the binary representation of the input number as a string.<br></br>
        ///     For negative numbers, if width is not given, a minus sign is added to the
        ///     front.<br></br>
        ///     If width is given, the two’s complement of the number is
        ///     returned, with respect to that width.<br></br>
        ///     In a two’s-complement system negative numbers are represented by the two’s
        ///     complement of the absolute value.<br></br>
        ///     This is the most common method of
        ///     representing signed integers on computers [1].<br></br>
        ///     A N-bit two’s-complement
        ///     system can represent every integer in the range
        ///     to .
        ///     Notes
        ///     binary_repr is equivalent to using base_repr with base 2, but about 25x
        ///     faster.<br></br>
        ///     References
        /// </summary>
        /// <param name="num">
        ///     Only an integer decimal number can be used.
        /// </param>
        /// <param name="width">
        ///     The length of the returned string if num is positive, or the length
        ///     of the two’s complement if num is negative, provided that width is
        ///     at least a sufficient number of bits for num to be represented in the
        ///     designated form.<br></br>
        ///     If the width value is insufficient, it will be ignored, and num will
        ///     be returned in binary (num &gt; 0) or two’s complement (num &lt; 0) form
        ///     with its width equal to the minimum number of bits needed to represent
        ///     the number in the designated form.<br></br>
        ///     This behavior is deprecated and will
        ///     later raise an error.
        /// </param>
        /// <returns>
        ///     Binary representation of num or two’s complement of num.
        /// </returns>
        public static string binary_repr(int num, int? width = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return cp.binary_repr(num, width);
            }
            else
            {
                return np.binary_repr(num, width);
            }
        }
    }
}
