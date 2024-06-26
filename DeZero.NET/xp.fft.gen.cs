﻿using Cupy;
using Numpy;

namespace DeZero.NET
{
    public static partial class xp
    {
        public static partial class fft
        {
            /// <summary>
            ///     Compute the one-dimensional discrete Fourier Transform for real input.<br></br>
            ///     This function computes the one-dimensional n-point discrete Fourier
            ///     Transform (DFT) of a real-valued array by means of an efficient algorithm
            ///     called the Fast Fourier Transform (FFT).<br></br>
            ///     Notes
            ///     When the DFT is computed for purely real input, the output is
            ///     Hermitian-symmetric, i.e.<br></br>
            ///     the negative frequency terms are just the complex
            ///     conjugates of the corresponding positive-frequency terms, and the
            ///     negative-frequency terms are therefore redundant.<br></br>
            ///     This function does not
            ///     compute the negative frequency terms, and the length of the transformed
            ///     axis of the output is therefore n//2 + 1.<br></br>
            ///     When A = rfft(a) and fs is the sampling frequency, A[0] contains
            ///     the zero-frequency term 0*fs, which is real due to Hermitian symmetry.<br></br>
            ///     If n is even, A[-1] contains the term representing both positive
            ///     and negative Nyquist frequency (+fs/2 and -fs/2), and must also be purely
            ///     real.<br></br>
            ///     If n is odd, there is no term at fs/2; A[-1] contains
            ///     the largest positive frequency (fs/2*(n-1)/n), and is complex in the
            ///     general case.<br></br>
            ///     If the input a contains an imaginary part, it is silently discarded.
            /// </summary>
            /// <param name="a">
            ///     Input array
            /// </param>
            /// <param name="n">
            ///     Number of points along transformation axis in the input to use.<br></br>
            ///     If n is smaller than the length of the input, the input is cropped.<br></br>
            ///     If it is larger, the input is padded with zeros.<br></br>
            ///     If n is not given,
            ///     the length of the input along the axis specified by axis is used.
            /// </param>
            /// <param name="axis">
            ///     Axis over which to compute the FFT.<br></br>
            ///     If not given, the last axis is
            ///     used.
            /// </param>
            /// <param name="norm">
            ///     Normalization mode (see Cupy.fft).<br></br>
            ///     Default is None.
            /// </param>
            /// <returns>
            ///     The truncated or zero-padded input, transformed along the axis
            ///     indicated by axis, or the last one if axis is not specified.<br></br>
            ///     If n is even, the length of the transformed axis is (n/2)+1.
            ///     If n is odd, the length is (n+1)/2.
            /// </returns>
            public static NDarray rfft(NDarray a, int? n = null, int? axis = -1, string norm = null)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(cp.fft.rfft(a.CupyNDarray, n, axis, norm));
                }
                else
                {
                    return new NDarray(np.fft.rfft(a.NumpyNDarray, n, axis, norm));
                }
            }
        }

        public static partial class fft
        {
            /// <summary>
            ///     Compute the inverse of the n-point DFT for real input.<br></br>
            ///     This function computes the inverse of the one-dimensional n-point
            ///     discrete Fourier Transform of real input computed by rfft.<br></br>
            ///     In other words, irfft(rfft(a), len(a)) == a to within numerical
            ///     accuracy.<br></br>
            ///     (See Notes below for why len(a) is necessary here.)
            ///     The input is expected to be in the form returned by rfft, i.e.<br></br>
            ///     the
            ///     real zero-frequency term followed by the complex positive frequency terms
            ///     in order of increasing frequency.<br></br>
            ///     Since the discrete Fourier Transform of
            ///     real input is Hermitian-symmetric, the negative frequency terms are taken
            ///     to be the complex conjugates of the corresponding positive frequency terms.<br></br>
            ///     Notes
            ///     Returns the real valued n-point inverse discrete Fourier transform
            ///     of a, where a contains the non-negative frequency terms of a
            ///     Hermitian-symmetric sequence.<br></br>
            ///     n is the length of the result, not the
            ///     input.<br></br>
            ///     If you specify an n such that a must be zero-padded or truncated, the
            ///     extra/removed values will be added/removed at high frequencies.<br></br>
            ///     One can
            ///     thus resample a series to m points via Fourier interpolation by:
            ///     a_resamp = irfft(rfft(a), m).
            /// </summary>
            /// <param name="a">
            ///     The input array.
            /// </param>
            /// <param name="n">
            ///     Length of the transformed axis of the output.<br></br>
            ///     For n output points, n//2+1 input points are necessary.<br></br>
            ///     If the
            ///     input is longer than this, it is cropped.<br></br>
            ///     If it is shorter than this,
            ///     it is padded with zeros.<br></br>
            ///     If n is not given, it is determined from
            ///     the length of the input along the axis specified by axis.
            /// </param>
            /// <param name="axis">
            ///     Axis over which to compute the inverse FFT.<br></br>
            ///     If not given, the last
            ///     axis is used.
            /// </param>
            /// <param name="norm">
            ///     Normalization mode (see Cupy.fft).<br></br>
            ///     Default is None.
            /// </param>
            /// <returns>
            ///     The truncated or zero-padded input, transformed along the axis
            ///     indicated by axis, or the last one if axis is not specified.<br></br>
            ///     The length of the transformed axis is n, or, if n is not given,
            ///     2*(m-1) where m is the length of the transformed axis of the
            ///     input.<br></br>
            ///     To get an odd number of output points, n must be specified.
            /// </returns>
            public static NDarray irfft(NDarray a, int? n = null, int? axis = -1, string norm = null)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(cp.fft.irfft(a.CupyNDarray, n, axis, norm));
                }
                else
                {
                    return new NDarray(np.fft.irfft(a.NumpyNDarray, n, axis, norm));
                }
            }
        }

        public static partial class fft
        {
            /// <summary>
            ///     Compute the 2-dimensional FFT of a real array.<br></br>
            ///     Notes
            ///     This is really just rfftn with different default behavior.<br></br>
            ///     For more details see rfftn.
            /// </summary>
            /// <param name="a">
            ///     Input array, taken to be real.
            /// </param>
            /// <param name="s">
            ///     Shape of the FFT.
            /// </param>
            /// <param name="axes">
            ///     Axes over which to compute the FFT.
            /// </param>
            /// <param name="norm">
            ///     Normalization mode (see Cupy.fft).<br></br>
            ///     Default is None.
            /// </param>
            /// <returns>
            ///     The result of the real 2-D FFT.
            /// </returns>
            public static NDarray rfft2(NDarray a, int[] s = null, int[] axes = null, string norm = null)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(cp.fft.rfft2(a.CupyNDarray, s, axes, norm));
                }
                else
                {
                    return new NDarray(np.fft.rfft2(a.NumpyNDarray, s, axes, norm));
                }
            }
        }

        public static partial class fft
        {
            /// <summary>
            ///     Compute the 2-dimensional inverse FFT of a real array.<br></br>
            ///     Notes
            ///     This is really irfftn with different defaults.<br></br>
            ///     For more details see irfftn.
            /// </summary>
            /// <param name="a">
            ///     The input array
            /// </param>
            /// <param name="s">
            ///     Shape of the inverse FFT.
            /// </param>
            /// <param name="axes">
            ///     The axis over which to compute the inverse fft.<br></br>
            ///     Default is the last two axis.
            /// </param>
            /// <param name="norm">
            ///     Normalization mode (see Cupy.fft).<br></br>
            ///     Default is None.
            /// </param>
            /// <returns>
            ///     The result of the inverse real 2-D FFT.
            /// </returns>
            public static NDarray irfft2(NDarray a, int[] s = null, int[] axes = null, string norm = null)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(cp.fft.irfft2(a.CupyNDarray, s, axes, norm));
                }
                else
                {
                    return new NDarray(np.fft.irfft2(a.NumpyNDarray, s, axes, norm));
                }
            }
        }

        public static partial class fft
        {
            /// <summary>
            ///     Compute the N-dimensional discrete Fourier Transform for real input.<br></br>
            ///     This function computes the N-dimensional discrete Fourier Transform over
            ///     any number of axis in an M-dimensional real array by means of the Fast
            ///     Fourier Transform (FFT).<br></br>
            ///     By default, all axis are transformed, with the
            ///     real transform performed over the last axis, while the remaining
            ///     transforms are complex.<br></br>
            ///     Notes
            ///     The transform for real input is performed over the last transformation
            ///     axis, as by rfft, then the transform over the remaining axis is
            ///     performed as by fftn.<br></br>
            ///     The order of the output is as for rfft for the
            ///     final transformation axis, and as for fftn for the remaining
            ///     transformation axis.<br></br>
            ///     See fft for details, definitions and conventions used.
            /// </summary>
            /// <param name="a">
            ///     Input array, taken to be real.
            /// </param>
            /// <param name="s">
            ///     Shape (length along each transformed axis) to use from the input.<br></br>
            ///     (s[0] refers to axis 0, s[1] to axis 1, etc.).<br></br>
            ///     The final element of s corresponds to n for rfft(x, n), while
            ///     for the remaining axis, it corresponds to n for fft(x, n).<br></br>
            ///     Along any axis, if the given shape is smaller than that of the input,
            ///     the input is cropped.<br></br>
            ///     If it is larger, the input is padded with zeros.<br></br>
            ///     if s is not given, the shape of the input along the axis specified
            ///     by axis is used.
            /// </param>
            /// <param name="axes">
            ///     Axes over which to compute the FFT.<br></br>
            ///     If not given, the last len(s)
            ///     axis are used, or all axis if s is also not specified.
            /// </param>
            /// <param name="norm">
            ///     Normalization mode (see Cupy.fft).<br></br>
            ///     Default is None.
            /// </param>
            /// <returns>
            ///     The truncated or zero-padded input, transformed along the axis
            ///     indicated by axis, or by a combination of s and a,
            ///     as explained in the parameters section above.<br></br>
            ///     The length of the last axis transformed will be s[-1]//2+1,
            ///     while the remaining transformed axis will have lengths according to
            ///     s, or unchanged from the input.
            /// </returns>
            public static NDarray rfftn(NDarray a, int[] s = null, int[] axes = null, string norm = null)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(cp.fft.rfftn(a.CupyNDarray, s, axes, norm));
                }
                else
                {
                    return new NDarray(np.fft.rfftn(a.NumpyNDarray, s, axes, norm));
                }
            }
        }

        public static partial class fft
        {
            /// <summary>
            ///     Compute the inverse of the N-dimensional FFT of real input.<br></br>
            ///     This function computes the inverse of the N-dimensional discrete
            ///     Fourier Transform for real input over any number of axis in an
            ///     M-dimensional array by means of the Fast Fourier Transform (FFT).<br></br>
            ///     In
            ///     other words, irfftn(rfftn(a), a.shape) == a to within numerical
            ///     accuracy.<br></br>
            ///     (The a.shape is necessary like len(a) is for irfft,
            ///     and for the same reason.)
            ///     The input should be ordered in the same way as is returned by rfftn,
            ///     i.e.<br></br>
            ///     as for irfft for the final transformation axis, and as for ifftn
            ///     along all the other axis.<br></br>
            ///     Notes
            ///     See fft for definitions and conventions used.<br></br>
            ///     See rfft for definitions and conventions used for real input.
            /// </summary>
            /// <param name="a">
            ///     Input array.
            /// </param>
            /// <param name="s">
            ///     Shape (length of each transformed axis) of the output
            ///     (s[0] refers to axis 0, s[1] to axis 1, etc.).<br></br>
            ///     s is also the
            ///     number of input points used along this axis, except for the last axis,
            ///     where s[-1]//2+1 points of the input are used.<br></br>
            ///     Along any axis, if the shape indicated by s is smaller than that of
            ///     the input, the input is cropped.<br></br>
            ///     If it is larger, the input is padded
            ///     with zeros.<br></br>
            ///     If s is not given, the shape of the input along the
            ///     axis specified by axis is used.
            /// </param>
            /// <param name="axes">
            ///     Axes over which to compute the inverse FFT.<br></br>
            ///     If not given, the last
            ///     len(s) axis are used, or all axis if s is also not specified.<br></br>
            ///     Repeated indices in axis means that the inverse transform over that
            ///     axis is performed multiple times.
            /// </param>
            /// <param name="norm">
            ///     Normalization mode (see Cupy.fft).<br></br>
            ///     Default is None.
            /// </param>
            /// <returns>
            ///     The truncated or zero-padded input, transformed along the axis
            ///     indicated by axis, or by a combination of s or a,
            ///     as explained in the parameters section above.<br></br>
            ///     The length of each transformed axis is as given by the corresponding
            ///     element of s, or the length of the input in every axis except for the
            ///     last one if s is not given.<br></br>
            ///     In the final transformed axis the length
            ///     of the output when s is not given is 2*(m-1) where m is the
            ///     length of the final transformed axis of the input.<br></br>
            ///     To get an odd
            ///     number of output points in the final axis, s must be specified.
            /// </returns>
            public static NDarray irfftn(NDarray a, int[] s = null, int[] axes = null, string norm = null)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(cp.fft.irfftn(a.CupyNDarray, s, axes, norm));
                }
                else
                {
                    return new NDarray(np.fft.irfftn(a.NumpyNDarray, s, axes, norm));
                }
            }
        }

        public static partial class fft
        {
            /// <summary>
            ///     Compute the FFT of a signal that has Hermitian symmetry, i.e., a real
            ///     spectrum.<br></br>
            ///     Notes
            ///     hfft/ihfft are a pair analogous to rfft/irfft, but for the
            ///     opposite case: here the signal has Hermitian symmetry in the time
            ///     domain and is real in the frequency domain.<br></br>
            ///     So here it’s hfft for
            ///     which you must supply the length of the result if it is to be odd.
            /// </summary>
            /// <param name="a">
            ///     The input array.
            /// </param>
            /// <param name="n">
            ///     Length of the transformed axis of the output.<br></br>
            ///     For n output
            ///     points, n//2 + 1 input points are necessary.<br></br>
            ///     If the input is
            ///     longer than this, it is cropped.<br></br>
            ///     If it is shorter than this, it is
            ///     padded with zeros.<br></br>
            ///     If n is not given, it is determined from the
            ///     length of the input along the axis specified by axis.
            /// </param>
            /// <param name="axis">
            ///     Axis over which to compute the FFT.<br></br>
            ///     If not given, the last
            ///     axis is used.
            /// </param>
            /// <param name="norm">
            ///     Normalization mode (see Cupy.fft).<br></br>
            ///     Default is None.
            /// </param>
            /// <returns>
            ///     The truncated or zero-padded input, transformed along the axis
            ///     indicated by axis, or the last one if axis is not specified.<br></br>
            ///     The length of the transformed axis is n, or, if n is not given,
            ///     2*m - 2 where m is the length of the transformed axis of
            ///     the input.<br></br>
            ///     To get an odd number of output points, n must be
            ///     specified, for instance as 2*m - 1 in the typical case,
            /// </returns>
            public static NDarray hfft(NDarray a, int? n = null, int? axis = -1, string norm = null)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(cp.fft.hfft(a.CupyNDarray, n, axis, norm));
                }
                else
                {
                    return new NDarray(np.fft.hfft(a.NumpyNDarray, n, axis, norm));
                }
            }
        }

        public static partial class fft
        {
            /// <summary>
            ///     Compute the inverse FFT of a signal that has Hermitian symmetry.<br></br>
            ///     Notes
            ///     hfft/ihfft are a pair analogous to rfft/irfft, but for the
            ///     opposite case: here the signal has Hermitian symmetry in the time
            ///     domain and is real in the frequency domain.<br></br>
            ///     So here it’s hfft for
            ///     which you must supply the length of the result if it is to be odd:
            /// </summary>
            /// <param name="a">
            ///     Input array.
            /// </param>
            /// <param name="n">
            ///     Length of the inverse FFT, the number of points along
            ///     transformation axis in the input to use.<br></br>
            ///     If n is smaller than
            ///     the length of the input, the input is cropped.<br></br>
            ///     If it is larger,
            ///     the input is padded with zeros.<br></br>
            ///     If n is not given, the length of
            ///     the input along the axis specified by axis is used.
            /// </param>
            /// <param name="axis">
            ///     Axis over which to compute the inverse FFT.<br></br>
            ///     If not given, the last
            ///     axis is used.
            /// </param>
            /// <param name="norm">
            ///     Normalization mode (see Cupy.fft).<br></br>
            ///     Default is None.
            /// </param>
            /// <returns>
            ///     The truncated or zero-padded input, transformed along the axis
            ///     indicated by axis, or the last one if axis is not specified.<br></br>
            ///     The length of the transformed axis is n//2 + 1.
            /// </returns>
            public static NDarray ihfft(NDarray a, int? n = null, int? axis = -1, string norm = null)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(cp.fft.ihfft(a.CupyNDarray, n, axis, norm));
                }
                else
                {
                    return new NDarray(np.fft.ihfft(a.NumpyNDarray, n, axis, norm));
                }
            }
        }

        public static partial class fft
        {
            /// <summary>
            ///     Return the Discrete Fourier Transform sample frequencies.<br></br>
            ///     The returned float array f contains the frequency bin centers in cycles
            ///     per unit of the sample spacing (with zero at the start).<br></br>
            ///     For instance, if
            ///     the sample spacing is in seconds, then the frequency unit is cycles/second.<br></br>
            ///     Given a window length n and a sample spacing d:
            /// </summary>
            /// <param name="n">
            ///     Window length.
            /// </param>
            /// <param name="d">
            ///     Sample spacing (inverse of the sampling rate).<br></br>
            ///     Defaults to 1.
            /// </param>
            /// <returns>
            ///     Array of length n containing the sample frequencies.
            /// </returns>
            public static NDarray fftfreq(int n, float? d = 1.0f)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(cp.fft.fftfreq(n, d));
                }
                else
                {
                    return new NDarray(np.fft.fftfreq(n, d));
                }
            }
        }

        public static partial class fft
        {
            /// <summary>
            ///     Return the Discrete Fourier Transform sample frequencies
            ///     (for usage with rfft, irfft).<br></br>
            ///     The returned float array f contains the frequency bin centers in cycles
            ///     per unit of the sample spacing (with zero at the start).<br></br>
            ///     For instance, if
            ///     the sample spacing is in seconds, then the frequency unit is cycles/second.<br></br>
            ///     Given a window length n and a sample spacing d:
            ///     Unlike fftfreq (but like scipy.fftpack.rfftfreq)
            ///     the Nyquist frequency component is considered to be positive.
            /// </summary>
            /// <param name="n">
            ///     Window length.
            /// </param>
            /// <param name="d">
            ///     Sample spacing (inverse of the sampling rate).<br></br>
            ///     Defaults to 1.
            /// </param>
            /// <returns>
            ///     Array of length n//2 + 1 containing the sample frequencies.
            /// </returns>
            public static NDarray rfftfreq(int n, float? d = 1.0f)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(cp.fft.rfftfreq(n, d));
                }
                else
                {
                    return new NDarray(np.fft.rfftfreq(n, d));
                }
            }
        }

        public static partial class fft
        {
            /// <summary>
            ///     Shift the zero-frequency component to the center of the spectrum.<br></br>
            ///     This function swaps half-spaces for all axis listed (defaults to all).<br></br>
            ///     Note that y[0] is the Nyquist component only if len(x) is even.
            /// </summary>
            /// <param name="x">
            ///     Input array.
            /// </param>
            /// <param name="axes">
            ///     Axes over which to shift.<br></br>
            ///     Default is None, which shifts all axis.
            /// </param>
            /// <returns>
            ///     The shifted array.
            /// </returns>
            public static NDarray fftshift(NDarray x, int[] axes = null)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(cp.fft.fftshift(x.CupyNDarray, axes));
                }
                else
                {
                    return new NDarray(np.fft.fftshift(x.NumpyNDarray, axes));
                }
            }
        }

        public static partial class fft
        {
            /// <summary>
            ///     The inverse of fftshift.<br></br>
            ///     Although identical for even-length x, the
            ///     functions differ by one sample for odd-length x.
            /// </summary>
            /// <param name="x">
            ///     Input array.
            /// </param>
            /// <param name="axes">
            ///     Axes over which to calculate.<br></br>
            ///     Defaults to None, which shifts all axis.
            /// </param>
            /// <returns>
            ///     The shifted array.
            /// </returns>
            public static NDarray ifftshift(NDarray x, int[] axes = null)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(cp.fft.ifftshift(x.CupyNDarray, axes));
                }
                else
                {
                    return new NDarray(np.fft.ifftshift(x.NumpyNDarray, axes));
                }
            }
        }
    }
}
