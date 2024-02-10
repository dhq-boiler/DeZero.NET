using Cupy;
using Numpy;

namespace DeZero.NET
{
    public static partial class xp
    {
        /// <summary>
        ///     IEEE 754 floating point representation of (positive) infinity.
        /// </summary>
        public static float inf => Core.GpuAvailable && Core.UseGpu ? cp.inf : np.inf;

        /// <summary>
        ///     IEEE 754 floating point representation of (positive) infinity.
        ///     Use cp.inf because Inf, Infinity, PINF and infty are aliases for inf.For more details, see inf.
        /// </summary>
        public static float Inf => Core.GpuAvailable && Core.UseGpu ? cp.Inf : np.Inf;

        /// <summary>
        ///     IEEE 754 floating point representation of (positive) infinity.
        ///     Use cp.inf because Inf, Infinity, PINF and infty are aliases for inf.For more details, see inf.
        /// </summary>
        public static float Infinity => Core.GpuAvailable && Core.UseGpu ? cp.Infinity : np.Infinity;

        /// <summary>
        ///     IEEE 754 floating point representation of (positive) infinity.
        ///     Use cp.inf because Inf, Infinity, PINF and infty are aliases for inf.For more details, see inf.
        /// </summary>
        public static float PINF => Core.GpuAvailable && Core.UseGpu ? cp.PINF : np.PINF;

        /// <summary>
        ///     IEEE 754 floating point representation of (positive) infinity.
        ///     Use cp.inf because Inf, Infinity, PINF and infty are aliases for inf.For more details, see inf.
        /// </summary>
        public static float infty => Core.GpuAvailable && Core.UseGpu ? cp.infty : np.infty;

        /// <summary>
        ///     IEEE 754 floating point representation of (positive) infinity.
        /// </summary>
        public static float NINF => Core.GpuAvailable && Core.UseGpu ? cp.NINF : np.NINF;

        /// <summary>
        ///     IEEE 754 floating point representation of Not a Number(NaN).
        /// </summary>
        public static float nan => Core.GpuAvailable && Core.UseGpu ? cp.nan : np.nan;

        /// <summary>
        ///     IEEE 754 floating point representation of Not a Number(NaN).
        ///     NaN and NAN are equivalent definitions of nan.Please use nan instead of NAN.
        /// </summary>
        public static float NaN => Core.GpuAvailable && Core.UseGpu ? cp.NaN : np.NaN;

        /// <summary>
        ///     IEEE 754 floating point representation of Not a Number(NaN).
        ///     NaN and NAN are equivalent definitions of nan.Please use nan instead of NAN.
        /// </summary>
        public static float NAN => Core.GpuAvailable && Core.UseGpu ? cp.NAN : np.NAN;

        /// <summary>
        ///     IEEE 754 floating point representation of negative zero.
        /// </summary>
        public static float NZERO => Core.GpuAvailable && Core.UseGpu ? cp.NZERO : np.NZERO;

        /// <summary>
        ///     IEEE 754 floating point representation of positive zero.
        /// </summary>
        public static float PZERO => Core.GpuAvailable && Core.UseGpu ? cp.PZERO : np.PZERO;

        /// <summary>
        ///     Euler’s constant, base of natural logarithms, Napier’s constant.
        /// </summary>
        public static float e => Core.GpuAvailable && Core.UseGpu ? cp.e : np.e;

        /// <summary>
        ///     γ = 0.5772156649015328606065120900824024310421...
        ///     https://en.wikipedia.org/wiki/Euler-Mascheroni_constant
        /// </summary>
        public static float euler_gamma => Core.GpuAvailable && Core.UseGpu ? cp.euler_gamma : np.euler_gamma;

        /// <summary>
        ///     A convenient alias for None, useful for indexing arrays.
        /// </summary>
        public static object newaxis => Core.GpuAvailable && Core.UseGpu ? cp.newaxis : np.newaxis;

        /// <summary>
        ///     pi = 3.1415926535897932384626433...
        /// </summary>
        public static float pi => Core.GpuAvailable && Core.UseGpu ? cp.pi : np.pi;
    }
}
