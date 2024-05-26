using Cupy;
using Numpy;

namespace DeZero.NET
{
    public static partial class xp
    {
        /// <summary>
        ///     IEEE 754 floating point representation of (positive) infinity.
        /// </summary>
        public static float inf => Gpu.Available && Gpu.Use ? cp.inf : np.inf;

        /// <summary>
        ///     IEEE 754 floating point representation of (positive) infinity.
        ///     Use cp.inf because Inf, Infinity, PINF and infty are aliases for inf.For more details, see inf.
        /// </summary>
        public static float Inf => Gpu.Available && Gpu.Use ? cp.Inf : np.Inf;

        /// <summary>
        ///     IEEE 754 floating point representation of (positive) infinity.
        ///     Use cp.inf because Inf, Infinity, PINF and infty are aliases for inf.For more details, see inf.
        /// </summary>
        public static float Infinity => Gpu.Available && Gpu.Use ? cp.Infinity : np.Infinity;

        /// <summary>
        ///     IEEE 754 floating point representation of (positive) infinity.
        ///     Use cp.inf because Inf, Infinity, PINF and infty are aliases for inf.For more details, see inf.
        /// </summary>
        public static float PINF => Gpu.Available && Gpu.Use ? cp.PINF : np.PINF;

        /// <summary>
        ///     IEEE 754 floating point representation of (positive) infinity.
        ///     Use cp.inf because Inf, Infinity, PINF and infty are aliases for inf.For more details, see inf.
        /// </summary>
        public static float infty => Gpu.Available && Gpu.Use ? cp.infty : np.infty;

        /// <summary>
        ///     IEEE 754 floating point representation of (positive) infinity.
        /// </summary>
        public static float NINF => Gpu.Available && Gpu.Use ? cp.NINF : np.NINF;

        /// <summary>
        ///     IEEE 754 floating point representation of Not a Number(NaN).
        /// </summary>
        public static float nan => Gpu.Available && Gpu.Use ? cp.nan : np.nan;

        /// <summary>
        ///     IEEE 754 floating point representation of Not a Number(NaN).
        ///     NaN and NAN are equivalent definitions of nan.Please use nan instead of NAN.
        /// </summary>
        public static float NaN => Gpu.Available && Gpu.Use ? cp.NaN : np.NaN;

        /// <summary>
        ///     IEEE 754 floating point representation of Not a Number(NaN).
        ///     NaN and NAN are equivalent definitions of nan.Please use nan instead of NAN.
        /// </summary>
        public static float NAN => Gpu.Available && Gpu.Use ? cp.NAN : np.NAN;

        /// <summary>
        ///     IEEE 754 floating point representation of negative zero.
        /// </summary>
        public static float NZERO => Gpu.Available && Gpu.Use ? cp.NZERO : np.NZERO;

        /// <summary>
        ///     IEEE 754 floating point representation of positive zero.
        /// </summary>
        public static float PZERO => Gpu.Available && Gpu.Use ? cp.PZERO : np.PZERO;

        /// <summary>
        ///     Euler’s constant, base of natural logarithms, Napier’s constant.
        /// </summary>
        public static float e => Gpu.Available && Gpu.Use ? cp.e : np.e;

        /// <summary>
        ///     γ = 0.5772156649015328606065120900824024310421...
        ///     https://en.wikipedia.org/wiki/Euler-Mascheroni_constant
        /// </summary>
        public static float euler_gamma => Gpu.Available && Gpu.Use ? cp.euler_gamma : np.euler_gamma;

        /// <summary>
        ///     A convenient alias for None, useful for indexing arrays.
        /// </summary>
        public static object newaxis => Gpu.Available && Gpu.Use ? cp.newaxis : np.newaxis;

        /// <summary>
        ///     pi = 3.1415926535897932384626433...
        /// </summary>
        public static float pi => Gpu.Available && Gpu.Use ? cp.pi : np.pi;
    }
}
