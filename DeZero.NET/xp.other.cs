using Cupy;
using Numpy;

namespace DeZero.NET
{
    public static partial class xp
    {
        /// <summary>
        ///     Return the roots of a polynomial with coefficients given in p.<br></br>
        ///     The values in the rank-1 array p are coefficients of a polynomial.<br></br>
        ///     If the length of p is n+1 then the polynomial is described by:
        ///     Notes
        ///     The algorithm relies on computing the eigenvalues of the
        ///     companion matrix [1].<br></br>
        ///     References
        /// </summary>
        /// <param name="p">
        ///     Rank-1 array of polynomial coefficients.
        /// </param>
        /// <returns>
        ///     An array containing the roots of the polynomial.
        /// </returns>
        public static NDarray roots(this NDarray p)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.roots(p.CupyNDarray));
            }
            else
            {
                return new NDarray(np.roots(p.NumpyNDarray));
            }
        }
    }
}
