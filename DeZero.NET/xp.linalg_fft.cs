﻿using Cupy;
using Numpy;

namespace DeZero.NET
{
    public static partial class xp
    {
        /// <summary>
        ///     Modified Bessel function of the first kind, order 0.<br></br>
        ///     Usually denoted .  This function does broadcast, but will not
        ///     “up-cast” int dtype arguments unless accompanied by at least one float or
        ///     complex dtype argument (see Raises below).<br></br>
        ///     Notes
        ///     We use the algorithm published by Clenshaw [1] and referenced by
        ///     Abramowitz and Stegun [2], for which the function domain is
        ///     partitioned into the two intervals [0,8] and (8,inf), and Chebyshev
        ///     polynomial expansions are employed in each interval.<br></br>
        ///     Relative error on
        ///     the domain [0,30] using IEEE arithmetic is documented [3] as having a
        ///     peak of 5.8e-16 with an rms of 1.4e-16 (n = 30000).<br></br>
        ///     References
        /// </summary>
        /// <param name="x">
        ///     Argument of the Bessel function.
        /// </param>
        /// <returns>
        ///     The modified Bessel function evaluated at each of the elements of x.
        /// </returns>
        public static NDarray i0(this NDarray x)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.i0(x.CupyNDarray));
            }
            else
            {
                return new NDarray(np.i0(x.NumpyNDarray));
            }
        }

        public static partial class linalg
        {
            /// <summary>
            ///     Cholesky decomposition.<br></br>
            ///     Return the Cholesky decomposition, L * L.H, of the square matrix a,
            ///     where L is lower-triangular and .H is the conjugate transpose operator
            ///     (which is the ordinary transpose if a is real-valued).<br></br>
            ///     a must be
            ///     Hermitian (symmetric if real-valued) and positive-definite.<br></br>
            ///     Only L is
            ///     actually returned.<br></br>
            ///     Notes
            ///     Broadcasting rules apply, see the Cupy.linalg documentation for
            ///     details.<br></br>
            ///     The Cholesky decomposition is often used as a fast way of solving
            ///     (when A is both Hermitian/symmetric and positive-definite).<br></br>
            ///     First, we solve for  in
            ///     and then for  in
            /// </summary>
            /// <param name="a">
            ///     Hermitian (symmetric if all elements are real), positive-definite
            ///     input matrix.
            /// </param>
            /// <returns>
            ///     Upper or lower-triangular Cholesky factor of a.<br></br>
            ///     Returns a
            ///     matrix object if a is a matrix object.
            /// </returns>
            public static NDarray cholesky(NDarray a)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(cp.linalg.cholesky(a.CupyNDarray));
                }
                else
                {
                    return new NDarray(np.linalg.cholesky(a.NumpyNDarray));
                }
            }
        }

        public static partial class linalg
        {
            /// <summary>
            ///     Compute the determinant of an array.<br></br>
            ///     Notes
            ///     Broadcasting rules apply, see the Cupy.linalg documentation for
            ///     details.<br></br>
            ///     The determinant is computed via LU factorization using the LAPACK
            ///     routine z/dgetrf.
            /// </summary>
            /// <param name="a">
            ///     Input array to compute determinants for.
            /// </param>
            /// <returns>
            ///     Determinant of a.
            /// </returns>
            public static NDarray det(NDarray a)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(cp.linalg.det(a.CupyNDarray));
                }
                else
                {
                    return new NDarray(np.linalg.det(a.NumpyNDarray));
                }
            }
        }

        public static partial class linalg
        {
            /// <summary>
            ///     Compute the eigenvalues and right eigenvectors of a square array.<br></br>
            ///     Notes
            ///     Broadcasting rules apply, see the Cupy.linalg documentation for
            ///     details.<br></br>
            ///     This is implemented using the _geev LAPACK routines which compute
            ///     the eigenvalues and eigenvectors of general square arrays.<br></br>
            ///     The number w is an eigenvalue of a if there exists a vector
            ///     v such that dot(a,v) = w * v.<br></br>
            ///     Thus, the arrays a, w, and
            ///     v satisfy the equations dot(a[:,:], v[:,i]) = w[i] * v[:,i]
            ///     for .
            ///     The array v of eigenvectors may not be of maximum rank, that is, some
            ///     of the columns may be linearly dependent, although round-off error may
            ///     obscure that fact.<br></br>
            ///     If the eigenvalues are all different, then theoretically
            ///     the eigenvectors are linearly independent.<br></br>
            ///     Likewise, the (complex-valued)
            ///     matrix of eigenvectors v is unitary if the matrix a is normal, i.e.,
            ///     if dot(a, a.H) = dot(a.H, a), where a.H denotes the conjugate
            ///     transpose of a.<br></br>
            ///     Finally, it is emphasized that v consists of the right (as in
            ///     right-hand side) eigenvectors of a.<br></br>
            ///     A vector y satisfying
            ///     dot(y.T, a) = z * y.T for some number z is called a left
            ///     eigenvector of a, and, in general, the left and right eigenvectors
            ///     of a matrix are not necessarily the (perhaps conjugate) transposes
            ///     of each other.<br></br>
            ///     References
            ///     G.<br></br>
            ///     Strang, Linear Algebra and Its Applications, 2nd Ed., Orlando, FL,
            ///     Academic Press, Inc., 1980, Various pp.
            /// </summary>
            /// <param name="a">
            ///     Matrices for which the eigenvalues and right eigenvectors will
            ///     be computed
            /// </param>
            /// <returns>
            ///     A tuple of:
            ///     w
            ///     The eigenvalues, each repeated according to its multiplicity.
            ///     The eigenvalues are not necessarily ordered. The resulting
            ///     array will be of complex type, unless the imaginary part is
            ///     zero in which case it will be cast to a real type. When a
            ///     is real the resulting eigenvalues will be real (0 imaginary
            ///     part) or occur in conjugate pairs
            ///     v
            ///     The normalized (unit “length”) eigenvectors, such that the
            ///     column v[:,i] is the eigenvector corresponding to the
            ///     eigenvalue w[i].
            /// </returns>
            public static (NDarray, NDarray) eig(NDarray a)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    throw new NotSupportedException();
                }
                else
                {
                    var ret = np.linalg.eig(a.NumpyNDarray);
                    return (new NDarray(ret.Item1), new NDarray(ret.Item2));
                }
            }
        }

        public static partial class linalg
        {
            /// <summary>
            ///     Return the eigenvalues and eigenvectors of a complex Hermitian
            ///     (conjugate symmetric) or a real symmetric matrix.<br></br>
            ///     Returns two objects, a 1-D array containing the eigenvalues of a, and
            ///     a 2-D square array or matrix (depending on the input type) of the
            ///     corresponding eigenvectors (in columns).<br></br>
            ///     Notes
            ///     Broadcasting rules apply, see the Cupy.linalg documentation for
            ///     details.<br></br>
            ///     The eigenvalues/eigenvectors are computed using LAPACK routines _syevd,
            ///     _heevd
            ///     The eigenvalues of real symmetric or complex Hermitian matrices are
            ///     always real.<br></br>
            ///     [1] The array v of (column) eigenvectors is unitary
            ///     and a, w, and v satisfy the equations
            ///     dot(a, v[:, i]) = w[i] * v[:, i].<br></br>
            ///     References
            /// </summary>
            /// <param name="a">
            ///     Hermitian or real symmetric matrices whose eigenvalues and
            ///     eigenvectors are to be computed.
            /// </param>
            /// <param name="UPLO">
            ///     Specifies whether the calculation is done with the lower triangular
            ///     part of a (‘L’, default) or the upper triangular part (‘U’).<br></br>
            ///     Irrespective of this value only the real parts of the diagonal will
            ///     be considered in the computation to preserve the notion of a Hermitian
            ///     matrix.<br></br>
            ///     It therefore follows that the imaginary part of the diagonal
            ///     will always be treated as zero.
            /// </param>
            /// <returns>
            ///     A tuple of:
            ///     w
            ///     The eigenvalues in ascending order, each repeated according to
            ///     its multiplicity.
            ///     v
            ///     The column v[:, i] is the normalized eigenvector corresponding
            ///     to the eigenvalue w[i].  Will return a matrix object if a is
            ///     a matrix object.
            /// </returns>
            public static (NDarray, NDarray) eigh(NDarray a, string UPLO = "L")
            {
                if (Gpu.Available && Gpu.Use)
                {
                    var ret = cp.linalg.eigh(a.CupyNDarray, UPLO);
                    return (new NDarray(ret.Item1), new NDarray(ret.Item2));
                }
                else
                {
                    var ret = np.linalg.eigh(a.NumpyNDarray, UPLO);
                    return (new NDarray(ret.Item1), new NDarray(ret.Item2));
                }
            }
        }

        public static partial class linalg
        {
            /// <summary>
            ///     Compute the eigenvalues of a general matrix.<br></br>
            ///     Main difference between eigvals and eig: the eigenvectors aren’t
            ///     returned.<br></br>
            ///     Notes
            ///     Broadcasting rules apply, see the Cupy.linalg documentation for
            ///     details.<br></br>
            ///     This is implemented using the _geev LAPACK routines which compute
            ///     the eigenvalues and eigenvectors of general square arrays.
            /// </summary>
            /// <param name="a">
            ///     A complex- or real-valued matrix whose eigenvalues will be computed.
            /// </param>
            /// <returns>
            ///     The eigenvalues, each repeated according to its multiplicity.<br></br>
            ///     They are not necessarily ordered, nor are they necessarily
            ///     real for real matrices.
            /// </returns>
            public static NDarray eigvals(NDarray a)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(cp.linalg.eigvals(a.CupyNDarray));
                }
                else
                {
                    return new NDarray(np.linalg.eigvals(a.NumpyNDarray));
                }
            }
        }

        public static partial class linalg
        {
            /// <summary>
            ///     Compute the eigenvalues of a complex Hermitian or real symmetric matrix.<br></br>
            ///     Main difference from eigh: the eigenvectors are not computed.<br></br>
            ///     Notes
            ///     Broadcasting rules apply, see the Cupy.linalg documentation for
            ///     details.<br></br>
            ///     The eigenvalues are computed using LAPACK routines _syevd, _heevd
            /// </summary>
            /// <param name="a">
            ///     A complex- or real-valued matrix whose eigenvalues are to be
            ///     computed.
            /// </param>
            /// <param name="UPLO">
            ///     Specifies whether the calculation is done with the lower triangular
            ///     part of a (‘L’, default) or the upper triangular part (‘U’).<br></br>
            ///     Irrespective of this value only the real parts of the diagonal will
            ///     be considered in the computation to preserve the notion of a Hermitian
            ///     matrix.<br></br>
            ///     It therefore follows that the imaginary part of the diagonal
            ///     will always be treated as zero.
            /// </param>
            /// <returns>
            ///     The eigenvalues in ascending order, each repeated according to
            ///     its multiplicity.
            /// </returns>
            public static NDarray eigvalsh(NDarray a, string UPLO = "L")
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(cp.linalg.eigvalsh(a.CupyNDarray, UPLO));
                }
                else
                {
                    return new NDarray(np.linalg.eigvalsh(a.NumpyNDarray, UPLO));
                }
            }
        }

        public static partial class linalg
        {
            /// <summary>
            ///     Compute the (multiplicative) inverse of a matrix.<br></br>
            ///     Given a square matrix a, return the matrix ainv satisfying
            ///     dot(a, ainv) = dot(ainv, a) = eye(a.shape[0]).<br></br>
            ///     Notes
            ///     Broadcasting rules apply, see the Cupy.linalg documentation for
            ///     details.
            /// </summary>
            /// <param name="a">
            ///     Matrix to be inverted.
            /// </param>
            /// <returns>
            ///     (Multiplicative) inverse of the matrix a.
            /// </returns>
            public static NDarray inv(NDarray a)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(cp.linalg.inv(a.CupyNDarray));
                }
                else
                {
                    return new NDarray(np.linalg.inv(a.NumpyNDarray));
                }
            }
        }

        public static partial class linalg
        {
            /// <summary>
            ///     Return the least-squares solution to a linear matrix equation.<br></br>
            ///     Solves the equation a x = b by computing a vector x that
            ///     minimizes the Euclidean 2-norm || b - a x ||^2.  The equation may
            ///     be under-, well-, or over- determined (i.e., the number of
            ///     linearly independent rows of a can be less than, equal to, or
            ///     greater than its number of linearly independent columns).<br></br>
            ///     If a
            ///     is square and of full rank, then x (but for round-off error) is
            ///     the “exact” solution of the equation.<br></br>
            ///     Notes
            ///     If b is a matrix, then all array results are returned as matrices.
            /// </summary>
            /// <param name="a">
            ///     “Coefficient” matrix.
            /// </param>
            /// <param name="b">
            ///     Ordinate or “dependent variable” values.<br></br>
            ///     If b is two-dimensional,
            ///     the least-squares solution is calculated for each of the K columns
            ///     of b.
            /// </param>
            /// <param name="rcond">
            ///     Cut-off ratio for small singular values of a.<br></br>
            ///     For the purposes of rank determination, singular values are treated
            ///     as zero if they are smaller than rcond times the largest singular
            ///     value of a.
            /// </param>
            /// <returns>
            ///     A tuple of:
            ///     x
            ///     Least-squares solution. If b is two-dimensional,
            ///     the solutions are in the K columns of x.
            ///     residuals
            ///     Sums of residuals; squared Euclidean 2-norm for each column in
            ///     b - a*x.
            ///     If the rank of a is &lt; N or M &lt;= N, this is an empty array.
            ///     If b is 1-dimensional, this is a (1,) shape array.
            ///     Otherwise the shape is (K,).
            ///     rank
            ///     Rank of matrix a.
            ///     s
            ///     Singular values of a.
            /// </returns>
            public static (NDarray, NDarray, int, NDarray) lstsq(NDarray a, NDarray b, float? rcond = null)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    var ret = cp.linalg.lstsq(a.CupyNDarray, b.CupyNDarray, rcond);
                    return (new NDarray(ret.Item1), new NDarray(ret.Item2), ret.Item3, new NDarray(ret.Item4));
                }
                else
                {
                    var ret = np.linalg.lstsq(a.NumpyNDarray, b.NumpyNDarray, rcond);
                    return (new NDarray(ret.Item1), new NDarray(ret.Item2), ret.Item3, new NDarray(ret.Item4));
                }
            }
        }

        public static partial class linalg
        {
            /// <summary>
            ///     Compute the (Moore-Penrose) pseudo-inverse of a matrix.<br></br>
            ///     Calculate the generalized inverse of a matrix using its
            ///     singular-value decomposition (SVD) and including all
            ///     large singular values.<br></br>
            ///     Notes
            ///     The pseudo-inverse of a matrix A, denoted , is
            ///     defined as: “the matrix that ‘solves’ [the least-squares problem]
            ///     ,” i.e., if  is said solution, then
            ///     is that matrix such that .
            ///     It can be shown that if  is the singular
            ///     value decomposition of A, then
            ///     , where  are
            ///     orthogonal matrices,  is a diagonal matrix consisting
            ///     of A’s so-called singular values, (followed, typically, by
            ///     zeros), and then  is simply the diagonal matrix
            ///     consisting of the reciprocals of A’s singular values
            ///     (again, followed by zeros).<br></br>
            ///     [1]
            ///     References
            /// </summary>
            /// <param name="a">
            ///     Matrix or stack of matrices to be pseudo-inverted.
            /// </param>
            /// <param name="rcond">
            ///     Cutoff for small singular values.<br></br>
            ///     Singular values smaller (in modulus) than
            ///     rcond * largest_singular_value (again, in modulus)
            ///     are set to zero.<br></br>
            ///     Broadcasts against the stack of matrices
            /// </param>
            /// <returns>
            ///     The pseudo-inverse of a.<br></br>
            ///     If a is a matrix instance, then so
            ///     is B.
            /// </returns>
            public static NDarray pinv(NDarray a, float rcond = 1e-15f)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(cp.linalg.pinv(a.CupyNDarray, rcond));
                }
                else
                {
                    return new NDarray(np.linalg.pinv(a.NumpyNDarray, rcond));
                }
            }
        }

        public static partial class linalg
        {
            /// <summary>
            ///     Solve a linear matrix equation, or system of linear scalar equations.<br></br>
            ///     Computes the “exact” solution, x, of the well-determined, i.e., full
            ///     rank, linear matrix equation ax = b.<br></br>
            ///     Notes
            ///     Broadcasting rules apply, see the Cupy.linalg documentation for
            ///     details.<br></br>
            ///     The solutions are computed using LAPACK routine _gesv
            ///     a must be square and of full-rank, i.e., all rows (or, equivalently,
            ///     columns) must be linearly independent; if either is not true, use
            ///     lstsq for the least-squares best “solution” of the
            ///     system/equation.<br></br>
            ///     References
            /// </summary>
            /// <param name="a">
            ///     Coefficient matrix.
            /// </param>
            /// <param name="b">
            ///     Ordinate or “dependent variable” values.
            /// </param>
            /// <returns>
            ///     Solution to the system a x = b.<br></br>
            ///     Returned shape is identical to b.
            /// </returns>
            public static NDarray solve(NDarray a, NDarray b)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(cp.linalg.solve(a.CupyNDarray, b.CupyNDarray));
                }
                else
                {
                    return new NDarray(np.linalg.solve(a.NumpyNDarray, b.NumpyNDarray));
                }
            }
        }

        public static partial class linalg
        {
            /// <summary>
            ///     Singular Value Decomposition.<br></br>
            ///     When a is a 2D array, it is factorized as u &#64; cp.diag(s) &#64; vh
            ///     = (u * s) &#64; vh, where u and vh are 2D unitary arrays and s is a 1D
            ///     array of a’s singular values.<br></br>
            ///     When a is higher-dimensional, SVD is
            ///     applied in stacked mode as explained below.<br></br>
            ///     Notes
            ///     The decomposition is performed using LAPACK routine _gesdd.<br></br>
            ///     SVD is usually described for the factorization of a 2D matrix .
            ///     The higher-dimensional case will be discussed below.<br></br>
            ///     In the 2D case, SVD is
            ///     written as , where , ,
            ///     and . The 1D array s
            ///     contains the singular values of a and u and vh are unitary.<br></br>
            ///     The rows
            ///     of vh are the eigenvectors of  and the columns of u are
            ///     the eigenvectors of . In both cases the corresponding
            ///     (possibly non-zero) eigenvalues are given by s**2.
            ///     If a has more than two dimensions, then broadcasting rules apply, as
            ///     explained in Linear algebra on several matrices at once.<br></br>
            ///     This means that SVD is
            ///     working in “stacked” mode: it iterates over all indices of the first
            ///     a.ndim - 2 dimensions and for each combination SVD is applied to the
            ///     last two indices.<br></br>
            ///     The matrix a can be reconstructed from the
            ///     decomposition with either (u * s[..., None, :]) &#64; vh or
            ///     u &#64; (s[..., None] * vh).<br></br>
            ///     (The &#64; operator can be replaced by the
            ///     function cp.matmul for python versions below 3.5.)
            ///     If a is a matrix object (as opposed to an ndarray), then so are
            ///     all the return values.
            /// </summary>
            /// <param name="a">
            ///     A real or complex array with a.ndim &gt;= 2.
            /// </param>
            /// <param name="full_matrices">
            ///     If True (default), u and vh have the shapes (..., M, M) and
            ///     (..., N, N), respectively.<br></br>
            ///     Otherwise, the shapes are
            ///     (..., M, K) and (..., K, N), respectively, where
            ///     K = min(M, N).
            /// </param>
            /// <param name="compute_uv">
            ///     Whether or not to compute u and vh in addition to s.<br></br>
            ///     True
            ///     by default.
            /// </param>
            /// <returns>
            ///     A tuple of:
            ///     u
            ///     Unitary array(s). The first a.ndim - 2 dimensions have the same
            ///     size as those of the input a. The size of the last two dimensions
            ///     depends on the value of full_matrices. Only returned when
            ///     compute_uv is True.
            ///     s
            ///     Vector(s) with the singular values, within each vector sorted in
            ///     descending order. The first a.ndim - 2 dimensions have the same
            ///     size as those of the input a.
            ///     vh
            ///     Unitary array(s). The first a.ndim - 2 dimensions have the same
            ///     size as those of the input a. The size of the last two dimensions
            ///     depends on the value of full_matrices. Only returned when
            ///     compute_uv is True.
            /// </returns>
            public static (NDarray, NDarray, NDarray) svd(NDarray a, bool? full_matrices = true,
                bool? compute_uv = true)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    var ret = cp.linalg.svd(a.CupyNDarray, full_matrices, compute_uv);
                    return (new NDarray(ret.Item1), new NDarray(ret.Item2), new NDarray(ret.Item3));
                }
                else
                {
                    var ret = np.linalg.svd(a.NumpyNDarray, full_matrices, compute_uv);
                    return (new NDarray(ret.Item1), new NDarray(ret.Item2), new NDarray(ret.Item3));
                }
            }
        }

        public static partial class fft
        {
            /// <summary>
            ///     Compute the one-dimensional discrete Fourier Transform.<br></br>
            ///     This function computes the one-dimensional n-point discrete Fourier
            ///     Transform (DFT) with the efficient Fast Fourier Transform (FFT)
            ///     algorithm [CT].<br></br>
            ///     Notes
            ///     FFT (Fast Fourier Transform) refers to a way the discrete Fourier
            ///     Transform (DFT) can be calculated efficiently, by using symmetries in the
            ///     calculated terms.<br></br>
            ///     The symmetry is highest when n is a power of 2, and
            ///     the transform is therefore most efficient for these sizes.<br></br>
            ///     The DFT is defined, with the conventions used in this implementation, in
            ///     the documentation for the Cupy.fft module.<br></br>
            ///     References
            /// </summary>
            /// <param name="a">
            ///     Input array, can be complex.
            /// </param>
            /// <param name="n">
            ///     Length of the transformed axis of the output.<br></br>
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
            ///     indicated by axis, or the last one if axis is not specified.
            /// </returns>
            public static NDarray fft_(NDarray a, int? n = null, int? axis = -1, string norm = null)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(cp.fft.fft_(a.CupyNDarray, n, axis, norm));
                }
                else
                {
                    return new NDarray(np.fft.fft_(a.NumpyNDarray, n, axis, norm));
                }
            }
        }

        public static partial class fft
        {
            /// <summary>
            ///     Compute the 2-dimensional discrete Fourier Transform
            ///     This function computes the n-dimensional discrete Fourier Transform
            ///     over any axis in an M-dimensional array by means of the
            ///     Fast Fourier Transform (FFT).<br></br>
            ///     By default, the transform is computed over
            ///     the last two axis of the input array, i.e., a 2-dimensional FFT.<br></br>
            ///     Notes
            ///     fft2 is just fftn with a different default for axis.<br></br>
            ///     The output, analogously to fft, contains the term for zero frequency in
            ///     the low-order corner of the transformed axis, the positive frequency terms
            ///     in the first half of these axis, the term for the Nyquist frequency in the
            ///     middle of the axis and the negative frequency terms in the second half of
            ///     the axis, in order of decreasingly negative frequency.<br></br>
            ///     See fftn for details and a plotting example, and Cupy.fft for
            ///     definitions and conventions used.
            /// </summary>
            /// <param name="a">
            ///     Input array, can be complex
            /// </param>
            /// <param name="s">
            ///     Shape (length of each transformed axis) of the output
            ///     (s[0] refers to axis 0, s[1] to axis 1, etc.).<br></br>
            ///     This corresponds to n for fft(x, n).<br></br>
            ///     Along each axis, if the given shape is smaller than that of the input,
            ///     the input is cropped.<br></br>
            ///     If it is larger, the input is padded with zeros.<br></br>
            ///     if s is not given, the shape of the input along the axis specified
            ///     by axis is used.
            /// </param>
            /// <param name="axes">
            ///     Axes over which to compute the FFT.<br></br>
            ///     If not given, the last two
            ///     axis are used.<br></br>
            ///     A repeated index in axis means the transform over
            ///     that axis is performed multiple times.<br></br>
            ///     A one-element sequence means
            ///     that a one-dimensional FFT is performed.
            /// </param>
            /// <param name="norm">
            ///     Normalization mode (see Cupy.fft).<br></br>
            ///     Default is None.
            /// </param>
            /// <returns>
            ///     The truncated or zero-padded input, transformed along the axis
            ///     indicated by axis, or the last two axis if axis is not given.
            /// </returns>
            public static NDarray fft2(NDarray a, int[] s = null, int[] axes = null, string norm = null)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(cp.fft.fft2(a.CupyNDarray, s, axes, norm));
                }
                else
                {
                    return new NDarray(np.fft.fft2(a.NumpyNDarray, s, axes, norm));
                }
            }
        }

        public static partial class fft
        {
            /// <summary>
            ///     Compute the N-dimensional discrete Fourier Transform.<br></br>
            ///     This function computes the N-dimensional discrete Fourier Transform over
            ///     any number of axis in an M-dimensional array by means of the Fast Fourier
            ///     Transform (FFT).<br></br>
            ///     Notes
            ///     The output, analogously to fft, contains the term for zero frequency in
            ///     the low-order corner of all axis, the positive frequency terms in the
            ///     first half of all axis, the term for the Nyquist frequency in the middle
            ///     of all axis and the negative frequency terms in the second half of all
            ///     axis, in order of decreasingly negative frequency.<br></br>
            ///     See Cupy.fft for details, definitions and conventions used.
            /// </summary>
            /// <param name="a">
            ///     Input array, can be complex.
            /// </param>
            /// <param name="s">
            ///     Shape (length of each transformed axis) of the output
            ///     (s[0] refers to axis 0, s[1] to axis 1, etc.).<br></br>
            ///     This corresponds to n for fft(x, n).<br></br>
            ///     Along any axis, if the given shape is smaller than that of the input,
            ///     the input is cropped.<br></br>
            ///     If it is larger, the input is padded with zeros.<br></br>
            ///     if s is not given, the shape of the input along the axis specified
            ///     by axis is used.
            /// </param>
            /// <param name="axes">
            ///     Axes over which to compute the FFT.<br></br>
            ///     If not given, the last len(s)
            ///     axis are used, or all axis if s is also not specified.<br></br>
            ///     Repeated indices in axis means that the transform over that axis is
            ///     performed multiple times.
            /// </param>
            /// <param name="norm">
            ///     Normalization mode (see Cupy.fft).<br></br>
            ///     Default is None.
            /// </param>
            /// <returns>
            ///     The truncated or zero-padded input, transformed along the axis
            ///     indicated by axis, or by a combination of s and a,
            ///     as explained in the parameters section above.
            /// </returns>
            public static NDarray fftn(NDarray a, int[] s = null, int[] axes = null, string norm = null)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(cp.fft.fftn(a.CupyNDarray, s, axes, norm));
                }
                else
                {
                    return new NDarray(np.fft.fftn(a.NumpyNDarray, s, axes, norm));
                }
            }
        }

        public static partial class fft
        {
            /// <summary>
            ///     Compute the one-dimensional inverse discrete Fourier Transform.<br></br>
            ///     This function computes the inverse of the one-dimensional n-point
            ///     discrete Fourier transform computed by fft.<br></br>
            ///     In other words,
            ///     ifft(fft(a)) == a to within numerical accuracy.<br></br>
            ///     For a general description of the algorithm and definitions,
            ///     see Cupy.fft.<br></br>
            ///     The input should be ordered in the same way as is returned by fft,
            ///     i.e.,
            ///     For an even number of input points, A[n//2] represents the sum of
            ///     the values at the positive and negative Nyquist frequencies, as the two
            ///     are aliased together.<br></br>
            ///     See Cupy.fft for details.<br></br>
            ///     Notes
            ///     If the input parameter n is larger than the size of the input, the input
            ///     is padded by appending zeros at the end.<br></br>
            ///     Even though this is the common
            ///     approach, it might lead to surprising results.<br></br>
            ///     If a different padding is
            ///     desired, it must be performed before calling ifft.
            /// </summary>
            /// <param name="a">
            ///     Input array, can be complex.
            /// </param>
            /// <param name="n">
            ///     Length of the transformed axis of the output.<br></br>
            ///     If n is smaller than the length of the input, the input is cropped.<br></br>
            ///     If it is larger, the input is padded with zeros.<br></br>
            ///     If n is not given,
            ///     the length of the input along the axis specified by axis is used.<br></br>
            ///     See notes about padding issues.
            /// </param>
            /// <param name="axis">
            ///     Axis over which to compute the inverse DFT.<br></br>
            ///     If not given, the last
            ///     axis is used.
            /// </param>
            /// <param name="norm">
            ///     Normalization mode (see Cupy.fft).<br></br>
            ///     Default is None.
            /// </param>
            /// <returns>
            ///     The truncated or zero-padded input, transformed along the axis
            ///     indicated by axis, or the last one if axis is not specified.
            /// </returns>
            public static NDarray ifft(NDarray a, int? n = null, int? axis = -1, string norm = null)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(cp.fft.ifft(a.CupyNDarray, n, axis, norm));
                }
                else
                {
                    return new NDarray(np.fft.ifft(a.NumpyNDarray, n, axis, norm));
                }
            }
        }

        public static partial class fft
        {
            /// <summary>
            ///     Compute the 2-dimensional inverse discrete Fourier Transform.<br></br>
            ///     This function computes the inverse of the 2-dimensional discrete Fourier
            ///     Transform over any number of axis in an M-dimensional array by means of
            ///     the Fast Fourier Transform (FFT).<br></br>
            ///     In other words, ifft2(fft2(a)) == a
            ///     to within numerical accuracy.<br></br>
            ///     By default, the inverse transform is
            ///     computed over the last two axis of the input array.<br></br>
            ///     The input, analogously to ifft, should be ordered in the same way as is
            ///     returned by fft2, i.e.<br></br>
            ///     it should have the term for zero frequency
            ///     in the low-order corner of the two axis, the positive frequency terms in
            ///     the first half of these axis, the term for the Nyquist frequency in the
            ///     middle of the axis and the negative frequency terms in the second half of
            ///     both axis, in order of decreasingly negative frequency.<br></br>
            ///     Notes
            ///     ifft2 is just ifftn with a different default for axis.<br></br>
            ///     See ifftn for details and a plotting example, and Cupy.fft for
            ///     definition and conventions used.<br></br>
            ///     Zero-padding, analogously with ifft, is performed by appending zeros to
            ///     the input along the specified dimension.<br></br>
            ///     Although this is the common
            ///     approach, it might lead to surprising results.<br></br>
            ///     If another form of zero
            ///     padding is desired, it must be performed before ifft2 is called.
            /// </summary>
            /// <param name="a">
            ///     Input array, can be complex.
            /// </param>
            /// <param name="s">
            ///     Shape (length of each axis) of the output (s[0] refers to axis 0,
            ///     s[1] to axis 1, etc.).<br></br>
            ///     This corresponds to n for ifft(x, n).<br></br>
            ///     Along each axis, if the given shape is smaller than that of the input,
            ///     the input is cropped.<br></br>
            ///     If it is larger, the input is padded with zeros.<br></br>
            ///     if s is not given, the shape of the input along the axis specified
            ///     by axis is used.<br></br>
            ///     See notes for issue on ifft zero padding.
            /// </param>
            /// <param name="axes">
            ///     Axes over which to compute the FFT.<br></br>
            ///     If not given, the last two
            ///     axis are used.<br></br>
            ///     A repeated index in axis means the transform over
            ///     that axis is performed multiple times.<br></br>
            ///     A one-element sequence means
            ///     that a one-dimensional FFT is performed.
            /// </param>
            /// <param name="norm">
            ///     Normalization mode (see Cupy.fft).<br></br>
            ///     Default is None.
            /// </param>
            /// <returns>
            ///     The truncated or zero-padded input, transformed along the axis
            ///     indicated by axis, or the last two axis if axis is not given.
            /// </returns>
            public static NDarray ifft2(NDarray a, int[] s = null, int[] axes = null, string norm = null)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(cp.fft.ifft2(a.CupyNDarray, s, axes, norm));
                }
                else
                {
                    return new NDarray(np.fft.ifft2(a.NumpyNDarray, s, axes, norm));
                }
            }
        }

        public static partial class fft
        {
            /// <summary>
            ///     Compute the N-dimensional inverse discrete Fourier Transform.<br></br>
            ///     This function computes the inverse of the N-dimensional discrete
            ///     Fourier Transform over any number of axis in an M-dimensional array by
            ///     means of the Fast Fourier Transform (FFT).<br></br>
            ///     In other words,
            ///     ifftn(fftn(a)) == a to within numerical accuracy.<br></br>
            ///     For a description of the definitions and conventions used, see Cupy.fft.<br></br>
            ///     The input, analogously to ifft, should be ordered in the same way as is
            ///     returned by fftn, i.e.<br></br>
            ///     it should have the term for zero frequency
            ///     in all axis in the low-order corner, the positive frequency terms in the
            ///     first half of all axis, the term for the Nyquist frequency in the middle
            ///     of all axis and the negative frequency terms in the second half of all
            ///     axis, in order of decreasingly negative frequency.<br></br>
            ///     Notes
            ///     See Cupy.fft for definitions and conventions used.<br></br>
            ///     Zero-padding, analogously with ifft, is performed by appending zeros to
            ///     the input along the specified dimension.<br></br>
            ///     Although this is the common
            ///     approach, it might lead to surprising results.<br></br>
            ///     If another form of zero
            ///     padding is desired, it must be performed before ifftn is called.
            /// </summary>
            /// <param name="a">
            ///     Input array, can be complex.
            /// </param>
            /// <param name="s">
            ///     Shape (length of each transformed axis) of the output
            ///     (s[0] refers to axis 0, s[1] to axis 1, etc.).<br></br>
            ///     This corresponds to n for ifft(x, n).<br></br>
            ///     Along any axis, if the given shape is smaller than that of the input,
            ///     the input is cropped.<br></br>
            ///     If it is larger, the input is padded with zeros.<br></br>
            ///     if s is not given, the shape of the input along the axis specified
            ///     by axis is used.<br></br>
            ///     See notes for issue on ifft zero padding.
            /// </param>
            /// <param name="axes">
            ///     Axes over which to compute the IFFT.<br></br>
            ///     If not given, the last len(s)
            ///     axis are used, or all axis if s is also not specified.<br></br>
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
            ///     as explained in the parameters section above.
            /// </returns>
            public static NDarray ifftn(NDarray a, int[] s = null, int[] axes = null, string norm = null)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(cp.fft.ifftn(a.CupyNDarray, s, axes, norm));
                }
                else
                {
                    return new NDarray(np.fft.ifftn(a.NumpyNDarray, s, axes, norm));
                }
            }
        }
    }
}
