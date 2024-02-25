﻿using Cupy;
using Numpy;

namespace DeZero.NET
{
    public static partial class xp
    {
        public static partial class random
        {
            /// <summary>
            ///     Random values in a given shape.<br></br>
            ///     Create an array of the given shape and populate it with
            ///     random samples from a uniform distribution
            ///     over [0, 1).<br></br>
            ///     Notes
            ///     This is a convenience function.<br></br>
            ///     If you want an interface that
            ///     takes a shape-tuple as the first argument, refer to
            ///     cp.random.random_sample .
            /// </summary>
            /// <returns>
            ///     Random values.
            /// </returns>
            public static NDarray rand(params int[] shape)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(cp.random.rand(shape));
                }
                else
                {
                    return new NDarray(np.random.rand(shape));
                }
            }

            /// <summary>
            ///     Return a sample (or samples) from the “standard normal” distribution.<br></br>
            ///     If positive, int_like or int-convertible arguments are provided,
            ///     randn generates an array of shape (d0, d1, ..., dn), filled
            ///     with random floats sampled from a univariate “normal” (Gaussian)
            ///     distribution of mean 0 and variance 1 (if any of the  are
            ///     floats, they are first converted to integers by truncation).<br></br>
            ///     A single
            ///     float randomly sampled from the distribution is returned if no
            ///     argument is provided.<br></br>
            ///     This is a convenience function.<br></br>
            ///     If you want an interface that takes a
            ///     tuple as the first argument, use Cupy.random.standard_normal instead.<br></br>
            ///     Notes
            ///     For random samples from , use:
            ///     sigma * cp.random.randn(...) + mu
            /// </summary>
            /// <returns>
            ///     A (d0, d1, ..., dn)-shaped array of floating-point samples from
            ///     the standard normal distribution, or a single such float if
            ///     no parameters were supplied.
            /// </returns>
            public static NDarray randn(params int[] shape)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(cp.random.randn(shape));
                }
                else
                {
                    return new NDarray(np.random.randn(shape));
                }
            }

            /// <summary>
            ///     Draw random samples from a normal (Gaussian) distribution.<br></br>
            ///     The probability density function of the normal distribution, first
            ///     derived by De Moivre and 200 years later by both Gauss and Laplace
            ///     independently [2], is often called the bell curve because of
            ///     its characteristic shape (see the example below).<br></br>
            ///     The normal distributions occurs often in nature.<br></br>
            ///     For example, it
            ///     describes the commonly occurring distribution of samples influenced
            ///     by a large number of tiny, random disturbances, each with its own
            ///     unique distribution [2].<br></br>
            ///     Notes
            ///     The probability density for the Gaussian distribution is
            ///     where  is the mean and  the standard
            ///     deviation.<br></br>
            ///     The square of the standard deviation, ,
            ///     is called the variance.<br></br>
            ///     The function has its peak at the mean, and its “spread” increases with
            ///     the standard deviation (the function reaches 0.607 times its maximum at
            ///     and  [2]).<br></br>
            ///     This implies that
            ///     Cupy.random.normal is more likely to return samples lying close to
            ///     the mean, rather than those far away.<br></br>
            ///     References
            /// </summary>
            /// <param name="loc">
            ///     Mean (“centre”) of the distribution.
            /// </param>
            /// <param name="scale">
            ///     Standard deviation (spread or “width”) of the distribution.
            /// </param>
            /// <param name="size">
            ///     Output shape.<br></br>
            ///     If the given shape is, e.g., (m, n, k), then
            ///     m * n * k samples are drawn.<br></br>
            ///     If size is None (default),
            ///     a single value is returned if loc and scale are both scalars.<br></br>
            ///     Otherwise, cp.broadcast(loc, scale).size samples are drawn.
            /// </param>
            /// <returns>
            ///     Drawn samples from the parameterized normal distribution.
            /// </returns>
            public static NDarray normal(NDarray<float> loc, NDarray<float> scale = null, int[] size = null)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(cp.random.normal(loc.CupyNDarray, scale.CupyNDarray, size));
                }
                else
                {
                    return new NDarray(np.random.normal(loc.NumpyNDarray, scale.NumpyNDarray, size));
                }
            }

            public static NDarray normal(float? loc = null, float? scale = null, int[] size = null)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(cp.random.normal(loc, scale, size));
                }
                else
                {
                    return new NDarray(np.random.normal(loc, scale, size));
                }
            }

            public static NDarray normal(float loc)
            {
                return normal(loc, null);
            }

            public static NDarray normal(float loc, float scale)
            {
                return normal(loc, scale, null);
            }

            public static NDarray normal(float loc, float scale, int size)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(cp.random.normal(loc, scale, size));
                }
                else
                {
                    return new NDarray(np.random.normal(loc, scale, size));
                }
            }
        }

        public static partial class random
        {
            /// <summary>
            ///     Random values in a given shape.<br></br>
            ///     Create an array of the given shape and populate it with
            ///     random samples from a uniform distribution
            ///     over [0, 1).<br></br>
            ///     Notes
            ///     This is a convenience function.<br></br>
            ///     If you want an interface that
            ///     takes a shape-tuple as the first argument, refer to
            ///     cp.random.random_sample .
            /// </summary>
            /// <returns>
            ///     Random values.
            /// </returns>
            public static float rand()
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return cp.random.rand();
                }
                else
                {
                    return np.random.rand();
                }
            }
        }

        public static partial class random
        {
            /// <summary>
            ///     Return a sample (or samples) from the “standard normal” distribution.<br></br>
            ///     If positive, int_like or int-convertible arguments are provided,
            ///     randn generates an array of shape (d0, d1, ..., dn), filled
            ///     with random floats sampled from a univariate “normal” (Gaussian)
            ///     distribution of mean 0 and variance 1 (if any of the  are
            ///     floats, they are first converted to integers by truncation).<br></br>
            ///     A single
            ///     float randomly sampled from the distribution is returned if no
            ///     argument is provided.<br></br>
            ///     This is a convenience function.<br></br>
            ///     If you want an interface that takes a
            ///     tuple as the first argument, use Cupy.random.standard_normal instead.<br></br>
            ///     Notes
            ///     For random samples from , use:
            ///     sigma * cp.random.randn(...) + mu
            /// </summary>
            /// <returns>
            ///     A (d0, d1, ..., dn)-shaped array of floating-point samples from
            ///     the standard normal distribution, or a single such float if
            ///     no parameters were supplied.
            /// </returns>
            public static float randn()
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return cp.random.randn();
                }
                else
                {
                    return np.random.randn();
                }
            }
        }

        public static partial class random
        {
            /// <summary>
            ///     Return random integers from low (inclusive) to high (exclusive).<br></br>
            ///     Return random integers from the “discrete uniform” distribution of
            ///     the specified dtype in the “half-open” interval [low, high).<br></br>
            ///     If
            ///     high is None (the default), then results are from [0, low).
            /// </summary>
            /// <param name="low">
            ///     Lowest (signed) integer to be drawn from the distribution (unless
            ///     high=None, in which case this parameter is one above the
            ///     highest such integer).
            /// </param>
            /// <param name="high">
            ///     If provided, one above the largest (signed) integer to be drawn
            ///     from the distribution (see above for behavior if high=None).
            /// </param>
            /// <param name="size">
            ///     Output shape.<br></br>
            ///     If the given shape is, e.g., (m, n, k), then
            ///     m * n * k samples are drawn.<br></br>
            ///     Default is None, in which case a
            ///     single value is returned.
            /// </param>
            /// <param name="dtype">
            ///     Desired dtype of the result.<br></br>
            ///     All dtypes are determined by their
            ///     name, i.e., ‘int64’, ‘int’, etc, so byteorder is not available
            ///     and a specific precision may have different C types depending
            ///     on the platform.<br></br>
            ///     The default value is ‘cp.int’.
            /// </param>
            /// <returns>
            ///     size-shaped array of random integers from the appropriate
            ///     distribution, or a single such random int if size not provided.
            /// </returns>
            public static NDarray<int> randint(int low, int? high = null, int[] size = null, Dtype dtype = null)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray<int>(cp.random.randint(low, high, size, dtype?.CupyDtype));
                }
                else
                {
                    return new NDarray<int>(np.random.randint(low, high, size, dtype?.NumpyDtype));
                }
            }
        }

        public static partial class random
        {
            /// <summary>
            ///     Random integers of type cp.int between low and high, inclusive.<br></br>
            ///     Return random integers of type cp.int from the “discrete uniform”
            ///     distribution in the closed interval [low, high].<br></br>
            ///     If high is
            ///     None (the default), then results are from [1, low].<br></br>
            ///     The cp.int
            ///     type translates to the C long type used by Python 2 for “short”
            ///     integers and its precision is platform dependent.<br></br>
            ///     This function has been deprecated.<br></br>
            ///     Use randint instead.<br></br>
            ///     Notes
            ///     To sample from N evenly spaced floating-point numbers between a and b,
            ///     use:
            /// </summary>
            /// <param name="low">
            ///     Lowest (signed) integer to be drawn from the distribution (unless
            ///     high=None, in which case this parameter is the highest such
            ///     integer).
            /// </param>
            /// <param name="high">
            ///     If provided, the largest (signed) integer to be drawn from the
            ///     distribution (see above for behavior if high=None).
            /// </param>
            /// <param name="size">
            ///     Output shape.<br></br>
            ///     If the given shape is, e.g., (m, n, k), then
            ///     m * n * k samples are drawn.<br></br>
            ///     Default is None, in which case a
            ///     single value is returned.
            /// </param>
            /// <returns>
            ///     size-shaped array of random integers from the appropriate
            ///     distribution, or a single such random int if size not provided.
            /// </returns>
            public static NDarray<int> random_integers(int low, int? high = null, int[] size = null)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray<int>(cp.random.random_integers(low, high, size));
                }
                else
                {
                    return new NDarray<int>(np.random.random_integers(low, high, size));
                }
            }
        }

        public static partial class random
        {
            /// <summary>
            ///     Return random floats in the half-open interval [0.0, 1.0).<br></br>
            ///     Results are from the “continuous uniform” distribution over the
            ///     stated interval.<br></br>
            ///     To sample  multiply
            ///     the output of random_sample by (b-a) and add a:
            /// </summary>
            /// <param name="size">
            ///     Output shape.<br></br>
            ///     If the given shape is, e.g., (m, n, k), then
            ///     m * n * k samples are drawn.<br></br>
            ///     Default is None, in which case a
            ///     single value is returned.
            /// </param>
            /// <returns>
            ///     Array of random floats of shape size (unless size=None, in which
            ///     case a single float is returned).
            /// </returns>
            public static NDarray<float> random_sample(params int[] size)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray<float>(cp.random.random_sample(size));
                }
                else
                {
                    return new NDarray<float>(np.random.random_sample(size));
                }
            }
        }

        public static partial class random
        {
            /// <summary>
            ///     Return random floats in the half-open interval [0.0, 1.0).<br></br>
            ///     Results are from the “continuous uniform” distribution over the
            ///     stated interval.<br></br>
            ///     To sample  multiply
            ///     the output of random_sample by (b-a) and add a:
            /// </summary>
            /// <param name="size">
            ///     Output shape.<br></br>
            ///     If the given shape is, e.g., (m, n, k), then
            ///     m * n * k samples are drawn.<br></br>
            ///     Default is None, in which case a
            ///     single value is returned.
            /// </param>
            /// <returns>
            ///     Array of random floats of shape size (unless size=None, in which
            ///     case a single float is returned).
            /// </returns>
            public static NDarray<float> random_(params int[] size)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray<float>(cp.random.random_(size));
                }
                else
                {
                    return new NDarray<float>(np.random.random_(size));
                }
            }
        }

        public static partial class random
        {
            /// <summary>
            ///     Return random floats in the half-open interval [0.0, 1.0).<br></br>
            ///     Results are from the “continuous uniform” distribution over the
            ///     stated interval.<br></br>
            ///     To sample  multiply
            ///     the output of random_sample by (b-a) and add a:
            /// </summary>
            /// <param name="size">
            ///     Output shape.<br></br>
            ///     If the given shape is, e.g., (m, n, k), then
            ///     m * n * k samples are drawn.<br></br>
            ///     Default is None, in which case a
            ///     single value is returned.
            /// </param>
            /// <returns>
            ///     Array of random floats of shape size (unless size=None, in which
            ///     case a single float is returned).
            /// </returns>
            public static NDarray<float> ranf(params int[] size)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray<float>(cp.random.ranf(size));
                }
                else
                {
                    return new NDarray<float>(np.random.ranf(size));
                }
            }
        }

        public static partial class random
        {
            /// <summary>
            ///     Return random floats in the half-open interval [0.0, 1.0).<br></br>
            ///     Results are from the “continuous uniform” distribution over the
            ///     stated interval.<br></br>
            ///     To sample  multiply
            ///     the output of random_sample by (b-a) and add a:
            /// </summary>
            /// <param name="size">
            ///     Output shape.<br></br>
            ///     If the given shape is, e.g., (m, n, k), then
            ///     m * n * k samples are drawn.<br></br>
            ///     Default is None, in which case a
            ///     single value is returned.
            /// </param>
            /// <returns>
            ///     Array of random floats of shape size (unless size=None, in which
            ///     case a single float is returned).
            /// </returns>
            public static NDarray<float> sample(params int[] size)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray<float>(cp.random.sample(size));
                }
                else
                {
                    return new NDarray<float>(np.random.sample(size));
                }
            }
        }

        public static partial class random
        {
            /// <summary>
            ///     Generates a random sample from a given 1-D array
            /// </summary>
            /// <param name="a">
            ///     If an ndarray, a random sample is generated from its elements.<br></br>
            ///     If an int, the random sample is generated as if a were cp.arange(a)
            /// </param>
            /// <param name="size">
            ///     Output shape.<br></br>
            ///     If the given shape is, e.g., (m, n, k), then
            ///     m * n * k samples are drawn.<br></br>
            ///     Default is None, in which case a
            ///     single value is returned.
            /// </param>
            /// <param name="replace">
            ///     Whether the sample is with or without replacement
            /// </param>
            /// <param name="p">
            ///     The probabilities associated with each entry in a.<br></br>
            ///     If not given the sample assumes a uniform distribution over all
            ///     entries in a.
            /// </param>
            /// <returns>
            ///     The generated random samples
            /// </returns>
            public static NDarray choice(NDarray a, int[] size = null, bool? replace = true, NDarray p = null)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(cp.random.choice(a.CupyNDarray, size, replace, p?.CupyNDarray));
                }
                else
                {
                    return new NDarray(np.random.choice(a.NumpyNDarray, size, replace, p?.NumpyNDarray));
                }
            }
        }

        public static partial class random
        {
            /// <summary>
            ///     Generates a random sample from a given 1-D array
            /// </summary>
            /// <param name="a">
            ///     If an ndarray, a random sample is generated from its elements.<br></br>
            ///     If an int, the random sample is generated as if a were cp.arange(a)
            /// </param>
            /// <param name="size">
            ///     Output shape.<br></br>
            ///     If the given shape is, e.g., (m, n, k), then
            ///     m * n * k samples are drawn.<br></br>
            ///     Default is None, in which case a
            ///     single value is returned.
            /// </param>
            /// <param name="replace">
            ///     Whether the sample is with or without replacement
            /// </param>
            /// <param name="p">
            ///     The probabilities associated with each entry in a.<br></br>
            ///     If not given the sample assumes a uniform distribution over all
            ///     entries in a.
            /// </param>
            /// <returns>
            ///     The generated random samples
            /// </returns>
            public static NDarray choice(int a, int[] size = null, bool? replace = true, NDarray p = null)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(cp.random.choice(a, size, replace, p?.CupyNDarray));
                }
                else
                {
                    return new NDarray(np.random.choice(a, size, replace, p?.NumpyNDarray));
                }
            }
        }

        public static partial class random
        {
            /// <summary>
            ///     Return random bytes.
            /// </summary>
            /// <param name="length">
            ///     Number of random bytes.
            /// </param>
            /// <returns>
            ///     String of length length.
            /// </returns>
            public static string bytes(int length)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return cp.random.bytes(length);
                }
                else
                {
                    return np.random.bytes(length);
                }
            }
        }

        public static partial class random
        {
            /// <summary>
            ///     Modify a sequence in-place by shuffling its contents.<br></br>
            ///     This function only shuffles the array along the first axis of a
            ///     multi-dimensional array.<br></br>
            ///     The order of sub-arrays is changed but
            ///     their contents remains the same.
            /// </summary>
            /// <param name="x">
            ///     The array or list to be shuffled.
            /// </param>
            public static void shuffle(NDarray x)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    cp.random.shuffle(x.CupyNDarray);
                }
                else
                {
                    np.random.shuffle(x.NumpyNDarray);
                }
            }
        }

        public static partial class random
        {
            /// <summary>
            ///     Randomly permute a sequence, or return a permuted range.<br></br>
            ///     If x is a multi-dimensional array, it is only shuffled along its
            ///     first index.
            /// </summary>
            /// <param name="x">
            ///     If x is an integer, randomly permute cp.arange(x).<br></br>
            ///     If x is an array, make a copy and shuffle the elements
            ///     randomly.
            /// </param>
            /// <returns>
            ///     Permuted sequence or array range.
            /// </returns>
            public static NDarray permutation(NDarray x)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(cp.random.permutation(x.CupyNDarray));
                }
                else
                {
                    return new NDarray(np.random.permutation(x.NumpyNDarray));
                }
            }
        }

        public static partial class random
        {
            /// <summary>
            ///     Randomly permute a sequence, or return a permuted range.<br></br>
            ///     If x is a multi-dimensional array, it is only shuffled along its
            ///     first index.
            /// </summary>
            /// <param name="x">
            ///     If x is an integer, randomly permute cp.arange(x).<br></br>
            ///     If x is an array, make a copy and shuffle the elements
            ///     randomly.
            /// </param>
            /// <returns>
            ///     Permuted sequence or array range.
            /// </returns>
            public static NDarray permutation(int x)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(cp.random.permutation(x));
                }
                else
                {
                    return new NDarray(np.random.permutation(x));
                }
            }
        }

        public static partial class random
        {
            /// <summary>
            ///     Draw samples from a Beta distribution.<br></br>
            ///     The Beta distribution is a special case of the Dirichlet distribution,
            ///     and is related to the Gamma distribution.<br></br>
            ///     It has the probability
            ///     distribution function
            ///     where the normalisation, B, is the beta function,
            ///     It is often seen in Bayesian inference and order statistics.
            /// </summary>
            /// <param name="a">
            ///     Alpha, positive (&gt;0).
            /// </param>
            /// <param name="b">
            ///     Beta, positive (&gt;0).
            /// </param>
            /// <param name="size">
            ///     Output shape.<br></br>
            ///     If the given shape is, e.g., (m, n, k), then
            ///     m * n * k samples are drawn.<br></br>
            ///     If size is None (default),
            ///     a single value is returned if a and b are both scalars.<br></br>
            ///     Otherwise, cp.broadcast(a, b).size samples are drawn.
            /// </param>
            /// <returns>
            ///     Drawn samples from the parameterized beta distribution.
            /// </returns>
            public static NDarray beta(NDarray<float> a, NDarray<float> b, int[] size = null)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(cp.random.beta(a.CupyNDarray, b.CupyNDarray, size));
                }
                else
                {
                    return new NDarray(np.random.beta(a.NumpyNDarray, b.NumpyNDarray, size));
                }
            }
        }

        public static partial class random
        {
            /// <summary>
            ///     Draw samples from a binomial distribution.<br></br>
            ///     Samples are drawn from a binomial distribution with specified
            ///     parameters, n trials and p probability of success where
            ///     n an integer &gt;= 0 and p is in the interval [0,1].<br></br>
            ///     (n may be
            ///     input as a float, but it is truncated to an integer in use)
            ///     Notes
            ///     The probability density for the binomial distribution is
            ///     where  is the number of trials,  is the probability
            ///     of success, and  is the number of successes.<br></br>
            ///     When estimating the standard error of a proportion in a population by
            ///     using a random sample, the normal distribution works well unless the
            ///     product p*n &lt;=5, where p = population proportion estimate, and n =
            ///     number of samples, in which case the binomial distribution is used
            ///     instead.<br></br>
            ///     For example, a sample of 15 people shows 4 who are left
            ///     handed, and 11 who are right handed.<br></br>
            ///     Then p = 4/15 = 27%. 0.27*15 = 4,
            ///     so the binomial distribution should be used in this case.<br></br>
            ///     References
            /// </summary>
            /// <param name="n">
            ///     Parameter of the distribution, &gt;= 0.<br></br>
            ///     Floats are also accepted,
            ///     but they will be truncated to integers.
            /// </param>
            /// <param name="p">
            ///     Parameter of the distribution, &gt;= 0 and &lt;=1.
            /// </param>
            /// <param name="size">
            ///     Output shape.<br></br>
            ///     If the given shape is, e.g., (m, n, k), then
            ///     m * n * k samples are drawn.<br></br>
            ///     If size is None (default),
            ///     a single value is returned if n and p are both scalars.<br></br>
            ///     Otherwise, cp.broadcast(n, p).size samples are drawn.
            /// </param>
            /// <returns>
            ///     Drawn samples from the parameterized binomial distribution, where
            ///     each sample is equal to the number of successes over the n trials.
            /// </returns>
            public static NDarray binomial(NDarray<int> n, NDarray<float> p, int[] size = null)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(cp.random.binomial(n.CupyNDarray, p.CupyNDarray, size));
                }
                else
                {
                    return new NDarray(np.random.binomial(n.NumpyNDarray, p.NumpyNDarray, size));
                }
            }
        }

        public static partial class random
        {
            /// <summary>
            ///     Draw samples from a binomial distribution.<br></br>
            ///     Samples are drawn from a binomial distribution with specified
            ///     parameters, n trials and p probability of success where
            ///     n an integer &gt;= 0 and p is in the interval [0,1].<br></br>
            ///     (n may be
            ///     input as a float, but it is truncated to an integer in use)
            ///     Notes
            ///     The probability density for the binomial distribution is
            ///     where  is the number of trials,  is the probability
            ///     of success, and  is the number of successes.<br></br>
            ///     When estimating the standard error of a proportion in a population by
            ///     using a random sample, the normal distribution works well unless the
            ///     product p*n &lt;=5, where p = population proportion estimate, and n =
            ///     number of samples, in which case the binomial distribution is used
            ///     instead.<br></br>
            ///     For example, a sample of 15 people shows 4 who are left
            ///     handed, and 11 who are right handed.<br></br>
            ///     Then p = 4/15 = 27%. 0.27*15 = 4,
            ///     so the binomial distribution should be used in this case.<br></br>
            ///     References
            /// </summary>
            /// <param name="n">
            ///     Parameter of the distribution, &gt;= 0.<br></br>
            ///     Floats are also accepted,
            ///     but they will be truncated to integers.
            /// </param>
            /// <param name="p">
            ///     Parameter of the distribution, &gt;= 0 and &lt;=1.
            /// </param>
            /// <param name="size">
            ///     Output shape.<br></br>
            ///     If the given shape is, e.g., (m, n, k), then
            ///     m * n * k samples are drawn.<br></br>
            ///     If size is None (default),
            ///     a single value is returned if n and p are both scalars.<br></br>
            ///     Otherwise, cp.broadcast(n, p).size samples are drawn.
            /// </param>
            /// <returns>
            ///     Drawn samples from the parameterized binomial distribution, where
            ///     each sample is equal to the number of successes over the n trials.
            /// </returns>
            public static NDarray binomial(int n, NDarray<float> p, int[] size = null)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(cp.random.binomial(n, p.CupyNDarray, size));
                }
                else
                {
                    return new NDarray(np.random.binomial(n, p.NumpyNDarray, size));
                }
            }
        }

        public static partial class random
        {
            /// <summary>
            ///     Draw samples from a chi-square distribution.<br></br>
            ///     When df independent random variables, each with standard normal
            ///     distributions (mean 0, variance 1), are squared and summed, the
            ///     resulting distribution is chi-square (see Notes).<br></br>
            ///     This distribution
            ///     is often used in hypothesis testing.<br></br>
            ///     Notes
            ///     The variable obtained by summing the squares of df independent,
            ///     standard normally distributed random variables:
            ///     is chi-square distributed, denoted
            ///     The probability density function of the chi-squared distribution is
            ///     where  is the gamma function,
            ///     References
            /// </summary>
            /// <param name="df">
            ///     Number of degrees of freedom, should be &gt; 0.
            /// </param>
            /// <param name="size">
            ///     Output shape.<br></br>
            ///     If the given shape is, e.g., (m, n, k), then
            ///     m * n * k samples are drawn.<br></br>
            ///     If size is None (default),
            ///     a single value is returned if df is a scalar.<br></br>
            ///     Otherwise,
            ///     cp.array(df).size samples are drawn.
            /// </param>
            /// <returns>
            ///     Drawn samples from the parameterized chi-square distribution.
            /// </returns>
            public static NDarray chisquare(NDarray<float> df, int[] size = null)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(cp.random.chisquare(df.CupyNDarray, size));
                }
                else
                {
                    return new NDarray(np.random.chisquare(df.NumpyNDarray, size));
                }
            }
        }

        public static partial class random
        {
            /// <summary>
            ///     Draw samples from the Dirichlet distribution.<br></br>
            ///     Draw size samples of dimension k from a Dirichlet distribution.<br></br>
            ///     A
            ///     Dirichlet-distributed random variable can be seen as a multivariate
            ///     generalization of a Beta distribution.<br></br>
            ///     Dirichlet pdf is the conjugate
            ///     prior of a multinomial in Bayesian inference.<br></br>
            ///     Notes
            ///     Uses the following property for computation: for each dimension,
            ///     draw a random sample y_i from a standard gamma generator of shape
            ///     alpha_i, then
            ///     is
            ///     Dirichlet distributed.<br></br>
            ///     References
            /// </summary>
            /// <param name="alpha">
            ///     Parameter of the distribution (k dimension for sample of
            ///     dimension k).
            /// </param>
            /// <param name="size">
            ///     Output shape.<br></br>
            ///     If the given shape is, e.g., (m, n, k), then
            ///     m * n * k samples are drawn.<br></br>
            ///     Default is None, in which case a
            ///     single value is returned.
            /// </param>
            /// <returns>
            ///     The drawn samples, of shape (size, alpha.ndim).
            /// </returns>
            public static NDarray dirichlet(NDarray alpha, int[] size = null)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(cp.random.dirichlet(alpha.CupyNDarray, size));
                }
                else
                {
                    return new NDarray(np.random.dirichlet(alpha.NumpyNDarray, size));
                }
            }
        }

        public static partial class random
        {
            /// <summary>
            ///     Draw samples from an exponential distribution.<br></br>
            ///     Its probability density function is
            ///     for x &gt; 0 and 0 elsewhere.<br></br>
            ///     is the scale parameter,
            ///     which is the inverse of the rate parameter .
            ///     The rate parameter is an alternative, widely used parameterization
            ///     of the exponential distribution [3].<br></br>
            ///     The exponential distribution is a continuous analogue of the
            ///     geometric distribution.<br></br>
            ///     It describes many common situations, such as
            ///     the size of raindrops measured over many rainstorms [1], or the time
            ///     between page requests to Wikipedia [2].<br></br>
            ///     References
            /// </summary>
            /// <param name="scale">
            ///     The scale parameter, .
            /// </param>
            /// <param name="size">
            ///     Output shape.<br></br>
            ///     If the given shape is, e.g., (m, n, k), then
            ///     m * n * k samples are drawn.<br></br>
            ///     If size is None (default),
            ///     a single value is returned if scale is a scalar.<br></br>
            ///     Otherwise,
            ///     cp.array(scale).size samples are drawn.
            /// </param>
            /// <returns>
            ///     Drawn samples from the parameterized exponential distribution.
            /// </returns>
            public static NDarray exponential(NDarray<float> scale = null, int[] size = null)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(cp.random.exponential(scale?.CupyNDarray, size));
                }
                else
                {
                    return new NDarray(np.random.exponential(scale?.NumpyNDarray, size));
                }
            }
        }

        public static partial class random
        {
            /// <summary>
            ///     Draw samples from an F distribution.<br></br>
            ///     Samples are drawn from an F distribution with specified parameters,
            ///     dfnum (degrees of freedom in numerator) and dfden (degrees of
            ///     freedom in denominator), where both parameters should be greater than
            ///     zero.<br></br>
            ///     The random variate of the F distribution (also known as the
            ///     Fisher distribution) is a continuous probability distribution
            ///     that arises in ANOVA tests, and is the ratio of two chi-square
            ///     variates.<br></br>
            ///     Notes
            ///     The F statistic is used to compare in-group variances to between-group
            ///     variances.<br></br>
            ///     Calculating the distribution depends on the sampling, and
            ///     so it is a function of the respective degrees of freedom in the
            ///     problem.<br></br>
            ///     The variable dfnum is the number of samples minus one, the
            ///     between-groups degrees of freedom, while dfden is the within-groups
            ///     degrees of freedom, the sum of the number of samples in each group
            ///     minus the number of groups.<br></br>
            ///     References
            /// </summary>
            /// <param name="dfnum">
            ///     Degrees of freedom in numerator, should be &gt; 0.
            /// </param>
            /// <param name="dfden">
            ///     Degrees of freedom in denominator, should be &gt; 0.
            /// </param>
            /// <param name="size">
            ///     Output shape.<br></br>
            ///     If the given shape is, e.g., (m, n, k), then
            ///     m * n * k samples are drawn.<br></br>
            ///     If size is None (default),
            ///     a single value is returned if dfnum and dfden are both scalars.<br></br>
            ///     Otherwise, cp.broadcast(dfnum, dfden).size samples are drawn.
            /// </param>
            /// <returns>
            ///     Drawn samples from the parameterized Fisher distribution.
            /// </returns>
            public static NDarray f(NDarray<float> dfnum, NDarray<float> dfden, int[] size = null)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(cp.random.f(dfnum.CupyNDarray, dfden.CupyNDarray, size));
                }
                else
                {
                    return new NDarray(np.random.f(dfnum.NumpyNDarray, dfden.NumpyNDarray, size));
                }
            }
        }

        public static partial class random
        {
            /// <summary>
            ///     Draw samples from a Gamma distribution.<br></br>
            ///     Samples are drawn from a Gamma distribution with specified parameters,
            ///     shape (sometimes designated “k”) and scale (sometimes designated
            ///     “theta”), where both parameters are &gt; 0.<br></br>
            ///     Notes
            ///     The probability density for the Gamma distribution is
            ///     where  is the shape and  the scale,
            ///     and  is the Gamma function.<br></br>
            ///     The Gamma distribution is often used to model the times to failure of
            ///     electronic components, and arises naturally in processes for which the
            ///     waiting times between Poisson distributed events are relevant.<br></br>
            ///     References
            /// </summary>
            /// <param name="shape">
            ///     The shape of the gamma distribution.<br></br>
            ///     Should be greater than zero.
            /// </param>
            /// <param name="scale">
            ///     The scale of the gamma distribution.<br></br>
            ///     Should be greater than zero.<br></br>
            ///     Default is equal to 1.
            /// </param>
            /// <param name="size">
            ///     Output shape.<br></br>
            ///     If the given shape is, e.g., (m, n, k), then
            ///     m * n * k samples are drawn.<br></br>
            ///     If size is None (default),
            ///     a single value is returned if shape and scale are both scalars.<br></br>
            ///     Otherwise, cp.broadcast(shape, scale).size samples are drawn.
            /// </param>
            /// <returns>
            ///     Drawn samples from the parameterized gamma distribution.
            /// </returns>
            public static NDarray gamma(Shape shape, NDarray<float> scale = null, int[] size = null)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(cp.random.gamma(shape.CupyShape, scale?.CupyNDarray, size));
                }
                else
                {
                    return new NDarray(np.random.gamma(shape.NumpyShape, scale?.NumpyNDarray, size));
                }
            }
        }

        public static partial class random
        {
            /// <summary>
            ///     Draw samples from the geometric distribution.<br></br>
            ///     Bernoulli trials are experiments with one of two outcomes:
            ///     success or failure (an example of such an experiment is flipping
            ///     a coin).<br></br>
            ///     The geometric distribution models the number of trials
            ///     that must be run in order to achieve success.<br></br>
            ///     It is therefore
            ///     supported on the positive integers, k = 1, 2, ....
            ///     The probability mass function of the geometric distribution is
            ///     where p is the probability of success of an individual trial.
            /// </summary>
            /// <param name="p">
            ///     The probability of success of an individual trial.
            /// </param>
            /// <param name="size">
            ///     Output shape.<br></br>
            ///     If the given shape is, e.g., (m, n, k), then
            ///     m * n * k samples are drawn.<br></br>
            ///     If size is None (default),
            ///     a single value is returned if p is a scalar.<br></br>
            ///     Otherwise,
            ///     cp.array(p).size samples are drawn.
            /// </param>
            /// <returns>
            ///     Drawn samples from the parameterized geometric distribution.
            /// </returns>
            public static NDarray geometric(NDarray<float> p, int[] size = null)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(cp.random.geometric(p.CupyNDarray, size));
                }
                else
                {
                    return new NDarray(np.random.geometric(p.NumpyNDarray, size));
                }
            }
        }

        public static partial class random
        {
            /// <summary>
            ///     Draw samples from a Gumbel distribution.<br></br>
            ///     Draw samples from a Gumbel distribution with specified location and
            ///     scale.<br></br>
            ///     For more information on the Gumbel distribution, see
            ///     Notes and References below.<br></br>
            ///     Notes
            ///     The Gumbel (or Smallest Extreme Value (SEV) or the Smallest Extreme
            ///     Value Type I) distribution is one of a class of Generalized Extreme
            ///     Value (GEV) distributions used in modeling extreme value problems.<br></br>
            ///     The Gumbel is a special case of the Extreme Value Type I distribution
            ///     for maximums from distributions with “exponential-like” tails.<br></br>
            ///     The probability density for the Gumbel distribution is
            ///     where  is the mode, a location parameter, and
            ///     is the scale parameter.<br></br>
            ///     The Gumbel (named for German mathematician Emil Julius Gumbel) was used
            ///     very early in the hydrology literature, for modeling the occurrence of
            ///     flood events.<br></br>
            ///     It is also used for modeling maximum wind speed and
            ///     rainfall rates.<br></br>
            ///     It is a “fat-tailed” distribution - the probability of
            ///     an event in the tail of the distribution is larger than if one used a
            ///     Gaussian, hence the surprisingly frequent occurrence of 100-year
            ///     floods.<br></br>
            ///     Floods were initially modeled as a Gaussian process, which
            ///     underestimated the frequency of extreme events.<br></br>
            ///     It is one of a class of extreme value distributions, the Generalized
            ///     Extreme Value (GEV) distributions, which also includes the Weibull and
            ///     Frechet.<br></br>
            ///     The function has a mean of  and a variance
            ///     of .
            ///     References
            /// </summary>
            /// <param name="loc">
            ///     The location of the mode of the distribution.<br></br>
            ///     Default is 0.
            /// </param>
            /// <param name="scale">
            ///     The scale parameter of the distribution.<br></br>
            ///     Default is 1.
            /// </param>
            /// <param name="size">
            ///     Output shape.<br></br>
            ///     If the given shape is, e.g., (m, n, k), then
            ///     m * n * k samples are drawn.<br></br>
            ///     If size is None (default),
            ///     a single value is returned if loc and scale are both scalars.<br></br>
            ///     Otherwise, cp.broadcast(loc, scale).size samples are drawn.
            /// </param>
            /// <returns>
            ///     Drawn samples from the parameterized Gumbel distribution.
            /// </returns>
            public static NDarray gumbel(NDarray<float> loc = null, NDarray<float> scale = null, int[] size = null)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(cp.random.gumbel(loc?.CupyNDarray, scale?.CupyNDarray, size));
                }
                else
                {
                    return new NDarray(np.random.gumbel(loc?.NumpyNDarray, scale?.NumpyNDarray, size));
                }
            }
        }

        public static partial class random
        {
            /// <summary>
            ///     Draw samples from a Hypergeometric distribution.<br></br>
            ///     Samples are drawn from a hypergeometric distribution with specified
            ///     parameters, ngood (ways to make a good selection), nbad (ways to make
            ///     a bad selection), and nsample = number of items sampled, which is less
            ///     than or equal to the sum ngood + nbad.<br></br>
            ///     Notes
            ///     The probability density for the Hypergeometric distribution is
            ///     where  and
            ///     for P(x) the probability of x successes, g = ngood, b = nbad, and
            ///     n = number of samples.<br></br>
            ///     Consider an urn with black and white marbles in it, ngood of them
            ///     black and nbad are white.<br></br>
            ///     If you draw nsample balls without
            ///     replacement, then the hypergeometric distribution describes the
            ///     distribution of black balls in the drawn sample.<br></br>
            ///     Note that this distribution is very similar to the binomial
            ///     distribution, except that in this case, samples are drawn without
            ///     replacement, whereas in the Binomial case samples are drawn with
            ///     replacement (or the sample space is infinite).<br></br>
            ///     As the sample space
            ///     becomes large, this distribution approaches the binomial.<br></br>
            ///     References
            /// </summary>
            /// <param name="ngood">
            ///     Number of ways to make a good selection.<br></br>
            ///     Must be nonnegative.
            /// </param>
            /// <param name="nbad">
            ///     Number of ways to make a bad selection.<br></br>
            ///     Must be nonnegative.
            /// </param>
            /// <param name="nsample">
            ///     Number of items sampled.<br></br>
            ///     Must be at least 1 and at most
            ///     ngood + nbad.
            /// </param>
            /// <param name="size">
            ///     Output shape.<br></br>
            ///     If the given shape is, e.g., (m, n, k), then
            ///     m * n * k samples are drawn.<br></br>
            ///     If size is None (default),
            ///     a single value is returned if ngood, nbad, and nsample
            ///     are all scalars.<br></br>
            ///     Otherwise, cp.broadcast(ngood, nbad, nsample).size
            ///     samples are drawn.
            /// </param>
            /// <returns>
            ///     Drawn samples from the parameterized hypergeometric distribution.
            /// </returns>
            public static NDarray hypergeometric(NDarray<int> ngood, NDarray<int> nbad, NDarray<int> nsample,
                int[] size = null)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(cp.random.hypergeometric(ngood.CupyNDarray, nbad.CupyNDarray, nsample.CupyNDarray, size));
                }
                else
                {
                    return new NDarray(np.random.hypergeometric(ngood.NumpyNDarray, nbad.NumpyNDarray, nsample.NumpyNDarray, size));
                }
            }
        }

        public static partial class random
        {
            /// <summary>
            ///     Draw samples from the Laplace or double exponential distribution with
            ///     specified location (or mean) and scale (decay).<br></br>
            ///     The Laplace distribution is similar to the Gaussian/normal distribution,
            ///     but is sharper at the peak and has fatter tails.<br></br>
            ///     It represents the
            ///     difference between two independent, identically distributed exponential
            ///     random variables.<br></br>
            ///     Notes
            ///     It has the probability density function
            ///     The first law of Laplace, from 1774, states that the frequency
            ///     of an error can be expressed as an exponential function of the
            ///     absolute magnitude of the error, which leads to the Laplace
            ///     distribution.<br></br>
            ///     For many problems in economics and health
            ///     sciences, this distribution seems to model the data better
            ///     than the standard Gaussian distribution.<br></br>
            ///     References
            /// </summary>
            /// <param name="loc">
            ///     The position, , of the distribution peak.<br></br>
            ///     Default is 0.
            /// </param>
            /// <param name="scale">
            ///     , the exponential decay.<br></br>
            ///     Default is 1.
            /// </param>
            /// <param name="size">
            ///     Output shape.<br></br>
            ///     If the given shape is, e.g., (m, n, k), then
            ///     m * n * k samples are drawn.<br></br>
            ///     If size is None (default),
            ///     a single value is returned if loc and scale are both scalars.<br></br>
            ///     Otherwise, cp.broadcast(loc, scale).size samples are drawn.
            /// </param>
            /// <returns>
            ///     Drawn samples from the parameterized Laplace distribution.
            /// </returns>
            public static NDarray laplace(NDarray<float> loc = null, NDarray<float> scale = null, int[] size = null)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(cp.random.laplace(loc?.CupyNDarray, scale?.CupyNDarray, size));
                }
                else
                {
                    return new NDarray(np.random.laplace(loc?.NumpyNDarray, scale?.NumpyNDarray, size));
                }
            }
        }

        public static partial class random
        {
            /// <summary>
            ///     Draw samples from a logistic distribution.<br></br>
            ///     Samples are drawn from a logistic distribution with specified
            ///     parameters, loc (location or mean, also median), and scale (&gt;0).<br></br>
            ///     Notes
            ///     The probability density for the Logistic distribution is
            ///     where  = location and  = scale.<br></br>
            ///     The Logistic distribution is used in Extreme Value problems where it
            ///     can act as a mixture of Gumbel distributions, in Epidemiology, and by
            ///     the World Chess Federation (FIDE) where it is used in the Elo ranking
            ///     system, assuming the performance of each player is a logistically
            ///     distributed random variable.<br></br>
            ///     References
            /// </summary>
            /// <param name="loc">
            ///     Parameter of the distribution.<br></br>
            ///     Default is 0.
            /// </param>
            /// <param name="scale">
            ///     Parameter of the distribution.<br></br>
            ///     Should be greater than zero.<br></br>
            ///     Default is 1.
            /// </param>
            /// <param name="size">
            ///     Output shape.<br></br>
            ///     If the given shape is, e.g., (m, n, k), then
            ///     m * n * k samples are drawn.<br></br>
            ///     If size is None (default),
            ///     a single value is returned if loc and scale are both scalars.<br></br>
            ///     Otherwise, cp.broadcast(loc, scale).size samples are drawn.
            /// </param>
            /// <returns>
            ///     Drawn samples from the parameterized logistic distribution.
            /// </returns>
            public static NDarray logistic(NDarray<float> loc = null, NDarray<float> scale = null, int[] size = null)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(cp.random.logistic(loc?.CupyNDarray, scale?.CupyNDarray, size));
                }
                else
                {
                    return new NDarray(np.random.logistic(loc?.NumpyNDarray, scale?.NumpyNDarray, size));
                }
            }
        }

        public static partial class random
        {
            /// <summary>
            ///     Draw samples from a log-normal distribution.<br></br>
            ///     Draw samples from a log-normal distribution with specified mean,
            ///     standard deviation, and array shape.<br></br>
            ///     Note that the mean and standard
            ///     deviation are not the values for the distribution itself, but of the
            ///     underlying normal distribution it is derived from.<br></br>
            ///     Notes
            ///     A variable x has a log-normal distribution if log(x) is normally
            ///     distributed.<br></br>
            ///     The probability density function for the log-normal
            ///     distribution is:
            ///     where  is the mean and  is the standard
            ///     deviation of the normally distributed logarithm of the variable.<br></br>
            ///     A log-normal distribution results if a random variable is the product
            ///     of a large number of independent, identically-distributed variables in
            ///     the same way that a normal distribution results if the variable is the
            ///     sum of a large number of independent, identically-distributed
            ///     variables.<br></br>
            ///     References
            /// </summary>
            /// <param name="mean">
            ///     Mean value of the underlying normal distribution.<br></br>
            ///     Default is 0.
            /// </param>
            /// <param name="sigma">
            ///     Standard deviation of the underlying normal distribution.<br></br>
            ///     Should
            ///     be greater than zero.<br></br>
            ///     Default is 1.
            /// </param>
            /// <param name="size">
            ///     Output shape.<br></br>
            ///     If the given shape is, e.g., (m, n, k), then
            ///     m * n * k samples are drawn.<br></br>
            ///     If size is None (default),
            ///     a single value is returned if mean and sigma are both scalars.<br></br>
            ///     Otherwise, cp.broadcast(mean, sigma).size samples are drawn.
            /// </param>
            /// <returns>
            ///     Drawn samples from the parameterized log-normal distribution.
            /// </returns>
            public static NDarray lognormal(NDarray<float> mean = null, NDarray<float> sigma = null, int[] size = null)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(cp.random.lognormal(mean?.CupyNDarray, sigma?.CupyNDarray, size));
                }
                else
                {
                    return new NDarray(np.random.lognormal(mean?.NumpyNDarray, sigma?.NumpyNDarray, size));
                }
            }
        }

        public static partial class random
        {
            /// <summary>
            ///     Draw samples from a logarithmic series distribution.<br></br>
            ///     Samples are drawn from a log series distribution with specified
            ///     shape parameter, 0 &lt; p &lt; 1.<br></br>
            ///     Notes
            ///     The probability density for the Log Series distribution is
            ///     where p = probability.<br></br>
            ///     The log series distribution is frequently used to represent species
            ///     richness and occurrence, first proposed by Fisher, Corbet, and
            ///     Williams in 1943 [2].<br></br>
            ///     It may also be used to model the numbers of
            ///     occupants seen in cars [3].<br></br>
            ///     References
            /// </summary>
            /// <param name="p">
            ///     Shape parameter for the distribution.<br></br>
            ///     Must be in the range (0, 1).
            /// </param>
            /// <param name="size">
            ///     Output shape.<br></br>
            ///     If the given shape is, e.g., (m, n, k), then
            ///     m * n * k samples are drawn.<br></br>
            ///     If size is None (default),
            ///     a single value is returned if p is a scalar.<br></br>
            ///     Otherwise,
            ///     cp.array(p).size samples are drawn.
            /// </param>
            /// <returns>
            ///     Drawn samples from the parameterized logarithmic series distribution.
            /// </returns>
            public static NDarray logseries(NDarray<float> p, int[] size = null)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(cp.random.logseries(p.CupyNDarray, size));
                }
                else
                {
                    return new NDarray(np.random.logseries(p.NumpyNDarray, size));
                }
            }
        }

        public static partial class random
        {
            /// <summary>
            ///     Draw samples from a multinomial distribution.<br></br>
            ///     The multinomial distribution is a multivariate generalisation of the
            ///     binomial distribution.<br></br>
            ///     Take an experiment with one of p
            ///     possible outcomes.<br></br>
            ///     An example of such an experiment is throwing a dice,
            ///     where the outcome can be 1 through 6.<br></br>
            ///     Each sample drawn from the
            ///     distribution represents n such experiments.<br></br>
            ///     Its values,
            ///     X_i = [X_0, X_1, ..., X_p], represent the number of times the
            ///     outcome was i.
            /// </summary>
            /// <param name="n">
            ///     Number of experiments.
            /// </param>
            /// <param name="pvals">
            ///     Probabilities of each of the p different outcomes.<br></br>
            ///     These
            ///     should sum to 1 (however, the last element is always assumed to
            ///     account for the remaining probability, as long as
            ///     sum(pvals[:-1]) &lt;= 1).
            /// </param>
            /// <param name="size">
            ///     Output shape.<br></br>
            ///     If the given shape is, e.g., (m, n, k), then
            ///     m * n * k samples are drawn.<br></br>
            ///     Default is None, in which case a
            ///     single value is returned.
            /// </param>
            /// <returns>
            ///     The drawn samples, of shape size, if that was provided.<br></br>
            ///     If not,
            ///     the shape is (N,).<br></br>
            ///     In other words, each entry out[i,j,...,:] is an N-dimensional
            ///     value drawn from the distribution.
            /// </returns>
            public static NDarray multinomial(int n, NDarray<float> pvals, int[] size = null)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(cp.random.multinomial(n, pvals.CupyNDarray, size));
                }
                else
                {
                    return new NDarray(np.random.multinomial(n, pvals.NumpyNDarray, size));
                }
            }
        }

        public static partial class random
        {
            /// <summary>
            ///     Draw random samples from a multivariate normal distribution.<br></br>
            ///     The multivariate normal, multinormal or Gaussian distribution is a
            ///     generalization of the one-dimensional normal distribution to higher
            ///     dimensions.<br></br>
            ///     Such a distribution is specified by its mean and
            ///     covariance matrix.<br></br>
            ///     These parameters are analogous to the mean
            ///     (average or “center”) and variance (standard deviation, or “width,”
            ///     squared) of the one-dimensional normal distribution.<br></br>
            ///     Notes
            ///     The mean is a coordinate in N-dimensional space, which represents the
            ///     location where samples are most likely to be generated.<br></br>
            ///     This is
            ///     analogous to the peak of the bell curve for the one-dimensional or
            ///     univariate normal distribution.<br></br>
            ///     Covariance indicates the level to which two variables vary together.<br></br>
            ///     From the multivariate normal distribution, we draw N-dimensional
            ///     samples, .  The covariance matrix
            ///     element  is the covariance of  and .
            ///     The element  is the variance of  (i.e.<br></br>
            ///     its
            ///     “spread”).<br></br>
            ///     Instead of specifying the full covariance matrix, popular
            ///     approximations include:
            ///     This geometrical property can be seen in two dimensions by plotting
            ///     generated data-points:
            ///     Diagonal covariance means that points are oriented along x or y-axis:
            ///     Note that the covariance matrix must be positive semidefinite (a.k.a.<br></br>
            ///     nonnegative-definite).<br></br>
            ///     Otherwise, the behavior of this method is
            ///     undefined and backwards compatibility is not guaranteed.<br></br>
            ///     References
            /// </summary>
            /// <param name="mean">
            ///     Mean of the N-dimensional distribution.
            /// </param>
            /// <param name="cov">
            ///     Covariance matrix of the distribution.<br></br>
            ///     It must be symmetric and
            ///     positive-semidefinite for proper sampling.
            /// </param>
            /// <param name="size">
            ///     Given a shape of, for example, (m,n,k), m*n*k samples are
            ///     generated, and packed in an m-by-n-by-k arrangement.<br></br>
            ///     Because
            ///     each sample is N-dimensional, the output shape is (m,n,k,N).<br></br>
            ///     If no shape is specified, a single (N-D) sample is returned.
            /// </param>
            /// <param name="check_valid">
            ///     Behavior when the covariance matrix is not positive semidefinite.
            /// </param>
            /// <param name="tol">
            ///     Tolerance when checking the singular values in covariance matrix.
            /// </param>
            /// <returns>
            ///     The drawn samples, of shape size, if that was provided.<br></br>
            ///     If not,
            ///     the shape is (N,).<br></br>
            ///     In other words, each entry out[i,j,...,:] is an N-dimensional
            ///     value drawn from the distribution.
            /// </returns>
            public static NDarray multivariate_normal(NDarray mean, NDarray cov, int[] size = null,
                string check_valid = null, float? tol = null)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(cp.random.multivariate_normal(mean.CupyNDarray, cov.CupyNDarray, size, check_valid, tol));
                }
                else
                {
                    return new NDarray(np.random.multivariate_normal(mean.NumpyNDarray, cov.NumpyNDarray, size, check_valid, tol));
                }
            }
        }

        public static partial class random
        {
            /// <summary>
            ///     Draw samples from a negative binomial distribution.<br></br>
            ///     Samples are drawn from a negative binomial distribution with specified
            ///     parameters, n successes and p probability of success where n is an
            ///     integer &gt; 0 and p is in the interval [0, 1].<br></br>
            ///     Notes
            ///     The probability density for the negative binomial distribution is
            ///     where  is the number of successes,  is the
            ///     probability of success, and  is the number of trials.<br></br>
            ///     The negative binomial distribution gives the probability of N
            ///     failures given n successes, with a success on the last trial.<br></br>
            ///     If one throws a die repeatedly until the third time a “1” appears,
            ///     then the probability distribution of the number of non-“1”s that
            ///     appear before the third “1” is a negative binomial distribution.<br></br>
            ///     References
            /// </summary>
            /// <param name="n">
            ///     Parameter of the distribution, &gt; 0.<br></br>
            ///     Floats are also accepted,
            ///     but they will be truncated to integers.
            /// </param>
            /// <param name="p">
            ///     Parameter of the distribution, &gt;= 0 and &lt;=1.
            /// </param>
            /// <param name="size">
            ///     Output shape.<br></br>
            ///     If the given shape is, e.g., (m, n, k), then
            ///     m * n * k samples are drawn.<br></br>
            ///     If size is None (default),
            ///     a single value is returned if n and p are both scalars.<br></br>
            ///     Otherwise, cp.broadcast(n, p).size samples are drawn.
            /// </param>
            /// <returns>
            ///     Drawn samples from the parameterized negative binomial distribution,
            ///     where each sample is equal to N, the number of failures that
            ///     occurred before a total of n successes was reached.
            /// </returns>
            public static NDarray negative_binomial(NDarray<int> n, NDarray<float> p, int[] size = null)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(cp.random.negative_binomial(n.CupyNDarray, p.CupyNDarray, size));
                }
                else
                {
                    return new NDarray(np.random.negative_binomial(n.NumpyNDarray, p.NumpyNDarray, size));
                }
            }
        }

        public static partial class random
        {
            /// <summary>
            ///     Draw samples from a noncentral chi-square distribution.<br></br>
            ///     The noncentral  distribution is a generalisation of
            ///     the  distribution.<br></br>
            ///     Notes
            ///     The probability density function for the noncentral Chi-square
            ///     distribution is
            ///     where  is the Chi-square with q degrees of freedom.<br></br>
            ///     In Delhi (2007), it is noted that the noncentral chi-square is
            ///     useful in bombing and coverage problems, the probability of
            ///     killing the point target given by the noncentral chi-squared
            ///     distribution.<br></br>
            ///     References
            /// </summary>
            /// <param name="df">
            ///     Degrees of freedom, should be &gt; 0.
            /// </param>
            /// <param name="nonc">
            ///     Non-centrality, should be non-negative.
            /// </param>
            /// <param name="size">
            ///     Output shape.<br></br>
            ///     If the given shape is, e.g., (m, n, k), then
            ///     m * n * k samples are drawn.<br></br>
            ///     If size is None (default),
            ///     a single value is returned if df and nonc are both scalars.<br></br>
            ///     Otherwise, cp.broadcast(df, nonc).size samples are drawn.
            /// </param>
            /// <returns>
            ///     Drawn samples from the parameterized noncentral chi-square distribution.
            /// </returns>
            public static NDarray noncentral_chisquare(NDarray<float> df, NDarray<float> nonc, int[] size = null)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(cp.random.noncentral_chisquare(df.CupyNDarray, nonc.CupyNDarray, size));
                }
                else
                {
                    return new NDarray(np.random.noncentral_chisquare(df.NumpyNDarray, nonc.NumpyNDarray, size));
                }
            }
        }

        public static partial class random
        {
            /// <summary>
            ///     Draw samples from the noncentral F distribution.<br></br>
            ///     Samples are drawn from an F distribution with specified parameters,
            ///     dfnum (degrees of freedom in numerator) and dfden (degrees of
            ///     freedom in denominator), where both parameters &gt; 1.<br></br>
            ///     nonc is the non-centrality parameter.<br></br>
            ///     Notes
            ///     When calculating the power of an experiment (power = probability of
            ///     rejecting the null hypothesis when a specific alternative is true) the
            ///     non-central F statistic becomes important.<br></br>
            ///     When the null hypothesis is
            ///     true, the F statistic follows a central F distribution.<br></br>
            ///     When the null
            ///     hypothesis is not true, then it follows a non-central F statistic.<br></br>
            ///     References
            /// </summary>
            /// <param name="dfnum">
            ///     Numerator degrees of freedom, should be &gt; 0.
            /// </param>
            /// <param name="dfden">
            ///     Denominator degrees of freedom, should be &gt; 0.
            /// </param>
            /// <param name="nonc">
            ///     Non-centrality parameter, the sum of the squares of the numerator
            ///     means, should be &gt;= 0.
            /// </param>
            /// <param name="size">
            ///     Output shape.<br></br>
            ///     If the given shape is, e.g., (m, n, k), then
            ///     m * n * k samples are drawn.<br></br>
            ///     If size is None (default),
            ///     a single value is returned if dfnum, dfden, and nonc
            ///     are all scalars.<br></br>
            ///     Otherwise, cp.broadcast(dfnum, dfden, nonc).size
            ///     samples are drawn.
            /// </param>
            /// <returns>
            ///     Drawn samples from the parameterized noncentral Fisher distribution.
            /// </returns>
            public static NDarray noncentral_f(NDarray<float> dfnum, NDarray<float> dfden, NDarray<float> nonc,
                int[] size = null)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(cp.random.noncentral_f(dfnum.CupyNDarray, dfden.CupyNDarray, nonc.CupyNDarray, size));
                }
                else
                {
                    return new NDarray(np.random.noncentral_f(dfnum.NumpyNDarray, dfden.NumpyNDarray, nonc.NumpyNDarray, size));
                }
            }
        }

        public static partial class random
        {
            /// <summary>
            ///     Draw samples from a Pareto II or Lomax distribution with
            ///     specified shape.<br></br>
            ///     The Lomax or Pareto II distribution is a shifted Pareto
            ///     distribution.<br></br>
            ///     The classical Pareto distribution can be
            ///     obtained from the Lomax distribution by adding 1 and
            ///     multiplying by the scale parameter m (see Notes).<br></br>
            ///     The
            ///     smallest value of the Lomax distribution is zero while for the
            ///     classical Pareto distribution it is mu, where the standard
            ///     Pareto distribution has location mu = 1.<br></br>
            ///     Lomax can also
            ///     be considered as a simplified version of the Generalized
            ///     Pareto distribution (available in SciPy), with the scale set
            ///     to one and the location set to zero.<br></br>
            ///     The Pareto distribution must be greater than zero, and is
            ///     unbounded above.<br></br>
            ///     It is also known as the “80-20 rule”.  In
            ///     this distribution, 80 percent of the weights are in the lowest
            ///     20 percent of the range, while the other 20 percent fill the
            ///     remaining 80 percent of the range.<br></br>
            ///     Notes
            ///     The probability density for the Pareto distribution is
            ///     where  is the shape and  the scale.<br></br>
            ///     The Pareto distribution, named after the Italian economist
            ///     Vilfredo Pareto, is a power law probability distribution
            ///     useful in many real world problems.<br></br>
            ///     Outside the field of
            ///     economics it is generally referred to as the Bradford
            ///     distribution.<br></br>
            ///     Pareto developed the distribution to describe
            ///     the distribution of wealth in an economy.<br></br>
            ///     It has also found
            ///     use in insurance, web page access statistics, oil field sizes,
            ///     and many other problems, including the download frequency for
            ///     projects in Sourceforge [1].<br></br>
            ///     It is one of the so-called
            ///     “fat-tailed” distributions.<br></br>
            ///     References
            /// </summary>
            /// <param name="a">
            ///     Shape of the distribution.<br></br>
            ///     Should be greater than zero.
            /// </param>
            /// <param name="size">
            ///     Output shape.<br></br>
            ///     If the given shape is, e.g., (m, n, k), then
            ///     m * n * k samples are drawn.<br></br>
            ///     If size is None (default),
            ///     a single value is returned if a is a scalar.<br></br>
            ///     Otherwise,
            ///     cp.array(a).size samples are drawn.
            /// </param>
            /// <returns>
            ///     Drawn samples from the parameterized Pareto distribution.
            /// </returns>
            public static NDarray pareto(NDarray<float> a, int[] size = null)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(cp.random.pareto(a.CupyNDarray, size));
                }
                else
                {
                    return new NDarray(np.random.pareto(a.NumpyNDarray, size));
                }
            }
        }

        public static partial class random
        {
            /// <summary>
            ///     Draw samples from a Poisson distribution.<br></br>
            ///     The Poisson distribution is the limit of the binomial distribution
            ///     for large N.<br></br>
            ///     Notes
            ///     The Poisson distribution
            ///     For events with an expected separation  the Poisson
            ///     distribution  describes the probability of
            ///     events occurring within the observed
            ///     interval .
            ///     Because the output is limited to the range of the C long type, a
            ///     ValueError is raised when lam is within 10 sigma of the maximum
            ///     representable value.<br></br>
            ///     References
            /// </summary>
            /// <param name="lam">
            ///     Expectation of interval, should be &gt;= 0.<br></br>
            ///     A sequence of expectation
            ///     intervals must be broadcastable over the requested size.
            /// </param>
            /// <param name="size">
            ///     Output shape.<br></br>
            ///     If the given shape is, e.g., (m, n, k), then
            ///     m * n * k samples are drawn.<br></br>
            ///     If size is None (default),
            ///     a single value is returned if lam is a scalar.<br></br>
            ///     Otherwise,
            ///     cp.array(lam).size samples are drawn.
            /// </param>
            /// <returns>
            ///     Drawn samples from the parameterized Poisson distribution.
            /// </returns>
            public static NDarray poisson(NDarray<float> lam = null, int[] size = null)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(cp.random.poisson(lam?.CupyNDarray, size));
                }
                else
                {
                    return new NDarray(np.random.poisson(lam?.NumpyNDarray, size));
                }
            }
        }

        public static partial class random
        {
            /// <summary>
            ///     Draws samples in [0, 1] from a power distribution with positive
            ///     exponent a - 1.<br></br>
            ///     Also known as the power function distribution.<br></br>
            ///     Notes
            ///     The probability density function is
            ///     The power function distribution is just the inverse of the Pareto
            ///     distribution.<br></br>
            ///     It may also be seen as a special case of the Beta
            ///     distribution.<br></br>
            ///     It is used, for example, in modeling the over-reporting of insurance
            ///     claims.<br></br>
            ///     References
            /// </summary>
            /// <param name="a">
            ///     Parameter of the distribution.<br></br>
            ///     Should be greater than zero.
            /// </param>
            /// <param name="size">
            ///     Output shape.<br></br>
            ///     If the given shape is, e.g., (m, n, k), then
            ///     m * n * k samples are drawn.<br></br>
            ///     If size is None (default),
            ///     a single value is returned if a is a scalar.<br></br>
            ///     Otherwise,
            ///     cp.array(a).size samples are drawn.
            /// </param>
            /// <returns>
            ///     Drawn samples from the parameterized power distribution.
            /// </returns>
            public static NDarray power(NDarray<float> a, int[] size = null)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(cp.random.power(a.CupyNDarray, size));
                }
                else
                {
                    return new NDarray(np.random.power(a.NumpyNDarray, size));
                }
            }
        }

        public static partial class random
        {
            /// <summary>
            ///     Draw samples from a Rayleigh distribution.<br></br>
            ///     The  and Weibull distributions are generalizations of the
            ///     Rayleigh.<br></br>
            ///     Notes
            ///     The probability density function for the Rayleigh distribution is
            ///     The Rayleigh distribution would arise, for example, if the East
            ///     and North components of the wind velocity had identical zero-mean
            ///     Gaussian distributions.<br></br>
            ///     Then the wind speed would have a Rayleigh
            ///     distribution.<br></br>
            ///     References
            /// </summary>
            /// <param name="scale">
            ///     Scale, also equals the mode.<br></br>
            ///     Should be &gt;= 0.<br></br>
            ///     Default is 1.
            /// </param>
            /// <param name="size">
            ///     Output shape.<br></br>
            ///     If the given shape is, e.g., (m, n, k), then
            ///     m * n * k samples are drawn.<br></br>
            ///     If size is None (default),
            ///     a single value is returned if scale is a scalar.<br></br>
            ///     Otherwise,
            ///     cp.array(scale).size samples are drawn.
            /// </param>
            /// <returns>
            ///     Drawn samples from the parameterized Rayleigh distribution.
            /// </returns>
            public static NDarray rayleigh(NDarray<float> scale = null, int[] size = null)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(cp.random.rayleigh(scale?.CupyNDarray, size));
                }
                else
                {
                    return new NDarray(np.random.rayleigh(scale?.NumpyNDarray, size));
                }
            }
        }

        public static partial class random
        {
            /// <summary>
            ///     Draw samples from a standard Cauchy distribution with mode = 0.<br></br>
            ///     Also known as the Lorentz distribution.<br></br>
            ///     Notes
            ///     The probability density function for the full Cauchy distribution is
            ///     and the Standard Cauchy distribution just sets  and
            ///     The Cauchy distribution arises in the solution to the driven harmonic
            ///     oscillator problem, and also describes spectral line broadening.<br></br>
            ///     It
            ///     also describes the distribution of values at which a line tilted at
            ///     a random angle will cut the x axis.<br></br>
            ///     When studying hypothesis tests that assume normality, seeing how the
            ///     tests perform on data from a Cauchy distribution is a good indicator of
            ///     their sensitivity to a heavy-tailed distribution, since the Cauchy looks
            ///     very much like a Gaussian distribution, but with heavier tails.<br></br>
            ///     References
            /// </summary>
            /// <param name="size">
            ///     Output shape.<br></br>
            ///     If the given shape is, e.g., (m, n, k), then
            ///     m * n * k samples are drawn.<br></br>
            ///     Default is None, in which case a
            ///     single value is returned.
            /// </param>
            /// <returns>
            ///     The drawn samples.
            /// </returns>
            public static NDarray standard_cauchy(params int[] size)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(cp.random.standard_cauchy(size));
                }
                else
                {
                    return new NDarray(np.random.standard_cauchy(size));
                }
            }
        }

        public static partial class random
        {
            /// <summary>
            ///     Draw samples from the standard exponential distribution.<br></br>
            ///     standard_exponential is identical to the exponential distribution
            ///     with a scale parameter of 1.
            /// </summary>
            /// <param name="size">
            ///     Output shape.<br></br>
            ///     If the given shape is, e.g., (m, n, k), then
            ///     m * n * k samples are drawn.<br></br>
            ///     Default is None, in which case a
            ///     single value is returned.
            /// </param>
            /// <returns>
            ///     Drawn samples.
            /// </returns>
            public static NDarray standard_exponential(params int[] size)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(cp.random.standard_exponential(size));
                }
                else
                {
                    return new NDarray(np.random.standard_exponential(size));
                }
            }
        }

        public static partial class random
        {
            /// <summary>
            ///     Draw samples from a standard Gamma distribution.<br></br>
            ///     Samples are drawn from a Gamma distribution with specified parameters,
            ///     shape (sometimes designated “k”) and scale=1.
            ///     Notes
            ///     The probability density for the Gamma distribution is
            ///     where  is the shape and  the scale,
            ///     and  is the Gamma function.<br></br>
            ///     The Gamma distribution is often used to model the times to failure of
            ///     electronic components, and arises naturally in processes for which the
            ///     waiting times between Poisson distributed events are relevant.<br></br>
            ///     References
            /// </summary>
            /// <param name="shape">
            ///     Parameter, should be &gt; 0.
            /// </param>
            /// <param name="size">
            ///     Output shape.<br></br>
            ///     If the given shape is, e.g., (m, n, k), then
            ///     m * n * k samples are drawn.<br></br>
            ///     If size is None (default),
            ///     a single value is returned if shape is a scalar.<br></br>
            ///     Otherwise,
            ///     cp.array(shape).size samples are drawn.
            /// </param>
            /// <returns>
            ///     Drawn samples from the parameterized standard gamma distribution.
            /// </returns>
            public static NDarray standard_gamma(Shape shape, int[] size = null)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(cp.random.standard_gamma(shape.CupyShape, size));
                }
                else
                {
                    return new NDarray(np.random.standard_gamma(shape.NumpyShape, size));
                }
            }
        }

        public static partial class random
        {
            /// <summary>
            ///     Draw samples from a standard Normal distribution (mean=0, stdev=1).
            /// </summary>
            /// <param name="size">
            ///     Output shape.<br></br>
            ///     If the given shape is, e.g., (m, n, k), then
            ///     m * n * k samples are drawn.<br></br>
            ///     Default is None, in which case a
            ///     single value is returned.
            /// </param>
            /// <returns>
            ///     Drawn samples.
            /// </returns>
            public static NDarray standard_normal(params int[] size)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(cp.random.standard_normal(size));
                }
                else
                {
                    return new NDarray(np.random.standard_normal(size));
                }
            }
        }

        public static partial class random
        {
            /// <summary>
            ///     Draw samples from a standard Student’s t distribution with df degrees
            ///     of freedom.<br></br>
            ///     A special case of the hyperbolic distribution.<br></br>
            ///     As df gets
            ///     large, the result resembles that of the standard normal
            ///     distribution (standard_normal).<br></br>
            ///     Notes
            ///     The probability density function for the t distribution is
            ///     The t test is based on an assumption that the data come from a
            ///     Normal distribution.<br></br>
            ///     The t test provides a way to test whether
            ///     the sample mean (that is the mean calculated from the data) is
            ///     a good estimate of the true mean.<br></br>
            ///     The derivation of the t-distribution was first published in
            ///     1908 by William Gosset while working for the Guinness Brewery
            ///     in Dublin.<br></br>
            ///     Due to proprietary issues, he had to publish under
            ///     a pseudonym, and so he used the name Student.<br></br>
            ///     References
            /// </summary>
            /// <param name="df">
            ///     Degrees of freedom, should be &gt; 0.
            /// </param>
            /// <param name="size">
            ///     Output shape.<br></br>
            ///     If the given shape is, e.g., (m, n, k), then
            ///     m * n * k samples are drawn.<br></br>
            ///     If size is None (default),
            ///     a single value is returned if df is a scalar.<br></br>
            ///     Otherwise,
            ///     cp.array(df).size samples are drawn.
            /// </param>
            /// <returns>
            ///     Drawn samples from the parameterized standard Student’s t distribution.
            /// </returns>
            public static NDarray standard_t(NDarray<float> df, int[] size = null)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(cp.random.standard_t(df.CupyNDarray, size));
                }
                else
                {
                    return new NDarray(np.random.standard_t(df.NumpyNDarray, size));
                }
            }
        }

        public static partial class random
        {
            /// <summary>
            ///     Draw samples from the triangular distribution over the
            ///     interval [left, right].<br></br>
            ///     The triangular distribution is a continuous probability
            ///     distribution with lower limit left, peak at mode, and upper
            ///     limit right.<br></br>
            ///     Unlike the other distributions, these parameters
            ///     directly define the shape of the pdf.<br></br>
            ///     Notes
            ///     The probability density function for the triangular distribution is
            ///     The triangular distribution is often used in ill-defined
            ///     problems where the underlying distribution is not known, but
            ///     some knowledge of the limits and mode exists.<br></br>
            ///     Often it is used
            ///     in simulations.<br></br>
            ///     References
            /// </summary>
            /// <param name="left">
            ///     Lower limit.
            /// </param>
            /// <param name="mode">
            ///     The value where the peak of the distribution occurs.<br></br>
            ///     The value should fulfill the condition left &lt;= mode &lt;= right.
            /// </param>
            /// <param name="right">
            ///     Upper limit, should be larger than left.
            /// </param>
            /// <param name="size">
            ///     Output shape.<br></br>
            ///     If the given shape is, e.g., (m, n, k), then
            ///     m * n * k samples are drawn.<br></br>
            ///     If size is None (default),
            ///     a single value is returned if left, mode, and right
            ///     are all scalars.<br></br>
            ///     Otherwise, cp.broadcast(left, mode, right).size
            ///     samples are drawn.
            /// </param>
            /// <returns>
            ///     Drawn samples from the parameterized triangular distribution.
            /// </returns>
            public static NDarray triangular(NDarray<float> left, NDarray<float> mode, NDarray<float> right,
                int[] size = null)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(cp.random.triangular(left.CupyNDarray, mode.CupyNDarray, right.CupyNDarray, size));
                }
                else
                {
                    return new NDarray(np.random.triangular(left.NumpyNDarray, mode.NumpyNDarray, right.NumpyNDarray, size));
                }
            }
        }

        public static partial class random
        {
            /// <summary>
            ///     Draw samples from a uniform distribution.<br></br>
            ///     Samples are uniformly distributed over the half-open interval
            ///     [low, high) (includes low, but excludes high).<br></br>
            ///     In other words,
            ///     any value within the given interval is equally likely to be drawn
            ///     by uniform.<br></br>
            ///     Notes
            ///     The probability density function of the uniform distribution is
            ///     anywhere within the interval [a, b), and zero elsewhere.<br></br>
            ///     When high == low, values of low will be returned.<br></br>
            ///     If high &lt; low, the results are officially undefined
            ///     and may eventually raise an error, i.e.<br></br>
            ///     do not rely on this
            ///     function to behave when passed arguments satisfying that
            ///     inequality condition.
            /// </summary>
            /// <param name="low">
            ///     Lower boundary of the output interval.<br></br>
            ///     All values generated will be
            ///     greater than or equal to low.<br></br>
            ///     The default value is 0.
            /// </param>
            /// <param name="high">
            ///     Upper boundary of the output interval.<br></br>
            ///     All values generated will be
            ///     less than high.<br></br>
            ///     The default value is 1.0.
            /// </param>
            /// <param name="size">
            ///     Output shape.<br></br>
            ///     If the given shape is, e.g., (m, n, k), then
            ///     m * n * k samples are drawn.<br></br>
            ///     If size is None (default),
            ///     a single value is returned if low and high are both scalars.<br></br>
            ///     Otherwise, cp.broadcast(low, high).size samples are drawn.
            /// </param>
            /// <returns>
            ///     Drawn samples from the parameterized uniform distribution.
            /// </returns>
            public static NDarray uniform(NDarray<float> low = null, NDarray<float> high = null, int[] size = null)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(cp.random.uniform(low?.CupyNDarray, high?.CupyNDarray, size));
                }
                else
                {
                    return new NDarray(np.random.uniform(low?.NumpyNDarray, high?.NumpyNDarray, size));
                }
            }
        }

        public static partial class random
        {
            /// <summary>
            ///     Draw samples from a von Mises distribution.<br></br>
            ///     Samples are drawn from a von Mises distribution with specified mode
            ///     (mu) and dispersion (kappa), on the interval [-pi, pi].<br></br>
            ///     The von Mises distribution (also known as the circular normal
            ///     distribution) is a continuous probability distribution on the unit
            ///     circle.<br></br>
            ///     It may be thought of as the circular analogue of the normal
            ///     distribution.<br></br>
            ///     Notes
            ///     The probability density for the von Mises distribution is
            ///     where  is the mode and  the dispersion,
            ///     and  is the modified Bessel function of order 0.<br></br>
            ///     The von Mises is named for Richard Edler von Mises, who was born in
            ///     Austria-Hungary, in what is now the Ukraine.<br></br>
            ///     He fled to the United
            ///     States in 1939 and became a professor at Harvard.<br></br>
            ///     He worked in
            ///     probability theory, aerodynamics, fluid mechanics, and philosophy of
            ///     science.<br></br>
            ///     References
            /// </summary>
            /// <param name="mu">
            ///     Mode (“center”) of the distribution.
            /// </param>
            /// <param name="kappa">
            ///     Dispersion of the distribution, has to be &gt;=0.
            /// </param>
            /// <param name="size">
            ///     Output shape.<br></br>
            ///     If the given shape is, e.g., (m, n, k), then
            ///     m * n * k samples are drawn.<br></br>
            ///     If size is None (default),
            ///     a single value is returned if mu and kappa are both scalars.<br></br>
            ///     Otherwise, cp.broadcast(mu, kappa).size samples are drawn.
            /// </param>
            /// <returns>
            ///     Drawn samples from the parameterized von Mises distribution.
            /// </returns>
            public static NDarray vonmises(NDarray<float> mu, NDarray<float> kappa, int[] size = null)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(cp.random.vonmises(mu.CupyNDarray, kappa.CupyNDarray, size));
                }
                else
                {
                    return new NDarray(np.random.vonmises(mu.NumpyNDarray, kappa.NumpyNDarray, size));
                }
            }
        }

        public static partial class random
        {
            /// <summary>
            ///     Draw samples from a Wald, or inverse Gaussian, distribution.<br></br>
            ///     As the scale approaches infinity, the distribution becomes more like a
            ///     Gaussian.<br></br>
            ///     Some references claim that the Wald is an inverse Gaussian
            ///     with mean equal to 1, but this is by no means universal.<br></br>
            ///     The inverse Gaussian distribution was first studied in relationship to
            ///     Brownian motion.<br></br>
            ///     In 1956 M.C.K.<br></br>
            ///     Tweedie used the name inverse Gaussian
            ///     because there is an inverse relationship between the time to cover a
            ///     unit distance and distance covered in unit time.<br></br>
            ///     Notes
            ///     The probability density function for the Wald distribution is
            ///     As noted above the inverse Gaussian distribution first arise
            ///     from attempts to model Brownian motion.<br></br>
            ///     It is also a
            ///     competitor to the Weibull for use in reliability modeling and
            ///     modeling stock returns and interest rate processes.<br></br>
            ///     References
            /// </summary>
            /// <param name="mean">
            ///     Distribution mean, should be &gt; 0.
            /// </param>
            /// <param name="scale">
            ///     Scale parameter, should be &gt;= 0.
            /// </param>
            /// <param name="size">
            ///     Output shape.<br></br>
            ///     If the given shape is, e.g., (m, n, k), then
            ///     m * n * k samples are drawn.<br></br>
            ///     If size is None (default),
            ///     a single value is returned if mean and scale are both scalars.<br></br>
            ///     Otherwise, cp.broadcast(mean, scale).size samples are drawn.
            /// </param>
            /// <returns>
            ///     Drawn samples from the parameterized Wald distribution.
            /// </returns>
            public static NDarray wald(NDarray<float> mean, NDarray<float> scale, int[] size = null)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(cp.random.wald(mean.CupyNDarray, scale.CupyNDarray, size));
                }
                else
                {
                    return new NDarray(np.random.wald(mean.NumpyNDarray, scale.NumpyNDarray, size));
                }
            }
        }

        public static partial class random
        {
            /// <summary>
            ///     Draw samples from a Weibull distribution.<br></br>
            ///     Draw samples from a 1-parameter Weibull distribution with the given
            ///     shape parameter a.<br></br>
            ///     Here, U is drawn from the uniform distribution over (0,1].<br></br>
            ///     The more common 2-parameter Weibull, including a scale parameter
            ///     is just .
            ///     Notes
            ///     The Weibull (or Type III asymptotic extreme value distribution
            ///     for smallest values, SEV Type III, or Rosin-Rammler
            ///     distribution) is one of a class of Generalized Extreme Value
            ///     (GEV) distributions used in modeling extreme value problems.<br></br>
            ///     This class includes the Gumbel and Frechet distributions.<br></br>
            ///     The probability density for the Weibull distribution is
            ///     where  is the shape and  the scale.<br></br>
            ///     The function has its peak (the mode) at
            ///     .
            ///     When a = 1, the Weibull distribution reduces to the exponential
            ///     distribution.<br></br>
            ///     References
            /// </summary>
            /// <param name="a">
            ///     Shape parameter of the distribution.<br></br>
            ///     Must be nonnegative.
            /// </param>
            /// <param name="size">
            ///     Output shape.<br></br>
            ///     If the given shape is, e.g., (m, n, k), then
            ///     m * n * k samples are drawn.<br></br>
            ///     If size is None (default),
            ///     a single value is returned if a is a scalar.<br></br>
            ///     Otherwise,
            ///     cp.array(a).size samples are drawn.
            /// </param>
            /// <returns>
            ///     Drawn samples from the parameterized Weibull distribution.
            /// </returns>
            public static NDarray weibull(NDarray<float> a, int[] size = null)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(cp.random.weibull(a.CupyNDarray, size));
                }
                else
                {
                    return new NDarray(np.random.weibull(a.NumpyNDarray, size));
                }
            }
        }

        public static partial class random
        {
            /// <summary>
            ///     Draw samples from a Zipf distribution.<br></br>
            ///     Samples are drawn from a Zipf distribution with specified parameter
            ///     a &gt; 1.<br></br>
            ///     The Zipf distribution (also known as the zeta distribution) is a
            ///     continuous probability distribution that satisfies Zipf’s law: the
            ///     frequency of an item is inversely proportional to its rank in a
            ///     frequency table.<br></br>
            ///     Notes
            ///     The probability density for the Zipf distribution is
            ///     where  is the Riemann Zeta function.<br></br>
            ///     It is named for the American linguist George Kingsley Zipf, who noted
            ///     that the frequency of any word in a sample of a language is inversely
            ///     proportional to its rank in the frequency table.<br></br>
            ///     References
            /// </summary>
            /// <param name="a">
            ///     Distribution parameter.<br></br>
            ///     Should be greater than 1.
            /// </param>
            /// <param name="size">
            ///     Output shape.<br></br>
            ///     If the given shape is, e.g., (m, n, k), then
            ///     m * n * k samples are drawn.<br></br>
            ///     If size is None (default),
            ///     a single value is returned if a is a scalar.<br></br>
            ///     Otherwise,
            ///     cp.array(a).size samples are drawn.
            /// </param>
            /// <returns>
            ///     Drawn samples from the parameterized Zipf distribution.
            /// </returns>
            public static NDarray zipf(NDarray<float> a, int[] size = null)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(cp.random.zipf(a.CupyNDarray, size));
                }
                else
                {
                    return new NDarray(np.random.zipf(a.NumpyNDarray, size));
                }
            }
        }

        public static partial class random
        {
            /// <summary>
            ///     Container for the Mersenne Twister pseudo-random number generator.<br></br>
            ///     RandomState exposes a number of methods for generating random numbers
            ///     drawn from a variety of probability distributions.<br></br>
            ///     In addition to the
            ///     distribution-specific arguments, each method takes a keyword argument
            ///     size that defaults to None.<br></br>
            ///     If size is None, then a single
            ///     value is generated and returned.<br></br>
            ///     If size is an integer, then a 1-D
            ///     array filled with generated values is returned.<br></br>
            ///     If size is a tuple,
            ///     then an array with that shape is filled and returned.<br></br>
            ///     Compatibility Guarantee
            ///     A fixed seed and a fixed series of calls to ‘RandomState’ methods using
            ///     the same parameters will always produce the same results up to roundoff
            ///     error except when the values were incorrect.<br></br>
            ///     Incorrect values will be
            ///     fixed and the Cupy version in which the fix was made will be noted in
            ///     the relevant docstring.<br></br>
            ///     Extension of existing parameter ranges and the
            ///     addition of new parameters is allowed as long the previous behavior
            ///     remains unchanged.<br></br>
            ///     Notes
            ///     The Python stdlib module “random” also contains a Mersenne Twister
            ///     pseudo-random number generator with a number of methods that are similar
            ///     to the ones available in RandomState.<br></br>
            ///     RandomState, besides being
            ///     Cupy-aware, has the advantage that it provides a much larger number
            ///     of probability distributions to choose from.<br></br>
            ///     Methods
            /// </summary>
            /// <param name="seed">
            ///     Random seed used to initialize the pseudo-random number generator.<br></br>
            ///     Can
            ///     be any integer between 0 and 2**32 - 1 inclusive, an array (or other
            ///     sequence) of such integers, or None (the default).<br></br>
            ///     If seed is
            ///     None, then RandomState will try to read data from
            ///     /dev/urandom (or the Windows analogue) if available or seed from
            ///     the clock otherwise.
            /// </param>
            public static void RandomState(int? seed = null)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    cp.random.RandomState(seed);
                }
                else
                {
                    np.random.RandomState(seed);
                }
            }
        }

        public static partial class random
        {
            /// <summary>
            ///     Container for the Mersenne Twister pseudo-random number generator.<br></br>
            ///     RandomState exposes a number of methods for generating random numbers
            ///     drawn from a variety of probability distributions.<br></br>
            ///     In addition to the
            ///     distribution-specific arguments, each method takes a keyword argument
            ///     size that defaults to None.<br></br>
            ///     If size is None, then a single
            ///     value is generated and returned.<br></br>
            ///     If size is an integer, then a 1-D
            ///     array filled with generated values is returned.<br></br>
            ///     If size is a tuple,
            ///     then an array with that shape is filled and returned.<br></br>
            ///     Compatibility Guarantee
            ///     A fixed seed and a fixed series of calls to ‘RandomState’ methods using
            ///     the same parameters will always produce the same results up to roundoff
            ///     error except when the values were incorrect.<br></br>
            ///     Incorrect values will be
            ///     fixed and the Cupy version in which the fix was made will be noted in
            ///     the relevant docstring.<br></br>
            ///     Extension of existing parameter ranges and the
            ///     addition of new parameters is allowed as long the previous behavior
            ///     remains unchanged.<br></br>
            ///     Notes
            ///     The Python stdlib module “random” also contains a Mersenne Twister
            ///     pseudo-random number generator with a number of methods that are similar
            ///     to the ones available in RandomState.<br></br>
            ///     RandomState, besides being
            ///     Cupy-aware, has the advantage that it provides a much larger number
            ///     of probability distributions to choose from.<br></br>
            ///     Methods
            /// </summary>
            /// <param name="seed">
            ///     Random seed used to initialize the pseudo-random number generator.<br></br>
            ///     Can
            ///     be any integer between 0 and 2**32 - 1 inclusive, an array (or other
            ///     sequence) of such integers, or None (the default).<br></br>
            ///     If seed is
            ///     None, then RandomState will try to read data from
            ///     /dev/urandom (or the Windows analogue) if available or seed from
            ///     the clock otherwise.
            /// </param>
            public static void RandomState(NDarray seed = null)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    cp.random.RandomState(seed?.CupyNDarray);
                }
                else
                {
                    np.random.RandomState(seed?.NumpyNDarray);
                }
            }
        }

        public static partial class random
        {
            /// <summary>
            ///     Seed the generator.<br></br>
            ///     This method is called when RandomState is initialized.<br></br>
            ///     It can be
            ///     called again to re-seed the generator.<br></br>
            ///     For details, see RandomState.
            /// </summary>
            /// <param name="seed">
            ///     Seed for RandomState.<br></br>
            ///     Must be convertible to 32 bit unsigned integers.
            /// </param>
            public static void seed(int? seed = null)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    cp.random.seed(seed);
                }
                else
                {
                    np.random.seed(seed);
                }
            }
        }

        public static partial class random
        {
            /// <summary>
            ///     Seed the generator.<br></br>
            ///     This method is called when RandomState is initialized.<br></br>
            ///     It can be
            ///     called again to re-seed the generator.<br></br>
            ///     For details, see RandomState.
            /// </summary>
            /// <param name="seed">
            ///     Seed for RandomState.<br></br>
            ///     Must be convertible to 32 bit unsigned integers.
            /// </param>
            public static void seed(NDarray seed = null)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    cp.random.seed(seed?.CupyNDarray);
                }
                else
                {
                    np.random.seed(seed?.NumpyNDarray);
                }
            }
        }

        /*
        public static partial class random {
            /// <summary>
            ///	Return a tuple representing the internal state of the generator.<br></br>
            ///	
            ///	For more details, see set_state.<br></br>
            ///	
            ///	Notes
            ///	
            ///	set_state and get_state are not needed to work with any of the
            ///	random distributions in Cupy.<br></br>
            ///	 If the internal state is manually altered,
            ///	the user should know exactly what he/she is doing.
            /// </summary>
            /// <param name="out">
            ///	The returned tuple has the following items:
            /// </param>
            /// <returns>
            ///	The returned tuple has the following items:
            /// </returns>
            public static tuple(str get_state(tuple(str @out = null)
            {
                //auto-generated code, do not change
                var random = self.GetAttr("random");
                var __self__=random;
                var pyargs=ToTuple(new object[]
                {
                });
                var kwargs=new PyDict();
                if (@out!=null) kwargs["out"]=ToPython(@out);
                dynamic py = __self__.InvokeMethod("get_state", pyargs, kwargs);
                return ToCsharp<tuple(str>(py);
            }
        }
        */

        /*
        public static partial class random {
            /// <summary>
            ///	Set the internal state of the generator from a tuple.<br></br>
            ///	
            ///	For use if one has reason to manually (re-)set the internal state of the
            ///	“Mersenne Twister”[1] pseudo-random number generating algorithm.<br></br>
            ///	
            ///	Notes
            ///	
            ///	set_state and get_state are not needed to work with any of the
            ///	random distributions in Cupy.<br></br>
            ///	 If the internal state is manually altered,
            ///	the user should know exactly what he/she is doing.<br></br>
            ///	
            ///	For backwards compatibility, the form (str, array of 624 uints, int) is
            ///	also accepted although it is missing some information about the cached
            ///	Gaussian value: state = ('MT19937', keys, pos).<br></br>
            ///	
            ///	References
            /// </summary>
            /// <param name="state">
            ///	The state tuple has the following items:
            /// </param>
            /// <returns>
            ///	Returns ‘None’ on success.
            /// </returns>
            public static None set_state(tuple(str state)
            {
                //auto-generated code, do not change
                var random = self.GetAttr("random");
                var __self__=random;
                var pyargs=ToTuple(new object[]
                {
                    state,
                });
                var kwargs=new PyDict();
                dynamic py = __self__.InvokeMethod("set_state", pyargs, kwargs);
                return ToCsharp<None>(py);
            }
        }
        */
    }
}
