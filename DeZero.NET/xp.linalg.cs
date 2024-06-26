﻿using Cupy;
using Numpy;
using Python.Runtime;

namespace DeZero.NET
{
    public static partial class xp
    {
        /// <summary>
        ///     Dot product of two arrays.<br></br>
        ///     Specifically,
        /// </summary>
        /// <param name="a">
        ///     First argument.
        /// </param>
        /// <param name="b">
        ///     Second argument.
        /// </param>
        /// <param name="out">
        ///     Output argument.<br></br>
        ///     This must have the exact kind that would be returned
        ///     if it was not used.<br></br>
        ///     In particular, it must have the right type, must be
        ///     C-contiguous, and its dtype must be the dtype that would be returned
        ///     for dot(a,b).<br></br>
        ///     This is a performance feature.<br></br>
        ///     Therefore, if these
        ///     conditions are not met, an exception is raised, instead of attempting
        ///     to be flexible.
        /// </param>
        /// <returns>
        ///     Returns the dot product of a and b.<br></br>
        ///     If a and b are both
        ///     scalars or both 1-D arrays then a scalar is returned; otherwise
        ///     an array is returned.<br></br>
        ///     If out is given, then it is returned.
        /// </returns>
        public static NDarray dot(this NDarray a, NDarray b, NDarray @out = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.dot(a.CupyNDarray, b.CupyNDarray, @out?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.dot(a.NumpyNDarray, b.NumpyNDarray, @out?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Return the dot product of two vectors.<br></br>
        ///     The vdot(a, b) function handles complex numbers differently than
        ///     dot(a, b).<br></br>
        ///     If the first argument is complex the complex conjugate
        ///     of the first argument is used for the calculation of the dot product.<br></br>
        ///     Note that vdot handles multidimensional arrays differently than dot:
        ///     it does not perform a matrix product, but flattens input arguments
        ///     to 1-D vectors first.<br></br>
        ///     Consequently, it should only be used for vectors.
        /// </summary>
        /// <param name="a">
        ///     If a is complex the complex conjugate is taken before calculation
        ///     of the dot product.
        /// </param>
        /// <param name="b">
        ///     Second argument to the dot product.
        /// </param>
        /// <returns>
        ///     Dot product of a and b.<br></br>
        ///     Can be an int, float, or
        ///     complex depending on the types of a and b.
        /// </returns>
        public static NDarray vdot(this NDarray a, NDarray b)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.vdot(a.CupyNDarray, b.CupyNDarray));
            }
            else
            {
                return new NDarray(np.vdot(a.NumpyNDarray, b.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Inner product of two arrays.<br></br>
        ///     Ordinary inner product of vectors for 1-D arrays (without complex
        ///     conjugation), in higher dimensions a sum product over the last axis.<br></br>
        ///     Notes
        ///     For vectors (1-D arrays) it computes the ordinary inner-product:
        ///     More generally, if ndim(a) = r &gt; 0 and ndim(b) = s &gt; 0:
        ///     or explicitly:
        ///     In addition a or b may be scalars, in which case:
        /// </summary>
        /// <param name="b">
        ///     If a and b are nonscalar, their last dimensions must match.
        /// </param>
        /// <param name="a">
        ///     If a and b are nonscalar, their last dimensions must match.
        /// </param>
        /// <returns>
        ///     out.shape = a.shape[:-1] + b.shape[:-1]
        /// </returns>
        public static NDarray inner(this NDarray b, NDarray a)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.inner(a.CupyNDarray, b.CupyNDarray));
            }
            else
            {
                return new NDarray(np.inner(a.NumpyNDarray, b.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Compute the outer product of two vectors.<br></br>
        ///     Given two vectors, a = [a0, a1, ..., aM] and
        ///     b = [b0, b1, ..., bN],
        ///     the outer product [1] is:
        ///     References
        /// </summary>
        /// <param name="a">
        ///     First input vector.<br></br>
        ///     Input is flattened if
        ///     not already 1-dimensional.
        /// </param>
        /// <param name="b">
        ///     Second input vector.<br></br>
        ///     Input is flattened if
        ///     not already 1-dimensional.
        /// </param>
        /// <param name="out">
        ///     A location where the result is stored
        /// </param>
        /// <returns>
        ///     out[i, j] = a[i] * b[j]
        /// </returns>
        public static NDarray outer(this NDarray a, NDarray b, NDarray @out = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.outer(a.CupyNDarray, b.CupyNDarray, @out?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.outer(a.NumpyNDarray, b.NumpyNDarray, @out?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Matrix product of two arrays.<br></br>
        ///     Notes
        ///     The behavior depends on the arguments in the following way.<br></br>
        ///     matmul differs from dot in two important ways:
        ///     The matmul function implements the semantics of the &#64; operator introduced
        ///     in Python 3.5 following PEP465.
        /// </summary>
        /// <param name="x2">
        ///     Input arrays, scalars not allowed.
        /// </param>
        /// <param name="x1">
        ///     Input arrays, scalars not allowed.
        /// </param>
        /// <param name="out">
        ///     A location into which the result is stored.<br></br>
        ///     If provided, it must have
        ///     a shape that matches the signature (n,k),(k,m)-&gt;(n,m).<br></br>
        ///     If not
        ///     provided or None, a freshly-allocated array is returned.
        /// </param>
        /// <returns>
        ///     The matrix product of the inputs.<br></br>
        ///     This is a scalar only when both x1, x2 are 1-d vectors.
        /// </returns>
        public static NDarray matmul(this NDarray x2, NDarray x1, NDarray @out = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.matmul(x2.CupyNDarray, x1.CupyNDarray, @out?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.matmul(x2.NumpyNDarray, x1.NumpyNDarray, @out?.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Compute tensor dot product along specified axis for arrays &gt;= 1-D.<br></br>
        ///     Given two tensors (arrays of dimension greater than or equal to one),
        ///     a and b, and an array_like object containing two array_like
        ///     objects, (a_axes, b_axes), sum the products of a’s and b’s
        ///     elements (components) over the axis specified by a_axes and
        ///     b_axes.<br></br>
        ///     The third argument can be a single non-negative
        ///     integer_like scalar, N; if it is such, then the last N
        ///     dimensions of a and the first N dimensions of b are summed
        ///     over.<br></br>
        ///     Notes
        ///     When axis is integer_like, the sequence for evaluation will be: first
        ///     the -Nth axis in a and 0th axis in b, and the -1th axis in a and
        ///     Nth axis in b last.<br></br>
        ///     When there is more than one axis to sum over - and they are not the last
        ///     (first) axis of a (b) - the argument axis should consist of
        ///     two sequences of the same length, with the first axis to sum over given
        ///     first in both sequences, the second axis second, and so forth.
        /// </summary>
        /// <param name="b">
        ///     Tensors to “dot”.
        /// </param>
        /// <param name="a">
        ///     Tensors to “dot”.
        /// </param>
        public static NDarray tensordot(this NDarray b, NDarray a, int[] axes = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.tensordot(b.ToCupyNDarray, a.ToCupyNDarray, axes));
            }
            else
            {
                return new NDarray(np.tensordot(b.ToNumpyNDarray, a.ToNumpyNDarray, axes));
            }
        }

        public static NDarray tensordot(this NDarray b, NDarray a, int[][] axes = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                using var __self__ = Py.Import("cupy");
                using var pyargs = ToTuple(new object[]
                {
                    b.ToCupyNDarray.PyObject,
                    a.ToCupyNDarray.PyObject,
                });
                using var kwargs = new PyDict();
                if (axes != null) kwargs["axes"] = ToTuple(axes).ToPython();
                dynamic py = __self__.InvokeMethod("tensordot", pyargs, kwargs);
                return new NDarray(ToCsharp<Cupy.NDarray>(py));
            }
            else
            {
                using var __self__ = Py.Import("numpy");
                using var pyargs = ToTuple(new object[]
                {
                    b.ToNumpyNDarray.PyObject,
                    a.ToNumpyNDarray.PyObject,
                });
                using var kwargs = new PyDict();
                if (axes != null) kwargs["axes"] = ToPython(axes);
                dynamic py = __self__.InvokeMethod("tensordot", pyargs, kwargs);
                return new NDarray(ToCsharp<Numpy.NDarray>(py));
            }
        }

        //auto-generated
        private static PyTuple ToTuple(Array input)
        {
            var array = new PyObject[input.Length];
            for (var i = 0; i < input.Length; i++) array[i] = ToPython(input.GetValue(i));
            return new PyTuple(array);
        }

        //auto-generated
        internal static PyObject ToPython(object obj)
        {
            if (obj == null) return Runtime.None;
            switch (obj)
            {
                // basic types
                case int o: return new PyInt(o);
                case long o: return new PyInt(o);
                case float o: return new PyFloat(o);
                case double o: return new PyFloat(o);
                case string o: return new PyString(o);
                case bool o: return o.ToPython();
                case int[][] o: return ToTuple(o);
                case NDarray o: return o.Array.ToPython();
                case PyObject o: return o;
                // sequence types
                case Array o: return ToTuple(o);
                // special types from 'ToPythonConversions'
                case Axis o: return o.Axes == null ? null : ToTuple(o.Axes);
                case Shape o: return ToTuple(o.Dimensions);
                case Slice o: return o.ToPython();
                case PythonObject o: return o.PyObject;
                case Dictionary<string, NDarray> o: return ToDict(o);
                case Cupy.NDarray o: return o.PyObject;
                case Numpy.NDarray o: return o.PyObject; 
                case Cupy.Dtype o: return o.PyObject;
                case Numpy.Dtype o: return o.PyObject;
                default:
                    throw new NotImplementedException(
                        $"Type is not yet supported: {obj.GetType().Name}. Add it to 'ToPythonConversions'");
            }
        }

        internal static T ToCsharp<T>(dynamic pyobj)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return ToCsharpCupy<T>(pyobj);
            }
            else
            {
                return ToCsharpNumpy<T>(pyobj);
            }
            
        }

        internal static T ToCsharpNumpy<T>(dynamic pyobj)
        {
            switch (typeof(T).Name)
            {
                // types from 'ToCsharpConversions'
                case "Dtype": return (T)(object)new Numpy.Dtype(pyobj);
                case "NDarray": return (T)(object)new Numpy.NDarray(pyobj);
                case "NDarray`1":
                    switch (typeof(T).GenericTypeArguments[0].Name)
                    {
                        case "Byte": return (T)(object)new Numpy.NDarray<byte>(pyobj);
                        case "Short": return (T)(object)new Numpy.NDarray<short>(pyobj);
                        case "Boolean": return (T)(object)new Numpy.NDarray<bool>(pyobj);
                        case "Int32": return (T)(object)new Numpy.NDarray<int>(pyobj);
                        case "Int64": return (T)(object)new Numpy.NDarray<long>(pyobj);
                        case "Single": return (T)(object)new Numpy.NDarray<float>(pyobj);
                        case "Double": return (T)(object)new Numpy.NDarray<double>(pyobj);
                        default:
                            throw new NotImplementedException(
                                $"Type NDarray<{typeof(T).GenericTypeArguments[0].Name}> missing. Add it to 'ToCsharpConversions'");
                    }

                    break;
                case "NDarray[]":
                    var po = pyobj as PyObject;
                    var len = po.Length();
                    var rv = new Numpy.NDarray[len];
                    for (var i = 0; i < len; i++)
                        rv[i] = ToCsharp<Numpy.NDarray>(po[i]);
                    return (T)(object)rv;
                case "Matrix": return (T)(object)new Matrix(pyobj);
                default:
                    var pyClass = $"{pyobj.__class__}";
                    if (pyClass == "<class 'str'>") return (T)(object)pyobj.ToString();
                    if (pyClass.StartsWith("<class 'Cupy")) return (pyobj.item() as PyObject).As<T>();
                    if (pyClass.StartsWith("<class 'cupy")) return (pyobj.item() as PyObject).As<T>();
                    try
                    {
                        return pyobj.As<T>();
                    }
                    catch (Exception e)
                    {
                        throw new NotImplementedException(
                            $"conversion from {pyobj.__class__} to {typeof(T).Name} not implemented", e);
                        return default;
                    }
            }
        }

        internal static T ToCsharpCupy<T>(dynamic pyobj)
        {
            switch (typeof(T).Name)
            {
                // types from 'ToCsharpConversions'
                case "Dtype": return (T)(object)new Cupy.Dtype(pyobj);
                case "NDarray": return (T)(object)new Cupy.NDarray(pyobj);
                case "NDarray`1":
                    switch (typeof(T).GenericTypeArguments[0].Name)
                    {
                        case "Byte": return (T)(object)new Cupy.NDarray<byte>(pyobj);
                        case "Short": return (T)(object)new Cupy.NDarray<short>(pyobj);
                        case "Boolean": return (T)(object)new Cupy.NDarray<bool>(pyobj);
                        case "Int32": return (T)(object)new Cupy.NDarray<int>(pyobj);
                        case "Int64": return (T)(object)new Cupy.NDarray<long>(pyobj);
                        case "Single": return (T)(object)new Cupy.NDarray<float>(pyobj);
                        case "Double": return (T)(object)new Cupy.NDarray<double>(pyobj);
                        default:
                            throw new NotImplementedException(
                                $"Type NDarray<{typeof(T).GenericTypeArguments[0].Name}> missing. Add it to 'ToCsharpConversions'");
                    }

                    break;
                case "NDarray[]":
                    var po = pyobj as PyObject;
                    var len = po.Length();
                    var rv = new Cupy.NDarray[len];
                    for (var i = 0; i < len; i++)
                        rv[i] = ToCsharp<Cupy.NDarray>(po[i]);
                    return (T)(object)rv;
                case "Matrix": return (T)(object)new Matrix(pyobj);
                default:
                    var pyClass = $"{pyobj.__class__}";
                    if (pyClass == "<class 'str'>") return (T)(object)pyobj.ToString();
                    if (pyClass.StartsWith("<class 'Cupy") || pyClass.StartsWith("<class 'cupy"))
                    {
                        var item = pyobj.item() as PyObject;
                        try
                        {
                            return item.As<T>();
                        }
                        finally
                        {
                            item = null;
                        }
                    }
                    try
                    {
                        return pyobj.As<T>();
                    }
                    catch (Exception e)
                    {
                        throw new NotImplementedException(
                            $"conversion from {pyobj.__class__} to {typeof(T).Name} not implemented", e);
                        return default;
                    }
            }
        }

        private static PyDict ToDict(Dictionary<string, NDarray> d)
        {
            var dict = new PyDict();
            foreach (var pair in d)
            {
                using var key = new PyString(pair.Key);
                dict[key] = pair.Value.self;
            }

            return dict;
        }

        /// <summary>
        ///     Kronecker product of two arrays.<br></br>
        ///     Computes the Kronecker product, a composite array made of blocks of the
        ///     second array scaled by the first.<br></br>
        ///     Notes
        ///     The function assumes that the number of dimensions of a and b
        ///     are the same, if necessary prepending the smallest with ones.<br></br>
        ///     If a.shape = (r0,r1,..,rN) and b.shape = (s0,s1,…,sN),
        ///     the Kronecker product has shape (r0*s0, r1*s1, …, rN*SN).<br></br>
        ///     The elements are products of elements from a and b, organized
        ///     explicitly by:
        ///     where:
        ///     In the common 2-D case (N=1), the block structure can be visualized:
        /// </summary>
        public static NDarray kron(this NDarray b, NDarray a)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.kron(b.CupyNDarray, a.CupyNDarray));
            }
            else
            {
                return new NDarray(np.kron(b.NumpyNDarray, a.NumpyNDarray));
            }
        }

        /// <summary>
        ///     Return the sum along diagonals of the array.<br></br>
        ///     If a is 2-D, the sum along its diagonal with the given offset
        ///     is returned, i.e., the sum of elements a[i,i+offset] for all i.<br></br>
        ///     If a has more than two dimensions, then the axis specified by axis1 and
        ///     axis2 are used to determine the 2-D sub-arrays whose traces are returned.<br></br>
        ///     The shape of the resulting array is the same as that of a with axis1
        ///     and axis2 removed.
        /// </summary>
        /// <param name="a">
        ///     Input array, from which the diagonals are taken.
        /// </param>
        /// <param name="offset">
        ///     Offset of the diagonal from the main diagonal.<br></br>
        ///     Can be both positive
        ///     and negative.<br></br>
        ///     Defaults to 0.
        /// </param>
        /// <param name="axis2">
        ///     Axes to be used as the first and second axis of the 2-D sub-arrays
        ///     from which the diagonals should be taken.<br></br>
        ///     Defaults are the first two
        ///     axis of a.
        /// </param>
        /// <param name="axis1">
        ///     Axes to be used as the first and second axis of the 2-D sub-arrays
        ///     from which the diagonals should be taken.<br></br>
        ///     Defaults are the first two
        ///     axis of a.
        /// </param>
        /// <param name="dtype">
        ///     Determines the data-type of the returned array and of the accumulator
        ///     where the elements are summed.<br></br>
        ///     If dtype has the value None and a is
        ///     of integer type of precision less than the default integer
        ///     precision, then the default integer precision is used.<br></br>
        ///     Otherwise,
        ///     the precision is the same as that of a.
        /// </param>
        /// <param name="out">
        ///     Array into which the output is placed.<br></br>
        ///     Its type is preserved and
        ///     it must be of the right shape to hold the output.
        /// </param>
        /// <returns>
        ///     If a is 2-D, the sum along the diagonal is returned.<br></br>
        ///     If a has
        ///     larger dimensions, then an array of sums along diagonals is returned.
        /// </returns>
        public static NDarray trace(this NDarray a, int? offset = 0, int? axis2 = null, int? axis1 = null,
            Dtype dtype = null, NDarray @out = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.trace(a.CupyNDarray, offset, axis2, axis1, dtype?.CupyDtype, @out?.CupyNDarray));
            }
            else
            {
                return new NDarray(np.trace(a.NumpyNDarray, offset, axis2, axis1, dtype?.NumpyDtype, @out?.NumpyNDarray));
            }
        }

        public static partial class linalg
        {
            /// <summary>
            ///     Compute the dot product of two or more arrays in a single function call,
            ///     while automatically selecting the fastest evaluation order.<br></br>
            ///     multi_dot chains Cupy.dot and uses optimal parenthesization
            ///     of the matrices [1] [2].<br></br>
            ///     Depending on the shapes of the matrices,
            ///     this can speed up the multiplication a lot.<br></br>
            ///     If the first argument is 1-D it is treated as a row vector.<br></br>
            ///     If the last argument is 1-D it is treated as a column vector.<br></br>
            ///     The other arguments must be 2-D.<br></br>
            ///     Think of multi_dot as:
            ///     Notes
            ///     The cost for a matrix multiplication can be calculated with the
            ///     following function:
            ///     Let’s assume we have three matrices
            ///     .
            ///     The costs for the two different parenthesizations are as follows:
            ///     References
            /// </summary>
            /// <param name="arrays">
            ///     If the first argument is 1-D it is treated as row vector.<br></br>
            ///     If the last argument is 1-D it is treated as column vector.<br></br>
            ///     The other arguments must be 2-D.
            /// </param>
            /// <returns>
            ///     Returns the dot product of the supplied arrays.
            /// </returns>
            public static NDarray multi_dot(params NDarray[] arrays)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(cp.linalg.multi_dot(arrays.Select(x => x.CupyNDarray).ToArray()));
                }
                else
                {
                    return new NDarray(np.linalg.multi_dot(arrays.Select(x => x.NumpyNDarray).ToArray()));
                }
            }
        }

        /*
        /// <summary>
        ///	Evaluates the lowest cost contraction order for an einsum expression by
        ///	considering the creation of intermediate arrays.<br></br>
        ///	
        ///	Notes
        ///	
        ///	The resulting path indicates which terms of the input contraction should be
        ///	contracted first, the result of this contraction is then appended to the
        ///	end of the contraction list.<br></br>
        ///	 This list can then be iterated over until all
        ///	intermediate contractions are complete.
        /// </summary>
        /// <param name="subscripts">
        ///	Specifies the subscripts for summation.
        /// </param>
        /// <param name="operands">
        ///	These are the arrays for the operation.
        /// </param>
        /// <param name="optimize">
        ///	Choose the type of path.<br></br>
        ///	If a tuple is provided, the second argument is
        ///	assumed to be the maximum intermediate size created.<br></br>
        ///	If only a single
        ///	argument is provided the largest input or output array size is used
        ///	as a maximum intermediate size.<br></br>
        ///	
        ///	Default is ‘greedy’.
        /// </param>
        /// <returns>
        /// A tuple of:
        /// path
        ///	A list representation of the einsum path.
        /// string_repr
        ///	A printable representation of the einsum path.
        /// </returns>
        public static (list of tuples, string) einsum_path(string subscripts, NDarray[] operands, string optimize = "greedy")
        {
            //auto-generated code, do not change
            var __self__=self;
            var pyargs=ToTuple(new object[]
            {
                subscripts,
                operands,
            });
            var kwargs=new PyDict();
            if (optimize!="greedy") kwargs["optimize"]=ToPython(optimize);
            dynamic py = __self__.InvokeMethod("einsum_path", pyargs, kwargs);
            return (ToCsharp<list of tuples>(py[0]), ToCsharp<string>(py[1]));
        }
        */

        public static partial class linalg
        {
            /// <summary>
            ///     Raise a square matrix to the (integer) power n.<br></br>
            ///     For positive integers n, the power is computed by repeated matrix
            ///     squarings and matrix multiplications.<br></br>
            ///     If n == 0, the identity matrix
            ///     of the same shape as M is returned.<br></br>
            ///     If n &lt; 0, the inverse
            ///     is computed and then raised to the abs(n).
            /// </summary>
            /// <param name="a">
            ///     Matrix to be “powered.”
            /// </param>
            /// <param name="n">
            ///     The exponent can be any integer or long integer, positive,
            ///     negative, or zero.
            /// </param>
            /// <returns>
            ///     The return value is the same shape and type as M;
            ///     if the exponent is positive or zero then the type of the
            ///     elements is the same as those of M.<br></br>
            ///     If the exponent is
            ///     negative the elements are floating-point.
            /// </returns>
            public static NDarray matrix_power(NDarray a, int n)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(cp.linalg.matrix_power(a.CupyNDarray, n));
                }
                else
                {
                    return new NDarray(np.linalg.matrix_power(a.NumpyNDarray, n));
                }
            }
        }

        /*
        public static partial class linalg {
            /// <summary>
            ///	Compute the condition number of a matrix.<br></br>
            ///	
            ///	This function is capable of returning the condition number using
            ///	one of seven different norms, depending on the value of p (see
            ///	Parameters below).<br></br>
            ///	
            ///	Notes
            ///	
            ///	The condition number of x is defined as the norm of x times the
            ///	norm of the inverse of x [1]; the norm can be the usual L2-norm
            ///	(root-of-sum-of-squares) or one of a number of other matrix norms.<br></br>
            ///	
            ///	References
            /// </summary>
            /// <param name="x">
            ///	The matrix whose condition number is sought.
            /// </param>
            /// <param name="p">
            ///	Order of the norm:
            ///	
            ///	inf means the Cupy.inf object, and the Frobenius norm is
            ///	the root-of-sum-of-squares norm.
            /// </param>
            /// <returns>
            ///	The condition number of the matrix.<br></br>
            ///	 May be infinite.
            /// </returns>
            public static {float cond(NDarray x, {None p = null)
            {
                //auto-generated code, do not change
                var linalg = self.GetAttr("linalg");
                var __self__=linalg;
                var pyargs=ToTuple(new object[]
                {
                    x,
                });
                var kwargs=new PyDict();
                if (p!=null) kwargs["p"]=ToPython(p);
                dynamic py = __self__.InvokeMethod("cond", pyargs, kwargs);
                return ToCsharp<{float>(py);
            }
        }
        */

        public static partial class linalg
        {
            /// <summary>
            ///     Return matrix rank of array using SVD method
            ///     Rank of the array is the number of singular values of the array that are
            ///     greater than tol.<br></br>
            ///     Notes
            ///     The default threshold to detect rank deficiency is a test on the magnitude
            ///     of the singular values of M.<br></br>
            ///     By default, we identify singular values less
            ///     than S.max() * max(M.shape) * eps as indicating rank deficiency (with
            ///     the symbols defined above).<br></br>
            ///     This is the algorithm MATLAB uses [1].<br></br>
            ///     It also
            ///     appears in Numerical recipes in the discussion of SVD solutions for linear
            ///     least squares [2].<br></br>
            ///     This default threshold is designed to detect rank deficiency accounting for
            ///     the numerical errors of the SVD computation.<br></br>
            ///     Imagine that there is a column
            ///     in M that is an exact (in floating point) linear combination of other
            ///     columns in M.<br></br>
            ///     Computing the SVD on M will not produce a singular value
            ///     exactly equal to 0 in general: any difference of the smallest SVD value from
            ///     0 will be caused by numerical imprecision in the calculation of the SVD.<br></br>
            ///     Our threshold for small SVD values takes this numerical imprecision into
            ///     account, and the default threshold will detect such numerical rank
            ///     deficiency.<br></br>
            ///     The threshold may declare a matrix M rank deficient even if
            ///     the linear combination of some columns of M is not exactly equal to
            ///     another column of M but only numerically very close to another column of
            ///     M.<br></br>
            ///     We chose our default threshold because it is in wide use.<br></br>
            ///     Other thresholds
            ///     are possible.<br></br>
            ///     For example, elsewhere in the 2007 edition of Numerical
            ///     recipes there is an alternative threshold of S.max() *
            ///     cp.finfo(M.dtype).eps / 2.<br></br>
            ///     * cp.sqrt(m + n + 1.).<br></br>
            ///     The authors describe
            ///     this threshold as being based on “expected roundoff error” (p 71).<br></br>
            ///     The thresholds above deal with floating point roundoff error in the
            ///     calculation of the SVD.<br></br>
            ///     However, you may have more information about the
            ///     sources of error in M that would make you consider other tolerance values
            ///     to detect effective rank deficiency.<br></br>
            ///     The most useful measure of the
            ///     tolerance depends on the operations you intend to use on your matrix.<br></br>
            ///     For
            ///     example, if your data come from uncertain measurements with uncertainties
            ///     greater than floating point epsilon, choosing a tolerance near that
            ///     uncertainty may be preferable.<br></br>
            ///     The tolerance may be absolute if the
            ///     uncertainties are absolute rather than relative.<br></br>
            ///     References
            /// </summary>
            /// <param name="M">
            ///     input vector or stack of matrices
            /// </param>
            /// <param name="tol">
            ///     threshold below which SVD values are considered zero.<br></br>
            ///     If tol is
            ///     None, and S is an array with singular values for M, and
            ///     eps is the epsilon value for datatype of S, then tol is
            ///     set to S.max() * max(M.shape) * eps.
            /// </param>
            /// <param name="hermitian">
            ///     If True, M is assumed to be Hermitian (symmetric if real-valued),
            ///     enabling a more efficient method for finding singular values.<br></br>
            ///     Defaults to False.
            /// </param>
            public static int matrix_rank(NDarray M, NDarray tol = null, bool? hermitian = false)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return cp.linalg.matrix_rank(M.CupyNDarray, tol?.CupyNDarray, hermitian);
                }
                else
                {
                    return np.linalg.matrix_rank(M.NumpyNDarray, tol?.NumpyNDarray, hermitian);
                }
            }
        }

        public static partial class linalg
        {
            /// <summary>
            ///     Compute the sign and (natural) logarithm of the determinant of an array.<br></br>
            ///     If an array has a very small or very large determinant, then a call to
            ///     det may overflow or underflow.<br></br>
            ///     This routine is more robust against such
            ///     issues, because it computes the logarithm of the determinant rather than
            ///     the determinant itself.<br></br>
            ///     Notes
            ///     Broadcasting rules apply, see the Cupy.linalg documentation for
            ///     details.<br></br>
            ///     The determinant is computed via LU factorization using the LAPACK
            ///     routine z/dgetrf.
            /// </summary>
            /// <param name="a">
            ///     Input array, has to be a square 2-D array.
            /// </param>
            /// <returns>
            ///     A tuple of:
            ///     sign
            ///     A number representing the sign of the determinant. For a real matrix,
            ///     this is 1, 0, or -1. For a complex matrix, this is a complex number
            ///     with absolute value 1 (i.e., it is on the unit circle), or else 0.
            ///     logdet
            ///     The natural log of the absolute value of the determinant.
            /// </returns>
            public static (NDarray, NDarray) slogdet(NDarray a)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    var ret = cp.linalg.slogdet(a.CupyNDarray);
                    return (new NDarray(ret.Item1), new NDarray(ret.Item2));
                }
                else
                {
                    var ret = np.linalg.slogdet(a.NumpyNDarray);
                    return (new NDarray(ret.Item1), new NDarray(ret.Item2));
                }
            }
        }

        public static partial class linalg
        {
            /// <summary>
            ///     Solve the tensor equation a x = b for x.<br></br>
            ///     It is assumed that all indices of x are summed over in the product,
            ///     together with the rightmost indices of a, as is done in, for example,
            ///     tensordot(a, x, axis=b.ndim).
            /// </summary>
            /// <param name="a">
            ///     Coefficient tensor, of shape b.shape + Q.<br></br>
            ///     Q, a tuple, equals
            ///     the shape of that sub-tensor of a consisting of the appropriate
            ///     number of its rightmost indices, and must be such that
            ///     prod(Q) == prod(b.shape) (in which sense a is said to be
            ///     ‘square’).
            /// </param>
            /// <param name="b">
            ///     Right-hand tensor, which can be of any shape.
            /// </param>
            /// <param name="axes">
            ///     Axes in a to reorder to the right, before inversion.<br></br>
            ///     If None (default), no reordering is done.
            /// </param>
            public static NDarray tensorsolve(NDarray a, NDarray b, int[] axes = null)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(cp.linalg.tensorsolve(a.CupyNDarray, b.CupyNDarray, axes));
                }
                else
                {
                    return new NDarray(np.linalg.tensorsolve(a.NumpyNDarray, b.NumpyNDarray, axes));
                }
            }
        }

        public static partial class linalg
        {
            /// <summary>
            ///     Compute the ‘inverse’ of an N-dimensional array.<br></br>
            ///     The result is an inverse for a relative to the tensordot operation
            ///     tensordot(a, b, ind), i.<br></br>
            ///     e., up to floating-point accuracy,
            ///     tensordot(tensorinv(a), a, ind) is the “identity” tensor for the
            ///     tensordot operation.
            /// </summary>
            /// <param name="a">
            ///     Tensor to ‘invert’. Its shape must be ‘square’, i.<br></br>
            ///     e.,
            ///     prod(a.shape[:ind]) == prod(a.shape[ind:]).
            /// </param>
            /// <param name="ind">
            ///     Number of first indices that are involved in the inverse sum.<br></br>
            ///     Must be a positive integer, default is 2.
            /// </param>
            /// <returns>
            ///     a’s tensordot inverse, shape a.shape[ind:] + a.shape[:ind].
            /// </returns>
            public static NDarray tensorinv(NDarray a, int? ind = 2)
            {
                if (Gpu.Available && Gpu.Use)
                {
                    return new NDarray(cp.linalg.tensorinv(a.CupyNDarray, ind));
                }
                else
                {
                    return new NDarray(np.linalg.tensorinv(a.NumpyNDarray, ind));
                }
            }
        }

        public static partial class linalg
        {
            /// <summary>
            ///     Generic Python-exception-derived object raised by linalg functions.<br></br>
            ///     General purpose exception class, derived from Python’s exception.Exception
            ///     class, programmatically raised in linalg functions when a Linear
            ///     Algebra-related condition would prevent further correct execution of the
            ///     function.
            /// </summary>
            public static void LinAlgError()
            {
                if (Gpu.Available && Gpu.Use)
                {
                    cp.linalg.LinAlgError();
                }
                else
                {
                    np.linalg.LinAlgError();
                }
            }
        }
    }
}
