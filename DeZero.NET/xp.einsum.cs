﻿using Cupy;
using Numpy;

namespace DeZero.NET
{
    public static partial class xp
    {
        /// <summary>
        ///     Evaluates the Einstein summation convention on the operands.<br></br>
        ///     Using the Einstein summation convention, many common multi-dimensional,
        ///     linear algebraic array operations can be represented in a simple fashion.<br></br>
        ///     In implicit mode einsum computes these values.<br></br>
        ///     In explicit mode, einsum provides further flexibility to compute
        ///     other array operations that might not be considered classical Einstein
        ///     summation operations, by disabling, or forcing summation over specified
        ///     subscript labels.<br></br>
        ///     See the notes and examples for clarification.<br></br>
        ///     Notes
        ///     The Einstein summation convention can be used to compute
        ///     many multi-dimensional, linear algebraic array operations.<br></br>
        ///     einsum
        ///     provides a succinct way of representing these.<br></br>
        ///     A non-exhaustive list of these operations,
        ///     which can be computed by einsum, is shown below along with examples:
        ///     The subscripts string is a comma-separated list of subscript labels,
        ///     where each label refers to a dimension of the corresponding operand.<br></br>
        ///     Whenever a label is repeated it is summed, so cp.einsum('i,i', a, b)
        ///     is equivalent to cp.inner(a,b).<br></br>
        ///     If a label
        ///     appears only once, it is not summed, so cp.einsum('i', a) produces a
        ///     view of a with no changes.<br></br>
        ///     A further example cp.einsum('ij,jk', a, b)
        ///     describes traditional matrix multiplication and is equivalent to
        ///     cp.matmul(a,b).<br></br>
        ///     Repeated subscript labels in one
        ///     operand take the diagonal.<br></br>
        ///     For example, cp.einsum('ii', a) is equivalent
        ///     to cp.trace(a).<br></br>
        ///     In implicit mode, the chosen subscripts are important
        ///     since the axes of the output are reordered alphabetically.<br></br>
        ///     This
        ///     means that cp.einsum('ij', a) doesn’t affect a 2D array, while
        ///     cp.einsum('ji', a) takes its transpose.<br></br>
        ///     Additionally,
        ///     cp.einsum('ij,jk', a, b) returns a matrix multiplication, while,
        ///     cp.einsum('ij,jh', a, b) returns the transpose of the
        ///     multiplication since subscript ‘h’ precedes subscript ‘i’.
        ///     In explicit mode the output can be directly controlled by
        ///     specifying output subscript labels.<br></br>
        ///     This requires the
        ///     identifier ‘-&gt;’ as well as the list of output subscript labels.<br></br>
        ///     This feature increases the flexibility of the function since
        ///     summing can be disabled or forced when required.<br></br>
        ///     The call
        ///     cp.einsum('i-&gt;', a) is like cp.sum(a, axis=-1),
        ///     and cp.einsum('ii-&gt;i', a) is like cp.diag(a).<br></br>
        ///     The difference is that einsum does not allow broadcasting by default.<br></br>
        ///     Additionally cp.einsum('ij,jh-&gt;ih', a, b) directly specifies the
        ///     order of the output subscript labels and therefore returns matrix
        ///     multiplication, unlike the example above in implicit mode.<br></br>
        ///     To enable and control broadcasting, use an ellipsis.<br></br>
        ///     Default
        ///     Cupy-style broadcasting is done by adding an ellipsis
        ///     to the left of each term, like cp.einsum('...ii-&gt;...i', a).<br></br>
        ///     To take the trace along the first and last axes,
        ///     you can do cp.einsum('i...i', a), or to do a matrix-matrix
        ///     product with the left-most indices instead of rightmost, one can do
        ///     cp.einsum('ij...,jk...-&gt;ik...', a, b).<br></br>
        ///     When there is only one operand, no axes are summed, and no output
        ///     parameter is provided, a view into the operand is returned instead
        ///     of a new array.<br></br>
        ///     Thus, taking the diagonal as cp.einsum('ii-&gt;i', a)
        ///     produces a view (changed in version 1.10.0).<br></br>
        ///     einsum also provides an alternative way to provide the subscripts
        ///     and operands as einsum(op0, sublist0, op1, sublist1, ..., [sublistout]).<br></br>
        ///     If the output shape is not provided in this format einsum will be
        ///     calculated in implicit mode, otherwise it will be performed explicitly.<br></br>
        ///     The examples below have corresponding einsum calls with the two
        ///     parameter methods.<br></br>
        ///     Views returned from einsum are now writeable whenever the input array
        ///     is writeable.<br></br>
        ///     For example, cp.einsum('ijk...-&gt;kji...', a) will now
        ///     have the same effect as cp.swapaxes(a, 0, 2)
        ///     and cp.einsum('ii-&gt;i', a) will return a writeable view of the diagonal
        ///     of a 2D array.<br></br>
        ///     Added the optimize argument which will optimize the contraction order
        ///     of an einsum expression.<br></br>
        ///     For a contraction with three or more operands this
        ///     can greatly increase the computational efficiency at the cost of a larger
        ///     memory footprint during computation.<br></br>
        ///     Typically a ‘greedy’ algorithm is applied which empirical tests have shown
        ///     returns the optimal path in the majority of cases.<br></br>
        ///     In some cases ‘optimal’
        ///     will return the superlative path through a more expensive, exhaustive search.<br></br>
        ///     For iterative calculations it may be advisable to calculate the optimal path
        ///     once and reuse that path by supplying it as an argument.<br></br>
        ///     An example is given
        ///     below.<br></br>
        ///     See Cupy.einsum_path for more details.
        /// </summary>
        /// <param name="subscripts">
        ///     Specifies the subscripts for summation as comma separated list of
        ///     subscript labels.<br></br>
        ///     An implicit (classical Einstein summation)
        ///     calculation is performed unless the explicit indicator ‘-&gt;’ is
        ///     included as well as subscript labels of the precise output form.
        /// </param>
        /// <param name="operands">
        ///     These are the arrays for the operation.
        /// </param>
        /// <param name="out">
        ///     If provided, the calculation is done into this array.
        /// </param>
        /// <param name="dtype">
        ///     If provided, forces the calculation to use the data type specified.<br></br>
        ///     Note that you may have to also give a more liberal casting
        ///     parameter to allow the conversions.<br></br>
        ///     Default is None.
        /// </param>
        /// <param name="order">
        ///     Controls the memory layout of the output.<br></br>
        ///     ‘C’ means it should
        ///     be C contiguous.<br></br>
        ///     ‘F’ means it should be Fortran contiguous,
        ///     ‘A’ means it should be ‘F’ if the inputs are all ‘F’, ‘C’ otherwise.<br></br>
        ///     ‘K’ means it should be as close to the layout as the inputs as
        ///     is possible, including arbitrarily permuted axes.<br></br>
        ///     Default is ‘K’.
        /// </param>
        /// <param name="casting">
        ///     Controls what kind of data casting may occur.<br></br>
        ///     Setting this to
        ///     ‘unsafe’ is not recommended, as it can adversely affect accumulations.<br></br>
        ///     Default is ‘safe’.
        /// </param>
        /// <param name="optimize">
        ///     Controls if intermediate optimization should occur.<br></br>
        ///     No optimization
        ///     will occur if False and True will default to the ‘greedy’ algorithm.<br></br>
        ///     Also accepts an explicit contraction list from the cp.einsum_path
        ///     function.<br></br>
        ///     See cp.einsum_path for more details.<br></br>
        ///     Defaults to False.
        /// </param>
        /// <returns>
        ///     The calculation based on the Einstein summation convention.
        /// </returns>
        public static NDarray einsum(string subscripts, NDarray[] operands, NDarray @out = null, Dtype dtype = null,
            string order = null, string casting = "safe", object optimize = null)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.einsum(subscripts, operands.Select(x => x.CupyNDarray).ToArray(), @out?.CupyNDarray, dtype?.CupyDtype, order, casting,
                    optimize));
            }
            else
            {
                return new NDarray(np.einsum(subscripts, operands.Select(x => x.NumpyNDarray).ToArray(), @out?.NumpyNDarray, dtype?.NumpyDtype, order, casting,
                    optimize));
            }
        }

        /// <summary>
        ///     Evaluates the Einstein summation convention on the operands.<br></br>
        ///     Using the Einstein summation convention, many common multi-dimensional,
        ///     linear algebraic array operations can be represented in a simple fashion.<br></br>
        ///     In implicit mode einsum computes these values.<br></br>
        ///     In explicit mode, einsum provides further flexibility to compute
        ///     other array operations that might not be considered classical Einstein
        ///     summation operations, by disabling, or forcing summation over specified
        ///     subscript labels.<br></br>
        ///     See the notes and examples for clarification.<br></br>
        ///     Notes
        ///     The Einstein summation convention can be used to compute
        ///     many multi-dimensional, linear algebraic array operations.<br></br>
        ///     einsum
        ///     provides a succinct way of representing these.<br></br>
        ///     A non-exhaustive list of these operations,
        ///     which can be computed by einsum, is shown below along with examples:
        ///     The subscripts string is a comma-separated list of subscript labels,
        ///     where each label refers to a dimension of the corresponding operand.<br></br>
        ///     Whenever a label is repeated it is summed, so cp.einsum('i,i', a, b)
        ///     is equivalent to cp.inner(a,b).<br></br>
        ///     If a label
        ///     appears only once, it is not summed, so cp.einsum('i', a) produces a
        ///     view of a with no changes.<br></br>
        ///     A further example cp.einsum('ij,jk', a, b)
        ///     describes traditional matrix multiplication and is equivalent to
        ///     cp.matmul(a,b).<br></br>
        ///     Repeated subscript labels in one
        ///     operand take the diagonal.<br></br>
        ///     For example, cp.einsum('ii', a) is equivalent
        ///     to cp.trace(a).<br></br>
        ///     In implicit mode, the chosen subscripts are important
        ///     since the axes of the output are reordered alphabetically.<br></br>
        ///     This
        ///     means that cp.einsum('ij', a) doesn’t affect a 2D array, while
        ///     cp.einsum('ji', a) takes its transpose.<br></br>
        ///     Additionally,
        ///     cp.einsum('ij,jk', a, b) returns a matrix multiplication, while,
        ///     cp.einsum('ij,jh', a, b) returns the transpose of the
        ///     multiplication since subscript ‘h’ precedes subscript ‘i’.
        ///     In explicit mode the output can be directly controlled by
        ///     specifying output subscript labels.<br></br>
        ///     This requires the
        ///     identifier ‘-&gt;’ as well as the list of output subscript labels.<br></br>
        ///     This feature increases the flexibility of the function since
        ///     summing can be disabled or forced when required.<br></br>
        ///     The call
        ///     cp.einsum('i-&gt;', a) is like cp.sum(a, axis=-1),
        ///     and cp.einsum('ii-&gt;i', a) is like cp.diag(a).<br></br>
        ///     The difference is that einsum does not allow broadcasting by default.<br></br>
        ///     Additionally cp.einsum('ij,jh-&gt;ih', a, b) directly specifies the
        ///     order of the output subscript labels and therefore returns matrix
        ///     multiplication, unlike the example above in implicit mode.<br></br>
        ///     To enable and control broadcasting, use an ellipsis.<br></br>
        ///     Default
        ///     Cupy-style broadcasting is done by adding an ellipsis
        ///     to the left of each term, like cp.einsum('...ii-&gt;...i', a).<br></br>
        ///     To take the trace along the first and last axes,
        ///     you can do cp.einsum('i...i', a), or to do a matrix-matrix
        ///     product with the left-most indices instead of rightmost, one can do
        ///     cp.einsum('ij...,jk...-&gt;ik...', a, b).<br></br>
        ///     When there is only one operand, no axes are summed, and no output
        ///     parameter is provided, a view into the operand is returned instead
        ///     of a new array.<br></br>
        ///     Thus, taking the diagonal as cp.einsum('ii-&gt;i', a)
        ///     produces a view (changed in version 1.10.0).<br></br>
        ///     einsum also provides an alternative way to provide the subscripts
        ///     and operands as einsum(op0, sublist0, op1, sublist1, ..., [sublistout]).<br></br>
        ///     If the output shape is not provided in this format einsum will be
        ///     calculated in implicit mode, otherwise it will be performed explicitly.<br></br>
        ///     The examples below have corresponding einsum calls with the two
        ///     parameter methods.<br></br>
        ///     Views returned from einsum are now writeable whenever the input array
        ///     is writeable.<br></br>
        ///     For example, cp.einsum('ijk...-&gt;kji...', a) will now
        ///     have the same effect as cp.swapaxes(a, 0, 2)
        ///     and cp.einsum('ii-&gt;i', a) will return a writeable view of the diagonal
        ///     of a 2D array.<br></br>
        ///     Added the optimize argument which will optimize the contraction order
        ///     of an einsum expression.<br></br>
        ///     For a contraction with three or more operands this
        ///     can greatly increase the computational efficiency at the cost of a larger
        ///     memory footprint during computation.<br></br>
        ///     Typically a ‘greedy’ algorithm is applied which empirical tests have shown
        ///     returns the optimal path in the majority of cases.<br></br>
        ///     In some cases ‘optimal’
        ///     will return the superlative path through a more expensive, exhaustive search.<br></br>
        ///     For iterative calculations it may be advisable to calculate the optimal path
        ///     once and reuse that path by supplying it as an argument.<br></br>
        ///     An example is given
        ///     below.<br></br>
        ///     See Cupy.einsum_path for more details.
        /// </summary>
        /// <param name="subscripts">
        ///     Specifies the subscripts for summation as comma separated list of
        ///     subscript labels.<br></br>
        ///     An implicit (classical Einstein summation)
        ///     calculation is performed unless the explicit indicator ‘-&gt;’ is
        ///     included as well as subscript labels of the precise output form.
        /// </param>
        /// <param name="operands">
        ///     These are the arrays for the operation.
        /// </param>
        /// <param name="out">
        ///     If provided, the calculation is done into this array.
        /// </param>
        /// <param name="dtype">
        ///     If provided, forces the calculation to use the data type specified.<br></br>
        ///     Note that you may have to also give a more liberal casting
        ///     parameter to allow the conversions.<br></br>
        ///     Default is None.
        /// </param>
        /// <param name="order">
        ///     Controls the memory layout of the output.<br></br>
        ///     ‘C’ means it should
        ///     be C contiguous.<br></br>
        ///     ‘F’ means it should be Fortran contiguous,
        ///     ‘A’ means it should be ‘F’ if the inputs are all ‘F’, ‘C’ otherwise.<br></br>
        ///     ‘K’ means it should be as close to the layout as the inputs as
        ///     is possible, including arbitrarily permuted axes.<br></br>
        ///     Default is ‘K’.
        /// </param>
        /// <param name="casting">
        ///     Controls what kind of data casting may occur.<br></br>
        ///     Setting this to
        ///     ‘unsafe’ is not recommended, as it can adversely affect accumulations.<br></br>
        ///     Default is ‘safe’.
        /// </param>
        /// <param name="optimize">
        ///     Controls if intermediate optimization should occur.<br></br>
        ///     No optimization
        ///     will occur if False and True will default to the ‘greedy’ algorithm.<br></br>
        ///     Also accepts an explicit contraction list from the cp.einsum_path
        ///     function.<br></br>
        ///     See cp.einsum_path for more details.<br></br>
        ///     Defaults to False.
        /// </param>
        /// <returns>
        ///     The calculation based on the Einstein summation convention.
        /// </returns>
        public static NDarray einsum(string subscripts, params NDarray[] operands)
        {
            if (Gpu.Available && Gpu.Use)
            {
                return new NDarray(cp.einsum(subscripts, operands.Select(x => x.CupyNDarray).ToArray()));
            }
            else
            {
                return new NDarray(np.einsum(subscripts, operands.Select(x => x.NumpyNDarray).ToArray()));
            }
        }
    }
}
