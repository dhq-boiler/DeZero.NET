using DeZero.NET.Core;
using DeZero.NET.Extensions;

namespace DeZero.NET.Functions
{
    public class Linear : Function
    {
        public override Variable[] Forward(Params args)
        {
            var x = args.Get<Variable>("x");
            var W = args.Get<Variable>("W");
            var b = args.Get<Variable>("b");

            try
            {
                if (x.ndim == 2 && W.ndim == 2)
                {
                    return ProcessTwoDimensional(x, W, b);
                }
                else if (x.ndim == 3 && W.ndim == 2)
                {
                    return ProcessThreeDimensional(x, W, b);
                }
                else if (x.ndim == 4 && W.ndim == 2)
                {
                    // 4次元入力を処理 (batch_size, channels, height, width)
                    var batchSize = x.Shape[0];
                    var channels = x.Shape[1];
                    var height = x.Shape[2];
                    var width = x.Shape[3];
                    var inputSize = channels * height * width;
                    var outputSize = W.Shape[1];

                    // 4次元を2次元に変換 (batch_size, channels * height * width)
                    var xReshaped = x.Data.Value.reshape([batchSize, inputSize]);

                    // 行列乗算
                    var y = xReshaped.dot(W.Data.Value);

                    if (b?.Data.Value is not null)
                    {
                        y += b.Data.Value;
                    }

                    return [y.ToVariable(this)];
                }
                else
                {
                    throw new ArgumentException($"Unsupported dimensions - x: {x.ndim}D, W: {W.ndim}D");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Linear forward error: {ex.Message}");
                Console.WriteLine($"Shapes - x: {string.Join("x", x.Shape)}, W: {string.Join("x", W.Shape)}");
                throw;
            }
        }

        private Variable[] ProcessTwoDimensional(Variable x, Variable W, Variable b)
        {
            var y = x.Data.Value.dot(W.Data.Value);
            if (b?.Data.Value is not null)
            {
                y += b.Data.Value;
            }
            return [y.ToVariable(this)];
        }

        private Variable[] ProcessThreeDimensional(Variable x, Variable W, Variable b)
        {
            var batchSize = x.Shape[0];
            var seqLen = x.Shape[1];
            var inputSize = x.Shape[2];
            var outputSize = W.Shape[1];

            var xReshaped = x.Data.Value.reshape([-1, inputSize]);
            var y = xReshaped.dot(W.Data.Value);

            if (b?.Data.Value is not null)
            {
                y += b.Data.Value;
            }

            y = y.reshape([batchSize, seqLen, outputSize]);
            return [y.ToVariable(this)];
        }

        public override Variable[] Backward(Params args)
        {
            var gys = args.Through;
            var gy = gys[0];
            var x = Inputs.ElementAt(0);
            var W = Inputs.ElementAt(1);
            var b = Inputs.ElementAt(2);

            Variable gb = null;
            Variable gx = null;
            Variable gW = null;

            try
            {
                if (x.NDarray.ndim == 2 && W.NDarray.ndim == 2)
                {
                    gb = b.Variable.Data.Value is null ? null : SumTo.Invoke(gy.Variable, b.Variable.Shape)[0];
                    gx = MatMul.Invoke(gy.Variable, W.Variable.T)[0];
                    gW = MatMul.Invoke(x.Variable.T, gy.Variable)[0];
                }
                else if (x.NDarray.ndim == 3 && W.NDarray.ndim == 2)
                {
                    var batchSize = x.NDarray.shape[0];
                    var seqLen = x.NDarray.shape[1];
                    var inputSize = x.NDarray.shape[2];
                    var outputSize = W.NDarray.shape[1];

                    var gyReshaped = gy.Variable.Data.Value.reshape(new int[] { -1, outputSize });
                    gb = b.Variable.Data.Value is null ? null : SumTo.Invoke(gy.Variable, b.Variable.Shape)[0];

                    var gxTemp = MatMul.Invoke(gyReshaped.ToVariable(this), W.Variable.T)[0];
                    gx = gxTemp.Data.Value.reshape(new int[] { batchSize, seqLen, inputSize }).ToVariable(this);

                    var xReshaped = x.Variable.Data.Value.reshape(new int[] { -1, inputSize });
                    gW = MatMul.Invoke(xReshaped.ToVariable(this).T, gyReshaped.ToVariable(this))[0];
                }
                else if (x.NDarray.ndim == 4 && W.NDarray.ndim == 2)
                {
                    var batchSize = x.NDarray.shape[0];
                    var channels = x.NDarray.shape[1];
                    var height = x.NDarray.shape[2];
                    var width = x.NDarray.shape[3];
                    var inputSize = channels * height * width;
                    var outputSize = W.NDarray.shape[1];

                    // Reshape gradient for bias
                    gb = b.Variable.Data.Value is null ? null : SumTo.Invoke(gy.Variable, b.Variable.Shape)[0];

                    // Calculate gradient for x
                    var gxTemp = MatMul.Invoke(gy.Variable, W.Variable.T)[0];
                    gx = gxTemp.Data.Value.reshape(new int[] { batchSize, channels, height, width }).ToVariable(this);

                    // Calculate gradient for W
                    var xReshaped = x.Variable.Data.Value.reshape(new int[] { batchSize, inputSize });
                    gW = MatMul.Invoke(xReshaped.ToVariable(this).T, gy.Variable)[0];
                }
                else
                {
                    throw new ArgumentException($"Unsupported dimensions in backward pass - x: {x.NDarray.ndim}D, W: {W.NDarray.ndim}D");
                }

                return new[] { gx, gW, gb };
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Linear backward error: {ex.Message}");
                Console.WriteLine($"Shapes - x: {string.Join("x", x.NDarray.shape)}, W: {string.Join("x", W.NDarray.shape)}, gy: {string.Join("x", gy.Variable.Shape)}");
                throw;
            }
        }

        public static Variable[] Invoke(Variable x, Variable W, Variable b = null)
        {
            return new Linear().Call(Params.New.SetKeywordArg(x, W, b));
        }
    }
}
