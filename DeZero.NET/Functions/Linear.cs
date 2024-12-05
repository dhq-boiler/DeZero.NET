using DeZero.NET.Core;
using DeZero.NET.Extensions;
using DeZero.NET.Log;

namespace DeZero.NET.Functions
{
    public class Linear : Function
    {
        private readonly ILogger _logger = new ConsoleLogger(LogLevel.Info, false);
        public override Variable[] Forward(Params args)
        {
            var x = args.Get<Variable>("x");
            var W = args.Get<Variable>("W");
            var b = args.Get<Variable>("b");

            try
            {
                // 1次元入力を2次元に変換
                if (x.ndim == 1)
                {
                    x = x.reshape(1, x.Shape[0])[0];
                }

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

                    return [y.Relay(this)];
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
            try
            {
                // 入力の次元チェック
                x = DimensionHelper.EnsureShape(x, 2, _logger);
                W = DimensionHelper.EnsureShape(W, 2, _logger);

                // 自動的にxとWの形状を判断して適切な計算を実行
                Variable y = default;
                if (x.Shape[1] == W.Shape[0])
                {
                    // 通常のケース: x.shape=(N,M), W.shape=(M,K) -> y.shape=(N,K)
                    y = x.Data.Value.dot(W.Data.Value).ToVariable(this);
                }
                else if (x.Shape[0] == W.Shape[0])
                {
                    // 転置が必要なケース: x.shape=(M,N), W.shape=(K,M) -> x.T.shape=(N,M) -> y.shape=(N,K) 
                    using var x_t = x.Data.Value.transpose();
                    y = x_t.dot(W.Data.Value).ToVariable(this);
                }
                else if (x.Shape[1] == W.Shape[1])
                {
                    // 転置が必要なケース: x.shape=(N,M), W.shape=(K,N) -> W.T.shape=(N,K) -> y.shape=(N,K)
                    using var w_t = W.Data.Value.transpose();
                    y = x.Data.Value.dot(w_t).ToVariable(this);
                }
                else if (x.Shape[0] == W.Shape[1])
                {
                    using var x_t = x.Data.Value.transpose();
                    using var w_t = W.Data.Value.transpose();
                    y = x_t.dot(w_t).ToVariable(this);
                }

                // バイアスの加算
                if (b?.Data.Value is not null)
                {
                    y += b;
                }

                return [y];
            }
            catch (Exception ex)
            {
                _logger.LogError($"Error in ProcessTwoDimensional: {ex.Message}");
                _logger.LogDebug($"Input shapes - x: {string.Join(",", x.Shape.Dimensions)}, W: {string.Join(",", W.Shape.Dimensions)}");
                throw;
            }
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
            var gy = gys[0].Variable;
            var x = Inputs.ElementAt(0).Variable;
            var W = Inputs.ElementAt(1);
            var b = Inputs.ElementAt(2);

            Variable gb = null;
            Variable gx = null;
            Variable gW = null;

            try
            {
                if (x.ndim == 2 && W.NDarray.ndim == 2)
                {
                    using var b_shape = b.Variable.Shape;

                    if (b?.Variable?.Data.Value is not null)
                    {
                        gb = SumTo.Invoke(gy, b_shape)[0];
                    }

                    var gyReshaped = gy;
                    if (gy.ndim == 1)
                    {
                        // バッチサイズ分の ones を作成
                        using var ones = xp.ones(x.Shape[0]);  // (32,)
                        var gyExpanded = gy.Data.Value.broadcast_to(new[] { x.Shape[0], gy.Shape[0] });  // (32, 3)
                        gyReshaped = gyExpanded.ToVariable(this);
                    }

                    gx = MatMul.Invoke(gyReshaped, W.Variable.T)[0];

                    gW = MatMul.Invoke(x.T, gyReshaped)[0];
                }
                else if (x.ndim == 3 && W.NDarray.ndim == 2)
                {
                    // 3次元入力の処理 - gy の reshape が必要
                    using var x_shape = x.Shape;
                    using var W_shape = W.NDarray.shape;
                    using var b_shape = b.Variable.Shape;

                    var batchSize = x_shape[0];
                    var seqLen = x_shape[1];
                    var inputSize = x_shape[2];
                    var outputSize = W_shape[1];

                    // gy を適切な形状に reshape
                    var gyReshaped = gy.Data.Value.reshape(new int[] { -1, outputSize });

                    if (b?.Variable?.Data.Value is not null)
                    {
                        gb = SumTo.Invoke(gy, b_shape)[0];
                    }

                    using var W_T = W.Variable.T;
                    var gxTemp = MatMul.Invoke(gyReshaped.ToVariable(this), W_T)[0];
                    gx = gxTemp.Data.Value.reshape(new int[] { batchSize, seqLen, inputSize }).ToVariable(this);

                    var xReshaped = x.Data.Value.reshape(new int[] { -1, inputSize });
                    gW = MatMul.Invoke(xReshaped.ToVariable(this).T, gyReshaped.ToVariable(this))[0];
                }
                else if (x.ndim == 4 && W.NDarray.ndim == 2)
                {
                    // 4次元入力の処理も同様に gy の reshape が必要
                    using var x_shape = x.Shape;
                    using var W_shape = W.NDarray.shape;
                    using var b_shape = b.Variable.Shape;

                    var batchSize = x_shape[0];
                    var channels = x_shape[1];
                    var height = x_shape[2];
                    var width = x_shape[3];
                    var inputSize = channels * height * width;
                    var outputSize = W_shape[1];

                    if (b?.Variable?.Data.Value is not null)
                    {
                        gb = SumTo.Invoke(gy, b_shape)[0];
                    }

                    using var W_T = W.Variable.T;
                    var gxTemp = MatMul.Invoke(gy, W_T)[0];
                    gx = gxTemp.Data.Value.reshape(new int[] { batchSize, channels, height, width }).ToVariable(this);

                    var xReshaped = x.Data.Value.reshape(new int[] { batchSize, inputSize });
                    gW = MatMul.Invoke(xReshaped.ToVariable(this).T, gy)[0];
                }
                else
                {
                    throw new ArgumentException($"Unsupported dimensions in backward pass - x: {x.ndim}D, W: {W.NDarray.ndim}D");
                }

                return new[] { gx, gW, gb };
            }
            catch (Exception ex)
            {
                using var x_shape = x.Shape;
                using var W_shape = W.NDarray.shape;
                using var gy_shape = gy.Shape;
                _logger.LogError($"Linear backward error: {ex.Message}");
                _logger.LogDebug($"Shapes - x: {string.Join("x", x_shape)}, W: {string.Join("x", W_shape)}, gy: {string.Join("x", gy_shape)}");
                throw;
            }
        }

        public static Variable[] Invoke(Variable x, Variable W, Variable b = null)
        {
            return new Linear().Call(Params.New.SetKeywordArg(x, W, b));
        }
    }
}
