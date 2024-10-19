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

            // 形状の確認と修正
            if (x.ndim == 2 && W.ndim == 2)
            {
                // 2次元の場合の通常の行列乗算
                var y = x.Data.Value.dot(W.Data.Value);
                if (b?.Data.Value is not null)
                {
                    y += b.Data.Value;
                }
                return [y.ToVariable(this)];
            }
            else if (x.ndim == 3 && W.ndim == 2)
            {
                // 3次元入力の場合、バッチ処理として扱う
                var batchSize = x.Shape[0];
                var inputSize = x.Shape[2];
                var outputSize = W.Shape[1];

                // x を (batchSize * x.Shape[1], inputSize) に reshape
                var xReshaped = x.Data.Value.reshape(new int[] { -1, inputSize });
                var y = xReshaped.dot(W.Data.Value);

                if (b?.Data.Value is not null)
                {
                    y += b.Data.Value;
                }

                // 結果を元の3次元形状に戻す
                y = y.reshape(new int[] { batchSize, x.Shape[1], outputSize });

                return [y.ToVariable(this)];
            }
            else
            {
                throw new ArgumentException("Unsupported input dimensions for Linear function");
            }
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

            if (x.NDarray.ndim == 2 && W.NDarray.ndim == 2)
            {
                // 2D input case
                gb = b.Variable.Data.Value is null ? null : SumTo.Invoke(gy.Variable, b.Variable.Shape)[0];
                gx = MatMul.Invoke(gy.Variable, W.Variable.T)[0];
                gW = MatMul.Invoke(x.Variable.T, gy.Variable)[0];
            }
            else if (x.NDarray.ndim == 3 && W.NDarray.ndim == 2)
            {
                // 3D input case
                var batchSize = x.NDarray.shape[0];
                var seqLen = x.NDarray.shape[1];
                var inputSize = x.NDarray.shape[2];
                var outputSize = W.NDarray.shape[1];

                // Reshape gy to (batchSize * seqLen, outputSize)
                var gyReshaped = gy.Variable.Data.Value.reshape(new int[] { -1, outputSize });

                // Calculate gb
                gb = b.Variable.Data.Value is null ? null : SumTo.Invoke(gy.Variable, b.Variable.Shape)[0];

                // Calculate gx
                var gxTemp = MatMul.Invoke(gyReshaped.ToVariable(this), W.Variable.T)[0];
                gx = gxTemp.Data.Value.reshape(new int[] { batchSize, seqLen, inputSize }).ToVariable(this);

                // Calculate gW
                var xReshaped = x.Variable.Data.Value.reshape(new int[] { -1, inputSize });
                gW = MatMul.Invoke(xReshaped.ToVariable(this).T, gyReshaped.ToVariable(this))[0];
            }
            else
            {
                throw new ArgumentException("Unsupported input dimensions for Linear function backward pass");
            }

            return new[] { gx, gW, gb };
        }

        public static Variable[] Invoke(Variable x, Variable W, Variable b = null)
        {
            return new Linear().Call(Params.New.SetKeywordArg(x, W, b));
        }
    }
}
