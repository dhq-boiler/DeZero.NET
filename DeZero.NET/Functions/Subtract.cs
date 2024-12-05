using DeZero.NET.Core;
using DeZero.NET.Extensions;

namespace DeZero.NET.Functions
{
    public class Subtract : Function
    {
        private Shape x0_shape;
        private Shape x1_shape;

        public Subtract()
        {
        }

        public override Variable[] Forward(Params args)
        {
            var xs = args.Through.Select(x => x.NDarray).ToArray();
            x0_shape = xs[0].shape;
            x1_shape = xs[1].shape;
            return [(xs[0] - xs[1]).copy().Relay(this, args.Through.Select(x => x.Variable).ToArray())];
        }

        public override Variable[] Backward(Params args)
        {
            var gy = args.Through[0].Variable;
            var gx0 = gy;
            var gx1 = -gy;

            // gx0の形状を調整
            if (x0_shape.Dimensions.Length < gx0.Shape.Dimensions.Length || x0_shape != gx0.Shape)
            {
                var axes0 = GetAxesForReduction(x0_shape, gx0.Shape);
                if (axes0.Any())
                {
                    gx0.Data.Value = gx0.Data.Value.sum(new Axis(axes0), keepdims: false);
                    gx0.Data.Value = gx0.Data.Value.reshape(x0_shape);
                }
            }

            // gx1の形状を調整
            if (x1_shape.Dimensions.Length < gx1.Shape.Dimensions.Length || x1_shape != gx1.Shape)
            {
                var axes1 = GetAxesForReduction(x1_shape, gx1.Shape);
                if (axes1.Any())
                {
                    gx1.Data.Value = gx1.Data.Value.sum(new Axis(axes1), keepdims: false);
                    gx1.Data.Value = gx1.Data.Value.reshape(x1_shape);
                }
            }

            return [gx0, gx1];
        }

        private int[] GetAxesForReduction(Shape inputShape, Shape gradShape)
        {
            var axes = new List<int>();

            // 入力のランクが勾配より小さい場合の処理
            for (int i = 0; i < gradShape.Dimensions.Length; i++)
            {
                // 入力の次元を超える軸、または同じ次元でもブロードキャストされた軸を特定
                if (i >= inputShape.Dimensions.Length ||
                    (i < inputShape.Dimensions.Length && inputShape.Dimensions[i] != gradShape.Dimensions[i]))
                {
                    axes.Add(i);
                }
            }

            return axes.ToArray();
        }

        public static Variable[] Invoke(Variable x0, Variable x1)
        {
            return new Subtract().Call(Params.New.SetPositionalArgs(x0, x1));
        }
    }
}
