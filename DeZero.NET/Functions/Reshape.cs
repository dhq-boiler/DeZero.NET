using DeZero.NET.Core;
using DeZero.NET.Extensions;
using DeZero.NET.Extentions;

namespace DeZero.NET.Functions
{
    public class Reshape : Function
    {
        public Shape Shape { get; set; }
        public Shape X_Shape { get; set; }

        public Reshape(Shape shape)
        {
            Shape = shape;
        }

        public override Variable[] Forward(Params args)
        {
            var x = args.Get<Variable>(0);
            X_Shape = x.Shape;
            var y = x.Data.Value.reshape(Shape);
            return [y.Relay(this)];
        }

        public override Variable[] Backward(Params args)
        {
            var gy = args.Through[0].Variable;

            // 入力と出力の要素数をチェック
            var input_size = X_Shape.Dimensions.Product();
            var grad_size = gy.Shape.Dimensions.Product();

            var gx = gy.Data.Value;
            if (input_size == grad_size)
            {
                // 要素数が同じ場合は単純にreshape
                gx = gx.reshape(X_Shape);
            }
            else
            {
                var mask = xp.array(gy.Shape.Dimensions) == xp.array(X_Shape.Dimensions);
                var maskArr = mask.GetData<bool[]>();
                // 要素数が異なる場合は総和を取る
                var axes = Enumerable.Range(0, gy.Shape.Dimensions.Length).ToArray();
                axes = axes.Where((_, i) => !maskArr[i]).ToArray();
                gx = gx.sum(new Axis(axes), keepdims: true);
                gx = gx.reshape(X_Shape);
            }

            return [gx.Relay(this)];
        }

        public static Variable[] Invoke(Variable x, Shape shape)
        {
            if (x.Shape == shape)
            {
                return [x];
            }
            return new Reshape(shape).Call(Params.New.SetKeywordArg(x));
        }
    }
}
