using DeZero.NET.Core;
using DeZero.NET.Extensions;

namespace DeZero.NET.Functions
{
    public class Transpose : Function
    {
        public Axis[] Axes { get; set; }

        public Transpose(params Axis[] axes)
        {
            Axes = axes;
        }

        public override Variable[] Forward(Params args)
        {
            var x = args.Get<Variable>(0).Data.Value; // 仮定: 入力は単一のVariableオブジェクト
            var y = x.transpose(Axes is not null ? Axes.SelectMany(ax => ax.Axes).ToArray() : null); // xp.Transposeを使用して行列を転置
            return [y.Relay(this)];
        }

        public override Variable[] Backward(Params args)
        {
            var gys = args.Through;

            // If no axes specified, return transpose of gradient
            if (Axes is null || Axes.Length == 0)
            {
                return Invoke(gys[0].Variable);
            }

            // Get the permutation axes used in forward pass
            var forwardAxes = Axes.SelectMany(ax => ax.Axes).ToArray();

            // Create inverse permutation
            var axesLength = forwardAxes.Length;
            var inverseAxes = new int[axesLength];

            // For each position i in forward axes, store i at the position indicated by axes[i]
            for (int i = 0; i < axesLength; i++)
            {
                inverseAxes[forwardAxes[i]] = i;
            }

            // Apply inverse permutation to gradients
            return gys.Select(gy =>
            {
                var transposed = xp.transpose(gy.Variable.Data.Value, inverseAxes);
                return new Variable(transposed);
            }).ToArray();
        }

        public static Variable[] Invoke(Variable x, Axis[] axes = null)
        {
            return new Transpose(axes).Call(Params.New.SetKeywordArg(x));
        }
    }
}
