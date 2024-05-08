using DeZero.NET.Core;

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
            var x = args.Get<Variable>("x").Data; // 仮定: 入力は単一のVariableオブジェクト
            var y = x.transpose(Axes is not null ? Axes.SelectMany(ax => ax.Axes).ToArray() : null); // xp.Transposeを使用して行列を転置
            return [y.ToVariable(this)];
        }

        public override Variable[] Backward(Params args)
        {
            var gys = args.Through;
            if (Axes is null)
            {
                return Invoke(gys[0].Variable);
            }

            var axes_len = Math.Max(Axes.Length, Axes[0].Axes.Length);
            var inv_axes = Axes.SelectMany(axe => axe.Axes).Select(ax => ax % axes_len).OrderBy(v => v).ToArray();
            return gys.Select(gy => new Variable(xp.transpose(gy.Variable.Data, inv_axes))).ToArray();
        }

        public static Variable[] Invoke(Variable x, Axis[] axes = null)
        {
            return new Transpose(axes).Call(Params.New.SetKeywordArg(x));
        }
    }
}
