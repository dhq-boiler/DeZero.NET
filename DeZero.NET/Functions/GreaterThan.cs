using DeZero.NET.Core;
using DeZero.NET.Extensions;

namespace DeZero.NET.Functions
{
    public class GreaterThan : Function
    {
        private Variable _x0;
        private Variable _x1;

        public override Variable[] Forward(Params args)
        {
            _x0 = args.Get<Variable>(0);
            _x1 = args.Get<Variable>(1);

            var y = xp.greater(_x0.Data.Value, _x1.Data.Value);
            return [y.ToVariable(this)];
        }

        public override Variable[] Backward(Params args)
        {
            // GreaterThan は勾配を持たない操作なので、
            // 入力と同じ形状のゼロ行列を返します。
            var gx0 = xp.zeros_like(_x0.Data.Value).ToVariable();
            var gx1 = xp.zeros_like(_x1.Data.Value).ToVariable();

            return new[] { gx0, gx1 };
        }

        public static (Variable[], Function) Invoke(Variable x0, Variable x1)
        {
            var f = new GreaterThan();
            var y = f.Call(Params.New.SetPositionalArgs(x0, x1));
            return (y, f);
        }
    }
}
