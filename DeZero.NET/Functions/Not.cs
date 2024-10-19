using DeZero.NET.Core;
using DeZero.NET.Extensions;

namespace DeZero.NET.Functions
{
    public class Not : Function
    {
        private Variable _x;

        public override Variable[] Forward(Params args)
        {
            _x = args.Get<Variable>(0);

            // xpライブラリにLogicalNotが実装されていない場合は、
            // 以下のように実装することもできます。
            // var y = xp.Equal(_x.Data, xp.ZerosLike(_x.Data));
            var y = xp.logical_not(_x.Data.Value);
            return [y.ToVariable(this)];
        }

        public override Variable[] Backward(Params args)
        {
            // Not は勾配を持たない操作なので、
            // 入力と同じ形状のゼロ行列を返します。
            var gx = xp.zeros_like(_x.Data.Value).ToVariable();

            return new[] { gx };
        }

        public static (Variable[], Function) Invoke(Variable x)
        {
            var f = new Not();
            var y = f.Call(Params.New.SetPositionalArgs(x));
            return (y, f);
        }
    }
}
