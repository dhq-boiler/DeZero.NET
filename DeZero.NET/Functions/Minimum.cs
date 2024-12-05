using DeZero.NET.Core;
using DeZero.NET.Extensions;

namespace DeZero.NET.Functions
{
    public class Minimum : Function
    {
        private Variable _x0;
        private Variable _x1;

        public override Variable[] Forward(Params args)
        {
            _x0 = args.Get<Variable>(0);
            _x1 = args.Get<Variable>(1);

            var y = xp.minimum(_x0.Data.Value, _x1.Data.Value);
            return [y.Relay(this)];
        }

        public override Variable[] Backward(Params args)
        {
            var gy = args.Through[0].Variable;
            var gx0 = DeZero.NET.Functions.Where.Invoke(
                DeZero.NET.Functions.LessThanOrEqual.Invoke(_x0, _x1).Item1[0],
                gy,
                xp.zeros_like(gy.Data.Value).ToVariable()
            ).Item1[0];

            var gx1 = DeZero.NET.Functions.Where.Invoke(
                DeZero.NET.Functions.GreaterThan.Invoke(_x0, _x1).Item1[0],
                gy,
                xp.zeros_like(gy.Data.Value).ToVariable()
            ).Item1[0];

            return new[] { gx0, gx1 };
        }

        public static (Variable[], Function) Invoke(Variable x0, Variable x1)
        {
            var f = new Minimum();
            var y = f.Call(Params.New.SetPositionalArgs(x0, x1));
            return (y, f);
        }
    }
}
