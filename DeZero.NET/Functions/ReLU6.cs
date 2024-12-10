using DeZero.NET;
using DeZero.NET.Core;
using DeZero.NET.Extensions;

namespace DeZero.NET.Functions
{
    public class ReLU6 : Function
    {
        public override Variable[] Forward(Params args)
        {
            var x = args.Get<Variable>(0);
            var y = DeZero.NET.Functions.Minimum.Invoke(
                DeZero.NET.Functions.Maximum.Invoke(x, xp.array(0).ToVariable(this)).Item1[0],
                DeZero.NET.Functions.Const.Invoke(6)[0]
            ).Item1[0];

            return new[] { y.Relay(this), x.Relay(this) };
        }

        public override Variable[] Backward(Params args)
        {
            var x = Inputs.ElementAt(0).Variable;
            var gy = args.Through[0].Variable;
            using var zero = xp.array(0).ToVariable();
            using var six = xp.array(6).ToVariable();
            using var a = DeZero.NET.Functions.GreaterThan.Invoke(x, zero).Item1[0];
            using var b = DeZero.NET.Functions.LessThan.Invoke(x, six).Item1[0];
            using var mask = DeZero.NET.Functions.And.Invoke(a, b).Item1[0];
            using var gx = DeZero.NET.Functions.Mul.Invoke(gy, mask)[0];
            return [gx.copy()];
        }

        public static Variable[] Invoke(Variable x)
        {
            var f = new ReLU6();
            var y = f.Call(Params.New.SetPositionalArgs(x))[0];
            return [y];
        }
    }
}
