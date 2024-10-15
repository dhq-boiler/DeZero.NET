using DeZero.NET.Core;

namespace DeZero.NET.Functions
{
    public class MeanAbsoluteError : Function
    {
        public override Variable[] Forward(Params args)
        {
            var x0 = args.Get<Variable>(0);
            var x1 = args.Get<Variable>(1);
            var diff = x0.Data.Value - x1.Data.Value;
            var y = Abs(diff).sum() / diff.len;
            return [y.ToVariable(this)];
        }

        public override Variable[] Backward(Params args)
        {
            var gy = args.Get<Variable>(0);
            var x0 = Inputs.ElementAt(0).Variable;
            var x1 = Inputs.ElementAt(1).Variable;
            var diff = x0.Data.Value - x1.Data.Value;
            var gx0 = gy * Sign(diff) * (1f / diff.len);
            var gx1 = -gx0;
            return [gx0, gx1];
        }

        private static NDarray Abs(NDarray x)
        {
            return xp.abs(x);
        }

        private static NDarray Sign(NDarray x)
        {
            return xp.sign(x);
        }

        public static Variable[] Invoke(Variable x0, Variable x1)
        {
            return new MeanAbsoluteError().Call(Params.New.SetPositionalArgs(x0, x1));
        }
    }
}
