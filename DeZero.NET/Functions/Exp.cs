using DeZero.NET.Core;
using DeZero.NET.Extensions;

namespace DeZero.NET.Functions
{
    public class Exp : Function
    {
        public override Variable[] Forward(Params args)
        {
            var x = args.Get<Variable>(0);
            var y = xp.exp(x.Data.Value).ToVariable(this);
            return [y];
        }

        public override Variable[] Backward(Params args)
        {
            var gy = args.Get<Variable>(0);
            var y = Outputs.ElementAt(0);
            var gx = gy * y;
            return [gx];
        }

        public static Variable[] Invoke(Variable x)
        {
            return new Exp().Call(Params.New.SetPositionalArgs(x));
        }
    }
}
