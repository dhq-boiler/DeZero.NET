using DeZero.NET.Core;
using DeZero.NET.Extensions;

namespace DeZero.NET.Functions
{
    public class Sigmoid : Function
    {
        public override Variable[] Forward(Params args)
        {
            var x = args.Get<Variable>("x");
            var y = (xp.tanh(x.Data.Value.copy() * 0.5) * 0.5 + 0.5).ToVariable(this);
            return [y];
        }

        public override Variable[] Backward(Params args)
        {
            var gy = args.Get<Variable>(0);
            var y = Outputs.ElementAt(0);
            var gx = gy * y * (-y + new NDarray(1));
            return [gx];
        }

        public static Variable[] Invoke(Variable x)
        {
            return new Sigmoid().Call(Params.New.SetPositionalArgs(x));
        }
    }
}
