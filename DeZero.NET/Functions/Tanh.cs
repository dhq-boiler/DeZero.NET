using DeZero.NET.Core;
using DeZero.NET.Extensions;

namespace DeZero.NET.Functions
{
    public class Tanh : Function
    {
        public override Variable[] Forward(Params args)
        {
            var xp = args.Get<Variable>(0).Data.Value;
            var y = xp.tanh();
            return [y.ToVariable(this)];
        }


        public override Variable[] Backward(Params args)
        {
            var gy = args.Get<Variable>(0).Data.Value;
            var y = Outputs.ElementAt(0).Data.Value;
            var gx = gy * y;
            return [gx.ToVariable()];
        }

        public static Variable[] Invoke(Variable x)
        {
            return new Tanh().Call(Params.New.SetPositionalArgs(x));
        }
    }
}
