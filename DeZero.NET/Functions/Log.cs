using DeZero.NET.Core;
using DeZero.NET.Extensions;

namespace DeZero.NET.Functions
{
    public class Log : Function
    {
        public override Variable[] Forward(Params args)
        {
            var x = args.Get<Variable>(0);
            using var y = xp.log(x.Data.Value).ToVariable(this);
            return [y.copy().Relay(this)];
        }

        public override Variable[] Backward(Params args)
        {
            var gy = args.Get<Variable>(0);
            var x = Inputs.ElementAt(0).Variable;
            using var gx = gy / x;
            return [gx.copy()];
        }

        public static Variable[] Invoke(Variable x)
        {
            return new Log().Call(Params.New.SetPositionalArgs(x));
        }
    }
}
