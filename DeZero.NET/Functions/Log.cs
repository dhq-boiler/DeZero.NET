using DeZero.NET.Core;
using DeZero.NET.Extensions;

namespace DeZero.NET.Functions
{
    public class Log : Function
    {
        public override Variable[] Forward(Params args)
        {
            var x = args.Get<Variable>(0);
            var y = xp.log(x.Data.Value).ToVariable(this);
            return [y];
        }

        public override Variable[] Backward(Params args)
        {
            var gy = args.Get<Variable>(0);
            var x = Inputs.ElementAt(0).Variable;
            var gx = gy / x;
            return [gx];
        }

        public static Variable[] Invoke(Variable x)
        {
            return new Log().Call(Params.New.SetPositionalArgs(x));
        }
    }
}
