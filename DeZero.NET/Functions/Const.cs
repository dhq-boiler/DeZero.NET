using DeZero.NET.Core;
using DeZero.NET.Extensions;

namespace DeZero.NET.Functions
{
    public class Const : Function
    {
        public override Variable[] Forward(Params args)
        {
            var y = args.Get<Variable>(0);
            return [y.Relay(this)];
        }

        public override Variable[] Backward(Params args)
        {
            var gx = xp.zeros(1);
            return [gx.ToVariable()];
        }

        public static Variable[] Invoke(float x0)
        {
            var constant = xp.array(x0).ToVariable();
            var y = new Const().Call(Params.New.SetPositionalArgs(constant));
            return y;
        }
    }
}
