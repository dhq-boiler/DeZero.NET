using DeZero.NET.Core;
using DeZero.NET.Extensions;

namespace DeZero.NET.Functions
{
    public class Equal : Function
    {
        public override Variable[] Forward(Params args)
        {
            var x0 = args.Get<Variable>(0);
            var x1 = args.Get<Variable>(1);

            return [(x0.Data.Value == x1.Data.Value).Relay(this)];
        }

        public override Variable[] Backward(Params args)
        {
            // Equal関数は離散的な比較を行うため、勾配は0とする
            var gy0 = args.Through.ElementAt(0);
            var gy1 = args.Through.ElementAt(1);

            var z0 = xp.zeros_like(gy0.NDarray).ToVariable(this);
            var z1 = xp.zeros_like(gy1.NDarray).ToVariable(this);
            return [z0, z1];
        }

        public static Variable[] Invoke(Variable x0, Variable x1)
        {
            return new Equal().Call(Params.New.SetPositionalArgs(x0, x1));
        }
    }
}
