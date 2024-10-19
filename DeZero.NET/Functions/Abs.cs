using DeZero.NET.Core;
using DeZero.NET.Extensions;

namespace DeZero.NET.Functions
{
    public class Abs : Function
    {
        public override Variable[] Forward(Params args)
        {
            var x = args.Get<Variable>(0);
            var y = x.Data.Value.abs();
            return new Variable[] { y.ToVariable(this) };
        }

        public override Variable[] Backward(Params args)
        {
            var gy = args.Get<Variable>(0);
            var x = Inputs.ElementAt(0).Variable;
            var gx = gy * Sign.Invoke(x)[0];
            return new Variable[] { gx };
        }

        public static Variable[] Invoke(Variable x)
        {
            return new Abs().Call(Params.New.SetPositionalArgs(x));
        }
    }
}
