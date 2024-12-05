using DeZero.NET.Core;
using DeZero.NET.Extensions;

namespace DeZero.NET.Functions
{
    public class Neg : Function
    {

        public override Variable[] Forward(Params args)
        {
            return args.Through.Select(x => (-(x.Variable.Data.Value)).Relay(this)).ToArray();
        }

        public override Variable[] Backward(Params args)
        {
            return args.Through.Select(gy => (-(gy.Variable.Data.Value)).ToVariable(this)).ToArray();
        }

        public static Variable[] Invoke(Variable x)
        {
            return new Neg().Call(Params.New.SetPositionalArgs(x));
        }
    }
}
