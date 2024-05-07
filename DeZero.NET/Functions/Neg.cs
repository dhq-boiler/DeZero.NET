using DeZero.NET.Core;

namespace DeZero.NET.Functions
{
    public class Neg : Function
    {

        public override Variable[] Forward(Params args)
        {
            return args.Through().Select(x => -x.Variable).ToArray();
        }

        public override Variable[] Backward(Params args)
        {
            return args.Through().Select(gy => -gy.Variable).ToArray();
        }

        public static Variable[] Invoke(Variable x)
        {
            return new Neg().Call(Params.New.SetPositionalArgs(x));
        }
    }
}
