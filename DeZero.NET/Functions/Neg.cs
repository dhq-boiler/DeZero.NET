using DeZero.NET.Core;

namespace DeZero.NET.Functions
{
    public class Neg : Function
    {

        public override Variable[] Forward(Params args)
        {
            return args.Through().Select(x => -x).ToArray();
        }

        public override Variable[] Backward(Params args)
        {
            return args.Through().Select(gy => -gy).ToArray();
        }

        public static Variable[] Invoke(Variable x)
        {
            return new Neg().Call(Params<Variable>.args(x));
        }
    }
}
