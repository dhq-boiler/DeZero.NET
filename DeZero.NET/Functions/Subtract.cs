using DeZero.NET.Core;
using DeZero.NET.Extensions;

namespace DeZero.NET.Functions
{
    public class Subtract : Function
    {
        public Subtract()
        {
        }

        public override Variable[] Forward(Params args)
        {
            var xs = args.Through.Select(x => x.NDarray).ToArray();
            return [(xs[0] - xs[1]).ToVariable(this)];
        }

        public override Variable[] Backward(Params args)
        {
            var gy = args.Through[0].Variable;
            return [gy, -gy];
        }

        public static Variable[] Invoke(Variable x0, Variable x1)
        {
            return new Subtract().Call(Params.New.SetPositionalArgs(x0, x1));
        }
    }
}
