using DeZero.NET.Core;
using DeZero.NET.Extensions;

namespace DeZero.NET.Functions
{
    public class Flatten : Function
    {
        public Shape OriginalShape { get; set; }

        public override Variable[] Forward(Params args)
        {
            var x = args.Get<Variable>(0);
            this.OriginalShape = x.Shape;
            using var shape = new Shape(x.Shape[0], -1);
            using var result = Reshape.Invoke(x, shape)[0];
            return [result.copy().Relay(this)];
        }

        public override Variable[] Backward(Params args)
        {
            var gy = args.Get<Variable>(0);
            using var result = Reshape.Invoke(gy, this.OriginalShape)[0];
            return [result.copy()];
        }

        public static Variable Invoke(Variable x)
        {
            return new Flatten().Call(Params.New.SetPositionalArgs(x))[0];
        }
    }
}
