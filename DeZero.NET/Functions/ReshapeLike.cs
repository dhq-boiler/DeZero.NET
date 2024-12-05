using DeZero.NET.Core;
using DeZero.NET.Extensions;

namespace DeZero.NET.Functions
{
    public class ReshapeLike : Function
    {
        public override Variable[] Forward(Params args)
        {
            var x = args.Get<Variable>(0);
            var y = args.Get<Variable>(1);
            using var y_shape = y.Data.Value.shape;
            var z = x.Data.Value.reshape(y_shape).ToVariable(this);
            return [z.Relay(this)];
        }

        public override Variable[] Backward(Params args)
        {
            var gx = args.Get<Variable>(0);
            using var input_0_shape = Inputs.ElementAt(0).NDarray.shape;
            return gx.reshape(input_0_shape);
        }

        public static Variable[] Invoke(Variable x, Variable y)
        {
            return new ReshapeLike().Call(Params.New.SetPositionalArgs(x, y));
        }
    }
}
