using DeZero.NET.Core;

namespace DeZero.NET.Functions
{
    public class ReshapeLike : Function
    {
        public override Variable[] Forward(Params args)
        {
            var x = args.Get<Variable>(0);
            var y = args.Get<Variable>(1);
            var z = x.Data.Value.reshape(y.Data.Value.shape).ToVariable(this);
            return [z];
        }

        public override Variable[] Backward(Params args)
        {
            var gx = args.Get<Variable>(0);
            return gx.reshape(Inputs.ElementAt(0).NDarray.shape);
        }

        public static Variable[] Invoke(Variable x, Variable y)
        {
            return new ReshapeLike().Call(Params.New.SetPositionalArgs(x, y));
        }
    }
}
