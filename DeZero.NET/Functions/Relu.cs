using DeZero.NET.Core;

namespace DeZero.NET.Functions
{
    public class ReLU : Function
    {
        public override Variable[] Forward(Params args)
        {
            var x = args.Get<Variable>("x");
            var y = xp.maximum(x.Data.Value, new NDarray(0.0f)).ToVariable(this);
            return [y];
        }

        public override Variable[] Backward(Params args)
        {
            var gy = args.Get<Variable>(0);
            var x = Inputs.ElementAt(0).Variable;
            var mask = x.Data.Value > 0;
            var gx = gy * mask;
            return [gx];
        }

        public static Variable[] Invoke(Variable x)
        {
            return new ReLU().Call(Params.New.SetPositionalArgs(x));
        }
    }
}