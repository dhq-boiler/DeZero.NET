using DeZero.NET.Core;

namespace DeZero.NET.Functions
{
    public class LeakyRelu : Function
    {
        public double Slope { get; set; }

        public LeakyRelu(double slope = 0.2)
        {
            Slope = slope;
        }

        public override Variable[] Forward(Params args)
        {
            var x = args.Get<Variable>("x");
            var y = x.Data.Value.copy().ToVariable(this);
            y.Data.Value[x.Data.Value <= 0f] *= Slope;
            return [y];
        }

        public override Variable[] Backward(Params args)
        {
            var gy = args.Get<Variable>(0);
            var x = Inputs.ElementAt(0).Variable;
            var mask = (x.Data.Value > 0f).astype(gy.Dtype);
            mask[mask <= 0f] = new NDarray(Slope);
            var gx = gy * mask;
            return [gx];
        }

        public static Variable[] Invoke(Variable x, double slope = 0.2)
        {
            return new LeakyRelu(slope).Call(Params.New.SetPositionalArgs(x));
        }
    }
}
