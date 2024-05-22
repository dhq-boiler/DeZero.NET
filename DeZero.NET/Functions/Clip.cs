using DeZero.NET.Core;

namespace DeZero.NET.Functions
{
    public class Clip : Function
    {
        public double x_max { get; set; }
        public double x_min { get; set; }

        public Clip(double x_min, double x_max)
        {
            this.x_min = x_min;
            this.x_max = x_max;
        }

        public override Variable[] Forward(Params args)
        {
            var x = args.Get<Variable>("x");
            var y = xp.clip(x.Data.Value, new NDarray(x_min), new NDarray(x_max)).ToVariable(this);
            return [y];
        }

        public override Variable[] Backward(Params args)
        {
            var gy = args.Get<Variable>(0);
            var x = Inputs.ElementAt(0).Variable;
            var mask = (x.Data.Value >= x_min) * (x.Data.Value <= x_max);
            var gx = gy * mask;
            return [gx];
        }

        public static Variable[] Invoke(Variable x, double min, double max)
        {
            return new Clip(min, max).Call(Params.New.SetPositionalArgs(x));
        }
    }
}
