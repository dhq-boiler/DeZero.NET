using DeZero.NET.Core;
using DeZero.NET.Extensions;

namespace DeZero.NET.Functions
{
    public class LayerNorm : Function
    {
        public NDarray<double> Mean { get; set; }
        public NDarray<double> Var { get; set; }
        public NDarray Std { get; set; }
        public float eps { get; }
        public Variable x { get; set; }
        public NDarray x_norm { get; set; }

        public LayerNorm(float eps = 1e-8f)
        {
            this.eps = eps;
        }


        public override Variable[] Forward(Params args)
        {
            this.x = args.Get<Variable>(0);
            this.Mean = xp.mean(x.Data.Value, new Axis(-1), keepdims: true);
            this.Var = xp.var(x.Data.Value, new Axis(-1), keepdims: true);
            this.Std = xp.sqrt(this.Var + this.eps);
            this.x_norm = (x.Data.Value - this.Mean) / this.Std;
            return [x_norm.Relay(this)];
        }

        public override Variable[] Backward(Params args)
        {
            var gy = args.Get<Variable>(0);
            var gx = gy / Std;
            var gvar = -0.5 * xp.sum((gy * (this.x - this.Mean)).Data.Value, new Axis(-1), keepdims: true) / (this.Std * this.Std * this.Std);
            var gmean = -xp.sum(gy.Data.Value / this.Std, new Axis(-1), keepdims: true) - 2 * gvar * xp.mean(this.x.Data.Value - this.Mean, new Axis(-1), keepdims: true);
            int N = gy.Shape[-1];
            gx += (2.0 / N) * gvar * (this.x.Data.Value - this.Mean) + (1.0 / N) * gmean;
            return [gx];
        }


        public static Variable Invoke(Variable x, float eps = 1e-8f)
        {
            return new LayerNorm(eps).Call(Params.New.SetPositionalArgs(x))[0];
        }
    }
}
