using System.Diagnostics;
using DeZero.NET.Core;

namespace DeZero.NET.Functions
{
    public class BatchNorm : Function
    {
        public Variable Mean { get; }
        public Variable Var { get; }
        public double Decay { get; }
        public double Eps { get; }

        private Variable AvgMean { get; set; } = new Variable(xp.array([0], xp.float32));
        private Variable AvgVar { get; set; } = new Variable(xp.array([0], xp.float32));
        private Variable InvStd { get; set; } = new Variable(xp.array([0], xp.float32));

        public BatchNorm(Func<Params, Variable[]> f)
            : base(f)
        { }

        public BatchNorm(ref Variable mean, ref Variable var, double decay, double eps)
        {
            AvgMean = mean;
            AvgVar = var;
            Decay = decay;
            Eps = eps;
            InvStd = null;
        }

        public override Variable[] Forward(Params args)
        {
            var x = args.Get<Variable>("x");
            var gamma = args.Get<Variable>("gamma");
            var beta = args.Get<Variable>("beta");
            Debug.Assert(x.ndim == 2 || x.ndim == 4);

            int N = 0;
            int C = 0;
            int H = 0;
            int W = 0;
            var x_ndim = x.ndim;
            if (x_ndim == 4)
            {
                N = x.Shape.Dimensions[0];
                C = x.Shape.Dimensions[1];
                H = x.Shape.Dimensions[2];
                W = x.Shape.Dimensions[3];
                x = x.transpose(0, 2, 3, 1)[0].reshape(-1, C)[0];
            }

            Variable xc;
            if (Config.Train)
            {
                var mean = x.Data.mean(axis: 0);
                var var = x.Data.var(axis: 0);
                var inv_std = 1f / xp.sqrt(var + Eps);
                xc = (x - mean) * inv_std;

                var m = (int)(x.size / gamma.size);
                var s = m - 1f > 1f ? m - 1f : 1f;
                var adjust = m / s;
                AvgMean *= Decay;
                AvgMean += (1 - Decay) * mean;
                AvgVar *= Decay;
                AvgVar += (1 - Decay) * adjust * var;
                InvStd = inv_std.ToVariable();
            }
            else
            {
                var inv_std = 1f / xp.sqrt(AvgVar.Data + Eps);
                xc = (x - AvgMean) * inv_std;
            }

            var y = gamma * xc + beta;

            if (x_ndim == 4)
            {
                y = y.reshape(N, H, W, C)[0].transpose(0, 3, 1, 2)[0];
            }

            return [y];
        }

        public override Variable[] Backward(Params args)
        {
            var gy = args.Get<Variable>("gy");
            var gy_ndim = gy.ndim;

            if (gy_ndim == 4)
            {
                var N = gy.Shape.Dimensions[0];
                var C = gy.Shape.Dimensions[1];
                var H = gy.Shape.Dimensions[2];
                var W = gy.Shape.Dimensions[3];
                gy = gy.transpose(0, 2, 3, 1)[0].reshape(-1, C)[0];
            }

            var x = Inputs.ElementAt(0);
            var gamma = Inputs.ElementAt(1);
            var beta = Inputs.ElementAt(2);
            var batch_size = gy.__len__;

            if (x.ndim == 4)
            {
                var N = x.Shape.Dimensions[0];
                var C = x.Shape.Dimensions[1];
                var H = x.Shape.Dimensions[2];
                var W = x.Shape.Dimensions[3];
                x = x.transpose(0, 2, 3, 1)[0].reshape(-1, C)[0];
            }

            var mean = x.Data.sum(axis: 0) / batch_size;
            var xc = (x - mean) * InvStd;

            var gbeta = Sum.Invoke(gy, axis: 0)[0];
            var ggamma = Sum.Invoke(xc * gy, axis: 0)[0];
            var gx = gy - gbeta / batch_size - xc * ggamma / batch_size;
            gx *= gamma * InvStd;

            if (gy_ndim == 4)
            {
                var N = gy.Shape.Dimensions[0];
                var C = gy.Shape.Dimensions[1];
                var H = gy.Shape.Dimensions[2];
                var W = gy.Shape.Dimensions[3];
                gx = gx.reshape(N, H, W, C)[0].transpose(0, 3, 1, 2)[0];
            }

            return [gx, ggamma, gbeta];
        }

        public static Variable[] Invoke(Variable x, Variable gamma, Variable beta, Variable mean, Variable var,
            double decay = 0.9, double eps = 2e-5)
        {
            var bn = new BatchNorm(ref mean, ref var, decay, eps);
            try
            {
                return bn.BaseForward(Params<Variable, Variable, Variable>.args(x, gamma, beta));
            }
            finally
            {
                mean.Data = bn.AvgMean.Data;
                var.Data = bn.AvgVar.Data;
            }
        }
    }
}
