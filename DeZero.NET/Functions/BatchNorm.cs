﻿using System.Diagnostics;
using System.Xml.Linq;

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

        public BatchNorm(Variable mean, Variable var, double decay, double eps)
        {
            Mean = mean;
            Var = var;
            Decay = decay;
            Eps = eps;
        }

        public override Variable[] Forward(params Variable[] xs)
        {
            var x = xs[0];
            var gamma = xs[1];
            var beta = xs[2];
            Debug.Assert(x.ndim == 2 || x.ndim == 4);

            var x_ndim = x.ndim;
            if (x_ndim == 4)
            {
                var N = x.Shape.Dimensions[0];
                var C = x.Shape.Dimensions[1];
                var H = x.Shape.Dimensions[2];
                var W = x.Shape.Dimensions[3];
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
                var N = x.Shape.Dimensions[0];
                var C = x.Shape.Dimensions[1];
                var H = x.Shape.Dimensions[2];
                var W = x.Shape.Dimensions[3];
                y = y.reshape(N, H, W, C)[0].transpose(0, 3, 1, 2)[0];
            }

            return [y];
        }

        public override Variable[] Backward(params Variable[] gys)
        {
            var gy = gys[0];
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
            return new BatchNorm(mean, var, decay, eps).BaseForward(x, gamma, beta);
        }
    }
}
