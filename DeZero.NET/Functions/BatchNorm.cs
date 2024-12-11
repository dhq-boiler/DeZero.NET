using DeZero.NET.Core;
using DeZero.NET.Extensions;
using System.Diagnostics;

namespace DeZero.NET.Functions
{
    public class BatchNorm : Function
    {
        public Variable Mean { get; }
        public Variable Var { get; }
        public double Decay { get; set; }
        public double Eps { get; set; }

        public Variable AvgMean { get; set; } = new Variable(xp.array([0], xp.float32));
        public Variable AvgVar { get; set; } = new Variable(xp.array([0], xp.float32));
        public Variable InvStd { get; set; } = new Variable(xp.array([0], xp.float32));
        public Variable InitAvgMean { get; set; }
        public Variable InitAvgVar { get; set; }
        public Variable InitInvStd { get; set; }

        public Func<Params, Variable[]> f
        {
            get => _f;
            set => _f = value;
        }

        public BatchNorm()
        { }

        public BatchNorm(Func<Params, Variable[]> f)
            : base(f)
        { }

        public BatchNorm(double decay = 0.9, double eps = 2e-5)
        {
            Decay = decay;
            Eps = eps;
        }

        public void Initialize(Variable mean, Variable var)
        {
            if (mean is null) throw new ArgumentNullException(nameof(mean));
            if (var is null) throw new ArgumentNullException(nameof(var));

            AvgMean = new Variable(mean.Data.Value.copy());
            AvgVar = new Variable(var.Data.Value.copy());
            InitAvgMean = new Variable(mean.Data.Value.copy());
            InitAvgVar = new Variable(var.Data.Value.copy());
            InvStd = null;
        }

        public override Variable[] Call(Params args)
        {
            InvStd = null;
            return base.Call(args);
        }

        public override Variable[] Forward(Params args)
        {
            var x = args.Get<Variable>("x");
            if (x is null) throw new ArgumentNullException(nameof(x));

            var gamma = args.Get<Variable>("gamma");
            if (gamma is null) throw new ArgumentNullException(nameof(gamma));

            var beta = args.Get<Variable>("beta");
            if (beta is null) throw new ArgumentNullException(nameof(beta));

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
                x = x.transpose(0, 2, 3, 1)[0].reshape(new Shape(-1, C))[0];
            }

            Variable xc;
            if (Config.Train)
            {
                var mean = x.Data.Value.mean(axis: 0);
                var var = x.Data.Value.var(axis: 0);
                var inv_std = 1f / xp.sqrt(var + Eps);
                xc = (x - mean) * inv_std;

                var m = (int)(x.size / gamma.size);
                var s = m - 1f > 1f ? m - 1f : 1f;
                var adjust = m / s;
                AvgMean.Data.Value *= Decay;
                AvgMean.Data.Value += (1 - Decay) * mean;
                AvgVar.Data.Value *= Decay;
                AvgVar.Data.Value += (1 - Decay) * adjust * var;
                InvStd = inv_std.ToVariable();
            }
            else
            {
                var inv_std = 1f / xp.sqrt(AvgVar.Data.Value + Eps);
                xc = (x - AvgMean) * inv_std;
            }

            var y = gamma * xc + beta;

            if (x_ndim == 4)
            {
                y = y.reshape(new Shape(N, H, W, C))[0].transpose(0, 3, 1, 2)[0];
            }

            return [y.Relay(this)];
        }

        public override Variable[] Backward(Params args)
        {
            var gy = args.Get<Variable>(0);
            var gy_ndim = gy.ndim;

            var original_N = gy.Shape.Dimensions[0];
            var original_C = gy.Shape.Dimensions[1];
            var original_H = gy_ndim == 4 ? gy.Shape.Dimensions[2] : 1;
            var original_W = gy_ndim == 4 ? gy.Shape.Dimensions[3] : 1;

            if (gy_ndim == 4)
            {
                var N = gy.Shape.Dimensions[0];
                var C = gy.Shape.Dimensions[1];
                var H = gy.Shape.Dimensions[2];
                var W = gy.Shape.Dimensions[3];
                gy = gy.transpose(0, 2, 3, 1)[0].reshape(new Shape(-1, C))[0];
            }

            var x = Inputs.ElementAt(0).Variable;
            var gamma = Inputs.ElementAt(1).Variable;
            var beta = Inputs.ElementAt(2).Variable;
            var batch_size = gy.__len__;

            if (x.ndim == 4)
            {
                var N = x.Shape.Dimensions[0];
                var C = x.Shape.Dimensions[1];
                var H = x.Shape.Dimensions[2];
                var W = x.Shape.Dimensions[3];
                x = x.transpose(0, 2, 3, 1)[0].reshape(new Shape(-1, C))[0];
            }

            var mean = x.Data.Value.sum(axis: 0) / batch_size;
            var xc = (x - mean) * InvStd;

            var gbeta = Sum.Invoke(gy, axis: 0)[0];
            var ggamma = Sum.Invoke(xc * gy, axis: 0)[0];
            var gx = gy - gbeta / batch_size - xc * ggamma / batch_size;
            gx *= gamma * InvStd;

            if (gy_ndim == 4)
            {
                var N = original_N;
                var C = original_C;
                var H = original_H;
                var W = original_W;
                gx = gx.reshape(new Shape(N, H, W, C))[0].transpose(0, 3, 1, 2)[0];
            }

            return [gx, ggamma, gbeta];
        }

        public void InitParams(Variable mean, Variable var, Variable gamma, Variable beta)
        {
            // 平均値と分散の初期状態を保存
            AvgMean = new Variable(mean.Data.Value.copy());
            AvgVar = new Variable(var.Data.Value.copy());
            InitAvgMean = new Variable(mean.Data.Value.copy());
            InitAvgVar = new Variable(var.Data.Value.copy());

            Decay = 0.9;  // デフォルト値
            Eps = 2e-5;   // デフォルト値
            InvStd = null;
        }

        public void InitParams(Shape shape, Dtype dtype = null)
        {
            if (dtype is null) dtype = Dtype.float32;

            AvgMean = new Variable(xp.zeros(shape, dtype: dtype).copy());
            AvgVar = new Variable(xp.zeros(shape, dtype: dtype).copy());
            InitAvgMean = new Variable(xp.zeros(shape, dtype: dtype).copy());
            InitAvgVar = new Variable(xp.zeros(shape, dtype: dtype).copy());

            Decay = 0.9;
            Eps = 2e-5;
            InvStd = null;
        }

        public void InitParams(int size, Dtype dtype = null)
        {
            InitParams(new Shape(size), dtype);
        }


        public override void ResetParams()
        {
            if (InitAvgMean is not null && InitAvgVar is not null)
            {
                AvgMean = new Variable(InitAvgMean.Data.Value.copy());
                AvgVar = new Variable(InitAvgVar.Data.Value.copy());
            }
            InvStd = null;
        }

        public static (Variable[], BatchNorm) Invoke(Variable x, Variable gamma, Variable beta, Variable mean, Variable var,
            double decay = 0.9, double eps = 2e-5)
        {
            var bn = new BatchNorm(decay, eps);
            bn.Initialize(mean, var);

            try
            {
                return (bn.Call(Params.New.SetKeywordArg(x, gamma, beta, mean, var)), bn);
            }
            finally
            {
                if (mean != null && bn.AvgMean != null)
                    mean.Data.Value = bn.AvgMean.Data.Value;
                if (var != null && bn.AvgVar != null)
                    var.Data.Value = bn.AvgVar.Data.Value;
            }
        }

        public static Variable[] Invoke(BatchNorm bn, Variable x, Variable gamma, Variable beta, Variable mean, Variable var,
            double decay = 0.9, double eps = 2e-5)
        {
            //Debug.WriteLine(gamma.__repr__, "gamma a");
            //var bn = new BatchNorm(ref mean, ref var, decay, eps);
            bn.AvgMean = bn.InitAvgMean = mean;
            bn.AvgVar = bn.InitAvgVar = var;
            bn.Decay = decay;
            bn.Eps = eps;
            bn.InvStd = null;
            try
            {
                return bn.Call(Params.New.SetKeywordArg(x, gamma, beta, mean, var));
            }
            finally
            {
                mean.Data.Value = bn.AvgMean.Data.Value;
                var.Data.Value = bn.AvgVar.Data.Value;
            }
        }
    }
}
