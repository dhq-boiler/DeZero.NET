using DeZero.NET.Core;

namespace DeZero.NET.Layers
{
    public class BatchNorm : Layer
    {
        public override Func<Variable[], Variable[]> F => xs => Forward(xs);
        public Property<Parameter> AvgMean { get; set; } = new(nameof(AvgMean), new Parameter(null, name: "avg_mean"));
        public Property<Parameter> AvgVar { get; set; } = new(nameof(AvgVar), new Parameter(null, name: "avg_var"));
        public Property<Parameter> Gamma { get; set; } = new(nameof(Gamma), new Parameter(null, name: "gamma"));
        public Property<Parameter> Beta { get; set; } = new(nameof(Beta), new Parameter(null, name: "beta"));

        public BatchNorm()
        {
            RegisterEvent(AvgMean, AvgVar, Gamma, Beta);
        }

        public void InitParams(Variable x)
        {
            var D = x.Shape[1];
            if (AvgMean.Value.Data.Value is null)
            {
                AvgMean.Value.Data.Value = xp.zeros(D, dtype: x.Dtype);
            }
            if (AvgVar.Value.Data.Value is null)
            {
                AvgVar.Value.Data.Value = xp.ones(D, dtype: x.Dtype);
            }
            if (Gamma.Value.Data.Value is null)
            {
                Gamma.Value.Data.Value = xp.ones(D, dtype: x.Dtype);
            }
            if (Beta.Value.Data.Value is null)
            {
                Beta.Value.Data.Value = xp.zeros(D, dtype: x.Dtype);
            }
        }

        public override Variable[] Forward(params Variable[] xs)
        {
            var x = xs[0];
            if (AvgMean.Value.Data.Value is null)
            {
                InitParams(x);
            }

            return Functions.BatchNorm.Invoke(x, Gamma.Value, Beta.Value, AvgMean.Value, AvgVar.Value).Item1;
        }
    }
}
