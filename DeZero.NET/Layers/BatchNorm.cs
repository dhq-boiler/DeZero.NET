namespace DeZero.NET.Layers
{
    public class BatchNorm : Layer
    {
        public override Func<Variable[], Variable[]> F => xs => Forward(xs);
        public Parameter AvgMean { get; set; }
        public Parameter AvgVar { get; set; }
        public Parameter Gamma { get; set; }
        public Parameter Beta { get; set; }

        public BatchNorm()
        {
            AvgMean = new Parameter(null, name: "avg_mean");
            AvgVar = new Parameter(null, name: "avg_var");
            Gamma = new Parameter(null, name: "gamma");
            Beta = new Parameter(null, name: "beta");
        }

        public void InitParams(Variable x)
        {
            var D = x.Shape[1];
            if (AvgMean.Data is null)
            {
                AvgMean.Data = xp.zeros(D, dtype: x.Dtype);
            }
            if (AvgVar.Data is null)
            {
                AvgVar.Data = xp.ones(D, dtype: x.Dtype);
            }
            if (Gamma.Data is null)
            {
                Gamma.Data = xp.ones(D, dtype: x.Dtype);
            }
            if (Beta.Data is null)
            {
                Beta.Data = xp.zeros(D, dtype: x.Dtype);
            }
        }

        public override Variable[] Forward(params Variable[] xs)
        {
            var x = xs[0];
            if (AvgMean.Data is null)
            {
                InitParams(x);
            }

            return Functions.BatchNorm.Invoke(x, Gamma, Beta, AvgMean, AvgVar);
        }
    }
}
