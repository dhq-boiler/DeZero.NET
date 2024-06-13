using DeZero.NET.Core;

namespace DeZero.NET.Layers.Normalization
{
    /// <summary>
    /// バッチ正規化（Batch Normalization）層
    /// 正規化層の一つ
    /// ※Softmax層やSigmoid層の前に挿入するとスケール情報が失われる
    /// ※回帰問題には使わない方が良い
    /// </summary>
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
            AvgMean.Value.Data.Value = xp.zeros(D, dtype: x.Dtype);
            AvgVar.Value.Data.Value = xp.ones(D, dtype: x.Dtype);
            Gamma.Value.Data.Value = xp.ones(D, dtype: x.Dtype);
            Beta.Value.Data.Value = xp.zeros(D, dtype: x.Dtype);
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
