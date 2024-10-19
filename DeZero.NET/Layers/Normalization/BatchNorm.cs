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
        public Property<Parameter> AvgMean { get; set; }
        public Property<Parameter> AvgVar { get; set; }
        public Property<Parameter> Gamma { get; set; }
        public Property<Parameter> Beta { get; set; }
        private int? Channels { get; set; }

        public BatchNorm(int? channels = null, Dtype dtype = null)
        {
            Channels = channels;

            AvgMean = new(nameof(AvgMean), new Parameter(null, name: "avg_mean"));
            AvgVar = new(nameof(AvgVar), new Parameter(null, name: "avg_var"));
            Gamma = new(nameof(Gamma), new Parameter(null, name: "gamma"));
            Beta = new(nameof(Beta), new Parameter(null, name: "beta"));

            RegisterEvent(AvgMean, AvgVar, Gamma, Beta);

            if (channels.HasValue)
            {
                InitParams(channels.Value, dtype);
            }
        }

        public void InitParams(int channels, Dtype dtype)
        {
            AvgMean.Value.Data.Value = xp.zeros(channels, dtype: dtype);
            AvgVar.Value.Data.Value = xp.ones(channels, dtype: dtype);
            Gamma.Value.Data.Value = xp.ones(channels, dtype: dtype);
            Beta.Value.Data.Value = xp.zeros(channels, dtype: dtype);
        }

        public override Variable[] Forward(params Variable[] xs)
        {
            var x = xs[0];
            if (x.Data.Value is null)
            {
                throw new ArgumentException("Input data cannot be null.", nameof(xs));
            }

            if (!Channels.HasValue)
            {
                Channels = x.Shape[1];
                InitParams(Channels.Value, x.Dtype);
            }
            else if (x.Shape[1] != Channels)
            {
                throw new ArgumentException($"Input channels ({x.Shape[1]}) do not match the specified channels ({Channels}).", nameof(xs));
            }

            return Functions.BatchNorm.Invoke(x, Gamma.Value, Beta.Value, AvgMean.Value, AvgVar.Value).Item1;
        }
    }
}
