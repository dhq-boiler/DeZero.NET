using DeZero.NET.Core;

namespace DeZero.NET.Layers.Standardization
{
    public class WeightStandardization : Layer, IWrap
    {
        public Property<IWeight> Layer { get; } = new(nameof(Layer));
        public Property<double> eps { get; } = new(nameof(eps));

        public WeightStandardization(IWeight layer, double eps = 1e-5)
        {
            RegisterEvent(Layer, this.eps);
            this.Layer.Value = layer;
            this.Layer.Value.WInitialized = () => Standardize();
            this.eps.Value = eps;
        }

        private void Standardize()
        {
            var W = this.Layer.Value.W.Value.Data.Value;
            using var mean = W.mean(axis: 1, keepdims: true);
            using var std = W.std(axis: 1, keepdims: true);
            var W_standardized = (W - mean) / (std + this.eps.Value);
            if (this.Layer.Value.W.Value.Data.Value is not null)
            {
                this.Layer.Value.W.Value.Data.Value.Dispose();
                this.Layer.Value.W.Value.Data.Value = null;
            }
            this.Layer.Value.W.Value.Data.Value = W_standardized;
        }

        public override Variable[] Forward(params Variable[] xs)
        {
            return this.Layer.Value.Call(xs);
        }
    }
}
