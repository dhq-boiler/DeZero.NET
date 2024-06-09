using DeZero.NET.Core;

namespace DeZero.NET.Layers.Normalization
{
    public class WeightNorm : Layer, IWrap
    {
        public Property<IWeight> Layer { get; } = new(nameof(Layer));
        public Property<Parameter> g { get; } = new(nameof(g));
        public WeightNorm(IWeight layer)
        {
            RegisterEvent(Layer, g);
            this.Layer.Value = layer;
            this.Layer.Value.WInitialized = () => Normalize();
        }

        private void Normalize()
        {
            this.g.Value = new Parameter(xp.ones([this.Layer.Value.W.Value.Data.Value.shape[0]]).ToVariable(), "Weight");
            var W = this.Layer.Value.W.Value.Data.Value;
            using var W_norm = W / xp.linalg.norm(W, axis: 1, keepdims: true);
            if (this.Layer.Value.W.Value.Data.Value is not null)
            {
                this.Layer.Value.W.Value.Data.Value.Dispose();
                this.Layer.Value.W.Value.Data.Value = null;
            }
            this.Layer.Value.W.Value.Data.Value = this.g.Value.Data.Value.reshape(-1, 1) * W_norm;
        }

        public override Variable[] Forward(params Variable[] xs)
        {
            return this.Layer.Value.Call(xs);
        }
    }
}
