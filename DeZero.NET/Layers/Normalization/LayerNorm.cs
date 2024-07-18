using DeZero.NET.Core;

namespace DeZero.NET.Layers.Normalization
{
    /// <summary>
    /// 層正規化（Layer Normalization）層
    /// 正規化層の一つ
    /// サンプルごとの統計量が大きく異なる場合に有効
    /// </summary>
    public class LayerNorm : Layer
    {
        public Property<float> eps { get; } = new(nameof(eps));

        public LayerNorm(float eps = 1e-8f)
        {
            RegisterEvent(this.eps);
            this.eps.Value = eps;
        }

        public override Variable[] Forward(params Variable[] xs)
        {
            var x = xs[0];
            return [Functions.LayerNorm.Invoke(x, eps.Value)];
        }
    }
}
