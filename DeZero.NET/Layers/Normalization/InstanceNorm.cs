using DeZero.NET.Core;

namespace DeZero.NET.Layers.Normalization
{
    /// <summary>
    /// インスタンス正規化（Instance Normalization）層
    /// 正規化層の一つ
    /// </summary>
    public class InstanceNorm : Layer
    {
        public Property<float> eps { get; } = new(nameof(eps));

        public InstanceNorm(float eps = 1e-8f)
        {
            RegisterEvent(this.eps);
            this.eps.Value = eps;
        }

        public override Variable[] Forward(params Variable[] xs)
        {
            var x = xs[0];
            return [Functions.InstanceNorm.Invoke(x, eps.Value)];
        }
    }
}
