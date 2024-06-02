namespace DeZero.NET.Layers.Normalization
{
    /// <summary>
    /// 層正規化（Layer Normalization）層
    /// 正規化層の一つ
    /// サンプルごとの統計量が大きく異なる場合に有効
    /// </summary>
    public class LayerNorm : Layer
    {
        public float eps { get; set; }

        public LayerNorm(float eps = 1e-8f)
        {
            this.eps = eps;
        }

        public override Variable[] Forward(params Variable[] xs)
        {
            var x = xs[0];
            return [Functions.LayerNorm.Invoke(x, eps)];
        }
    }
}
