namespace DeZero.NET.Layers
{
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
