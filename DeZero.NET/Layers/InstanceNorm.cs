namespace DeZero.NET.Layers
{
    public class InstanceNorm : Layer
    {
        public float eps { get; set; }
     
        public InstanceNorm(float eps = 1e-8f)
        {
            this.eps = eps;
        }

        public override Variable[] Forward(params Variable[] xs)
        {
            var x = xs[0];
            return [Functions.InstanceNorm.Invoke(x, eps)];
        }
    }
}
