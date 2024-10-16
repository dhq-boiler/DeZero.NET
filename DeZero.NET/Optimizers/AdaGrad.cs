using DeZero.NET.Extensions;

namespace DeZero.NET.Optimizers
{
    public class AdaGrad : Optimizer
    {
        public float lr { get; set; }
        public float Eps { get; }
        public Dictionary<int, Variable> hs { get; set; }

        public AdaGrad(float lr = 0.001f, float eps = 1e-8f) : base()
        {
            this.lr = lr;
            Eps = eps;
            this.hs = new Dictionary<int, Variable>();
        }

        public override void UpdateOne(Parameter param)
        {
            var h_key = param.GetHashCode();
            if (hs.ContainsKey(h_key))
            {
                hs[h_key] = xp.zeros_like(param.Data.Value).ToVariable();
            }

            var lr = this.lr;
            var eps = this.Eps;
            var grad = param.Grad.Value.Data.Value;
            var h = hs[h_key];

            h += grad * grad;
            param.Data.Value -= lr * grad / (xp.sqrt(h.Data.Value) + eps);
        }
    }
}
