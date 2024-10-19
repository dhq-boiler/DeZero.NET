using DeZero.NET.Core;
using DeZero.NET.Extensions;

namespace DeZero.NET.Optimizers
{
    public class AdaGrad : Optimizer
    {
        public float lr { get; set; }
        public float Eps { get; }
        public Property<Dictionary<int, Variable>> hs { get; } = new(nameof(hs));

        public AdaGrad(float lr = 0.001f, float eps = 1e-8f) : base()
        {
            this.lr = lr;
            Eps = eps;
            this.hs.Value = new Dictionary<int, Variable>();
            RegisterNonVolatileParameters(this.hs);
        }

        public override void UpdateOne(Parameter param)
        {
            var h_key = param.GetHashCode();
            if (hs.Value.ContainsKey(h_key))
            {
                hs.Value[h_key] = xp.zeros_like(param.Data.Value).ToVariable();
            }

            var lr = this.lr;
            var eps = this.Eps;
            var grad = param.Grad.Value.Data.Value;
            var h = hs.Value[h_key];

            h += grad * grad;
            param.Data.Value -= lr * grad / (xp.sqrt(h.Data.Value) + eps);
        }
    }
}
