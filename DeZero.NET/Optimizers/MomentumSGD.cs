using DeZero.NET.Extensions;

namespace DeZero.NET.Optimizers
{
    public class MomentumSGD : Optimizer
    {
        public float lr { get; private set; }
        public float Momentum { get; }
        public Dictionary<int, Variable> vs { get; set; }

        public MomentumSGD(float lr = 0.01f, float momentum = 0.9f) : base()
        {
            this.lr = lr;
            this.Momentum = momentum;
            this.vs = new Dictionary<int, Variable>();
        }

        public override void UpdateOne(Parameter param)
        {
            var v_key = param.GetHashCode();
            if (vs.ContainsKey(v_key))
            {
                vs[v_key] = xp.zeros_like(param.Data.Value).ToVariable();
            }

            var v = vs[v_key];
            v *= Momentum;
            v -= lr * param.Grad.Value.Data.Value;
            param.Data.Value += v.Data.Value;
        }

        public override void SetNewLr(float newLr)
        {
            this.lr = newLr;
        }
    }
}
