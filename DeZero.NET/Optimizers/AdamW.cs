using DeZero.NET.Core;

namespace DeZero.NET.Optimizers
{
    public class AdamW : Optimizer
    {
        public int t { get; set; }
        public float alpha { get; set; }
        public float beta1 { get; set; }
        public float beta2 { get; set; }
        public float eps { get; set; }
        public float WeightDecay { get; }
        public Property<Dictionary<string, Variable>> ms { get; } = new(nameof(ms));
        public Property<Dictionary<string, Variable>> vs { get; } = new(nameof(vs));
        
        public AdamW(float alpha = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f, float weight_decay = 0.01f) : base()
        {
            this.t = 0;
            this.alpha = alpha;
            this.beta1 = beta1;
            this.beta2 = beta2;
            this.eps = eps;
            this.WeightDecay = weight_decay;
            this.ms.Value = new Dictionary<string, Variable>();
            this.vs.Value = new Dictionary<string, Variable>();
            RegisterNonVolatileParameters(this.ms, this.vs);
        }

        public override void Update(Params args)
        {
            this.t += 1;
            base.Update(args);
        }

        public override void UpdateOne(Parameter param)
        {
            var key = param.Title.ToString();
            if (!this.ms.Value.ContainsKey(key))
            {
                this.ms.Value[key] = xp.zeros_like(param.Data.Value).ToVariable();
                this.vs.Value[key] = xp.zeros_like(param.Data.Value).ToVariable();
            }

            //Weight Decayを先に適用
            param.Data.Value -= this.alpha * this.WeightDecay * param.Data.Value;

            var m = this.ms.Value[key];
            var v = this.vs.Value[key];
            var beta1 = this.beta1;
            var beta2 = this.beta2;
            var eps = this.eps;
            var grad = param.Grad.Value.Data.Value;

            m += (1 - beta1) * (grad - m.Data.Value);
            v += (1 - beta2) * (grad * grad - v.Data.Value);
            using var v_sqrt = xp.sqrt(v.Data.Value);
            param.Data.Value -= this.alpha * m.Data.Value / (v_sqrt + eps);
        }
    }
}
