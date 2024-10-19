using DeZero.NET.Core;
using DeZero.NET.Extensions;

namespace DeZero.NET.Optimizers
{
    public class Adam : Optimizer
    {
        public int t { get; set; }
        public float alpha { get; set; }
        public float beta1 { get; set; }
        public float beta2 { get; set; }
        public float eps { get; set; }
        public Property<Dictionary<int, Variable>> ms { get; } = new(nameof(ms));
        public Property<Dictionary<int, Variable>> vs { get; } = new(nameof(vs));

        public Adam(float alpha = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f) : base()
        {
            this.t = 0;
            this.alpha = alpha;
            this.beta1 = beta1;
            this.beta2 = beta2;
            this.eps = eps;
            this.ms.Value = new Dictionary<int, Variable>();
            this.vs.Value = new Dictionary<int, Variable>();
            RegisterNonVolatileParameters(this.ms, this.vs);
        }

        public override void Update(Params args)
        {
            this.t += 1;
            base.Update(args);
        }

        public NDarray lr
        {
            get
            {
                var fix1 = 1f - Math.Pow(this.beta1, this.t);
                var fix2 = 1f - Math.Pow(this.beta2, this.t);
                return new NDarray(this.alpha * (float)Math.Sqrt(fix2) / fix1);
            }
        }

        public override void UpdateOne(Parameter param)
        {
            var key = param.Title.GetHashCode();
            if (!this.ms.Value.ContainsKey(key))
            {
                this.ms.Value[key] = xp.zeros_like(param.Data.Value).ToVariable();
                this.vs.Value[key] = xp.zeros_like(param.Data.Value).ToVariable();
            }

            var m = this.ms.Value[key];
            var v = this.vs.Value[key];
            var beta1 = this.beta1;
            var beta2 = this.beta2;
            var eps = this.eps;
            var grad = param.Grad.Value.Data.Value;

            m += (1 - beta1) * (grad - m.Data.Value);
            v += (1 - beta2) * (grad * grad - v.Data.Value);
            using var v_sqrt = xp.sqrt(v.Data.Value);
            param.Data.Value -= this.lr * m.Data.Value / (v_sqrt + eps);
        }
    }
}
