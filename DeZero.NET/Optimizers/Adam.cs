using DeZero.NET.Core;

namespace DeZero.NET.Optimizers
{
    public class Adam : Optimizer
    {
        public int t { get; set; }
        public float alpha { get; set; }
        public float beta1 { get; set; }
        public float beta2 { get; set; }
        public float eps { get; set; }
        public Dictionary<int, Variable> ms { get; set; }
        public Dictionary<int, Variable> vs { get; set; }
        
        public Adam(float alpha = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f) : base()
        {
            this.t = 0;
            this.alpha = alpha;
            this.beta1 = beta1;
            this.beta2 = beta2;
            this.eps = eps;
            this.ms = new Dictionary<int, Variable>();
            this.vs = new Dictionary<int, Variable>();
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
            var key = param.GetHashCode();
            if (!this.ms.ContainsKey(key))
            {
                this.ms[key] = xp.zeros_like(param.Data).ToVariable();
                this.vs[key] = xp.zeros_like(param.Data).ToVariable();
            }

            var m = this.ms[key];
            var v = this.vs[key];
            var beta1 = this.beta1;
            var beta2 = this.beta2;
            var eps = this.eps;
            var grad = param.Grad.Data;

            m += (1 - beta1) * (grad - m.Data);
            v += (1 - beta2) * (grad * grad - v.Data);
            param.Data -= this.lr * m.Data / (xp.sqrt(v.Data) + eps);
        }
    }
}
