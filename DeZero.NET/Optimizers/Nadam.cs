using DeZero.NET.Core;
using DeZero.NET.Extensions;

namespace DeZero.NET.Optimizers
{
    public class Nadam : Optimizer
    {
        public float lr { get; set; }
        public float beta1 { get; set; }
        public float beta2 { get; set; }
        public float eps { get; set; }
        public List<Variable> m { get; set; }
        public List<Variable> v { get; set; }
        public int t { get; set; }

        public Nadam(float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f) : base()
        {
            this.lr = lr;
            this.beta1 = beta1;
            this.beta2 = beta2;
            this.eps = eps;
            this.m = null;
            this.v = null;
            this.t = 0;
        }

        public override void Update(Params args)
        {
            if (this.m is null)
            {
                this.m = new List<Variable>();
                this.v = new List<Variable>();
                foreach (var param in this.Target.Params().Where(p => p.Grad.Value is not null))
                {
                    this.m.Add(xp.zeros_like(param.Data.Value).ToVariable());
                    this.v.Add(xp.zeros_like(param.Data.Value).ToVariable());
                }
            }

            this.t += 1;
            //var lr_t = this.lr * Math.Sqrt(1.0f - Math.Pow(this.beta2, this.t) / (1.0f - Math.Pow(this.beta1, this.t)));
            var lr_t = this.lr * (float)Math.Sqrt(1.0 - Math.Pow(this.beta2, this.t)) / (1.0f - (float)Math.Pow(this.beta1, this.t));

            foreach (var (i, param) in this.Target.Params().Where(p => p.Grad.Value is not null).Select((p, i) => (i, p)))
            {
                var grad = param.Grad.Value.Data.Value;
                var m_t = (this.beta1 * this.m[i].Data.Value) + (1f - this.beta1) * grad;
                var v_t = (this.beta2 * this.v[i].Data.Value) + (1f - this.beta2) * grad * grad;
                //var m_cap = m_t / (1f - Math.Pow(this.beta1, this.t + 1));
                //var v_cap = v_t / (1f - Math.Pow(this.beta2, this.t));
                var m_cap = m_t / (1f - (float)Math.Pow(this.beta1, this.t));
                var v_cap = v_t / (1f - (float)Math.Pow(this.beta2, this.t));

                var g_prime = grad / (1 - Math.Pow(this.beta1, this.t));
                var m_bar = (1 - this.beta1) * g_prime + this.beta1 * m_cap;

                param.Data.Value -= lr_t * m_bar / (xp.sqrt(v_cap) + this.eps);
                this.m[i].Data.Value = m_t;
                this.v[i].Data.Value = v_t;
            }
        }

        public override void UpdateOne(Parameter param)
        {
            throw new NotSupportedException();
        }

        public override void SetNewLr(float newLr)
        {
            this.lr = newLr;
        }
    }
}
