using DeZero.NET.Extensions;

namespace DeZero.NET.Optimizers
{
    /// <summary>
    /// モーメンタムを共有化し、必要メモリ量を削減するオプティマイザ
    /// </summary>
    public class NovoGrad : Optimizer
    {
        public float Alpha { get; set; }
        public float Beta1 { get; set; }
        public float Beta2 { get; set; }
        public float Eps { get; set; }
        public Variable m { get; set; }
        public Variable v { get; set; }
        public int t { get; set; }

        public NovoGrad(float alpha = 0.01f, float beta1 = 0.95f, float beta2=0.98f, float eps = 1e-8f) : base()
        {
            this.Alpha = alpha;
            this.Beta1 = beta1;
            this.Beta2 = beta2;
            this.Eps = eps;
            this.m = null;
            this.v = null;
            this.t = 0;
        }

        public override void UpdateOne(Parameter param)
        {
            if (this.m is null)
            {
                this.m = xp.zeros_like(param.Data.Value).ToVariable();
                this.v = xp.zeros_like(param.Data.Value).ToVariable();
            }

            this.t += 1;
            var g = param.Grad.Value;
            this.m = this.Beta1 * this.m + (1 - this.Beta1) * g;
            this.v = (this.Beta2 * this.v + (1 - this.Beta2) * g).pow(2);
            var m_hat = this.m / (1 - Math.Pow(this.Beta1, this.t));
            var v_hat = this.v / (1 - Math.Pow(this.Beta2, this.t));
            var g_hat = g / (xp.sqrt(v_hat.Data.Value) + this.Eps);
            param.Data.Value -= this.Alpha * (m_hat / (xp.sqrt(v_hat.Data.Value) + this.Eps) + g_hat).Data.Value;
        }

        public override void SetNewLr(float newLr)
        {
            this.Alpha = newLr;
        }
    }
}
