using DeZero.NET.Core;
using DeZero.NET.Extensions;

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
        public override float Lr => this.alpha;
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

            try
            {
                var grad = param.Grad.Value?.Data.Value;
                if (grad is null) return;

                // 形状の不一致をチェック
                if (!param.Data.Value.shape.Equals(grad.shape))
                {
                    Console.WriteLine($"Warning: Parameter shape {param.Data.Value.shape} doesn't match gradient shape {grad.shape}");
                    return;  // 形状が一致しない場合は更新をスキップ
                }

                // zeros_likeはパラメータの形状に合わせる
                if (!this.ms.Value.ContainsKey(key))
                {
                    this.ms.Value[key] = xp.zeros_like(param.Data.Value).ToVariable();
                    this.vs.Value[key] = xp.zeros_like(param.Data.Value).ToVariable();
                }

                // クリッピングを追加
                var grad_norm = xp.linalg.norm(grad).asscalar<float>();
                if (grad_norm > 1.0f)
                {
                    grad = grad * (1.0f / grad_norm);
                }

                var m = this.ms.Value[key];
                var v = this.vs.Value[key];

                // モーメンタムの更新
                m.Data.Value = beta1 * m.Data.Value + (1 - beta1) * grad;
                v.Data.Value = beta2 * v.Data.Value + (1 - beta2) * (grad * grad);

                // バイアス補正
                float bc1 = Math.Max(1e-7f, 1 - (float)Math.Pow(beta1, t));
                float bc2 = Math.Max(1e-7f, 1 - (float)Math.Pow(beta2, t));
                var m_hat = m.Data.Value / bc1;
                var v_hat = v.Data.Value / bc2;

                using var v_sqrt = xp.sqrt(v_hat + eps);
                var update = this.alpha * m_hat / v_sqrt;

                // アップデートのクリッピング
                var update_norm = xp.linalg.norm(update).asscalar<float>();
                if (update_norm > 1.0f)
                {
                    update = update * (1.0f / update_norm);
                }

                var weight_decay_term = this.alpha * this.WeightDecay * param.Data.Value;  // 修正: grad ではなく param を使用
                param.Data.Value -= (update + weight_decay_term);
            }
            catch (Exception e)
            {
                Console.WriteLine($"Error updating parameter {key}: {e.Message}");
                Console.WriteLine($"Stack trace: {e.StackTrace}");
            }
        }

        public override void SetNewLr(float newLr)
        {
            this.alpha = newLr;
        }
    }
}
