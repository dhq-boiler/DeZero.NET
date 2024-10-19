using DeZero.NET.Core;
using DeZero.NET.Extensions;

namespace DeZero.NET.Optimizers
{
    /// <summary>
    /// 先読みしてから更新するオプティマイザ
    /// </summary>
    public class LookaheadOptimizer : Optimizer
    {
        public Optimizer BaseOptimizer { get; set; }
        public int k { get; set; }
        public float alpha { get; set; }
        public List<Variable> fast_params { get; set; }
        public List<Variable> slow_params { get; set; }
        public int iter { get; set; }

        public LookaheadOptimizer(Optimizer baseOptimizer, int k = 5, float alpha = 0.5f) : base()
        {
            this.BaseOptimizer = baseOptimizer;
            this.k = k;
            this.alpha = alpha;
            this.fast_params = new List<Variable>();
            this.slow_params = new List<Variable>();
            this.iter = 0;
        }

        public override void Update(Params param)
        {
            if (this.fast_params is null)
            {
                this.fast_params = this.Target.Params().Select(p => p.Data.Value.copy().ToVariable()).ToList();
                this.slow_params = this.Target.Params().Select(x => x).Cast<Variable>().ToList();
            }

            foreach (var (fast, slow) in this.fast_params.Zip(this.slow_params))
            {
                this.BaseOptimizer.UpdateOne(new Parameter(fast));
            }

            this.iter += 1;

            if (this.iter % this.k == 0)
            {
                foreach (var (fast, slow) in this.fast_params.Zip(this.slow_params))
                {
                    slow.Data.Value = this.alpha * slow.Data.Value + (1 - this.alpha) * fast.Data.Value;
                    fast.Data.Value = slow.Data.Value.copy();
                }
            }
        }

        public override void UpdateOne(Parameter param)
        {
            throw new NotSupportedException();
        }
    }
}
