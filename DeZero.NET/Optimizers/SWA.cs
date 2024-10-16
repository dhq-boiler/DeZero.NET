using DeZero.NET.Core;
using DeZero.NET.Extensions;

namespace DeZero.NET.Optimizers
{   
    /// <summary>
    /// Stochastic Weight Averaging
    /// </summary>
    public class SWA : Optimizer
    {
        public Optimizer BaseOptimizer { get; set; }
        public int swa_start { get; set; }
        public int swa_freq { get; set; }
        public float swa_lr { get; set; }
        public List<Variable> swa_model { get; set; }
        public int iter { get; set; }

        public SWA(Optimizer baseOptimizer, int swa_start = 100, int swa_freq = 10, float swa_lr = 0.05f) : base()
        {
            this.BaseOptimizer = baseOptimizer;
            this.swa_start = swa_start;
            this.swa_freq = swa_freq;
            this.swa_lr = swa_lr;
            this.swa_model = null;
            this.iter = 0;
        }

        public override void Update(Params param)
        {
            this.BaseOptimizer.Update(param);
            this.iter += 1;

            if (this.iter >= this.swa_start && this.iter % this.swa_freq == 0)
            {
                if (this.swa_model is null)
                {
                    this.swa_model = this.BaseOptimizer.Target.Params().Select(p => p.Data.Value.copy().ToVariable()).ToList();
                }
                else
                {
                    foreach (var (swa_p, p) in this.swa_model.Zip(this.BaseOptimizer.Target.Params()))
                    {
                        swa_p.Data.Value +=
                            (swa_p.Data.Value * (this.iter - this.swa_start) / this.swa_freq + p.Data.Value) /
                            ((this.iter - this.swa_start) / this.swa_freq + 1);
                    }
                }
            }
        }

        public override void UpdateOne(Parameter param)
        {
            throw new NotSupportedException();
        }

        public void SwapSWAParams(Params _params)
        {
            if (this.swa_model is null)
            {
                return;
            }

            foreach (var (swa_p , p) in this.swa_model.Zip(_params.Through))
            {
                p.Variable.Data.Value = swa_p.Data.Value.copy();
            }
        }

        public override void SaveParameters()
        {
            BaseOptimizer.SaveParameters();

            var i = 0;
            foreach (var parameter in this.swa_model)
            {
                var filename = Path.Combine("optimizer", Uri.EscapeDataString($"SWA__swa_model__{i}.npy")).Replace("%2F", "_");
                Console.Write($"\n {filename} ...");
                var ndarray = parameter.Data.Value;
                xp.save(filename, ndarray);
                Console.Write("Done.");
                i++;
            }
        }

        public override void LoadParameters()
        {
            BaseOptimizer.LoadParameters();
            
            this.swa_model = this.BaseOptimizer.Target.Params().Select(p => p.Data.Value.copy().ToVariable()).ToList();

            var i = 0;
            foreach (var parameter in this.swa_model)
            {
                var filename = Path.Combine("optimizer", Uri.EscapeDataString($"SWA__swa_model__{i}.npy")).Replace("%2F", "_");
                Console.Write($"\n {filename} ...");
                var ndarray = xp.load(filename);
                parameter.Data.Value = ndarray;
                Console.Write("Done.");
                i++;
            }
        }
    }
}
