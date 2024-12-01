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
            this.iter = 0;
            InitializeSWAModel();
        }

        private void InitializeSWAModel()
        {
            try
            {
                if (this.BaseOptimizer?.Target != null)
                {
                    var tempParams = this.BaseOptimizer.Target.Params();
                    this.swa_model = tempParams.Select(p => {
                        var copy = p.Data.Value.copy();
                        return copy.ToVariable();
                    }).ToList();
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Warning: Failed to initialize SWA model: {ex.Message}");
                this.swa_model = null;
            }
        }

        private bool ValidateShapes(NDarray a, NDarray b)
        {
            using var a_shape = a.shape;
            using var b_shape = b.shape;
            if (a_shape.Dimensions.Length != b_shape.Dimensions.Length)
                return false;

            for (int i = 0; i < a_shape.Dimensions.Length; i++)
            {
                if (a_shape[i] != b_shape[i])
                    return false;
            }
            return true;
        }

        public override void Update(Params param)
        {
            try
            {
                this.BaseOptimizer.Update(param);
                this.iter += 1;

                if (this.swa_model is null)
                {
                    InitializeSWAModel();
                }

                if (this.iter >= this.swa_start && this.iter % this.swa_freq == 0 && this.swa_model != null)
                {
                    var @params = this.BaseOptimizer.Target.Params();
                    var paramPairs = this.swa_model.Zip(@params).ToList();

                    foreach (var (swa_p, p) in paramPairs)
                    {
                        if (!ValidateShapes(swa_p.Data.Value, p.Data.Value))
                        {
                            using var swa_shape = swa_p.Data.Value.shape;
                            using var p_shape = p.Data.Value.shape;
                            throw new InvalidOperationException(
                                $"Shape mismatch: SWA parameter shape {string.Join(",", swa_shape)} " +
                                $"!= current parameter shape {string.Join(",", p_shape)}");
                        }

                        var n = (this.iter - this.swa_start) / this.swa_freq;
                        var factor = 1.0f / (n + 1);

                        // n/(n+1) * swa_param + 1/(n+1) * param
                        var newValue = (swa_p.Data.Value * n + p.Data.Value) * factor;
                        swa_p.Data.Value = newValue;
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error in SWA Update: {ex.Message}");
                // 元の形状情報もログに出力
                if (this.swa_model != null)
                {
                    var @params = this.BaseOptimizer.Target.Params();
                    Console.WriteLine("Current shapes:");
                    for (int i = 0; i < this.swa_model.Count; i++)
                    {
                        using var swa_model_i_shape = this.swa_model[i].Data.Value.shape;
                        using var param_i_shape = @params.ElementAt(i).Data.Value.shape;
                        Console.WriteLine($"SWA model [{i}]: {string.Join(",", swa_model_i_shape)}");
                        Console.WriteLine($"Current params [{i}]: {string.Join(",", param_i_shape)}");
                    }
                }
                throw;
            }
        }

        public override void SaveParameters()
        {
            try
            {
                BaseOptimizer.SaveParameters();

                // swa_modelがnullの場合は初期化を試みる
                if (this.swa_model is null)
                {
                    InitializeSWAModel();
                }

                if (this.swa_model != null)
                {
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
                else
                {
                    Console.WriteLine("Warning: SWA model is not initialized, skipping parameter saving.");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error in SaveParameters: {ex.Message}");
                throw;
            }
        }

        public override void LoadParameters()
        {
            try
            {
                BaseOptimizer.LoadParameters();

                // swa_modelの初期化
                InitializeSWAModel();

                if (this.swa_model != null)
                {
                    var i = 0;
                    foreach (var parameter in this.swa_model)
                    {
                        var filename = Path.Combine("optimizer", Uri.EscapeDataString($"SWA__swa_model__{i}.npy")).Replace("%2F", "_");
                        if (File.Exists(filename))
                        {
                            try
                            {
                                Console.Write($"\n {filename} ...");
                                var ndarray = xp.load(filename, allow_pickle: true);
                                parameter.Data.Value = ndarray;
                                Console.Write("Done.");
                            }
                            catch (Exception e)
                            {
                                Console.WriteLine("Error while loading parameters.");
                                Console.WriteLine(e.Message);
                            }
                        }
                        i++;
                    }
                }
                else
                {
                    Console.WriteLine("Warning: Failed to initialize SWA model during parameter loading.");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error in LoadParameters: {ex.Message}");
                throw;
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
                Console.WriteLine("Warning: Cannot swap parameters as SWA model is not initialized.");
                return;
            }

            foreach (var (swa_p, p) in this.swa_model.Zip(_params.Through))
            {
                p.Variable.Data.Value = swa_p.Data.Value.copy();
            }
        }

        public override void SetNewLr(float newLr)
        {
            this.swa_lr = newLr;
        }
    }
}
