namespace DeZero.NET.Optimizers.HookFunctions
{
    public class ClipGrad : HookFunction
    {
        public float MaxNorm { get; }

        public ClipGrad(float max_norm)
        {
            MaxNorm = max_norm;
        }

        public override void Call(Parameter[] @params)
        {
            var total_norm = 0d;
            foreach (var param in @params)
            {
                total_norm += (param.Grad.Value.Data.Value * param.Grad.Value.Data.Value).sum().asscalar<float>();
            }

            total_norm = Math.Sqrt(total_norm);

            var rate = MaxNorm / (total_norm + 1e-6f);
            if (rate < 1)
            {
                foreach (var param in @params)
                {
                    param.Grad.Value.Data.Value *= rate;
                }
            }
        }
    }
}
