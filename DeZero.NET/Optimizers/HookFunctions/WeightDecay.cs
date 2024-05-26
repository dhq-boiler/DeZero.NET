namespace DeZero.NET.Optimizers.HookFunctions
{
    public class WeightDecay : HookFunction
    {
        public float Rate { get; }

        public WeightDecay(float rate)
        {
            Rate = rate;
        }

        public override void Call(Parameter[] @params)
        {
            foreach (var param in @params)
            {
                if (param.Grad.Value is null)
                {
                    param.Grad.Value = new Variable(xp.ones_like(param.Data.Value));
                }

                param.Grad.Value.Data.Value += Rate * param.Data.Value;
            }
        }
    }
}
