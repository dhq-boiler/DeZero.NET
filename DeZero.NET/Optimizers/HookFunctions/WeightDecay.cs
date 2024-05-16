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
                param.Grad.Data += Rate * param.Data;
            }
        }
    }
}
