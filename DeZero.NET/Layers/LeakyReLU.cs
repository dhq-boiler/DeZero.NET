namespace DeZero.NET.Layers
{
    public class LeakyReLU : Layer
    {
        public override Variable[] Forward(params Variable[] xs)
        {
            var x = xs[0];
            return Functions.LeakyReLU.Invoke(x);
        }
    }
}
