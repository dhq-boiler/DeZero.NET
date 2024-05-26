namespace DeZero.NET.Layers
{
    public class Tanh : Layer
    {
        public override Variable[] Forward(params Variable[] xs)
        {
            var x = xs[0];
            return Functions.Tanh.Invoke(x);
        }
    }
}
