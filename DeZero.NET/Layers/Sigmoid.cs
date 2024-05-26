namespace DeZero.NET.Layers
{
    public class Sigmoid : Layer
    {
        public override Variable[] Forward(params Variable[] xs)
        {
            var x = xs[0];
            return Functions.Sigmoid.Invoke(x);
        }
    }
}
