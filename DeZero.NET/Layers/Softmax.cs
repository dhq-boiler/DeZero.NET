namespace DeZero.NET.Layers
{
    public class Softmax : Layer
    {
        public override Variable[] Forward(params Variable[] xs)
        {
            var x = xs[0];
            return Functions.Softmax.Invoke(x);
        }
    }
}
