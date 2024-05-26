namespace DeZero.NET.Layers
{
    public class ReLU : Layer
    {
        public override Variable[] Forward(params Variable[] xs)
        {
            var x = xs[0];
            return Functions.Relu.Invoke(x);
        }
    }
}
