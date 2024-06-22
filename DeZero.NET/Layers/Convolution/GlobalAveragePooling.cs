using F = DeZero.NET.Functions;

namespace DeZero.NET.Layers.Convolution
{
    public class GlobalAveragePooling : Layers.Layer
    {
        public F.GlobalAveragePooling F { get; } = new();

        public override Variable[] Forward(params Variable[] xs)
        {
            var x = xs[0];
            return Functions.GlobalAveragePooling.Invoke(x);
        }
    }
}
