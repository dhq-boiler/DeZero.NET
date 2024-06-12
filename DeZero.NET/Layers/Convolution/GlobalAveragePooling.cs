using F = DeZero.NET.Functions;
using L = DeZero.NET.Layers;

namespace DeZero.NET.Layers.Convolution
{
    public class GlobalAveragePooling : L.Layer
    {
        public F.GlobalAveragePooling F { get; } = new();

        public override Variable[] Forward(params Variable[] xs)
        {
            var x = xs[0];
            return F.Forward(Core.Params.New.SetPositionalArgs(x));
        }

        public override Variable[] Backward(params Variable[] gys)
        {
            return F.Backward(Core.Params.New.SetPositionalArgs(gys[0]));
        }
    }
}
