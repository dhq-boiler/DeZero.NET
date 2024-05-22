using DeZero.NET.Core;

namespace DeZero.NET.Layers
{
    public class EmbedID : Layer
    {
        public Property<Parameter> W { get; } = new();

        public EmbedID(int in_size, int out_size) : base()
        {
            W.Value = new Parameter(xp.random.randn(in_size, out_size).ToVariable(), "W");
        }
        
        public override Variable[] Forward(params Variable[] xs)
        {
            var y = W.Value.Data.Value[xs[0].Data.Value];
            return [y.ToVariable()];
        }
    }
}
