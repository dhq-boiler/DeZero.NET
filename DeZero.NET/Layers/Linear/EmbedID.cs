using DeZero.NET.Core;

namespace DeZero.NET.Layers.Linear
{
    public class EmbedID : Layer
    {
        public Property<Parameter> W { get; } = new(nameof(W));

        public EmbedID() : base()
        {
            RegisterEvent(W);
        }

        public EmbedID(int in_size, int out_size) : this()
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
