namespace DeZero.NET.Layers
{
    public class EmbedID : Layer
    {
        public Parameter W { get; set; }

        public EmbedID(int in_size, int out_size) : base()
        {
            W = new Parameter(xp.random.randn(in_size, out_size).ToVariable(), "W");
        }
        
        public override Variable[] Forward(params Variable[] xs)
        {
            var y = W.Data[xs[0].Data];
            return [y.ToVariable()];
        }
    }
}
