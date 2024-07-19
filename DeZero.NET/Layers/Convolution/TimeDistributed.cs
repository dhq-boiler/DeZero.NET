namespace DeZero.NET.Layers.Convolution
{
    public class TimeDistributed : Layer
    {
        private readonly Functions.TimeDistributed _funcTimeDistributed;

        public TimeDistributed(Layer layer)
        {
            _funcTimeDistributed = new Functions.TimeDistributed(layer);
        }

        public override Variable[] Forward(params Variable[] xs)
        {
            return _funcTimeDistributed.Forward(Core.Params.New.SetPositionalArgs(xs[0], arg1Name: "x"));
        }
    }
}
