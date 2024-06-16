namespace DeZero.NET.Layers.Convolution
{
    public class Conv2dResNet : Layers.Convolution.Conv2d
    {
        private string Name { get; set; }
        public Conv2dResNet(string name, int out_channels, int kernel_size, Dtype dtype, int stride = 1, int pad = 0, bool nobias = false,
            int? in_channels = null) : base(out_channels, kernel_size, dtype, stride, pad, nobias, in_channels)
        {
            Name = name;
        }

        public override Variable[] Forward(params Variable[] xs)
        {
            if (W.Value.Data.Value is null)
            {
                InChannels.Value = xs[0].Shape[1];
                _init_W();
            }

            WInitialized?.Invoke();

            var y = Functions.Conv2dResNet.Invoke(xs[0], W.Value, b.Value, stride: (Stride.Value, Stride.Value), pad: (Pad.Value, Pad.Value));
            return y;
        }
    }
}
