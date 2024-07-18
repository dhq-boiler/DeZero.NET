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
            var x = xs[0];
            // 入力のチャンネル数が InChannels と一致しているかチェック
            if (x.Shape[1] != InChannels.Value)
            {
                InChannels.Value = x.Shape[1];
                //throw new ArgumentException($"Input channel does not match. Expected: {InChannels.Value}, Actual: {x.Shape[1]}");
            }

            if (W.Value.Data.Value is null)
            {
                _init_W();
            }

            WInitialized?.Invoke();

            var y = Functions.Conv2dResNet.Invoke(xs[0], W.Value, b.Value, stride: (Stride.Value, Stride.Value), pad: (Pad.Value, Pad.Value));
            return y;
        }
    }
}
