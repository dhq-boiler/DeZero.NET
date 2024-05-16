namespace DeZero.NET.Layers
{
    public class Deconv2d : Layer
    {
        public int? InChannels { get; private set; }
        public int OutChannels { get; }
        public int KernelSize { get; }
        public Dtype Dtype { get; }
        public int Stride { get; }
        public int Pad { get; }
        public Parameter b { get; set; }
        public Parameter W { get; set; }

        public Deconv2d(int out_channels, int kernel_size, Dtype dtype, int stride = 1, int pad = 0, bool nobias = false, int? in_channels = null)
        {
            InChannels = in_channels;
            OutChannels = out_channels;
            KernelSize = kernel_size;
            Dtype = dtype;
            Stride = stride;
            Pad = pad;

            W = new Parameter(null, name: "W");
            if (in_channels is not null)
            {
                _init_W();
            }

            if (nobias)
            {
                b = null;
            }
            else
            {
                b = new Parameter(xp.zeros(OutChannels, dtype: dtype).ToVariable(), name: "b");
            }
        }

        private void _init_W()
        {
            int C = InChannels.Value, OC = OutChannels;
            int KH = KernelSize, KW = KernelSize;
            float scale = xp.sqrt(new NDarray(1f / (C * KH * KW))).asscalar<float>();
            var W_data = xp.random.randn(C, OC, KH, KW).astype(Dtype) * scale;
            W.Data = W_data;
        }

        public override Variable[] Forward(params Variable[] xs)
        {
            if (W.Data is null)
            {
                InChannels = xs[0].Shape[1];
                _init_W();
            }

            var y = Functions.Deconv2d.Invoke(xs[0], W, b, stride: (Stride, Stride), pad: (Pad, Pad));
            return y;
        }
    }
}
