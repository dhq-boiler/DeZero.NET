﻿using DeZero.NET.Core;
using DeZero.NET.Extensions;

namespace DeZero.NET.Layers.Convolution
{
    public class Deconv2d : Layer, IWbOwner
    {
        public Property<int?> InChannels { get; private set; } = new(nameof(InChannels));
        public Property<int> OutChannels { get; } = new(nameof(OutChannels));
        public Property<int> KernelSize { get; } = new(nameof(KernelSize));
        public Property<Dtype> Dtype { get; } = new(nameof(Dtype));
        public Property<int> Stride { get; } = new(nameof(Stride));
        public Property<int> Pad { get; } = new(nameof(Pad));
        public Property<Parameter> b { get; set; } = new(nameof(b));
        public Property<Parameter> W { get; set; } = new(nameof(W));
        public Action WInitialized { get; set; }

        public Deconv2d()
        {
            RegisterEvent(InChannels, OutChannels, KernelSize, Dtype, Stride, Pad, b, W);
        }

        public Deconv2d(int out_channels, int kernel_size, Dtype dtype, int stride = 1, int pad = 0, bool nobias = false, int? in_channels = null)
        {
            InChannels.Value = in_channels;
            OutChannels.Value = out_channels;
            KernelSize.Value = kernel_size;
            Dtype.Value = dtype;
            Stride.Value = stride;
            Pad.Value = pad;

            W.Value = new Parameter(null, name: "W");
            if (in_channels is not null)
            {
                _init_W();
            }

            if (nobias)
            {
                b.Value = null;
            }
            else
            {
                b.Value = new Parameter(xp.zeros(OutChannels.Value, dtype: dtype).ToVariable(), name: "b");
            }
        }

        private void _init_W()
        {
            int C = InChannels.Value.Value, OC = OutChannels.Value;
            int KH = KernelSize.Value, KW = KernelSize.Value;
            float scale = xp.sqrt(new NDarray(1f / (C * KH * KW))).asscalar<float>();
            var W_data = xp.random.randn(C, OC, KH, KW).astype(Dtype.Value) * scale;
            W.Value.Data.Value = W_data;
        }

        public override Variable[] Forward(params Variable[] xs)
        {
            if (W.Value.Data.Value is null)
            {
                InChannels.Value = xs[0].Shape[1];
                _init_W();
            }

            WInitialized?.Invoke();

            var y = Functions.Deconv2d.Invoke(xs[0], W.Value, b.Value, stride: (Stride.Value, Stride.Value), pad: (Pad.Value, Pad.Value));
            return y;
        }
    }
}
