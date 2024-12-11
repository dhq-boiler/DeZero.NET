using DeZero.NET.Core;
using DeZero.NET.Extensions;

namespace DeZero.NET.Layers.Convolution
{
    public class Conv2d : Layer, IWbOwner, IDisposable
    {
        private bool disposed = false;

        public Property<int?> InChannels { get; } = new(nameof(InChannels));
        public Property<int> OutChannels { get; } = new(nameof(OutChannels));
        public Property<int> KernelSize { get; } = new(nameof(KernelSize));
        public Property<Dtype> Dtype { get; } = new(nameof(Dtype));
        public Property<int> Stride { get; } = new(nameof(Stride));
        public Property<int> Pad { get; } = new(nameof(Pad));
        public Property<Parameter> b { get; set; } = new(nameof(b));
        public Property<Parameter> W { get; set; } = new(nameof(W));
        public Action WInitialized { get; set; }

        public Conv2d()
        {
            RegisterEvent(InChannels, OutChannels, KernelSize, Dtype, Stride, Pad, b, W);
        }

        public Conv2d(int out_channels, int kernel_size, Dtype dtype, int stride = 1, int pad = 0, bool nobias = false, int? in_channels = null) : this()
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

            b.Value?.Dispose();
            if (nobias)
            {
                b.Value = null;
            }
            else
            {
                b.Value = new Parameter(xp.zeros(OutChannels.Value, dtype: dtype).ToVariable(), name: "b");
            }
        }

        protected void _init_W()
        {
            int C = InChannels.Value.Value, OC = OutChannels.Value;
            int KH = KernelSize.Value, KW = KernelSize.Value;

            using (var scope = new ComputationScope())
            {
                using var w_data = xp.random.randn(OC, C, KH, KW);
                using var s = xp.sqrt(new NDarray(1f / (C * KH * KW)));
                float scale = s.asscalar<float>();

                // 既存の重みをクリーンアップ
                W.Value.Data.Value?.Dispose();

                // 新しい重みを設定
                W.Value.Data.Value = (w_data * scale).astype(Dtype.Value);
            }
        }

        public override Variable[] Forward(params Variable[] xs)
        {
            if (xs == null || xs.Length == 0 || xs[0] == null)
                return null;

            var x = xs[0];

            // 入力チャンネル数の検証と重みの初期化
            if (InChannels.Value != x.Shape[1] || W.Value.Data.Value is null)
            {
                InChannels.Value = x.Shape[1];
                _init_W();
                WInitialized?.Invoke();
            }

            using (var scope = new ComputationScope())
            {
                using var result = Functions.Conv2d.Invoke(x, W.Value, b.Value,
                    stride: Stride.Value, pad: Pad.Value)[0];
                return [result.copy()];
            }
        }

        public void Dispose()
        {
            if (!disposed)
            {
                // パラメータの解放
                W.Value?.Data?.Value?.Dispose();
                b.Value?.Data?.Value?.Dispose();
                disposed = true;
            }
            GC.SuppressFinalize(this);
        }
    }
}
