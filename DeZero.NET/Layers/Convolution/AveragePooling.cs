using DeZero.NET.Core;

namespace DeZero.NET.Layers.Convolution
{
    /// <summary>
    /// 平均値を出力するプーリング層
    /// </summary>
    public class AveragePooling : Layer
    {
        public Property<(int, int)> KernelSize { get; } = new(nameof(KernelSize));
        public Property<int> Stride { get; } = new(nameof(Stride));
        public Property<int> Pad { get; } = new(nameof(Pad));

        public AveragePooling((int, int) kernelSize, int stride, int pad)
        {
            RegisterEvent(KernelSize, Stride, Pad);
            this.KernelSize.Value = kernelSize;
            this.Stride.Value = stride;
            this.Pad.Value = pad;
        }

        public override Variable[] Forward(params Variable[] xs)
        {
            var x = xs[0];
            return Functions.AveragePooling.Invoke(x, KernelSize.Value, Stride.Value, Pad.Value);
        }
    }
}
