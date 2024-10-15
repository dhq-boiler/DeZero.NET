using DeZero.NET.Core;

namespace DeZero.NET.Layers.Convolution
{
    /// <summary>
    /// 最大値を出力するプーリング層
    /// </summary>
    public class MaxPooling : Layer
    {
        public Property<(int, int)> KernelSize { get; } = new(nameof(KernelSize));
        public Property<(int, int)> Stride { get; } = new(nameof(Stride));
        public Property<(int, int)> Pad { get; } = new(nameof(Pad));

        public MaxPooling((int, int) kernelSize, (int, int) stride, (int, int) pad)
        {
            RegisterEvent(KernelSize, Stride, Pad);
            this.KernelSize.Value = kernelSize;
            this.Stride.Value = stride;
            this.Pad.Value = pad;
        }

        public override Variable[] Forward(params Variable[] xs)
        {
            var x = xs[0];
            return Functions.MaxPooling.Invoke(x, KernelSize.Value, Stride.Value, Pad.Value);
        }
    }
}
