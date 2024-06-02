namespace DeZero.NET.Layers.Convolution
{
    /// <summary>
    /// 最大値を出力するプーリング層
    /// </summary>
    public class MaxPooling : Layer
    {
        public (int, int) KernelSize { get; set; }
        public (int, int) Stride { get; set; }
        public (int, int) Pad { get; set; }

        public MaxPooling((int, int) kernelSize, (int, int) stride, (int, int) pad)
        {
            this.KernelSize = kernelSize;
            this.Stride = stride;
            this.Pad = pad;
        }

        public override Variable[] Forward(params Variable[] xs)
        {
            var x = xs[0];
            return [Functions.MaxPooling.Invoke(x, KernelSize, Stride, Pad)];
        }
    }
}
