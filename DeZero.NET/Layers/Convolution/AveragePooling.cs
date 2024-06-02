namespace DeZero.NET.Layers.Convolution
{
    /// <summary>
    /// 平均値を出力するプーリング層
    /// </summary>
    public class AveragePooling : Layer
    {
        public (int, int) KernelSize { get; }
        public int Stride { get; }
        public int Pad { get; }

        public AveragePooling((int, int) kernelSize, int stride, int pad)
        {
            this.KernelSize = kernelSize;
            this.Stride = stride;
            this.Pad = pad;
        }

        public override Variable[] Forward(params Variable[] xs)
        {
            var x = xs[0];
            return Functions.AveragePooling.Invoke(x, KernelSize, Stride, Pad);
        }
    }
}
