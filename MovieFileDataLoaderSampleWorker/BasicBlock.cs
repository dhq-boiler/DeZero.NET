using DeZero.NET;
using DeZero.NET.Layers.Convolution;
using DeZero.NET.Layers.Normalization;
using DeZero.NET.Models;

namespace MovieFileDataLoaderSampleWorker
{
    public class BasicBlock : Model
    {
        private Conv2d Conv1 { get; set; }
        private BatchNorm Bn1 { get; set; }
        private Conv2d Conv2 { get; set; }
        private BatchNorm Bn2 { get; set; }
        private Conv2d ConvShortcut { get; set; }
        private BatchNorm BnShortcut { get; set; }

        public BasicBlock(int in_channels, int out_channels, int stride = 1, Dtype dtype = null)
        {
            Conv1 = new Conv2d(out_channels, kernel_size: 3, stride: stride, pad: 1, in_channels: in_channels, dtype: dtype);
            Bn1 = new BatchNorm(out_channels);
            Conv2 = new Conv2d(out_channels, kernel_size: 3, stride: 1, pad: 1, in_channels: out_channels, dtype: dtype);
            Bn2 = new BatchNorm(out_channels);

            if (stride != 1 || in_channels != out_channels)
            {
                ConvShortcut = new Conv2d(out_channels, kernel_size: 1, stride: stride, in_channels: in_channels, dtype: dtype);
                BnShortcut = new BatchNorm(out_channels);
            }
        }

        public override Variable[] Forward(params Variable[] inputs)
        {
            var x = inputs[0];
            var shortcut = x;

            x = DeZero.NET.Functions.ReLU.Invoke(Bn1.Forward(Conv1.Forward(x)[0])[0])[0];
            x = Bn2.Forward(Conv2.Forward(x)[0])[0];

            if (ConvShortcut != null)
            {
                shortcut = BnShortcut.Forward(ConvShortcut.Forward(shortcut)[0])[0];
            }

            x = DeZero.NET.Functions.Add.Invoke(x, shortcut).Item1[0];
            x = DeZero.NET.Functions.ReLU.Invoke(x)[0];

            return new[] { x };
        }
    }
}
