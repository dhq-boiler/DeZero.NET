using DeZero.NET;
using DeZero.NET.Models;

namespace MovieFileDataLoaderSampleWorker
{
    public class BottleneckBlock : Model
    {
        private DeZero.NET.Layers.Convolution.Conv2d Conv1 { get; set; }
        private DeZero.NET.Layers.Normalization.BatchNorm Bn1 { get; set; }
        private DeZero.NET.Layers.Convolution.Conv2d Conv2 { get; set; }
        private DeZero.NET.Layers.Normalization.BatchNorm Bn2 { get; set; }
        private DeZero.NET.Layers.Convolution.Conv2d Conv3 { get; set; }
        private DeZero.NET.Layers.Normalization.BatchNorm Bn3 { get; set; }
        private DeZero.NET.Layers.Convolution.Conv2d ConvSkip { get; set; }
        private DeZero.NET.Layers.Normalization.BatchNorm BnSkip { get; set; }

        public BottleneckBlock(int in_channels, int mid_channels, int out_channels, int stride = 1, bool is_first_block = false, Dtype dtype = null)
        {
            Conv1 = new DeZero.NET.Layers.Convolution.Conv2d(mid_channels, kernel_size: 1, stride: stride, in_channels: in_channels, dtype: dtype);
            Bn1 = new DeZero.NET.Layers.Normalization.BatchNorm(mid_channels);
            Conv2 = new DeZero.NET.Layers.Convolution.Conv2d(mid_channels, kernel_size: 3, pad: 1, in_channels: mid_channels, dtype: dtype);
            Bn2 = new DeZero.NET.Layers.Normalization.BatchNorm(mid_channels);
            Conv3 = new DeZero.NET.Layers.Convolution.Conv2d(out_channels, kernel_size: 1, in_channels: mid_channels, dtype: dtype);
            Bn3 = new DeZero.NET.Layers.Normalization.BatchNorm(out_channels);

            if (is_first_block || in_channels != out_channels)
            {
                ConvSkip = new DeZero.NET.Layers.Convolution.Conv2d(out_channels, kernel_size: 1, stride: stride, in_channels: in_channels, dtype: dtype);
                BnSkip = new DeZero.NET.Layers.Normalization.BatchNorm(out_channels);
            }
        }

        public override Variable[] Forward(params Variable[] inputs)
        {
            var x = inputs[0];
            var residual = x;

            x = DeZero.NET.Functions.ReLU.Invoke(Bn1.Forward(Conv1.Forward(x)[0])[0])[0];
            x = DeZero.NET.Functions.ReLU.Invoke(Bn2.Forward(Conv2.Forward(x)[0])[0])[0];
            x = Bn3.Forward(Conv3.Forward(x)[0])[0];

            if (ConvSkip != null)
            {
                residual = BnSkip.Forward(ConvSkip.Forward(residual)[0])[0];
            }

            x = DeZero.NET.Functions.Add.Invoke(x, residual).Item1[0];
            x = DeZero.NET.Functions.ReLU.Invoke(x)[0];

            return [x];
        }
    }
}
