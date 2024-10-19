using DeZero.NET;
using DeZero.NET.Functions;
using DeZero.NET.Models;

namespace MovieFileDataLoaderSampleWorker
{
    public class ResNet50 : Model
    {
        private DeZero.NET.Layers.Convolution.Conv2d Conv1 { get; set; }
        private DeZero.NET.Layers.Normalization.BatchNorm Bn1 { get; set; }
        private List<BottleneckBlock> Layers { get; set; }
        private DeZero.NET.Layers.Linear.Linear Fc { get; set; }

        public ResNet50(Dtype dtype = null)
        {
            Conv1 = new DeZero.NET.Layers.Convolution.Conv2d(64, kernel_size: 7, stride: 2, pad: 3, dtype: dtype, in_channels: 3);
            Bn1 = new DeZero.NET.Layers.Normalization.BatchNorm(64);
            Layers = new List<BottleneckBlock>();

            // Layer1
            Layers.Add(new BottleneckBlock(64, 64, 256, stride: 1, is_first_block: true, dtype: dtype));
            for (int i = 0; i < 2; i++)
                Layers.Add(new BottleneckBlock(256, 64, 256, dtype: dtype));

            // Layer2
            Layers.Add(new BottleneckBlock(256, 128, 512, stride: 2, dtype: dtype));
            for (int i = 0; i < 3; i++)
                Layers.Add(new BottleneckBlock(512, 128, 512, dtype: dtype));

            // Layer3
            Layers.Add(new BottleneckBlock(512, 256, 1024, stride: 2, dtype: dtype));
            for (int i = 0; i < 5; i++)
                Layers.Add(new BottleneckBlock(1024, 256, 1024, dtype: dtype));

            // Layer4
            Layers.Add(new BottleneckBlock(1024, 512, 2048, stride: 2, dtype: dtype));
            for (int i = 0; i < 2; i++)
                Layers.Add(new BottleneckBlock(2048, 512, 2048, dtype: dtype));

            Fc = new DeZero.NET.Layers.Linear.Linear(1000, in_size: 2048, dtype: dtype);
        }

        public override Variable[] Forward(params Variable[] inputs)
        {
            var x = inputs[0];
            x = ReLU.Invoke(Bn1.Forward(Conv1.Forward(x)[0])[0])[0];
            x = DeZero.NET.Functions.MaxPooling.Invoke(x, kernelSize: (3, 3), stride: (2, 2), pad: (1, 1))[0];

            foreach (var layer in Layers)
            {
                x = layer.Forward(x)[0];
            }

            x = DeZero.NET.Functions.GlobalAveragePooling.Invoke(x)[0];
            x = Fc.Forward(x)[0];

            return [x];
        }
    }
}
