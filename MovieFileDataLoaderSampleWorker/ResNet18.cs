using DeZero.NET;
using DeZero.NET.Layers.Convolution;
using DeZero.NET.Layers.Linear;
using DeZero.NET.Layers.Normalization;
using DeZero.NET.Models;

namespace MovieFileDataLoaderSampleWorker
{
    public class ResNet18 : Model
    {
        private Conv2d Conv1 { get; set; }
        private BatchNorm Bn1 { get; set; }
        private List<BasicBlock> Layers { get; set; }
        private Linear Fc { get; set; }

        public ResNet18(int num_classes = 1000, Dtype dtype = null)
        {
            Conv1 = new Conv2d(64, kernel_size: 7, stride: 2, pad: 3, in_channels: 3, dtype: dtype);
            Bn1 = new BatchNorm(64);
            Layers = new List<BasicBlock>();

            // Layer1
            Layers.Add(new BasicBlock(64, 64, dtype: dtype));
            Layers.Add(new BasicBlock(64, 64, dtype: dtype));

            // Layer2
            Layers.Add(new BasicBlock(64, 128, stride: 2, dtype: dtype));
            Layers.Add(new BasicBlock(128, 128, dtype: dtype));

            // Layer3
            Layers.Add(new BasicBlock(128, 256, stride: 2, dtype: dtype));
            Layers.Add(new BasicBlock(256, 256, dtype: dtype));

            // Layer4
            Layers.Add(new BasicBlock(256, 512, stride: 2, dtype: dtype));
            Layers.Add(new BasicBlock(512, 512, dtype: dtype));

            Fc = new Linear(num_classes, in_size: 512, dtype: dtype);
        }

        public override Variable[] Forward(params Variable[] inputs)
        {
            var x = inputs[0];
            x = DeZero.NET.Functions.ReLU.Invoke(Bn1.Forward(Conv1.Forward(x)[0])[0])[0];
            x = DeZero.NET.Functions.MaxPooling.Invoke(x, kernelSize: (3, 3), stride: (2, 2), pad: (1, 1))[0];

            foreach (var layer in Layers)
            {
                x = layer.Forward(x)[0];
            }

            x = DeZero.NET.Functions.GlobalAveragePooling.Invoke(x)[0];
            x = Fc.Forward(x)[0];

            return new[] { x };
        }
    }
}
