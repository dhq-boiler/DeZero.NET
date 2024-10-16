using DeZero.NET;
using DeZero.NET.Layers;
using DeZero.NET.Models;

namespace MovieFileDataLoaderSampleWorker
{
    public class MobileNet : Model
    {
        private readonly List<Layer> _layers = new List<Layer>();

        public MobileNet(int num_classes = 1000, float width_mult = 1.0f)
        {
            // 初期レイヤー：3チャンネル（RGB）入力から32チャンネル出力
            AddConvBNReLU(3, 32, stride: 2, width_mult: width_mult);

            // Inverted Residual Blocksの追加
            AddInvertedResidual(32, 16, 1, 1, width_mult);
            AddInvertedResidual(16, 24, 2, 6, width_mult);
            AddInvertedResidual(24, 24, 1, 6, width_mult);
            AddInvertedResidual(24, 32, 2, 6, width_mult);
            AddInvertedResidual(32, 32, 1, 6, width_mult);
            AddInvertedResidual(32, 32, 1, 6, width_mult);
            AddInvertedResidual(32, 64, 2, 6, width_mult);
            AddInvertedResidual(64, 64, 1, 6, width_mult);
            AddInvertedResidual(64, 64, 1, 6, width_mult);
            AddInvertedResidual(64, 64, 1, 6, width_mult);
            AddInvertedResidual(64, 96, 1, 6, width_mult);
            AddInvertedResidual(96, 96, 1, 6, width_mult);
            AddInvertedResidual(96, 96, 1, 6, width_mult);
            AddInvertedResidual(96, 160, 2, 6, width_mult);
            AddInvertedResidual(160, 160, 1, 6, width_mult);
            AddInvertedResidual(160, 160, 1, 6, width_mult);
            AddInvertedResidual(160, 320, 1, 6, width_mult);

            // 最後の畳み込み層
            AddConvBNReLU(240, 960, kernel: 1, stride: 1, padding: 0, width_mult: width_mult);

            // グローバル平均プーリング
            _layers.Add(new DeZero.NET.Layers.Convolution.AveragePooling((7, 7), stride: 1, pad: 0));
        }

        private void AddConvBNReLU(int in_channels, int out_channels, int kernel = 3, int stride = 1, int padding = 1, float width_mult = 1.0f)
        {
            int oc = (int)(out_channels * width_mult);
            _layers.Add(new Conv2dMobileNet(oc, kernel, Dtype.float32, stride: stride, pad: padding, in_channels: in_channels));
            _layers.Add(new DeZero.NET.Layers.Normalization.BatchNorm(oc));
            _layers.Add(new DeZero.NET.Layers.Activation.ReLU());
        }

        private void AddInvertedResidual(int in_channels, int out_channels, int stride, int expand_ratio, float width_mult)
        {
            int ic = (int)(in_channels * width_mult);
            int oc = (int)(out_channels * width_mult);
            int hidden_dim = ic * expand_ratio;

            var layers = new List<Layer>();

            if (expand_ratio != 1)
            {
                layers.Add(new Conv2dMobileNet(hidden_dim, 1, Dtype.float32, in_channels: ic));
                layers.Add(new DeZero.NET.Layers.Normalization.BatchNorm(hidden_dim));
                layers.Add(new DeZero.NET.Layers.Activation.ReLU());
            }

            layers.Add(new Conv2dMobileNet(hidden_dim, 3, Dtype.float32, stride: stride, pad: 1, in_channels: hidden_dim));
            layers.Add(new DeZero.NET.Layers.Normalization.BatchNorm(hidden_dim));
            layers.Add(new DeZero.NET.Layers.Activation.ReLU());

            layers.Add(new Conv2dMobileNet(oc, 1, Dtype.float32, in_channels: hidden_dim));
            layers.Add(new DeZero.NET.Layers.Normalization.BatchNorm(oc));

            if (ic == oc && stride == 1)
            {
                _layers.Add(new InvertedResidualBlock(layers, ic));
            }
            else
            {
                _layers.AddRange(layers);
            }
        }

        public override Variable[] Forward(params Variable[] inputs)
        {
            var x = inputs[0];
            foreach (var layer in _layers.Zip(Enumerable.Range(0, _layers.Count)))
            {
                Console.WriteLine($"Before Layer {layer.Second}: {string.Join(", ", x.Shape.Dimensions)}");
                x = layer.First.Forward(x)[0];
                Console.WriteLine($"After Layer {layer.Second}: {string.Join(", ", x.Shape.Dimensions)}");
                Console.WriteLine($"Layer {layer.Second} type: {layer.First.GetType().Name}");
            }
            return new[] { x };
        }
    }

    public class InvertedResidualBlock : Layer
    {
        private readonly List<Layer> _layers;
        private readonly int _in_channels;

        public InvertedResidualBlock(List<Layer> layers, int in_channels)
        {
            _layers = layers;
            _in_channels = in_channels;
        }

        public override Variable[] Forward(params Variable[] inputs)
        {
            var x = inputs[0];
            var identity = x;

            foreach (var layer in _layers)
            {
                x = layer.Forward(x)[0];
            }

            x = DeZero.NET.Functions.Add.Invoke(x, identity).Item1[0];
            return new[] { x };
        }
    }
}
